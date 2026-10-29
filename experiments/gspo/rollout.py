from __future__ import annotations
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any
import pytrio as trio
from data import MathExample
from reward import RewardResult, score_answer


QUESTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
ADVANTAGE_EPSILON = 1e-8


@dataclass(frozen=True)
class RolloutConfig:
    """采样与 group 构造参数（GSPO 无候选倍率 / overlong 字段）。"""

    group_size: int = 8
    max_prompt_tokens: int = 512
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42

    def validate(self) -> None:
        """在发起远端采样前检查配置。"""
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.max_prompt_tokens < 1 or self.max_tokens < 1:
            raise ValueError("max_prompt_tokens and max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass
class RolloutSample:
    """一条 completion 及其训练所需的旧策略信息。

    ``sampling_logprobs`` 是 sampler 对每个 completion token 返回的 logprob，
    GSPO 的序列级重要性比率以它为旧策略端点（见 loss.py）。
    """

    example: MathExample
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    text: str
    reward_result: RewardResult
    advantage: float = 0.0


@dataclass
class RolloutGroup:
    """同一道题的一组 completion（一个 GSPO 组）。"""

    example: MathExample
    prompt_tokens: list[int]
    samples: list[RolloutSample]
    expected_group_size: int

    @property
    def correct_count(self) -> int:
        return sum(int(sample.reward_result.correct) for sample in self.samples)

    @property
    def usable(self) -> bool:
        """完整返回且每条 completion 都有 token，才允许进入训练。"""
        return (
            len(self.samples) == self.expected_group_size
            and all(sample.completion_tokens for sample in self.samples)
        )

    @property
    def degenerate(self) -> bool:
        """组内 reward 全相同的退化组：advantage 全 0，对梯度没有贡献。

        GSPO 论文未使用 Dynamic Sampling，退化组不补采。训练侧可跳过
        它们的远端计算，但会在 loss 的原始 batch 分母中保留其零目标。
        """
        rewards = {sample.reward_result.shaped_reward for sample in self.samples}
        return len(rewards) <= 1


@dataclass(frozen=True)
class RolloutBatch:
    """一个训练 step 的 rollout 结果。

    ``train_groups`` 包含退化组在内的全部采样组，是否剔除由调用方决定。
    """

    requested_groups: int
    train_groups: list[RolloutGroup]

    @property
    def effective_groups(self) -> list[RolloutGroup]:
        """完整可用且非退化的组，即实际能产生梯度的组。"""
        return [
            group
            for group in self.train_groups
            if group.usable and not group.degenerate
        ]

    @property
    def effective_group_ratio(self) -> float:
        return len(self.effective_groups) / max(len(self.train_groups), 1)


def build_prompt_tokens(tokenizer: Any, question: str) -> list[int]:
    """用模型 chat template 构造单轮数学题 prompt（模板与后缀同 experiments/dapo）。

    长度上限不在此处检查，由 ``sample_group_async`` 按
    ``RolloutConfig.max_prompt_tokens`` 校验。
    """
    content = question.strip() + QUESTION_SUFFIX
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not tokens:
        raise ValueError("prompt tokens are empty")
    return tokens


def stop_sequences(tokenizer: Any) -> list[str]:
    """返回 chat 终止字符串。"""
    return [tokenizer.eos_token] if tokenizer.eos_token else ["<|im_end|>"]


def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """读取 PyTRIO completion，并严格校验 token/logprob 对齐。"""
    tokens = [int(token) for token in sequence.tokens]
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise ValueError(
            f"completion token/logprob mismatch: {len(tokens)} != {len(logprobs)}"
        )
    text = sequence.text
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=False)
    return tokens, logprobs, str(text)


def assign_group_advantages(
    group: RolloutGroup,
    *,
    epsilon: float = ADVANTAGE_EPSILON,
) -> None:
    """在组内对 reward 做标准化，得到每条序列共享的 advantage 标量。

    advantage = (r - mean) / (std + epsilon)，其中 std 使用 N-1 分母的
    样本标准差（对齐 verl 的 torch.std 默认行为）。组内 reward 全相同时
    std 为 0，整组 advantage 置 0，即退化组。
    """
    if not group.samples:
        return
    rewards = [sample.reward_result.shaped_reward for sample in group.samples]
    reward_mean = sum(rewards) / len(rewards)
    if len(rewards) == 1:
        reward_std = 0.0
    else:
        variance = sum(
            (reward - reward_mean) ** 2 for reward in rewards
        ) / (len(rewards) - 1)
        reward_std = math.sqrt(variance)
    if reward_std == 0.0:
        for sample in group.samples:
            sample.advantage = 0.0
        return
    denominator = reward_std + epsilon
    for sample in group.samples:
        sample.advantage = (
            sample.reward_result.shaped_reward - reward_mean
        ) / denominator


async def sample_group_async(
    sampling_client: Any,
    tokenizer: Any,
    example: MathExample,
    config: RolloutConfig,
    *,
    group_index: int,
) -> RolloutGroup:
    """为一道题采样一个完整 group 并计算 reward/advantage。"""
    prompt_tokens = build_prompt_tokens(tokenizer, example.question)
    if len(prompt_tokens) > config.max_prompt_tokens:
        raise ValueError(
            f"prompt has {len(prompt_tokens)} tokens, "
            f"exceeding {config.max_prompt_tokens}"
        )
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=config.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=config.max_tokens,
            seed=config.seed + group_index,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=stop_sequences(tokenizer),
        ),
        return_text=True,
    )

    samples: list[RolloutSample] = []
    for sequence in response.sequences:
        tokens, sampling_logprobs, text = read_sequence(sequence, tokenizer)
        reward = score_answer(
            text,
            example.answer,
            completion_tokens=len(tokens),
        )
        samples.append(
            RolloutSample(
                example=example,
                completion_tokens=tokens,
                sampling_logprobs=sampling_logprobs,
                text=text,
                reward_result=reward,
            )
        )

    group = RolloutGroup(
        example=example,
        prompt_tokens=prompt_tokens,
        samples=samples,
        expected_group_size=config.group_size,
    )
    assign_group_advantages(group)
    return group


async def collect_rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[MathExample],
    config: RolloutConfig,
    *,
    concurrency: int,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """对给定题目一次性并发采齐所有组，不做 Dynamic Sampling 补采。

    GSPO 论文未使用 Dynamic Sampling：每道题只采一轮 ``group_size`` 条，
    退化组（组内 reward 全同）保留在返回的 batch 里；训练侧跳过其远端
    计算，同时保留原始 batch 分母。
    并发度由调用方传入并用 semaphore 限流，返回顺序与输入题目顺序一致。
    """
    config.validate()
    if not examples:
        raise ValueError("examples must not be empty")

    semaphore = asyncio.Semaphore(concurrency)

    async def sample_one(group_index: int, example: MathExample) -> RolloutGroup:
        async with semaphore:
            group = await sample_group_async(
                sampling_client,
                tokenizer,
                example,
                config,
                group_index=group_index,
            )
            if progress_callback is not None:
                progress_callback(1)
            return group

    groups = list(
        await asyncio.gather(
            *(
                sample_one(group_index, example)
                for group_index, example in enumerate(examples)
            )
        )
    )
    return RolloutBatch(requested_groups=len(examples), train_groups=groups)
