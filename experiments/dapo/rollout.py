"""GRPO/DAPO 共用的 group rollout 与 Dynamic Sampling。"""

from __future__ import annotations
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Literal
import pytrio as trio
from data import MathExample
from reward import RewardResult, score_answer


Algorithm = Literal["grpo", "dapo"]
QUESTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


@dataclass(frozen=True)
class RolloutConfig:
    """采样与 group 构造参数。"""

    max_prompt_tokens: int
    max_tokens: int
    overlong_cache: int
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    concurrency: int = 16
    seed: int = 42
    advantage_epsilon: float = 1e-8

    def validate(self) -> None:
        """在发起远端采样前检查配置。"""
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")
        if self.max_prompt_tokens < 1 or self.max_tokens < 1:
            raise ValueError("max_prompt_tokens and max_tokens must be >= 1")
        if not 1 <= self.overlong_cache <= self.max_tokens:
            raise ValueError("overlong_cache must be in [1, max_tokens]")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be > 0")


@dataclass
class RolloutSample:
    """一条 completion 及其训练所需的旧策略信息。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: RewardResult
    stop_reason: str | None
    advantage: float = 0.0


@dataclass
class RolloutGroup:
    """同一道题的一组 completion。"""

    example: MathExample
    prompt_tokens: list[int]
    samples: list[RolloutSample]
    expected_group_size: int
    candidate_index: int

    @property
    def correct_count(self) -> int:
        return sum(int(sample.reward.correct) for sample in self.samples)

    @property
    def usable(self) -> bool:
        """完整返回且每条 completion 都有 token，才允许进入训练。"""
        return (
            len(self.samples) == self.expected_group_size
            and all(sample.tokens for sample in self.samples)
        )


@dataclass(frozen=True)
class RolloutBatch:
    """一个训练 step 的候选组和最终训练组。"""

    algorithm: Algorithm
    requested_groups: int
    candidate_groups: list[RolloutGroup]
    train_groups: list[RolloutGroup]

    @property
    def effective_group_ratio(self) -> float:
        return len(self.train_groups) / max(len(self.candidate_groups), 1)

    @property
    def effective_fill_ratio(self) -> float:
        """实际有效组数量占目标训练组数量的比例。"""
        return len(self.train_groups) / max(self.requested_groups, 1)

    @property
    def oversample_ratio(self) -> float:
        return len(self.candidate_groups) / max(self.requested_groups, 1)


def build_prompt_tokens(
    tokenizer: Any,
    question: str,
    *,
    max_prompt_tokens: int,
) -> list[int]:
    """用模型 chat template 构造单轮数学题 prompt。"""
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
    if len(tokens) > max_prompt_tokens:
        raise ValueError(
            f"prompt has {len(tokens)} tokens, exceeding {max_prompt_tokens}"
        )
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
    epsilon: float,
) -> None:
    """用 shaped reward 在完整 group 内计算标准化 advantage。"""
    if not group.samples:
        return
    rewards = [sample.reward.shaped_reward for sample in group.samples]
    reward_mean = sum(rewards) / len(rewards)
    if len(rewards) == 1:
        reward_std = 0.0
    else:
        # 对齐 verl 的 torch.std 默认行为：使用 N-1 分母的样本标准差。
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
        sample.advantage = (sample.reward.shaped_reward - reward_mean) / denominator


def is_effective_group(group: RolloutGroup) -> bool:
    """Dynamic Sampling 只按原始正确性判断是否为非退化组。"""
    return (
        group.usable
        and 0 < group.correct_count < group.expected_group_size
    )


async def sample_group_async(
    sampling_client: Any,
    tokenizer: Any,
    example: MathExample,
    config: RolloutConfig,
    *,
    candidate_index: int,
    enable_overlong: bool,
) -> RolloutGroup:
    """为一道题采样一个完整 group 并计算 reward/advantage。"""
    prompt_tokens = build_prompt_tokens(
        tokenizer,
        example.question,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    response = await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=config.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=config.max_tokens,
            seed=config.seed + candidate_index,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=stop_sequences(tokenizer),
        ),
        return_text=True,
    )

    samples: list[RolloutSample] = []
    for sequence in response.sequences:
        tokens, logprobs, text = read_sequence(sequence, tokenizer)
        reward = score_answer(
            text,
            example.answer,
            completion_tokens=len(tokens),
            max_tokens=config.max_tokens,
            overlong_cache=config.overlong_cache,
            enable_overlong=enable_overlong,
        )
        stop_reason = getattr(sequence, "stop_reason", None)
        samples.append(
            RolloutSample(
                tokens=tokens,
                logprobs=logprobs,
                text=text,
                reward=reward,
                stop_reason=str(stop_reason) if stop_reason is not None else None,
            )
        )

    group = RolloutGroup(
        example=example,
        prompt_tokens=prompt_tokens,
        samples=samples,
        expected_group_size=config.group_size,
        candidate_index=candidate_index,
    )
    assign_group_advantages(group, epsilon=config.advantage_epsilon)
    return group


async def sample_groups_async(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[MathExample],
    config: RolloutConfig,
    *,
    candidate_start: int,
    enable_overlong: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> list[RolloutGroup]:
    """在设定并发度内采样多道题，并保持输入顺序。"""
    semaphore = asyncio.Semaphore(config.concurrency)

    async def sample_one(offset: int, example: MathExample) -> RolloutGroup:
        async with semaphore:
            group = await sample_group_async(
                sampling_client,
                tokenizer,
                example,
                config,
                candidate_index=candidate_start + offset,
                enable_overlong=enable_overlong,
            )
            if progress_callback is not None:
                progress_callback(1)
            return group

    return list(
        await asyncio.gather(
            *(
                sample_one(offset, example)
                for offset, example in enumerate(examples)
            )
        )
    )


async def collect_rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    take_examples: Callable[[int], list[MathExample]],
    *,
    algorithm: Algorithm,
    requested_groups: int,
    config: RolloutConfig,
    max_candidate_groups: int,
    progress_callback: Callable[[int], None] | None = None,
    progress_total_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """执行固定 GRPO batch，或在候选上限内补采 DAPO 有效组。"""
    config.validate()
    if algorithm not in {"grpo", "dapo"}:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if requested_groups < 1:
        raise ValueError("requested_groups must be >= 1")
    if max_candidate_groups < requested_groups:
        raise ValueError("max_candidate_groups must be >= requested_groups")

    candidate_groups: list[RolloutGroup] = []
    train_groups: list[RolloutGroup] = []
    enable_overlong = algorithm == "dapo"

    if algorithm == "grpo":
        examples = take_examples(requested_groups)
        if progress_total_callback is not None:
            progress_total_callback(len(examples))
        candidate_groups = await sample_groups_async(
            sampling_client,
            tokenizer,
            examples,
            config,
            candidate_start=0,
            enable_overlong=False,
            progress_callback=progress_callback,
        )
        train_groups = [
            group for group in candidate_groups if is_effective_group(group)
        ]
    else:
        while len(train_groups) < requested_groups:
            available = max_candidate_groups - len(candidate_groups)
            if available <= 0:
                # 候选预算耗尽后返回已经收集到的有效组：部分 batch 仍可训练，
                # 一个有效组都没有时由训练循环跳过本 step 的参数更新。
                break
            round_size = min(requested_groups - len(train_groups), available)
            examples = take_examples(round_size)
            if progress_total_callback is not None:
                progress_total_callback(len(examples))
            new_groups = await sample_groups_async(
                sampling_client,
                tokenizer,
                examples,
                config,
                candidate_start=len(candidate_groups),
                enable_overlong=enable_overlong,
                progress_callback=progress_callback,
            )
            candidate_groups.extend(new_groups)
            train_groups.extend(
                group for group in new_groups if is_effective_group(group)
            )

    return RolloutBatch(
        algorithm=algorithm,
        requested_groups=requested_groups,
        candidate_groups=candidate_groups,
        train_groups=train_groups,
    )
