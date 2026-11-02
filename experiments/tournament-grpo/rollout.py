"""多轮采样 Tournament-GRPO 轨迹：工具 observation 直接接在 assistant 文本后面。

Adapted from experiments/search-r1 and PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pytrio as trio

from data import ResearchExample
from protocol import MAX_PROMPT_LENGTH, TOOL_STOP, ToolCall, build_prompt, parse_tool_call, tool_observation
from search import ResearchTools


@dataclass(frozen=True)
class RolloutConfig:
    group_size: int = 8
    max_response_tokens: int = 18000  # 和作者一样，response 长度包含 observation
    max_model_tokens: int = 25000
    max_assistant_turns: int = 6
    max_tool_calls: int = 4  # 搜索和浏览合计；最后一次工具返回时提示模型收尾
    tool_concurrency: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42


@dataclass
class Trajectory:
    example: ResearchExample
    question_index: int
    group_index: int
    prompt_tokens: list[int]
    response_tokens: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)  # 1 = 模型生成，0 = 工具 observation
    logprobs: list[float] = field(default_factory=list)  # 采样端 logprob，observation 记 0
    assistant_turns: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_errors: int = 0
    finish_reason: str = ""  # 非空表示轨迹结束：no_tool_call / max_turns / max_tool_calls / max_response_tokens
    reward: float = 0.0
    advantage: float = 0.0

    @property
    def done(self) -> bool:
        return bool(self.finish_reason)


def consume_assistant(trajectory: Trajectory, sequence: Any, tokenizer: Any, config: RolloutConfig) -> ToolCall | None:
    """记录本轮生成的 token 和 logprob，返回要执行的工具调用；轨迹结束时返回 None。"""
    # PyTRIO 的 text 会去掉停止符，但 tokens 和 logprobs 保留了 </call_tool>，和作者 include_stop_str_in_output 一致。
    trajectory.response_tokens.extend(sequence.tokens)
    trajectory.response_mask.extend([1] * len(sequence.tokens))
    trajectory.logprobs.extend(sequence.logprobs)
    trajectory.assistant_turns += 1
    # 先检查长度和轮数上限：达到上限时即使生成了工具调用也不再执行。
    if len(trajectory.response_tokens) >= config.max_response_tokens:
        trajectory.finish_reason = "max_response_tokens"
        return None
    if trajectory.assistant_turns >= config.max_assistant_turns:
        trajectory.finish_reason = "max_turns"
        return None
    call = parse_tool_call(tokenizer.decode(sequence.tokens))
    if call is None:
        trajectory.finish_reason = "no_tool_call"
        return None
    if len(trajectory.tool_calls) >= config.max_tool_calls:
        trajectory.finish_reason = "max_tool_calls"
        return None
    return call


def append_observation(trajectory: Trajectory, text: str, tokenizer: Any, config: RolloutConfig) -> None:
    """observation 和预算提示均不参与训练（mask 0）；放不下时直接结束。"""
    observation = tool_observation(text, tools_exhausted=len(trajectory.tool_calls) >= config.max_tool_calls)
    tokens = tokenizer.encode(observation, add_special_tokens=False)
    if len(trajectory.response_tokens) + len(tokens) >= config.max_response_tokens:
        trajectory.finish_reason = "max_response_tokens"
        return
    trajectory.response_tokens.extend(tokens)
    trajectory.response_mask.extend([0] * len(tokens))
    trajectory.logprobs.extend([0.0] * len(tokens))


def token_budget(context_length: int, config: RolloutConfig) -> int:
    """作者的每轮生成上限：response 余量、prompt+response 余量和模型上下文余量取最小。"""
    return min(
        config.max_response_tokens,
        MAX_PROMPT_LENGTH + config.max_response_tokens - context_length,
        config.max_model_tokens - context_length,
    )


def sample_turn(sampling_client: Any, tokens: list[int], config: RolloutConfig, num_samples: int, seed: int) -> Any:
    """接着当前上下文采样下一段 assistant 文本，遇到 </call_tool> 停止。"""
    return sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(tokens),
        num_samples=num_samples,
        sampling_params=trio.SamplingParams(
            max_tokens=token_budget(len(tokens), config), stop=[TOOL_STOP],
            temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, seed=seed,
        ),
        return_text=False,
    )


async def rollout_batch_async(
    sampling_client: Any,
    tokenizer: Any,
    tools: ResearchTools,
    examples: list[ResearchExample],
    config: RolloutConfig,
    on_finished: Callable[[int], None],
) -> list[Trajectory]:
    """第一轮对共享 prompt 一次采样 group_size 条，之后每条轨迹各自继续。

    on_finished(n) 在每轮结束时报告新完成的轨迹数，供进度条使用。
    """
    prompts = [build_prompt(tokenizer, example.query) for example in examples]
    responses = await asyncio.gather(*[
        sample_turn(sampling_client, prompt, config, config.group_size, config.seed + index)
        for index, prompt in enumerate(prompts)
    ])
    trajectories = [
        Trajectory(example, question_index, group_index, prompt)
        for question_index, (example, prompt) in enumerate(zip(examples, prompts, strict=True))
        for group_index in range(config.group_size)
    ]
    # sequences[i] 是第 i 条轨迹本轮新生成的内容；已结束的轨迹为 None。
    sequences = [sequence for response in responses for sequence in response.sequences]
    finished = 0
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=config.tool_concurrency) as pool:
        for turn in range(1, config.max_assistant_turns + 1):
            # 1. 记录本轮生成，收集需要执行的工具调用。
            pending = []
            for trajectory, sequence in zip(trajectories, sequences, strict=True):
                if sequence is None:
                    continue
                call = consume_assistant(trajectory, sequence, tokenizer, config)
                if call is not None:
                    pending.append((trajectory, call))

            # 2. 并发执行工具，把 observation 接到各自的上下文后面。
            results = await asyncio.gather(*[loop.run_in_executor(pool, tools.execute, call) for _, call in pending])
            for (trajectory, call), result in zip(pending, results, strict=True):
                trajectory.tool_calls.append(call)
                trajectory.tool_errors += int(not result.ok)
                append_observation(trajectory, result.text, tokenizer, config)

            done = sum(t.done for t in trajectories)
            on_finished(done - finished)
            finished = done

            # 3. 未结束的轨迹各自采样下一轮。
            active = [index for index, trajectory in enumerate(trajectories) if not trajectory.done]
            if not active:
                break
            responses = await asyncio.gather(*[
                sample_turn(sampling_client, trajectories[index].prompt_tokens + trajectories[index].response_tokens,
                            config, 1, config.seed + index + turn * 10000)
                for index in active
            ])
            sequences = [None] * len(trajectories)
            for index, response in zip(active, responses, strict=True):
                sequences[index] = response.sequences[0]

    # 每轮预算按 prompt + response 总长计算，prompt 不足 4096 时 response 可能越过上限；
    # 和作者一样，最后统一截断到 max_response_tokens。
    for trajectory in trajectories:
        trajectory.response_tokens = trajectory.response_tokens[:config.max_response_tokens]
        trajectory.response_mask = trajectory.response_mask[:config.max_response_tokens]
        trajectory.logprobs = trajectory.logprobs[:config.max_response_tokens]
    return trajectories


def rollout_batch(
    sampling_client: Any, tokenizer: Any, tools: ResearchTools,
    examples: list[ResearchExample], config: RolloutConfig, on_finished: Callable[[int], None],
) -> list[Trajectory]:
    """同步入口，和其他章节的训练循环保持一致。"""
    return asyncio.run(rollout_batch_async(sampling_client, tokenizer, tools, examples, config, on_finished))


def trajectory_record(trajectory: Trajectory, tokenizer: Any, **extra: Any) -> dict[str, Any]:
    """保存一条完整轨迹，附解码后的 response 文本，方便人工复盘。"""
    return {
        **asdict(trajectory), **extra,
        "response_text": tokenizer.decode(trajectory.response_tokens, skip_special_tokens=True),
    }
