"""采样并执行 text-only ALFWorld 的同游戏分组轨迹。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pytrio as trio

from advantages import AdvantageStats, assign_advantages
from data import GameExample
from environment import ALFWorldGroup, EnvironmentState, EnvironmentStep
from protocol import (
    build_next_prompt,
    build_prompt,
    environment_tool_content,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)


INVALID_ENV_ACTION = "__invalid_tool_call__"
INVALID_ACTION_PENALTY = 0.1
MAX_SEQUENCE_TOKENS = 12_000


class EnvironmentGroup(Protocol):
    """rollout 所需的最小环境接口，便于用 fake environment 做单元测试。"""

    def reset(self) -> list[EnvironmentState]: ...

    def step(self, actions: list[str]) -> list[EnvironmentStep]: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[..., EnvironmentGroup]


@dataclass(frozen=True)
class RolloutConfig:
    """保存环境和采样参数。"""

    group_size: int = 8
    max_steps: int = 50
    max_trajectory_tokens: int = MAX_SEQUENCE_TOKENS
    max_assistant_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    include_admissible_actions: bool = True
    environment_asynchronous: bool = True

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError("group_size 必须大于等于 1")
        if self.max_steps < 1:
            raise ValueError("max_steps 必须大于等于 1")
        if self.max_trajectory_tokens < 1 or self.max_assistant_tokens < 1:
            raise ValueError("token 上限必须大于等于 1")
        if self.temperature < 0.0:
            raise ValueError("temperature 不能为负数")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")


@dataclass
class StepRecord:
    """一次 action 前后的完整训练和环境信息。"""

    index: int
    observation: str
    admissible_actions: tuple[str, ...]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    assistant_text: str
    call_id: str
    action: str
    valid_format: bool
    admissible: bool
    valid_action: bool
    tool_response: str
    next_observation: str
    score: float
    done: bool
    won: bool


@dataclass
class Trajectory:
    """一条独立环境轨迹。"""

    example: GameExample
    group_id: str
    game_index: int
    group_index: int
    task: str
    initial_observation: str
    current_observation: str
    current_admissible_actions: tuple[str, ...]
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    steps: list[StepRecord] = field(default_factory=list)
    done: bool = False
    won: bool = False
    truncated: bool = False
    stop_reason: str = ""
    invalid_action_count: int = 0
    reward: float = 0.0
    advantage: float = 0.0
    progress_reported: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class RolloutBatch:
    """一批轨迹及其 advantage 分组统计。"""

    trajectories: list[Trajectory]
    advantage_stats: AdvantageStats


@dataclass(frozen=True)
class SampleRequest:
    """描述一个共享 prompt 或一条分叉轨迹的采样请求。"""

    targets: tuple[tuple[int, int], ...]
    prompt_tokens: list[int]
    num_samples: int
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class GeneratedAction:
    """从 sampler response 中读取的一次待执行动作。"""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str
    action: str
    valid_format: bool


@dataclass
class GroupRuntime:
    """一个游戏组的环境和 K 条分支。"""

    environment: EnvironmentGroup
    trajectories: list[Trajectory]


def _default_environment_factory(
    example: GameExample,
    group_size: int,
    max_steps: int,
    *,
    seed: int,
    asynchronous: bool,
) -> EnvironmentGroup:
    return ALFWorldGroup(
        example,
        group_size,
        max_steps,
        seed=seed,
        asynchronous=asynchronous,
    )


def prompt_for_trajectory(
    tokenizer: Any,
    trajectory: Trajectory,
) -> list[int]:
    """返回某条分支包含完整工具历史的真实 token 前缀。"""
    if trajectory.next_prompt_tokens is not None:
        return trajectory.next_prompt_tokens
    return build_prompt(tokenizer, trajectory.messages)


def _generation_budget(
    prompt_tokens: list[int],
    config: RolloutConfig,
) -> int:
    """在 16K 轨迹上限内计算本轮最多可采样的 token 数。"""
    return min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )


def _request_for_trajectory(
    tokenizer: Any,
    trajectory: Trajectory,
    target: tuple[int, int],
    config: RolloutConfig,
    *,
    seed: int,
) -> SampleRequest | None:
    prompt_tokens = prompt_for_trajectory(tokenizer, trajectory)
    max_tokens = _generation_budget(prompt_tokens, config)
    if max_tokens < 1:
        trajectory.done = True
        trajectory.truncated = True
        trajectory.stop_reason = "trajectory_too_long"
        return None
    return SampleRequest((target,), prompt_tokens, 1, max_tokens, seed)


async def _sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    tokenizer: Any,
    config: RolloutConfig,
) -> list[Any]:
    """并发提交不同 prompt；同一轨迹内部仍保持严格顺序。"""
    stop = stop_sequences(tokenizer)
    calls = [
        sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(request.prompt_tokens),
            num_samples=request.num_samples,
            sampling_params=trio.SamplingParams(
                max_tokens=request.max_tokens,
                seed=request.seed,
                stop=stop,
                temperature=config.temperature,
                top_p=config.top_p,
            ),
            return_text=True,
        )
        for request in requests
    ]
    return list(await asyncio.gather(*calls))


def _read_sequence(
    sequence: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
) -> GeneratedAction:
    tokens = [int(token) for token in sequence.tokens]
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise ValueError("采样 completion token 与 old logprob 长度不一致")
    text = sequence.text
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = parse_assistant(str(text))
    return GeneratedAction(
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        logprobs=logprobs,
        text=str(text),
        action=parsed.action if parsed.action is not None else INVALID_ENV_ACTION,
        valid_format=parsed.valid_format,
    )


def _consume_responses(
    requests: list[SampleRequest],
    responses: list[Any],
    tokenizer: Any,
) -> dict[tuple[int, int], GeneratedAction]:
    """将 response 中的候选按 (game, branch) 放回原位置。"""
    pending: dict[tuple[int, int], GeneratedAction] = {}
    for request, response in zip(requests, responses, strict=True):
        sequences = list(response.sequences)
        if len(sequences) != request.num_samples:
            raise ValueError(
                f"采样数量 {len(sequences)} != 请求数量 {request.num_samples}"
            )
        if len(request.targets) != request.num_samples:
            raise ValueError("SampleRequest targets 与 num_samples 不一致")
        for target, sequence in zip(request.targets, sequences, strict=True):
            pending[target] = _read_sequence(
                sequence,
                tokenizer,
                request.prompt_tokens,
            )
    return pending


def _step_environments(
    runtimes: list[GroupRuntime],
    pending: dict[tuple[int, int], GeneratedAction],
) -> dict[int, list[EnvironmentStep]]:
    """不同游戏组可并发执行；组内一次提交 K 个动作。"""
    calls: list[tuple[int, EnvironmentGroup, list[str]]] = []
    for game_index, runtime in enumerate(runtimes):
        targets = [
            (branch_index, pending.get((game_index, branch_index)))
            for branch_index in range(len(runtime.trajectories))
        ]
        if not any(generated is not None for _, generated in targets):
            continue
        actions = [
            generated.action if generated is not None else "look"
            for _, generated in targets
        ]
        calls.append((game_index, runtime.environment, actions))

    if not calls:
        return {}
    results: dict[int, list[EnvironmentStep]] = {}
    with ThreadPoolExecutor(max_workers=min(len(calls), 32)) as pool:
        futures = [
            (game_index, pool.submit(environment.step, actions))
            for game_index, environment, actions in calls
        ]
        for game_index, future in futures:
            results[game_index] = future.result()
    return results


def _apply_environment_results(
    runtimes: list[GroupRuntime],
    pending: dict[tuple[int, int], GeneratedAction],
    results: dict[int, list[EnvironmentStep]],
    tokenizer: Any,
    config: RolloutConfig,
) -> None:
    """写入 StepRecord，并更新各分支的下一状态。"""
    for (game_index, branch_index), generated in pending.items():
        trajectory = runtimes[game_index].trajectories[branch_index]
        environment_step = results[game_index][branch_index]
        valid_action = generated.valid_format and environment_step.admissible
        step_index = len(trajectory.steps) + 1
        call_id = (
            f"alfworld-{trajectory.game_index}-{trajectory.group_index}-{step_index}"
        )
        content = environment_tool_content(
            step=step_index,
            observation=environment_step.observation,
            admissible_actions=environment_step.admissible_actions,
            valid_format=generated.valid_format,
            admissible=environment_step.admissible,
            done=environment_step.done,
            won=environment_step.won,
            include_admissible_actions=config.include_admissible_actions,
        )
        next_tool_message = tool_message(call_id, content)
        next_prompt_tokens = build_next_prompt(
            tokenizer,
            list(trajectory.messages),
            generated.prompt_tokens,
            generated.completion_tokens,
            next_tool_message,
        )
        trajectory.messages.append(
            {"role": "assistant", "content": generated.text}
        )
        trajectory.messages.append(next_tool_message)
        prompt_fits = len(next_prompt_tokens) <= config.max_trajectory_tokens
        trajectory.next_prompt_tokens = next_prompt_tokens if prompt_fits else None

        trajectory.steps.append(
            StepRecord(
                index=step_index,
                observation=trajectory.current_observation,
                admissible_actions=trajectory.current_admissible_actions,
                prompt_tokens=generated.prompt_tokens,
                completion_tokens=generated.completion_tokens,
                logprobs=generated.logprobs,
                assistant_text=generated.text,
                call_id=call_id,
                action=generated.action,
                valid_format=generated.valid_format,
                admissible=environment_step.admissible,
                valid_action=valid_action,
                tool_response=content,
                next_observation=environment_step.observation,
                score=environment_step.score,
                done=environment_step.done,
                won=environment_step.won,
            )
        )
        trajectory.current_observation = environment_step.observation
        trajectory.current_admissible_actions = environment_step.admissible_actions
        trajectory.won = environment_step.won

        reached_limit = len(trajectory.steps) >= config.max_steps
        if environment_step.done or reached_limit:
            trajectory.done = True
            trajectory.truncated = not trajectory.won
            if trajectory.won:
                trajectory.stop_reason = "won"
            elif environment_step.done:
                trajectory.stop_reason = "environment_done"
            else:
                trajectory.stop_reason = "max_steps"
        elif not prompt_fits:
            trajectory.done = True
            trajectory.truncated = True
            trajectory.stop_reason = "trajectory_too_long"


def _report_finished(
    trajectories: list[Trajectory],
    callback: Callable[[int], None] | None,
) -> None:
    newly_finished = 0
    for trajectory in trajectories:
        if trajectory.done and not trajectory.progress_reported:
            trajectory.progress_reported = True
            newly_finished += 1
    if newly_finished and callback is not None:
        callback(newly_finished)


def _initialize_runtimes(
    examples: list[GameExample],
    config: RolloutConfig,
    environment_factory: EnvironmentFactory,
) -> list[GroupRuntime]:
    runtimes: list[GroupRuntime] = []
    try:
        for game_index, example in enumerate(examples):
            environment = environment_factory(
                example,
                config.group_size,
                config.max_steps,
                seed=config.seed + game_index,
                asynchronous=config.environment_asynchronous,
            )
            try:
                states = environment.reset()
                if len(states) != config.group_size:
                    raise ValueError("环境 reset 返回数量与 group_size 不一致")
                if len({state.observation for state in states}) != 1:
                    raise RuntimeError("同游戏 group 的初始 observation 不一致")
                if len({state.task for state in states}) != 1:
                    raise RuntimeError("同游戏 group 的初始 task 不一致")
            except Exception:
                environment.close()
                raise

            group_id = f"{game_index}:{example.id}"
            trajectories = [
                Trajectory(
                    example=example,
                    group_id=group_id,
                    game_index=game_index,
                    group_index=branch_index,
                    task=state.task,
                    initial_observation=state.observation,
                    current_observation=state.observation,
                    current_admissible_actions=state.admissible_actions,
                    messages=initial_messages(
                        task=state.task,
                        observation=state.observation,
                        admissible_actions=state.admissible_actions,
                        include_admissible_actions=config.include_admissible_actions,
                    ),
                )
                for branch_index, state in enumerate(states)
            ]
            runtimes.append(GroupRuntime(environment, trajectories))
    except Exception:
        for runtime in runtimes:
            runtime.environment.close()
        raise
    return runtimes


def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[GameExample],
    config: RolloutConfig,
    *,
    environment_factory: EnvironmentFactory | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """完成一批同游戏分组轨迹，再统一计算轨迹级 advantage。"""
    if not examples:
        empty_stats = AdvantageStats(0, 0)
        return RolloutBatch([], empty_stats)
    factory = environment_factory or _default_environment_factory
    runtimes: list[GroupRuntime] = []
    try:
        runtimes = _initialize_runtimes(examples, config, factory)
        trajectories = [
            trajectory
            for runtime in runtimes
            for trajectory in runtime.trajectories
        ]

        for round_index in range(config.max_steps):
            if all(trajectory.done for trajectory in trajectories):
                break

            requests: list[SampleRequest] = []
            if round_index == 0:
                # 同游戏 K 条根轨迹共享 prompt，一次请求分叉出 K 个候选。
                for game_index, runtime in enumerate(runtimes):
                    root = runtime.trajectories[0]
                    prompt_tokens = prompt_for_trajectory(tokenizer, root)
                    max_tokens = _generation_budget(prompt_tokens, config)
                    if max_tokens < 1:
                        for trajectory in runtime.trajectories:
                            trajectory.done = True
                            trajectory.truncated = True
                            trajectory.stop_reason = "trajectory_too_long"
                        continue
                    for trajectory in runtime.trajectories[1:]:
                        if prompt_for_trajectory(tokenizer, trajectory) != prompt_tokens:
                            raise RuntimeError("同游戏 group 的首轮 prompt 不一致")
                    targets = tuple(
                        (game_index, branch_index)
                        for branch_index in range(config.group_size)
                    )
                    requests.append(
                        SampleRequest(
                            targets=targets,
                            prompt_tokens=prompt_tokens,
                            num_samples=config.group_size,
                            max_tokens=max_tokens,
                            seed=config.seed + game_index,
                        )
                    )
            else:
                # 分叉后每条轨迹状态不同，各自请求一个候选；
                # 不同轨迹的请求之间并发。
                for game_index, runtime in enumerate(runtimes):
                    for branch_index, trajectory in enumerate(runtime.trajectories):
                        if trajectory.done:
                            continue
                        request = _request_for_trajectory(
                            tokenizer,
                            trajectory,
                            (game_index, branch_index),
                            config,
                            seed=(
                                config.seed
                                + game_index * 1_000_000
                                + branch_index * 10_000
                                + round_index
                            ),
                        )
                        if request is not None:
                            requests.append(request)

            _report_finished(trajectories, progress_callback)
            if not requests:
                break
            responses = asyncio.run(
                _sample_requests_async(sampling_client, requests, tokenizer, config)
            )
            pending = _consume_responses(requests, responses, tokenizer)
            environment_results = _step_environments(runtimes, pending)
            _apply_environment_results(
                runtimes,
                pending,
                environment_results,
                tokenizer,
                config,
            )
            _report_finished(trajectories, progress_callback)

        for trajectory in trajectories:
            if not trajectory.done:
                trajectory.done = True
                trajectory.truncated = True
                trajectory.stop_reason = "max_steps"
            trajectory.invalid_action_count = sum(
                not step.valid_action for step in trajectory.steps
            )
            trajectory.reward = (
                (1.0 if trajectory.won else 0.0)
                - INVALID_ACTION_PENALTY * trajectory.invalid_action_count
            )
        _report_finished(trajectories, progress_callback)
    finally:
        for runtime in runtimes:
            runtime.environment.close()

    advantage_stats = assign_advantages(trajectories)
    return RolloutBatch(trajectories, advantage_stats)


def trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    """把轨迹转成适合 JSONL 落盘和人工 review 的结构。"""
    return {
        "game_id": trajectory.example.id,
        "game_file": str(trajectory.example.game_file),
        "split": trajectory.example.split,
        "task_type": trajectory.example.task_type,
        "group_id": trajectory.group_id,
        "group_index": trajectory.group_index,
        "task": trajectory.task,
        "initial_observation": trajectory.initial_observation,
        "messages": trajectory.messages,
        "won": trajectory.won,
        "truncated": trajectory.truncated,
        "stop_reason": trajectory.stop_reason,
        "invalid_action_count": trajectory.invalid_action_count,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "steps": [
            {
                "step": step.index,
                "observation": step.observation,
                "admissible_actions": list(step.admissible_actions),
                "assistant_text": step.assistant_text,
                "call_id": step.call_id,
                "action": step.action,
                "valid_format": step.valid_format,
                "admissible": step.admissible,
                "valid_action": step.valid_action,
                "tool_response": step.tool_response,
                "next_observation": step.next_observation,
                "environment_score": step.score,
                "done": step.done,
                "won": step.won,
                "prompt_tokens": len(step.prompt_tokens),
                "completion_tokens": len(step.completion_tokens),
                "mean_old_logprob": (
                    sum(step.logprobs) / len(step.logprobs)
                    if step.logprobs
                    else 0.0
                ),
            }
            for step in trajectory.steps
        ],
    }
