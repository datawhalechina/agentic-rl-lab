"""从保存的 macro-step 起点采样 N 条长度为 H 的分支轨迹。

与第 8 篇完整轨迹 rollout 的两点区别：
1. 每条分支只向前走 H 轮，停在 macro_boundary 而不是终局；
2. 起点可以是一个保存的中间状态，环境通过重放动作历史恢复，
   续跑 prompt 使用保存的真实 token 前缀，保证严格前缀扩展。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import pytrio as trio

from data import GameExample
from environment import ALFWorldGroup, EnvironmentStep
from protocol import (
    build_next_prompt,
    build_prompt,
    environment_tool_content,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)
from states import MacroState


INVALID_ENV_ACTION = "__invalid_tool_call__"
MAX_SEQUENCE_TOKENS = 12_000


@dataclass(frozen=True)
class RolloutConfig:
    """保存环境、macro-step 和采样参数。"""

    branches: int = 4                 # N：每个起点的分支数（actor 的 group 大小）
    macro_rounds: int = 10            # H：一个 macro-step 的交互轮数
    max_episode_steps: int = 50       # T：单局交互上限
    max_trajectory_tokens: int = MAX_SEQUENCE_TOKENS
    max_assistant_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    include_admissible_actions: bool = True
    environment_asynchronous: bool = True

    def __post_init__(self) -> None:
        if self.branches < 1:
            raise ValueError("branches 必须大于等于 1")
        if self.macro_rounds < 1 or self.max_episode_steps < 1:
            raise ValueError("macro_rounds 和 max_episode_steps 必须大于等于 1")
        if self.max_episode_steps % self.macro_rounds != 0:
            raise ValueError("max_episode_steps 必须是 macro_rounds 的整数倍")
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
    """从某个起点出发的一条分支。"""

    example: GameExample
    start_state: MacroState
    group_id: str
    game_index: int
    group_index: int
    task: str
    initial_observation: str
    start_round: int
    current_observation: str
    current_admissible_actions: tuple[str, ...]
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    steps: list[StepRecord] = field(default_factory=list)
    done: bool = False
    won: bool = False
    truncated: bool = False
    stop_reason: str = ""
    # TEMPO 信号字段，由 tempo.py 拼装。
    endpoint_value: float = 0.0
    segment_reward: float = 0.0
    branch_return: float = 0.0
    advantage: float = 0.0
    progress_reported: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class MacroGroup:
    """同一个 macro-step 起点的 N 条分支。"""

    start_state: MacroState
    trajectories: list[Trajectory]


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
    """一个起点组的环境和 N 条分支。"""

    environment: ALFWorldGroup
    start_state: MacroState
    trajectories: list[Trajectory]


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
    """在轨迹 token 上限内计算本轮最多可采样的 token 数。"""
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
    """不同起点组可并发执行；组内一次提交 N 个动作。"""
    calls: list[tuple[int, ALFWorldGroup, list[str]]] = []
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


def _finish_trajectory(
    trajectory: Trajectory,
    environment_step: EnvironmentStep,
    *,
    prompt_fits: bool,
    config: RolloutConfig,
) -> None:
    """按终局 / T 上限 / H 边界 / token 预算的优先级结束一条分支。"""
    segment_rounds = len(trajectory.steps)
    absolute_round = trajectory.start_round + segment_rounds
    if environment_step.done:
        trajectory.done = True
        trajectory.truncated = not environment_step.won
        trajectory.stop_reason = "won" if environment_step.won else "environment_done"
    elif absolute_round >= config.max_episode_steps:
        trajectory.done = True
        trajectory.truncated = not environment_step.won
        trajectory.stop_reason = "max_steps"
    elif segment_rounds >= config.macro_rounds:
        trajectory.done = True
        trajectory.truncated = False
        trajectory.stop_reason = "macro_boundary"
    elif not prompt_fits:
        trajectory.done = True
        trajectory.truncated = True
        trajectory.stop_reason = "trajectory_too_long"


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
        step_index = trajectory.start_round + len(trajectory.steps) + 1
        call_id = (
            f"tempo-{trajectory.game_index}-{trajectory.group_index}-{step_index}"
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
        trajectory.won = trajectory.won or environment_step.won

        if not trajectory.done:
            _finish_trajectory(
                trajectory,
                environment_step,
                prompt_fits=prompt_fits,
                config=config,
            )


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


def _replay_environment(
    environment: ALFWorldGroup,
    state: MacroState,
    config: RolloutConfig,
) -> None:
    """reset 后按保存的动作历史重放，并校验恢复到的 observation。"""
    env_states = environment.reset()
    if len({item.observation for item in env_states}) != 1:
        raise RuntimeError("同组 ALFWorld 环境的初始 observation 不一致")

    last: list[EnvironmentStep] | None = None
    for action in state.action_history:
        last = environment.step([action] * config.branches)

    if last is None:
        restored = env_states[0].observation
    else:
        # 重放是组内锁步的，所有分支必然落在同一 observation 上。
        observations = {step.observation for step in last}
        if len(observations) != 1:
            raise RuntimeError("重放后分支 observation 不一致")
        restored = observations.pop()
    if restored != state.current_observation:
        raise RuntimeError(
            "重放后的 observation 与保存状态不一致，环境无法恢复: "
            f"expected={state.current_observation!r}, actual={restored!r}"
        )


def _trajectories_from_state(
    state: MacroState,
    game_index: int,
    config: RolloutConfig,
) -> list[Trajectory]:
    group_id = f"{game_index}:{state.example.id}"
    return [
        Trajectory(
            example=state.example,
            start_state=state,
            group_id=group_id,
            game_index=game_index,
            group_index=branch_index,
            task=state.task,
            initial_observation=state.initial_observation,
            start_round=state.round_index,
            current_observation=state.current_observation,
            current_admissible_actions=state.current_admissible_actions,
            messages=state.fork_messages(),
            next_prompt_tokens=(
                list(state.token_prefix) if state.token_prefix is not None else None
            ),
        )
        for branch_index in range(config.branches)
    ]


def _restored_runtime(
    state: MacroState,
    game_index: int,
    config: RolloutConfig,
) -> GroupRuntime:
    environment = ALFWorldGroup(
        state.example,
        config.branches,
        config.max_episode_steps,
        seed=state.environment_seed,
        asynchronous=config.environment_asynchronous,
    )
    try:
        _replay_environment(environment, state, config)
    except Exception:
        environment.close()
        raise
    return GroupRuntime(
        environment,
        state,
        _trajectories_from_state(state, game_index, config),
    )


def _fresh_runtime(
    example: GameExample,
    game_index: int,
    config: RolloutConfig,
    *,
    seed: int,
) -> GroupRuntime:
    environment = ALFWorldGroup(
        example,
        config.branches,
        config.max_episode_steps,
        seed=seed,
        asynchronous=config.environment_asynchronous,
    )
    try:
        env_states = environment.reset()
    except Exception:
        environment.close()
        raise
    state = env_states[0]
    messages = initial_messages(
        task=state.task,
        observation=state.observation,
        admissible_actions=state.admissible_actions,
        include_admissible_actions=config.include_admissible_actions,
    )
    macro_state = MacroState(
        example=example,
        task=state.task,
        initial_observation=state.observation,
        action_history=(),
        messages=messages,
        token_prefix=None,
        round_index=0,
        macro_index=0,
        environment_seed=seed,
        current_observation=state.observation,
        current_admissible_actions=state.admissible_actions,
    )
    return GroupRuntime(
        environment,
        macro_state,
        _trajectories_from_state(macro_state, game_index, config),
    )


def _initialize_runtimes(
    start_states: list[MacroState],
    fresh_examples: list[GameExample],
    config: RolloutConfig,
    *,
    seed_offset: int,
) -> list[GroupRuntime]:
    runtimes: list[GroupRuntime] = []
    try:
        for state in start_states:
            runtimes.append(_restored_runtime(state, len(runtimes), config))
        for example in fresh_examples:
            runtimes.append(
                _fresh_runtime(
                    example,
                    len(runtimes),
                    config,
                    seed=config.seed + seed_offset + len(runtimes),
                )
            )
    except Exception:
        for runtime in runtimes:
            runtime.environment.close()
        raise
    return runtimes


def rollout_macro_steps(
    sampling_client: Any,
    tokenizer: Any,
    start_states: list[MacroState],
    fresh_examples: list[GameExample],
    config: RolloutConfig,
    *,
    seed_offset: int = 0,
    progress_callback: Callable[[int], None] | None = None,
) -> list[MacroGroup]:
    """对每个起点恢复环境并采满一个 macro-step，返回按起点分组的分支。

    warm-up 阶段传入 macro_rounds == max_episode_steps 的 config 和
    全 fresh 游戏，即退化为第 8 篇的完整轨迹 rollout。
    """
    if not start_states and not fresh_examples:
        return []
    runtimes: list[GroupRuntime] = []
    try:
        runtimes = _initialize_runtimes(
            start_states,
            fresh_examples,
            config,
            seed_offset=seed_offset,
        )
        trajectories = [
            trajectory
            for runtime in runtimes
            for trajectory in runtime.trajectories
        ]

        for round_index in range(config.macro_rounds):
            if all(trajectory.done for trajectory in trajectories):
                break

            requests: list[SampleRequest] = []
            if round_index == 0:
                # 同一起点的 N 条分支共享 prompt，一次请求分叉出 N 个候选。
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
                            raise RuntimeError("同组分支的续跑 prompt 不一致")
                    targets = tuple(
                        (game_index, branch_index)
                        for branch_index in range(config.branches)
                    )
                    requests.append(
                        SampleRequest(
                            targets=targets,
                            prompt_tokens=prompt_tokens,
                            num_samples=config.branches,
                            max_tokens=max_tokens,
                            seed=config.seed + seed_offset + game_index,
                        )
                    )
            else:
                # 分叉后每条分支状态不同，各自请求一个候选；
                # 不同分支的请求之间并发。
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
                                + seed_offset
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
                trajectory.truncated = not trajectory.won
                trajectory.stop_reason = "max_steps"
            trajectory.segment_reward = 1.0 if trajectory.won else 0.0
        _report_finished(trajectories, progress_callback)
    finally:
        for runtime in runtimes:
            runtime.environment.close()

    return [
        MacroGroup(runtime.start_state, runtime.trajectories)
        for runtime in runtimes
    ]
