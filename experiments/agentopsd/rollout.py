"""采样并执行 text-only ALFWorld 的同游戏分组轨迹。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pytrio as trio

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
# 平台单条 sequence 的硬上限（采样、Teacher 打分、训练共用）。
MAX_SEQUENCE_TOKENS = 16_384
# 为 Teacher 独有的技能前缀预留的 token 余量，保证顶到 rollout 上限的
# student 轨迹替换前缀后仍能放进一次 Teacher 打分请求。
TEACHER_PREFIX_TOKEN_ALLOWANCE = 2_048
MAX_STUDENT_TRAJECTORY_TOKENS = MAX_SEQUENCE_TOKENS - TEACHER_PREFIX_TOKEN_ALLOWANCE


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
    max_turns: int = 50
    max_trajectory_tokens: int = MAX_SEQUENCE_TOKENS
    max_assistant_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    include_admissible_actions: bool = True
    environment_asynchronous: bool = True


@dataclass
class StepRecord:
    """一次 action 前后的完整训练和环境信息。"""

    index: int
    observation: str
    admissible_actions: tuple[str, ...]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    action_start: int
    action_end: int
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
    teacher_logprobs: list[float] = field(default_factory=list)
    teacher_evidence: float = 0.0
    turn_advantage: float = 0.0


@dataclass
class Trajectory:
    """一条独立环境轨迹。"""

    example: GameExample
    group_id: str
    game_index: int
    group_index: int
    task: str
    initial_observation: str
    policy_snapshot_id: str
    include_admissible_actions: bool
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
    sequence_advantage: float = 0.0
    initial_belief: float = 0.0
    student_initial_tokens: list[int] = field(default_factory=list)
    student_initial_length: int = 0
    full_student_tokens: list[int] = field(default_factory=list)
    turn_evidences: list[float] = field(default_factory=list)
    turn_credits: list[Any] = field(default_factory=list)
    skill_key: str = ""
    teacher_scored_tokens: int = 0
    progress_reported: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class RolloutBatch:
    """A rollout batch produced by one frozen policy snapshot."""

    trajectories: list[Trajectory]
    policy_snapshot_id: str


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
    max_turns: int,
    *,
    seed: int,
    asynchronous: bool,
) -> EnvironmentGroup:
    return ALFWorldGroup(
        example,
        group_size,
        max_turns,
        seed=seed,
        asynchronous=asynchronous,
    )


def prompt_for_trajectory(
    tokenizer: Any,
    trajectory: Trajectory,
) -> list[int]:
    """返回某条分支包含完整工具历史的真实 token 前缀。"""
    if trajectory.next_prompt_tokens is not None:
        prompt_tokens = trajectory.next_prompt_tokens
    else:
        prompt_tokens = build_prompt(tokenizer, trajectory.messages)
    if not trajectory.student_initial_tokens:
        trajectory.student_initial_tokens = list(prompt_tokens)
        trajectory.student_initial_length = len(prompt_tokens)
    return prompt_tokens


def _generation_budget(
    prompt_tokens: list[int],
    config: RolloutConfig,
) -> int:
    """在配置的完整轨迹上限内计算本轮最多可采样的 token 数。"""
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
        raise ValueError("采样 token 数量与 old logprob 数量不一致")
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
    for request, response in zip(requests, responses):
        sequences = list(response.sequences)
        if len(sequences) != request.num_samples:
            raise ValueError("采样返回数量与请求数量不一致")
        if len(request.targets) != request.num_samples:
            raise ValueError("采样目标数量与请求数量不一致")
        for target, sequence in zip(request.targets, sequences):
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
                action_start=len(generated.prompt_tokens),
                action_end=(
                    len(generated.prompt_tokens) + len(generated.completion_tokens)
                ),
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

        reached_limit = len(trajectory.steps) >= config.max_turns
        if environment_step.done or reached_limit:
            trajectory.done = True
            trajectory.truncated = not trajectory.won
            if trajectory.won:
                trajectory.stop_reason = "won"
            elif environment_step.done:
                trajectory.stop_reason = "environment_done"
            else:
                trajectory.stop_reason = "max_turns"
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
    policy_snapshot_id: str,
) -> list[GroupRuntime]:
    runtimes: list[GroupRuntime] = []
    try:
        for game_index, example in enumerate(examples):
            environment = environment_factory(
                example,
                config.group_size,
                config.max_turns,
                seed=config.seed + game_index,
                asynchronous=config.environment_asynchronous,
            )
            try:
                states = environment.reset()
                if len(states) != config.group_size:
                    raise ValueError("环境返回的分支数量与 group_size 不一致")
                if len({state.observation for state in states}) != 1:
                    raise RuntimeError("同组分支的初始 observation 不一致")
                if len({state.task for state in states}) != 1:
                    raise RuntimeError("同组分支的初始 task 不一致")
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
                    policy_snapshot_id=policy_snapshot_id,
                    include_admissible_actions=config.include_admissible_actions,
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


def assemble_trajectory_tokens(trajectory: Trajectory) -> list[int]:
    """按严格前缀关系拼接完整 Student 轨迹。"""

    if not trajectory.steps:
        return list(trajectory.student_initial_tokens)

    full_tokens = list(trajectory.student_initial_tokens)
    for step in trajectory.steps:
        if step.prompt_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError(f"第 {step.index} 轮 prompt 改写了已有轨迹")
        full_tokens.extend(step.prompt_tokens[len(full_tokens) :])
        if step.action_start != len(full_tokens):
            raise ValueError(f"第 {step.index} 轮 action_start 错位")
        full_tokens.extend(step.completion_tokens)
        if step.action_end != len(full_tokens):
            raise ValueError(f"第 {step.index} 轮 action_end 错位")

    if trajectory.next_prompt_tokens is not None:
        if trajectory.next_prompt_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError("最后一条环境消息改写了已有轨迹")
        full_tokens.extend(trajectory.next_prompt_tokens[len(full_tokens) :])
    if trajectory.student_initial_length != len(trajectory.student_initial_tokens):
        raise ValueError("Student 初始 prompt 长度记录不一致")
    return full_tokens


def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[GameExample],
    config: RolloutConfig,
    *,
    policy_snapshot_id: str | None = None,
    environment_factory: EnvironmentFactory | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutBatch:
    """Complete grouped Student trajectories without assigning advantages."""
    snapshot_id = str(
        policy_snapshot_id or getattr(sampling_client, "task_id", "")
    ).strip()
    if not snapshot_id:
        raise ValueError("rollout 需要明确的 policy snapshot ID")
    factory = environment_factory or _default_environment_factory
    runtimes: list[GroupRuntime] = []
    try:
        runtimes = _initialize_runtimes(
            examples,
            config,
            factory,
            snapshot_id,
        )
        trajectories = [
            trajectory
            for runtime in runtimes
            for trajectory in runtime.trajectories
        ]

        for round_index in range(config.max_turns):
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
                            raise RuntimeError("同组分支的首轮 prompt 不一致")
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
                trajectory.stop_reason = "max_turns"
            trajectory.invalid_action_count = sum(
                not step.valid_action for step in trajectory.steps
            )
            trajectory.reward = 1.0 if trajectory.won else 0.0
            trajectory.full_student_tokens = assemble_trajectory_tokens(trajectory)
        _report_finished(trajectories, progress_callback)
    finally:
        for runtime in runtimes:
            runtime.environment.close()

    return RolloutBatch(trajectories, snapshot_id)


def trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    """把轨迹转成适合 JSONL 落盘和人工 review 的结构。"""
    return {
        "game_id": trajectory.example.id,
        "game_file": str(trajectory.example.game_file),
        "split": trajectory.example.split,
        "task_type": trajectory.example.task_type,
        "group_id": trajectory.group_id,
        "group_index": trajectory.group_index,
        "policy_snapshot_id": trajectory.policy_snapshot_id,
        "skill_key": trajectory.skill_key,
        "task": trajectory.task,
        "initial_observation": trajectory.initial_observation,
        "messages": trajectory.messages,
        "won": trajectory.won,
        "truncated": trajectory.truncated,
        "stop_reason": trajectory.stop_reason,
        "invalid_action_count": trajectory.invalid_action_count,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "sequence_advantage": trajectory.sequence_advantage,
        "initial_belief": trajectory.initial_belief,
        "student_tokens": len(trajectory.full_student_tokens),
        "teacher_scored_tokens": trajectory.teacher_scored_tokens,
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
                "action_start": step.action_start,
                "action_end": step.action_end,
                "teacher_evidence": step.teacher_evidence,
                "turn_advantage": step.turn_advantage,
                "mean_old_logprob": (
                    sum(step.logprobs) / len(step.logprobs)
                ),
            }
            for step in trajectory.steps
        ],
    }
