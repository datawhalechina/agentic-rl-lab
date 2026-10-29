"""构造 Teacher 专用技能提示，并对齐整条轨迹的对数概率。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import pytrio as trio

from protocol import build_prompt, teacher_initial_messages
from rollout import Trajectory
from skills import SkillProvider, SkillSelection


@dataclass(frozen=True)
class TeacherTrajectoryScore:
    """一条 Teacher 请求的打分结果。"""

    group_id: str
    group_index: int
    policy_snapshot_id: str
    skill_key: str
    teacher_input_tokens: int
    action_tokens: int
    evidences: tuple[float, ...]


@dataclass(frozen=True)
class TeacherBatchStats:
    """一次更新中的 Teacher 请求与 token 统计。"""

    requests: int
    teacher_input_tokens: int
    action_tokens: int
    elapsed_seconds: float


def build_teacher_trajectory_tokens(
    tokenizer: Any,
    trajectory: Trajectory,
    selection: SkillSelection,
) -> tuple[list[int], int]:
    """构造带技能的 Teacher 轨迹 token。"""

    initial_actions = trajectory.steps[0].admissible_actions
    teacher_messages = teacher_initial_messages(
        skill_text=selection.text,
        task=trajectory.task,
        observation=trajectory.initial_observation,
        admissible_actions=initial_actions,
        include_admissible_actions=trajectory.include_admissible_actions,
    )
    teacher_initial_tokens = build_prompt(tokenizer, teacher_messages)
    student_suffix = trajectory.full_student_tokens[
        trajectory.student_initial_length :
    ]
    teacher_full_tokens = [*teacher_initial_tokens, *student_suffix]
    return teacher_full_tokens, len(teacher_initial_tokens)


def _score_turns(
    trajectory: Trajectory,
    teacher_logprobs: Sequence[float | None],
    teacher_initial_length: int,
) -> tuple[tuple[float, ...], int]:
    """计算每一轮的 Teacher 证据。"""

    evidences: list[float] = []
    action_tokens = 0
    prefix_offset = teacher_initial_length - trajectory.student_initial_length

    for step in trajectory.steps:
        teacher_start = step.action_start + prefix_offset
        teacher_end = step.action_end + prefix_offset
        values = teacher_logprobs[teacher_start:teacher_end]
        numeric = [float(value) for value in values]
        evidence = sum(
            teacher - student
            for teacher, student in zip(numeric, step.logprobs)
        )
        step.teacher_logprobs = numeric
        step.teacher_evidence = float(evidence)
        evidences.append(float(evidence))
        action_tokens += len(numeric)
    return tuple(evidences), action_tokens


async def score_trajectory_async(
    sampling_client: Any,
    tokenizer: Any,
    trajectory: Trajectory,
    selection: SkillSelection,
    semaphore: asyncio.Semaphore,
) -> TeacherTrajectoryScore:
    """使用 Teacher 为一条轨迹打分。"""

    teacher_tokens, teacher_initial_length = build_teacher_trajectory_tokens(
        tokenizer,
        trajectory,
        selection,
    )
    async with semaphore:
        teacher_logprobs = await sampling_client.compute_logprobs_async(
            trio.ModelInput.from_ints(teacher_tokens)
        )
    evidences, action_tokens = _score_turns(
        trajectory,
        teacher_logprobs,
        teacher_initial_length,
    )
    trajectory.skill_key = selection.key
    trajectory.turn_evidences = list(evidences)
    trajectory.teacher_scored_tokens = len(teacher_tokens)
    return TeacherTrajectoryScore(
        group_id=trajectory.group_id,
        group_index=trajectory.group_index,
        policy_snapshot_id=trajectory.policy_snapshot_id,
        skill_key=selection.key,
        teacher_input_tokens=len(teacher_tokens),
        action_tokens=action_tokens,
        evidences=evidences,
    )


async def score_trajectories_async(
    sampling_client: Any,
    tokenizer: Any,
    trajectories: Sequence[Trajectory],
    skill_provider: SkillProvider,
    *,
    max_concurrency: int = 16,
) -> tuple[list[TeacherTrajectoryScore], TeacherBatchStats]:
    """并发为多条轨迹进行 Teacher 打分。"""

    started = perf_counter()
    semaphore = asyncio.Semaphore(max_concurrency)
    scoreable = [trajectory for trajectory in trajectories if trajectory.steps]
    scores = list(
        await asyncio.gather(
            *(
                score_trajectory_async(
                    sampling_client,
                    tokenizer,
                    trajectory,
                    skill_provider.resolve(trajectory.example.task_type),
                    semaphore,
                )
                for trajectory in scoreable
            )
        )
    )
    stats = TeacherBatchStats(
        requests=len(scores),
        teacher_input_tokens=sum(item.teacher_input_tokens for item in scores),
        action_tokens=sum(item.action_tokens for item in scores),
        elapsed_seconds=perf_counter() - started,
    )
    return scores, stats
