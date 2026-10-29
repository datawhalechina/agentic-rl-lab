"""计算 AgentOPSD 的轨迹级优势和轮级信用。

本模块不访问远程服务，只使用 NumPy/Python 对终局奖励以及已经对齐的
Teacher/Student 对数概率证据进行确定性计算。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class AgentOPSDConfig:
    """Paper-aligned hyperparameters for turn advantage reshaping."""

    gamma: float = 0.95
    reshape_lambda: float = 0.5
    weight_bound: float = 0.2
    epsilon: float = 1e-4


@dataclass(frozen=True)
class TurnCredit:
    """Every intermediate value used to reshape one turn's advantage."""

    turn_index: int
    evidence: float
    cumulative_evidence: float
    belief_before: float
    belief_after: float
    delta_belief: float
    outcome_credit: float
    normalized_credit: float
    weight: float
    advantage: float


@dataclass(frozen=True)
class TrajectoryCredit:
    """Sequence-level GRPO advantage plus all per-turn credits."""

    sequence_advantage: float
    initial_belief: float
    turns: tuple[TurnCredit, ...]


@dataclass(frozen=True)
class AdvantageStats:
    """Group-level diagnostics returned after assigning all trajectories."""

    groups: int
    degenerate_groups: int


def group_sequence_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float,
) -> tuple[list[float], float]:
    """计算组内 GRPO advantage 和初始 belief。"""

    mean_reward = sum(rewards) / len(rewards)
    std_reward = float(np.std(rewards, ddof=0))
    advantages = [
        (reward - mean_reward) / (std_reward + epsilon)
        for reward in rewards
    ]
    return advantages, mean_reward


def _sigmoid(value: float) -> float:
    """计算 sigmoid。"""

    if value >= 0.0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = exp(value)
    return forward / (1.0 + forward)


def reshape_turn_advantages(
    sequence_advantage: float,
    initial_belief: float,
    evidences: Sequence[float],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> TrajectoryCredit:
    """根据每轮证据重塑 advantage。"""

    b0 = min(max(initial_belief, config.epsilon), 1.0 - config.epsilon)
    logit_b0 = log(b0 / (1.0 - b0))
    cumulative = 0.0
    belief_before = b0
    raw: list[tuple[float, float, float, float]] = []

    for evidence in evidences:
        cumulative = config.gamma * cumulative + evidence
        belief_after = _sigmoid(logit_b0 + cumulative)
        raw.append((evidence, cumulative, belief_before, belief_after))
        belief_before = belief_after

    sign = 1.0 if sequence_advantage > 0.0 else (
        -1.0 if sequence_advantage < 0.0 else 0.0
    )
    outcome_credits = [
        sign * (belief_after - belief_before_value)
        for _, _, belief_before_value, belief_after in raw
    ]
    credit_mean = sum(outcome_credits) / len(outcome_credits)
    credit_std = float(np.std(outcome_credits, ddof=0))
    normalized = [
        (credit - credit_mean) / (credit_std + config.epsilon)
        for credit in outcome_credits
    ]

    lower_weight = 1.0 - config.weight_bound
    upper_weight = 1.0 + config.weight_bound
    turns: list[TurnCredit] = []
    for index, (
        (evidence, cumulative_value, before, after),
        outcome_credit,
        normalized_credit,
    ) in enumerate(zip(raw, outcome_credits, normalized), start=1):
        weight = min(
            max(1.0 + config.weight_bound * normalized_credit, lower_weight),
            upper_weight,
        )
        multiplier = (
            (1.0 - config.reshape_lambda)
            + config.reshape_lambda * weight
        )
        turn_advantage = sequence_advantage * multiplier
        turns.append(
            TurnCredit(
                turn_index=index,
                evidence=evidence,
                cumulative_evidence=cumulative_value,
                belief_before=before,
                belief_after=after,
                delta_belief=after - before,
                outcome_credit=outcome_credit,
                normalized_credit=normalized_credit,
                weight=weight,
                advantage=turn_advantage,
            )
        )

    return TrajectoryCredit(
        sequence_advantage=sequence_advantage,
        initial_belief=b0,
        turns=tuple(turns),
    )


def assign_agentopsd_advantages(
    trajectories: Sequence[Any],
    config: AgentOPSDConfig = AgentOPSDConfig(),
) -> AdvantageStats:
    """为每组轨迹写入序列和 turn advantage。"""

    groups: dict[str, list[Any]] = defaultdict(list)
    for trajectory in trajectories:
        groups[str(trajectory.group_id)].append(trajectory)

    degenerate_groups = 0
    for group in groups.values():
        sequence_advantages, initial_belief = group_sequence_advantages(
            [trajectory.reward for trajectory in group],
            epsilon=config.epsilon,
        )
        if all(abs(value) <= config.epsilon for value in sequence_advantages):
            degenerate_groups += 1

        for trajectory, sequence_advantage in zip(group, sequence_advantages):
            credit = reshape_turn_advantages(
                sequence_advantage,
                initial_belief,
                trajectory.turn_evidences,
                config,
            )
            trajectory.advantage = sequence_advantage
            trajectory.sequence_advantage = sequence_advantage
            trajectory.initial_belief = credit.initial_belief
            trajectory.turn_credits = list(credit.turns)
            for step, turn in zip(trajectory.steps, credit.turns):
                step.teacher_evidence = turn.evidence
                step.turn_advantage = turn.advantage

    return AdvantageStats(
        groups=len(groups),
        degenerate_groups=degenerate_groups,
    )
