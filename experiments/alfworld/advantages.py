"""根据同一游戏的终局奖励计算轨迹级 group-relative advantage。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdvantageStats:
    """记录本次轨迹级 advantage 的分组统计。"""

    groups: int
    degenerate_groups: int


def assign_advantages(trajectories: list[Any]) -> AdvantageStats:
    """在相同游戏的 K 条轨迹内执行 reward 减均值。"""
    groups: dict[str, list[Any]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.group_id].append(trajectory)

    degenerate_groups = 0
    for group in groups.values():
        mean_reward = (
            sum(trajectory.reward for trajectory in group) / len(group)
        )
        advantages = [
            float(trajectory.reward - mean_reward)
            for trajectory in group
        ]
        for trajectory, advantage in zip(group, advantages, strict=True):
            trajectory.advantage = advantage
        if all(abs(value) <= 1e-12 for value in advantages):
            degenerate_groups += 1

    return AdvantageStats(
        groups=len(groups),
        degenerate_groups=degenerate_groups,
    )
