"""GiGPO mean-only variant on top of the unchanged ALFWorld GRPO baseline.

Reference: https://arxiv.org/abs/2505.10978, equations 3, 5--8.
Official reference: langfengQ/verl-agent, gigpo/core_gigpo.py.
This is an independent implementation for full-history PyTRIO trajectories.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from reward import discounted_returns


@dataclass(frozen=True)
class AdvantageStats:
    groups: int
    degenerate_groups: int
    anchor_groups: int = 0
    matched_step_fraction: float = 0.0
    cross_trajectory_step_fraction: float = 0.0
    noninitial_matched_fraction: float = 0.0
    nonzero_micro_fraction: float = 0.0
    micro_abs_mean: float = 0.0
    mean_anchor_size: float = 0.0


def anchor_key(step: Any, mode: str) -> Any:
    """Use the PRE-action observation, never the chosen action or future result.

    observation matches the official ALFWorld anchor: exact raw observation.
    observation_actions is an optional stricter ablation, NOT the main recipe.
    Neither is a guarantee of identical latent PDDL state in a POMDP.
    """
    if mode == "observation":
        return step.observation
    if mode == "observation_actions":
        return step.observation, tuple(sorted(set(step.admissible_actions)))
    raise ValueError(f"Unknown anchor mode: {mode}")


def assign_advantages(
    trajectories: list[Any], *, algorithm: str = "grpo", gamma: float = 0.95,
    step_weight: float = 1.0, anchor_mode: str = "observation",
) -> AdvantageStats:
    """Compute A_it = (R_i - mean_group R) + w * (G_it - mean_anchor G).

    Macro means weight each trajectory equally, exactly as experiments/alfworld does.
    Anchors are scoped to one game group and can span different time indices.
    All occurrences are retained, including repeated visits within a trajectory,
    matching official exact-match grouping. Singleton micro advantages are zero.
    No standard deviation normalization is used in either arm.
    """
    if algorithm not in {"grpo", "gigpo"}:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if not math.isfinite(step_weight) or step_weight < 0:
        raise ValueError("step_weight must be finite and nonnegative")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if anchor_mode not in {"observation", "observation_actions"}:
        raise ValueError("Unknown anchor mode")
    groups: dict[str, list[Any]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.group_id].append(trajectory)
    anchors: dict[Any, list[tuple[int, Any]]] = defaultdict(list)
    degenerate = 0
    total_steps = noninitial = 0
    for group_id, group in groups.items():
        mean_reward = sum(item.reward for item in group) / len(group)
        for branch, trajectory in enumerate(group):
            trajectory.advantage = float(trajectory.reward - mean_reward)
            returns = discounted_returns([step.reward for step in trajectory.steps], gamma)
            for step, value in zip(trajectory.steps, returns, strict=True):
                step.discounted_return = value
                step.micro_advantage = 0.0
                step.advantage = trajectory.advantage
                step.anchor_group_size = 1
                total_steps += 1
                noninitial += step.index > 1
                anchors[(group_id, anchor_key(step, anchor_mode))].append((branch, step))
        degenerate += all(abs(item.advantage) <= 1e-12 for item in group)

    matched = cross_trajectory = noninitial_matched = nonzero = 0
    micro_abs_sum = 0.0
    for members in anchors.values():
        size = len(members)
        mean_return = sum(step.discounted_return for _, step in members) / size
        has_multiple_branches = len({branch for branch, _ in members}) > 1
        for _, step in members:
            step.anchor_group_size = size
            micro = step.discounted_return - mean_return
            if algorithm == "gigpo":
                step.micro_advantage = micro
                step.advantage += step_weight * micro
            matched += size > 1
            cross_trajectory += has_multiple_branches
            noninitial_matched += size > 1 and step.index > 1
            nonzero += abs(step.micro_advantage) > 1e-12
            micro_abs_sum += abs(step.micro_advantage)
    return AdvantageStats(
        groups=len(groups), degenerate_groups=degenerate,
        anchor_groups=len(anchors), matched_step_fraction=matched / max(total_steps, 1),
        cross_trajectory_step_fraction=cross_trajectory / max(total_steps, 1),
        noninitial_matched_fraction=noninitial_matched / max(noninitial, 1),
        nonzero_micro_fraction=nonzero / max(total_steps, 1),
        micro_abs_mean=micro_abs_sum / max(total_steps, 1),
        mean_anchor_size=total_steps / max(len(anchors), 1),
    )
