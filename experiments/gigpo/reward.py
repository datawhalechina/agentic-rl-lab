"""Shared GRPO/GiGPO reward; no model judge and no additional rollouts."""

from __future__ import annotations

from typing import Any

INVALID_ACTION_PENALTY = 0.1


def assign_rewards(trajectory: Any) -> None:
    """Preserve experiments/alfworld's return; expose its per-step decomposition.

    Success is paid once, at the final step. Each invalid action incurs -0.1
    at that step. The undiscounted sum therefore equals the original return.
    A timeout/token truncation is a terminal failure, without bootstrapping.
    """
    trajectory.invalid_action_count = sum(not step.valid_action for step in trajectory.steps)
    trajectory.reward = float(trajectory.won) - INVALID_ACTION_PENALTY * trajectory.invalid_action_count
    if trajectory.won and not trajectory.steps:
        raise ValueError("A successful trajectory must contain an action")
    for step in trajectory.steps:
        step.reward = -INVALID_ACTION_PENALTY if not step.valid_action else 0.0
    if trajectory.steps:
        trajectory.steps[-1].reward += float(trajectory.won)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute G_t = r_t + gamma * G_(t+1), including future invalid penalties."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    returns = [0.0] * len(rewards)
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        future = float(rewards[index]) + gamma * future
        returns[index] = future
    return returns
