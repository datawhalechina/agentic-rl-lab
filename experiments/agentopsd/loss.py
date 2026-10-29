"""为 AgentOPSD 构造只训练 action token 的 PyTRIO 内置 PPO Datum。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytrio as trio

from rollout import Trajectory


@dataclass(frozen=True)
class TrainingDatum:
    """保存 PPO 数据和 token 统计信息。"""

    datum: trio.Datum
    num_tokens: int
    action_tokens: int
    game_id: str
    group_index: int
    policy_snapshot_id: str


@dataclass(frozen=True)
class BuiltinPPOConfig:
    """保存 PyTRIO 内置 PPO 的裁剪参数。"""

    clip_low: float = 0.8
    clip_high: float = 1.24

    def as_loss_fn_config(self) -> dict[str, float]:
        """转换为 PyTRIO PPO 配置。"""

        return {
            "clip_low_threshold": self.clip_low,
            "clip_high_threshold": self.clip_high,
        }


def build_agentopsd_ppo_datum(trajectory: Trajectory) -> TrainingDatum:
    """构造只训练 action token 的 PPO 数据。"""

    full_tokens = trajectory.full_student_tokens
    old_logprobs_by_token = [0.0] * len(full_tokens)
    advantages_by_token = [0.0] * len(full_tokens)
    action_tokens = 0

    for step, credit in zip(trajectory.steps, trajectory.turn_credits):
        for offset, old_logprob in enumerate(step.logprobs):
            position = step.action_start + offset
            old_logprobs_by_token[position] = old_logprob
            advantages_by_token[position] = credit.advantage
        action_tokens += len(step.logprobs)

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        action_tokens=action_tokens,
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
        policy_snapshot_id=trajectory.policy_snapshot_id,
    )


def build_training_datums(
    trajectories: Sequence[Trajectory],
) -> list[TrainingDatum]:
    """批量构造非零 advantage 的 PPO 数据。"""

    return [
        build_agentopsd_ppo_datum(trajectory)
        for trajectory in trajectories
        if trajectory.sequence_advantage != 0.0
    ]
