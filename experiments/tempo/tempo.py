"""TEMPO 的信号拼装与训练 Datum 构造。

本文件对应博客中 TEMPO 区别于普通 GRPO 的全部本地逻辑：
- actor 的 return = 段内环境奖励 + 终点 critic 估值（截断尾部由 V̂ 补齐）；
- G（TD target）= 同一起点 N 条分支 return 的均值；
- critic 的 reward = -|V̂ - G| / R_max（解析失败按最差处理）；
- 两侧都做组内中心化，用同一个 importance_sampling loss 更新。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytrio as trio

from critic import CriticSample
from rollout import MacroGroup, Trajectory
from states import MacroState


R_MAX = 1.0
ADVANTAGE_EPSILON = 1e-12


@dataclass(frozen=True)
class ActorSignals:
    """一个起点组内 N 条分支的 return 与 advantage。"""

    segment_rewards: list[float]
    endpoint_values: list[float]
    returns: list[float]
    value_target: float
    advantages: list[float]
    degenerate: bool
    # 段内奖励相同（全 0）且非终局分支间的估值差 —— 博客"放置骑士"指标。
    zero_reward_value_std: float


@dataclass(frozen=True)
class CriticSignals:
    """同一起点上 K 次独立估值的误差奖励与 advantage。"""

    rewards: list[float]
    advantages: list[float]
    errors: list[float]
    degenerate: bool
    parse_failures: int
    mean_estimate: float | None


def _terminal(trajectory: Trajectory) -> bool:
    """macro_boundary 之外的停机原因都视为终局（没有可 bootstrap 的未来）。"""
    return trajectory.stop_reason != "macro_boundary"


def assemble_actor_group(group: MacroGroup) -> ActorSignals:
    """计算 N 条分支的 return、G 与组内中心化 advantage。"""
    trajectories = group.trajectories
    rewards = [trajectory.segment_reward for trajectory in trajectories]
    # 终局分支没有未来，终点估值恒为 0；只有停在 macro_boundary 的分支
    # 由 critic 估出的 endpoint_value 参与 bootstrap。
    values = [
        0.0 if _terminal(trajectory) else trajectory.endpoint_value
        for trajectory in trajectories
    ]
    returns = [reward + value for reward, value in zip(rewards, values, strict=True)]
    value_target = sum(returns) / len(returns)
    mean_return = value_target
    advantages = [item - mean_return for item in returns]
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        trajectory.branch_return = (
            trajectory.segment_reward
            + (0.0 if _terminal(trajectory) else trajectory.endpoint_value)
        )
        trajectory.advantage = advantage

    zero_reward_values = [
        value
        for reward, value, trajectory in zip(
            rewards, values, trajectories, strict=True
        )
        if abs(reward) <= ADVANTAGE_EPSILON and not _terminal(trajectory)
    ]
    zero_reward_value_std = (
        float(np.std(np.asarray(zero_reward_values, dtype=np.float64)))
        if len(zero_reward_values) >= 2
        else 0.0
    )

    return ActorSignals(
        segment_rewards=rewards,
        endpoint_values=values,
        returns=returns,
        value_target=value_target,
        advantages=advantages,
        degenerate=all(
            abs(advantage) <= ADVANTAGE_EPSILON for advantage in advantages
        ),
        zero_reward_value_std=zero_reward_value_std,
    )


def assemble_critic_group(
    samples: Sequence[CriticSample],
    value_target: float,
) -> CriticSignals:
    """估值误差转奖励并组内中心化；解析失败的样本按最差 reward 处理。"""
    rewards: list[float] = []
    errors: list[float] = []
    parse_failures = 0
    for sample in samples:
        if sample.value is None:
            parse_failures += 1
            rewards.append(-R_MAX)
            errors.append(R_MAX)
        else:
            error = abs(sample.value - value_target) / R_MAX
            errors.append(error)
            rewards.append(-error)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    advantages = [reward - mean_reward for reward in rewards]

    valid = [sample.value for sample in samples if sample.value is not None]
    return CriticSignals(
        rewards=rewards,
        advantages=advantages,
        errors=errors,
        degenerate=all(
            abs(advantage) <= ADVANTAGE_EPSILON for advantage in advantages
        ),
        parse_failures=parse_failures,
        mean_estimate=(sum(valid) / len(valid)) if valid else None,
    )


@dataclass(frozen=True)
class TrainingDatum:
    """PyTRIO Datum 及其装箱、追踪元数据。"""

    datum: trio.Datum
    num_tokens: int
    loss_tokens: int
    kind: str  # "actor" | "critic"
    game_id: str
    group_index: int


def build_segment_datum(
    start_state: MacroState,
    trajectory: Trajectory,
    *,
    max_sequence_tokens: int,
) -> TrainingDatum:
    """把一个 macro-step 分支构造成右移对齐的 Datum。

    基座是保存的 token_prefix（初始状态时为空，由第一步 prompt 兜底）。
    每轮 prompt 相对上一轮只增加 assistant 闭合符和 tool observation，
    这些环境 token 的 old logprob/advantage 均为 0；本段所有 assistant
    completion 使用采样时的 old logprob 和该分支的组内中心化 advantage。
    """
    if not trajectory.steps:
        raise ValueError("不能用空分支构造 Datum")

    base = start_state.token_prefix
    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0

    for step in trajectory.steps:
        if not step.prompt_tokens:
            raise ValueError(f"第 {step.index} 步 prompt 不能为空")
        if not step.completion_tokens:
            continue
        if len(step.completion_tokens) != len(step.logprobs):
            raise ValueError(
                f"第 {step.index} 步 completion token 与 old logprob 长度不一致"
            )

        if not full_tokens:
            if base is None:
                delta_environment = step.prompt_tokens
            elif step.prompt_tokens[: len(base)] == base:
                # 保存的 token 前缀是上下文的一部分，必须完整保留，
                # 训练信号为 0。
                full_tokens.extend(base)
                old_logprobs_by_token.extend([0.0] * len(base))
                advantages_by_token.extend([0.0] * len(base))
                delta_environment = step.prompt_tokens[len(base) :]
            else:
                raise ValueError("分支首步 prompt 不是保存前缀的扩展")
        elif step.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_environment = step.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(f"第 {step.index} 步 prompt 不是已有轨迹的前缀扩展")

        full_tokens.extend(delta_environment)
        old_logprobs_by_token.extend([0.0] * len(delta_environment))
        advantages_by_token.extend([0.0] * len(delta_environment))

        full_tokens.extend(step.completion_tokens)
        old_logprobs_by_token.extend(step.logprobs)
        advantages_by_token.extend(
            [trajectory.advantage] * len(step.completion_tokens)
        )
        assistant_token_count += len(step.completion_tokens)

    # macro_boundary 分支保留最后一段 tool observation 作为上下文，
    # 训练信号明确为 0。
    if trajectory.next_prompt_tokens is not None:
        if trajectory.next_prompt_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError("最终 tool observation 不是已有轨迹的前缀扩展")
        final_environment = trajectory.next_prompt_tokens[len(full_tokens) :]
        full_tokens.extend(final_environment)
        old_logprobs_by_token.extend([0.0] * len(final_environment))
        advantages_by_token.extend([0.0] * len(final_environment))

    if assistant_token_count == 0:
        raise ValueError("不能用没有 assistant token 的分支构造 Datum")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError("分支的 token、logprob 和 advantage 长度不一致")
    if len(full_tokens) > max_sequence_tokens:
        raise ValueError(
            f"分支为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}"
        )

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError("Datum input/target/logprobs/advantages 长度不一致")

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
        loss_tokens=assistant_token_count,
        kind="actor",
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
    )


def build_critic_datum(
    sample: CriticSample,
    advantage: float,
    *,
    game_id: str,
    group_index: int,
) -> TrainingDatum:
    """把一次估值样本构造成 Datum：prompt 仅作上下文，整段 completion 训练。"""
    prompt_tokens = sample.prompt_tokens
    completion_tokens = sample.completion_tokens
    if not completion_tokens:
        raise ValueError("critic 样本没有 completion token")
    if len(completion_tokens) != len(sample.logprobs):
        raise ValueError("critic completion token 与 old logprob 长度不一致")

    full_tokens = [*prompt_tokens, *completion_tokens]
    old_logprobs_by_token = [0.0] * len(prompt_tokens) + list(sample.logprobs)
    advantages_by_token = [0.0] * len(prompt_tokens) + [advantage] * len(
        completion_tokens
    )

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
        loss_tokens=len(completion_tokens),
        kind="critic",
        game_id=game_id,
        group_index=group_index,
    )


def build_actor_datums(
    group: MacroGroup,
    signals: ActorSignals,
    *,
    max_sequence_tokens: int,
) -> list[TrainingDatum]:
    """跳过零 advantage 分支后，为起点组构造 actor Datum。"""
    datums: list[TrainingDatum] = []
    for trajectory in group.trajectories:
        trainable = any(step.completion_tokens for step in trajectory.steps)
        if not trainable or abs(trajectory.advantage) <= ADVANTAGE_EPSILON:
            continue
        datums.append(
            build_segment_datum(
                group.start_state,
                trajectory,
                max_sequence_tokens=max_sequence_tokens,
            )
        )
    return datums


def build_critic_datums(
    samples: Sequence[CriticSample],
    signals: CriticSignals,
    *,
    game_id: str,
    group_index: int,
) -> list[TrainingDatum]:
    """跳过退化组与零 advantage 样本后，构造 critic Datum。"""
    if signals.degenerate or not samples:
        return []
    datums: list[TrainingDatum] = []
    for sample, advantage in zip(samples, signals.advantages, strict=True):
        if abs(advantage) <= ADVANTAGE_EPSILON or not sample.completion_tokens:
            continue
        datums.append(
            build_critic_datum(
                sample,
                advantage,
                game_id=game_id,
                group_index=group_index,
            )
        )
    return datums
