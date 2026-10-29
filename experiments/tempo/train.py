"""使用 PyTRIO + TEMPO 在 ALFWorld 上做 macro-step 粒度的 actor-critic 训练。

训练分两阶段：
1. warm-up：完整轨迹 rollout，Monte Carlo return 作为 critic 的 value
   target（无需 bootstrap），只更新 critic；
2. TD 阶段：从 StateStore 取保存的 macro-step 起点续跑 H 轮，生成式
   critic 估终点价值，actor return = 段内奖励 + V̂(终点)，actor 与
   critic 的 Datum 合并成 batch 更新，非终局终点重新入库。

先在本目录下载数据（同第 8 篇）：

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"

冒烟测试：
uv run --extra alfworld python train.py \
    --warmup-updates 4 \
    --td-updates 4 \
    --warmup-games-per-batch 4 \
    --states-per-batch 4 \
    --branches 4 \
    --critic-samples 4 \
    --endpoint-samples 1 \
    --swanlab-mode disabled

正式训练：
uv run --extra alfworld python train.py \
    --warmup-updates 4 --td-updates 40 \
    --swanlab-mode online
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from critic import (
    CriticRequest,
    critic_prompt_for_state,
    endpoint_value,
    sample_critic_values,
)
from data import (
    SPLIT_DIRECTORIES,
    default_data_root,
    load_walkthrough,
    shuffled_games,
    take_batch,
)
from rollout import (
    MAX_SEQUENCE_TOKENS,
    MacroGroup,
    RolloutConfig,
    Trajectory,
    rollout_macro_steps,
)
from states import (
    MacroState,
    StateStore,
    boundary_states,
    endpoint_state,
)
from tempo import (
    ActorSignals,
    CriticSignals,
    TrainingDatum,
    assemble_actor_group,
    assemble_critic_group,
    build_actor_datums,
    build_critic_datums,
)


MAX_TRAIN_BATCH_SIZE = 128


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析两阶段训练、rollout、critic 和日志参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--train-split", choices=list(SPLIT_DIRECTORIES), default="train"
    )
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="最多使用多少局；0 表示使用完整训练 split",
    )
    # ---- 两阶段 ----
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=4,
        help="warm-up 更新次数（完整轨迹 + MC target，只训 critic）",
    )
    parser.add_argument(
        "--td-updates", type=int, default=40, help="TD 阶段更新次数"
    )
    parser.add_argument(
        "--warmup-games-per-batch",
        type=int,
        default=4,
        help="每次 warm-up 更新采多少局完整轨迹",
    )
    # ---- macro-step / 采样 ----
    parser.add_argument("--states-per-batch", type=int, default=4)
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="每个起点的分支数 N（actor 的 group 大小）",
    )
    parser.add_argument(
        "--critic-samples",
        type=int,
        default=4,
        help="同一起点独立估值次数 K（critic 的 group 大小）",
    )
    parser.add_argument(
        "--endpoint-samples",
        type=int,
        default=2,
        help="每个非终局终点采几次估值取均值",
    )
    parser.add_argument(
        "--macro-rounds",
        type=int,
        default=10,
        help="macro-step 长度 H",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=50, help="单局交互上限 T"
    )
    parser.add_argument("--state-store-capacity", type=int, default=256)
    parser.add_argument(
        "--include-admissible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--environment-asynchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-trajectory-tokens", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--max-assistant-tokens", type=int, default=512)
    parser.add_argument("--critic-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # ---- 优化器 / 模型 ----
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每隔多少次 update 保存；0 关闭中间保存，最终产物始终保存",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="tempo-alfworld-qwen35-4b")
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-tempo")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
    )
    args = parser.parse_args(argv)
    args.data_root = _path(args.data_root)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "warmup_updates": args.warmup_updates,
        "td_updates": args.td_updates,
        "warmup_games_per_batch": args.warmup_games_per_batch,
        "states_per_batch": args.states_per_batch,
        "branches": args.branches,
        "critic_samples": args.critic_samples,
        "endpoint_samples": args.endpoint_samples,
        "macro_rounds": args.macro_rounds,
        "max_episode_steps": args.max_episode_steps,
        "state_store_capacity": args.state_store_capacity,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "critic_max_tokens": args.critic_max_tokens,
        "lora_rank": args.lora_rank,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} 必须大于等于 1")
    if args.max_episode_steps % args.macro_rounds != 0:
        raise ValueError("max_episode_steps 必须是 macro_rounds 的整数倍")
    if args.max_train_games < 0 or args.save_every < 0:
        raise ValueError("max_train_games 和 save_every 不能为负数")
    if args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS:
        raise ValueError(
            f"max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}"
        )
    if args.states_per_batch * (args.branches + args.critic_samples) > MAX_TRAIN_BATCH_SIZE:
        raise ValueError(
            f"states_per_batch * (branches + critic_samples) 超过平台 batch 上限 "
            f"{MAX_TRAIN_BATCH_SIZE}"
        )
    if args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("temperature 不能为负数，top_p 必须位于 (0, 1]")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须大于 0")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or population_std(xs) <= 0.0 or population_std(ys) <= 0.0:
        return None
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """把 PyTRIO 单次 batch 返回的数值指标统一加上 trainer 前缀。"""
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in dict(result.metrics).items():
            if isinstance(value, (int, float, np.number)):
                values[key].append(float(value))
    return {f"trainer/{key}": mean(items) for key, items in values.items()}


def pick_mean_loss(metrics: dict[str, float]) -> float | None:
    for key in ("trainer/loss_mean", "trainer/loss/mean"):
        if key in metrics:
            return metrics[key]
    return None


def forward_backward_chunked(
    training_client: Any,
    datums: list[TrainingDatum],
) -> list[Any]:
    """按平台 batch 上限分块前向反向，梯度累积后由调用方统一 optim_step。"""
    results: list[Any] = []
    for start in range(0, len(datums), MAX_TRAIN_BATCH_SIZE):
        chunk = datums[start : start + MAX_TRAIN_BATCH_SIZE]
        results.append(
            training_client.forward_backward(
                [item.datum for item in chunk],
                loss_fn="importance_sampling",
            ).result()
        )
    return results


def _critic_prompt_budget(args: argparse.Namespace) -> int:
    return args.max_trajectory_tokens - args.critic_max_tokens


def _trajectory_steps(trajectories: Sequence[Trajectory]) -> list[Any]:
    return [step for item in trajectories for step in item.steps]


def _rollout_quality_metrics(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    steps = _trajectory_steps(trajectories)
    return {
        "rollout/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        "rollout/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        "rollout/valid_action_rate": mean(
            [float(step.valid_action) for step in steps]
        ),
    }


def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存训练 state 和 sampler weights。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(
        name=f"{name}-weights"
    ).result()
    print(f"Saved state: {state.path}")
    print(f"Saved sampler weights: {weights.path}")


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


class WarmupRunner:
    """warm-up 阶段：完整轨迹 + Monte Carlo target，只更新 critic。"""

    def __init__(self, args: argparse.Namespace, rollout_config: RolloutConfig) -> None:
        self.args = args
        # warm-up 让 macro 边界退化为终局，即完整轨迹 rollout。
        self.config = replace(rollout_config, macro_rounds=args.max_episode_steps)

    def run_update(
        self,
        training_client: Any,
        sampling_client: Any,
        tokenizer: Any,
        games: list[Any],
        cursor: int,
        *,
        update_index: int,
        seed_offset: int,
    ) -> tuple[dict[str, float], int]:
        """一次 warm-up 更新；返回 (指标, 本轮消耗的游戏数)。"""
        args = self.args
        fresh = take_batch(games, cursor, args.warmup_games_per_batch)
        with tqdm(
            total=args.warmup_games_per_batch * args.branches,
            desc=f"Update {update_index + 1} warmup rollout",
            unit="trajectory",
            position=1,
            leave=False,
        ) as rollout_progress:
            groups = rollout_macro_steps(
                sampling_client,
                tokenizer,
                [],
                fresh,
                self.config,
                seed_offset=seed_offset,
                progress_callback=rollout_progress.update,
            )

        # 轨迹终局即 MC return；边界状态由 states.boundary_states 重建。
        # 注意局部变量不要也叫 boundary_states，会遮蔽导入的函数。
        snapshots: list[MacroState] = []
        mc_returns: list[float] = []
        for group in groups:
            for trajectory in group.trajectories:
                won = 1.0 if trajectory.won else 0.0
                for state in boundary_states(
                    group.start_state,
                    trajectory,
                    macro_rounds=args.macro_rounds,
                    max_trajectory_tokens=args.max_trajectory_tokens,
                ):
                    snapshots.append(state)
                    mc_returns.append(won)

        requests: list[CriticRequest] = []
        kept: list[tuple[MacroState, float]] = []
        prompt_too_long = 0
        for index, (state, mc_return) in enumerate(
            zip(snapshots, mc_returns, strict=True)
        ):
            prompt = critic_prompt_for_state(
                tokenizer,
                state.messages,
                task=state.task,
                walkthrough=load_walkthrough(state.example.game_file),
            )
            if len(prompt) > _critic_prompt_budget(args):
                prompt_too_long += 1
                continue
            kept.append((state, mc_return))
            requests.append(
                CriticRequest(
                    prompt_tokens=prompt,
                    num_samples=args.critic_samples,
                    seed=args.seed + seed_offset + index,
                )
            )

        batches = sample_critic_values(
            sampling_client,
            tokenizer,
            requests,
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        datums: list[TrainingDatum] = []
        abs_errors: list[float] = []
        parse_failures = 0
        total_samples = 0
        estimates: list[float] = []
        outcomes: list[float] = []
        for (state, mc_return), samples in zip(kept, batches, strict=True):
            signals = assemble_critic_group(samples, mc_return)
            abs_errors.extend(signals.errors)
            parse_failures += signals.parse_failures
            total_samples += len(samples)
            if signals.mean_estimate is not None:
                estimates.append(signals.mean_estimate)
                outcomes.append(mc_return)
            datums.extend(
                build_critic_datums(
                    samples,
                    signals,
                    game_id=state.example.id,
                    group_index=state.macro_index,
                )
            )

        trainer_results: list[Any] = []
        if datums:
            trainer_results = forward_backward_chunked(training_client, datums)
            training_client.optim_step(
                trio.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                )
            ).result()

        trajectories = [t for group in groups for t in group.trajectories]
        metrics: dict[str, float] = {
            "phase": 0.0,
            "warmup/success_rate": mean(
                [float(item.won) for item in trajectories]
            ),
            "warmup/steps_mean": mean(
                [float(len(item.steps)) for item in trajectories]
            ),
            "warmup/boundary_states": float(len(snapshots)),
            "target/mc_return_mean": mean(mc_returns),
            "critic/abs_error_mean": mean(abs_errors),
            "critic/parse_failure_rate": (
                parse_failures / total_samples if total_samples else 0.0
            ),
            "critic/prompt_too_long_rate": (
                prompt_too_long / len(snapshots) if snapshots else 0.0
            ),
            "critic/mean_estimate_mean": mean(estimates),
            "train/batch_size": float(len(datums)),
            "train/tokens": float(sum(item.num_tokens for item in datums)),
            "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
        }
        correlation = pearson(estimates, outcomes)
        if correlation is not None:
            metrics["critic/value_corr"] = correlation
        metrics.update(_rollout_quality_metrics(trajectories))
        metrics.update(merge_trainer_metrics(trainer_results))
        return metrics, len(fresh)


class TDRunner:
    """TD 阶段：macro-step rollout + 生成式 critic bootstrap。"""

    def __init__(self, args: argparse.Namespace, rollout_config: RolloutConfig) -> None:
        self.args = args
        self.config = rollout_config
        self.store = StateStore(args.state_store_capacity, seed=args.seed + 1)

    def run_update(
        self,
        training_client: Any,
        sampling_client: Any,
        tokenizer: Any,
        games: list[Any],
        cursor: int,
        *,
        update_index: int,
        seed_offset: int,
    ) -> dict[str, float]:
        """一次 TD 更新：rollout -> 估终点 -> 拼 return/G -> 更新 -> 入库。"""
        args = self.args
        start_states = self.store.sample(args.states_per_batch)
        fresh_count = args.states_per_batch - len(start_states)
        fresh = take_batch(games, cursor, fresh_count) if fresh_count > 0 else []
        with tqdm(
            total=(len(start_states) + len(fresh)) * args.branches,
            desc=f"Update {update_index + 1} TD rollout",
            unit="branch",
            position=1,
            leave=False,
        ) as rollout_progress:
            groups = rollout_macro_steps(
                sampling_client,
                tokenizer,
                start_states,
                fresh,
                self.config,
                seed_offset=seed_offset,
                progress_callback=rollout_progress.update,
            )

        # ---- 终点估值：非终局分支的截断尾部由 critic 补齐 ----
        endpoint_requests: list[tuple[int, int, CriticRequest]] = []
        prompt_too_long = 0
        for game_index, group in enumerate(groups):
            for branch_index, trajectory in enumerate(group.trajectories):
                if trajectory.stop_reason == "macro_boundary":
                    prompt = critic_prompt_for_state(
                        tokenizer,
                        trajectory.messages,
                        task=trajectory.task,
                        walkthrough=load_walkthrough(trajectory.example.game_file),
                    )
                    if len(prompt) > _critic_prompt_budget(args):
                        prompt_too_long += 1
                        continue
                    endpoint_requests.append(
                        (
                            game_index,
                            branch_index,
                            CriticRequest(
                                prompt_tokens=prompt,
                                num_samples=args.endpoint_samples,
                                seed=(
                                    args.seed
                                    + seed_offset
                                    + game_index * 10_000
                                    + branch_index
                                ),
                            ),
                        )
                    )
        endpoint_batches = sample_critic_values(
            sampling_client,
            tokenizer,
            [request for _, _, request in endpoint_requests],
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        endpoint_parse_failures = 0
        endpoint_values: list[float] = []
        for (game_index, branch_index, _), samples in zip(
            endpoint_requests, endpoint_batches, strict=True
        ):
            value, failures = endpoint_value(samples)
            groups[game_index].trajectories[branch_index].endpoint_value = value
            endpoint_parse_failures += failures
            endpoint_values.append(value)

        # ---- 信号拼装：actor return/G，随后同一起点的 critic 训练组 ----
        actor_signals: list[ActorSignals] = [
            assemble_actor_group(group) for group in groups
        ]
        critic_requests: list[CriticRequest] = []
        critic_states: list[tuple[MacroGroup, float]] = []
        start_prompt_too_long = 0
        for index, group in enumerate(groups):
            prompt = critic_prompt_for_state(
                tokenizer,
                group.start_state.messages,
                task=group.start_state.task,
                walkthrough=load_walkthrough(group.start_state.example.game_file),
            )
            if len(prompt) > _critic_prompt_budget(args):
                start_prompt_too_long += 1
                continue
            critic_states.append((group, actor_signals[index].value_target))
            critic_requests.append(
                CriticRequest(
                    prompt_tokens=prompt,
                    num_samples=args.critic_samples,
                    seed=args.seed + seed_offset + 100_000 + index,
                )
            )
        critic_batches = sample_critic_values(
            sampling_client,
            tokenizer,
            critic_requests,
            max_tokens=args.critic_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        datums: list[TrainingDatum] = []
        for group, signals in zip(groups, actor_signals, strict=True):
            datums.extend(
                build_actor_datums(
                    group,
                    signals,
                    max_sequence_tokens=args.max_trajectory_tokens,
                )
            )

        critic_signals: list[CriticSignals] = []
        abs_errors: list[float] = []
        parse_failures = 0
        total_samples = 0
        for (group, value_target), samples in zip(
            critic_states, critic_batches, strict=True
        ):
            signals = assemble_critic_group(samples, value_target)
            critic_signals.append(signals)
            abs_errors.extend(signals.errors)
            parse_failures += signals.parse_failures
            total_samples += len(samples)
            datums.extend(
                build_critic_datums(
                    samples,
                    signals,
                    game_id=group.start_state.example.id,
                    group_index=group.start_state.macro_index,
                )
            )

        trainer_results: list[Any] = []
        if datums:
            trainer_results = forward_backward_chunked(training_client, datums)
            training_client.optim_step(
                trio.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                )
            ).result()

        # ---- 终点入库，供后续训练继续向后推进 ----
        saved_endpoints: list[MacroState] = []
        for group in groups:
            for trajectory in group.trajectories:
                state = endpoint_state(
                    group.start_state,
                    trajectory,
                    config=self.config,
                )
                if state is not None:
                    saved_endpoints.append(state)
        self.store.add(saved_endpoints)

        trajectories = [t for group in groups for t in group.trajectories]
        total_macro_boundary = sum(
            trajectory.stop_reason == "macro_boundary" for trajectory in trajectories
        )
        metrics: dict[str, float] = {
            "phase": 1.0,
            "td/store_size": float(len(self.store)),
            "td/new_endpoints": float(len(saved_endpoints)),
            "td/fresh_states": float(len(fresh)),
            "td/macro_index_mean": mean(
                [float(group.start_state.macro_index) for group in groups]
            ),
            "actor/segment_reward_mean": mean(
                [item.segment_reward for item in trajectories]
            ),
            "actor/terminal_won_rate": mean(
                [float(item.won) for item in trajectories]
            ),
            "actor/macro_boundary_rate": (
                total_macro_boundary / len(trajectories) if trajectories else 0.0
            ),
            "actor/degenerate_group_rate": mean(
                [float(signal.degenerate) for signal in actor_signals]
            ),
            "actor/value_target_mean": mean(
                [signal.value_target for signal in actor_signals]
            ),
            "actor/value_target_std": population_std(
                [signal.value_target for signal in actor_signals]
            ),
            "actor/value_gap_zero_reward": mean(
                [signal.zero_reward_value_std for signal in actor_signals]
            ),
            "critic/abs_error_mean": mean(abs_errors),
            "critic/parse_failure_rate": (
                parse_failures / total_samples if total_samples else 0.0
            ),
            "critic/endpoint_value_mean": mean(endpoint_values),
            "critic/endpoint_parse_failures": float(endpoint_parse_failures),
            "critic/prompt_too_long_rate": (
                (prompt_too_long + start_prompt_too_long)
                / max(2 * len(groups), 1)
            ),
            "train/actor_datums": float(
                sum(item.kind == "actor" for item in datums)
            ),
            "train/critic_datums": float(
                sum(item.kind == "critic" for item in datums)
            ),
            "train/batch_size": float(len(datums)),
            "train/tokens": float(sum(item.num_tokens for item in datums)),
            "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
        }
        metrics.update(_rollout_quality_metrics(trajectories))
        metrics.update(merge_trainer_metrics(trainer_results))
        return metrics


def _log_update(
    progress: tqdm,
    label: str,
    metrics: dict[str, float],
    started: float,
) -> None:
    progress.update(1)
    loss = pick_mean_loss(metrics)
    loss_text = (
        "skipped"
        if metrics.get("train/batch_size", 0.0) == 0.0
        else (f"{loss:.4f}" if loss is not None else "missing")
    )
    progress.set_postfix(
        success=f"{metrics.get('actor/segment_reward_mean', metrics.get('warmup/success_rate', 0.0)):.3f}",
        loss=loss_text,
        step_s=f"{perf_counter() - started:.1f}",
        refresh=True,
    )
    tqdm.write(
        f"{label} "
        + " ".join(
            f"{key}={value:.3f}"
            for key, value in sorted(metrics.items())
            if isinstance(value, float)
            and (
                key.startswith(("warmup/", "actor/", "critic/abs", "td/store"))
            )
        )
        + f" loss={loss_text}"
    )


def main(args: argparse.Namespace) -> None:
    """warm-up -> TD 两阶段训练闭环。"""
    games = shuffled_games(
        args.data_root,
        args.train_split,
        args.seed,
        max_games=args.max_train_games,
    )
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()

    rollout_config = RolloutConfig(
        branches=args.branches,
        macro_rounds=args.macro_rounds,
        max_episode_steps=args.max_episode_steps,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )
    warmup = WarmupRunner(args, rollout_config)
    td = TDRunner(args, rollout_config)

    total_updates = args.warmup_updates + args.td_updates
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
        tags=["pytrio", "alfworld", "tempo", "macro-step"],
    )
    cursor = 0
    update_index = 0
    try:
        with tqdm(
            total=total_updates, desc="TEMPO training", unit="update"
        ) as progress:
            for _ in range(args.warmup_updates):
                started = perf_counter()
                progress.set_postfix(phase="sampler", refresh=True)
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                progress.set_postfix(phase="warmup-rollout", refresh=True)
                metrics, used = warmup.run_update(
                    training_client,
                    sampling_client,
                    tokenizer,
                    games,
                    cursor,
                    update_index=update_index,
                    seed_offset=update_index * 7919,
                )
                cursor += used
                metrics["time/update_seconds"] = perf_counter() - started
                swanlab.log(metrics, step=update_index)
                _log_update(progress, f"warmup={update_index + 1}", metrics, started)
                update_index += 1

            for _ in range(args.td_updates):
                started = perf_counter()
                progress.set_postfix(phase="sampler", refresh=True)
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                progress.set_postfix(phase="td-rollout", refresh=True)
                metrics = td.run_update(
                    training_client,
                    sampling_client,
                    tokenizer,
                    games,
                    cursor,
                    update_index=update_index,
                    seed_offset=update_index * 7919,
                )
                cursor += int(metrics["td/fresh_states"])
                metrics["time/update_seconds"] = perf_counter() - started
                swanlab.log(metrics, step=update_index)
                _log_update(progress, f"td={update_index + 1}", metrics, started)
                update_index += 1

                if args.save_every > 0 and update_index % args.save_every == 0:
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-update-{update_index}",
                    )

        save_checkpoint(training_client, f"{args.run_name}-final")
    except KeyboardInterrupt:
        swanlab.finish(state="aborted")
        raise
    except Exception as error:
        swanlab.finish(state="crashed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        swanlab.finish()


if __name__ == "__main__":
    main(parse_args())
