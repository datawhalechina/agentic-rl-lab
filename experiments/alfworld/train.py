"""使用 PyTRIO 在 text-only ALFWorld 上训练逐步决策策略。

先下载环境数据，再从本目录启动一次最小训练：

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"

测试:
uv run --extra alfworld python train.py \
    --max-steps 10 \
    --games-per-batch 4 \
    --group-size 8 \
    --save-every 5 \
    --swanlab-mode disabled

正式训练：（预计需要12小时）
uv run --extra alfworld python train.py \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --save-every 20 \
    --swanlab-mode online
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from data import SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch
from rollout import (
    MAX_SEQUENCE_TOKENS,
    RolloutBatch,
    RolloutConfig,
    Trajectory,
    rollout_batch,
)


MAX_TRAIN_BATCH_SIZE = 128


@dataclass(frozen=True)
class TrainingDatum:
    """PyTRIO Datum 及其装箱、追踪元数据。"""

    datum: trio.Datum
    num_tokens: int
    loss_tokens: int
    game_id: str
    group_index: int
    step_count: int


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析训练、rollout、奖励和日志参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument(
        "--train-split",
        choices=list(SPLIT_DIRECTORIES),
        default="train",
    )
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="最多使用多少局；0 表示使用完整训练 split",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="最大训练迭代次数，与单条轨迹的 --max-episode-steps 不同",
    )
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
    )
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
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=MAX_SEQUENCE_TOKENS,
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ppo-clip-low",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--ppo-clip-high",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help=(
            "每隔多少次 update 保存 state 和 sampler weights；"
            "0 关闭中间保存，最终产物始终保存"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-name",
        default="alfworld-agent-rl-qwen35-4b",
    )
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-alfworld",
    )
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
        "max_steps": args.max_steps,
        "games_per_batch": args.games_per_batch,
        "group_size": args.group_size,
        "max_episode_steps": args.max_episode_steps,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "lora_rank": args.lora_rank,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} 必须大于等于 1")
    if not 0.0 < args.ppo_clip_low <= 1.0 <= args.ppo_clip_high:
        raise ValueError("PPO clip 阈值必须满足 0 < low <= 1 <= high")
    if args.max_train_games < 0 or args.save_every < 0:
        raise ValueError("max_train_games 和 save_every 不能为负数")
    if args.max_trajectory_tokens > MAX_SEQUENCE_TOKENS:
        raise ValueError(
            f"max_trajectory_tokens 不能超过平台单序列上限 {MAX_SEQUENCE_TOKENS}"
        )
    rollout_batch_size = args.games_per_batch * args.group_size
    if rollout_batch_size > MAX_TRAIN_BATCH_SIZE:
        raise ValueError(
            f"games_per_batch * group_size 不能超过平台 batch 上限 "
            f"{MAX_TRAIN_BATCH_SIZE}"
        )
    if args.temperature < 0.0 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("temperature 不能为负数，top_p 必须位于 (0, 1]")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须大于 0")


def build_trajectory_datum(
    trajectory: Trajectory,
    *,
    max_sequence_tokens: int,
) -> TrainingDatum:
    """把完整工具轨迹构造成一个右移对齐的 PPO Datum。

    每轮 prompt 相对上一轮只增加 assistant 闭合符和 tool observation。
    这些环境 token 的 old logprob/advantage 均为 0；所有 assistant completion
    使用采样时保存的 old logprob 和整条轨迹的 group-relative advantage。
    """
    if not trajectory.steps:
        raise ValueError("不能用空轨迹构造 Datum")

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0
    step_count = 0

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
            delta_environment = step.prompt_tokens
        elif step.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_environment = step.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(
                f"第 {step.index} 步 prompt 不是已有轨迹的前缀扩展"
            )

        full_tokens.extend(delta_environment)
        old_logprobs_by_token.extend([0.0] * len(delta_environment))
        advantages_by_token.extend([0.0] * len(delta_environment))

        full_tokens.extend(step.completion_tokens)
        old_logprobs_by_token.extend(step.logprobs)
        advantages_by_token.extend(
            [trajectory.advantage] * len(step.completion_tokens)
        )
        assistant_token_count += len(step.completion_tokens)
        step_count += 1

    # 最后一次工具返回后可能不再生成 assistant，但它仍属于完整轨迹，
    # 因此保留真实 token，并明确把训练信号设为 0。
    if trajectory.next_prompt_tokens is not None:
        if trajectory.next_prompt_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError("最终 tool observation 不是已有轨迹的前缀扩展")
        final_environment = trajectory.next_prompt_tokens[len(full_tokens) :]
        full_tokens.extend(final_environment)
        old_logprobs_by_token.extend([0.0] * len(final_environment))
        advantages_by_token.extend([0.0] * len(final_environment))

    if assistant_token_count == 0:
        raise ValueError("不能用没有 assistant token 的轨迹构造 Datum")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError("完整轨迹的 token、logprob 和 advantage 长度不一致")
    if len(full_tokens) > max_sequence_tokens:
        raise ValueError(
            f"轨迹为 {len(full_tokens)} tokens，超过上限 {max_sequence_tokens}"
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
    datum.validate_loss_inputs("ppo")
    return TrainingDatum(
        datum=datum,
        num_tokens=len(input_tokens),
        loss_tokens=assistant_token_count,
        game_id=trajectory.example.id,
        group_index=trajectory.group_index,
        step_count=step_count,
    )


def logical_step_count(trajectories: list[Trajectory]) -> int:
    """统计有 completion 的全部 step；零 advantage 也属于逻辑 batch。"""
    return sum(
        bool(step.completion_tokens)
        for trajectory in trajectories
        for step in trajectory.steps
    )


def build_training_datums(
    trajectories: list[Trajectory],
    *,
    max_sequence_tokens: int,
) -> list[TrainingDatum]:
    """每条轨迹构造一个 Datum，并跳过 advantage 为零的轨迹。"""
    datums: list[TrainingDatum] = []
    for trajectory in trajectories:
        trainable_steps = [
            step for step in trajectory.steps if step.completion_tokens
        ]
        if not trainable_steps or abs(trajectory.advantage) <= 1e-12:
            continue
        datums.append(
            build_trajectory_datum(
                trajectory,
                max_sequence_tokens=max_sequence_tokens,
            )
        )
    return datums


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def rollout_metrics(
    batch: RolloutBatch,
    datums: list[TrainingDatum],
    logical_steps: int,
) -> dict[str, float]:
    """汇总环境、终局奖励、advantage 与完整轨迹 Datum 指标。"""
    trajectories = batch.trajectories
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    rewards = [trajectory.reward for trajectory in trajectories]
    advantages = [trajectory.advantage for trajectory in trajectories]
    logical_trajectories = sum(
        any(step.completion_tokens for step in trajectory.steps)
        for trajectory in trajectories
    )
    metrics = {
        "reward/mean": mean(rewards),
        "reward/success_rate": mean([float(item.won) for item in trajectories]),
        "rollout/steps_mean": mean(
            [float(len(trajectory.steps)) for trajectory in trajectories]
        ),
        "rollout/truncated_rate": mean(
            [float(item.truncated) for item in trajectories]
        ),
        "rollout/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        "rollout/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        "rollout/valid_action_rate": mean(
            [float(step.valid_action) for step in steps]
        ),
        "rollout/invalid_actions_mean": mean(
            [float(item.invalid_action_count) for item in trajectories]
        ),
        "advantage/std": population_std(advantages),
        "advantage/groups": float(batch.advantage_stats.groups),
        "advantage/degenerate_group_rate": (
            batch.advantage_stats.degenerate_groups
            / max(batch.advantage_stats.groups, 1)
        ),
        "train/logical_steps": float(logical_steps),
        "train/logical_trajectories": float(logical_trajectories),
        "train/nonzero_trajectory_datums": float(len(datums)),
        "train/batch_size": float(len(datums)),
        "train/tokens": float(sum(item.num_tokens for item in datums)),
        "train/loss_tokens": float(sum(item.loss_tokens for item in datums)),
        "train/max_sequence_tokens": float(
            max((item.num_tokens for item in datums), default=0)
        ),
    }
    by_task: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[trajectory.example.task_type].append(float(trajectory.won))
    for task_type, values in sorted(by_task.items()):
        metrics[f"reward/success_rate/{task_type}"] = mean(values)
    return metrics


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    """把 PyTRIO 单次 batch 返回的数值指标统一加上 trainer 前缀。"""
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in dict(result.metrics).items():
            if isinstance(value, (int, float, np.number)):
                values[key].append(float(value))
    merged: dict[str, float] = {}
    for key, items in values.items():
        merged[f"trainer/{key}"] = mean(items)
    return merged


def pick_mean_loss(metrics: dict[str, float]) -> float | None:
    for key in ("trainer/loss_mean", "trainer/loss/mean"):
        if key in metrics:
            return metrics[key]
    return None


def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存训练 state 和供 eval.py 使用的 sampler weights。"""
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


def main(args: argparse.Namespace) -> None:
    """运行 rollout -> advantage -> PPO backward -> optimizer 的训练闭环。"""
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
        group_size=args.group_size,
        max_steps=args.max_episode_steps,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )
    ppo_config = {
        "clip_low_threshold": args.ppo_clip_low,
        "clip_high_threshold": args.ppo_clip_high,
    }

    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
        tags=["pytrio", "alfworld", "grpo"],
    )
    try:
        with tqdm(total=args.max_steps, desc="Training", unit="update") as progress:
            for update_index in range(args.max_steps):
                started = perf_counter()
                games_batch = take_batch(
                    games,
                    update_index * args.games_per_batch,
                    args.games_per_batch,
                )
                progress.set_postfix(phase="sampler", refresh=True)
                sampling_client = training_client.save_weights_and_get_sampling_client()

                progress.set_postfix(phase="rollout", refresh=True)
                with tqdm(
                    total=len(games_batch) * args.group_size,
                    desc=f"Update {update_index + 1} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    batch = rollout_batch(
                        sampling_client,
                        tokenizer,
                        games_batch,
                        rollout_config,
                        progress_callback=rollout_progress.update,
                    )
                progress.set_postfix(phase="datums", refresh=True)
                logical_steps = logical_step_count(batch.trajectories)
                datums = build_training_datums(
                    batch.trajectories,
                    max_sequence_tokens=args.max_trajectory_tokens,
                )
                if len(datums) > MAX_TRAIN_BATCH_SIZE:
                    raise ValueError(
                        f"训练 Datum 数 {len(datums)} 超过平台 batch 上限 "
                        f"{MAX_TRAIN_BATCH_SIZE}"
                    )

                progress.set_postfix(phase="backward", refresh=True)
                trainer_results = []
                if datums:
                    result = training_client.forward_backward(
                        [item.datum for item in datums],
                        loss_fn="ppo",
                        loss_fn_config=ppo_config,
                    ).result()
                    trainer_results.append(result)
                    progress.set_postfix(phase="optimizer", refresh=True)
                    training_client.optim_step(adam_params).result()

                if args.save_every > 0 and (update_index + 1) % args.save_every == 0:
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-update-{update_index + 1}",
                    )

                metrics = rollout_metrics(
                    batch,
                    datums,
                    logical_steps,
                )
                metrics.update(merge_trainer_metrics(trainer_results))
                metrics["train/update_skipped"] = float(not datums)
                metrics["time/update_seconds"] = perf_counter() - started
                swanlab.log(metrics, step=update_index)

                loss = pick_mean_loss(metrics)
                loss_text = "skipped" if not datums else (
                    f"{loss:.4f}" if loss is not None else "missing"
                )
                progress.update(1)
                progress.set_postfix(
                    success=f"{metrics['reward/success_rate']:.3f}",
                    loss=loss_text,
                    step_s=f"{metrics['time/update_seconds']:.1f}",
                    refresh=True,
                )
                tqdm.write(
                    f"update={update_index + 1}/{args.max_steps} "
                    f"success={metrics['reward/success_rate']:.3f} "
                    f"reward={metrics['reward/mean']:.3f} "
                    f"valid_action={metrics['rollout/valid_action_rate']:.3f} "
                    f"mean_steps={metrics['rollout/steps_mean']:.2f} "
                    f"loss={loss_text}"
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
