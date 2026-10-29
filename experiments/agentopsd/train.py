"""使用 PyTRIO 内置 PPO 在纯文本 ALFWorld 上训练 AgentOPSD。

下载 ALFWorld 数据后，在当前目录运行：

uv run --extra alfworld python train.py \
    --max-steps 10 \
    --tasks-per-update 8 \
    --group-size 8 \
    --reshape-lambda 0.5 \
    --save-every 5 \
    --swanlab-mode disabled

同一个训练 step 中的 Student rollout、特权 Teacher 打分和旧策略对数概率始终共享
同一个冻结的 SamplingClient snapshot。Student rollout 永远不会看到技能。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from advantages import AgentOPSDConfig, AdvantageStats, assign_agentopsd_advantages
from data import SPLIT_DIRECTORIES, default_data_root, shuffled_games, take_batch
from loss import BuiltinPPOConfig, TrainingDatum, build_training_datums
from rollout import (
    MAX_STUDENT_TRAJECTORY_TOKENS,
    RolloutBatch,
    RolloutConfig,
    Trajectory,
    rollout_batch,
)
from skills import (
    DEFAULT_SKILLS_DIR,
    SKILL_SOURCE_COMMIT,
    SkillProvider,
)
from teacher import TeacherBatchStats, score_trajectories_async


trio.configure(timeout=1800)

PPO_EPOCHS = 1


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 AgentOPSD 训练和 PyTRIO 运行参数。"""

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示帮助信息并退出。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="ALFWorld 数据根目录。",
    )
    parser.add_argument(
        "--train-split",
        choices=list(SPLIT_DIRECTORIES),
        default="train",
        help="训练使用的数据划分。",
    )
    parser.add_argument(
        "--max-train-games",
        type=int,
        default=0,
        help="打乱后最多使用多少个游戏；0 表示使用整个数据划分。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="训练更新次数，不控制单条轨迹的环境交互轮数。",
    )
    parser.add_argument(
        "--tasks-per-update",
        type=int,
        default=16,
        help="每次训练更新抽取的 ALFWorld 游戏数量。",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="每个游戏并行采样的轨迹数量。",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="单条轨迹最多与环境交互的轮数。",
    )
    parser.add_argument(
        "--include-admissible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 prompt 中展示环境当前允许执行的动作。",
    )
    parser.add_argument(
        "--environment-asynchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 TextWorld 异步子进程环境。",
    )
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=MAX_STUDENT_TRAJECTORY_TOKENS,
        help="单条 Student 完整轨迹允许的最大 token 数。",
    )
    parser.add_argument(
        "--max-action-tokens",
        type=int,
        default=512,
        help="每轮 assistant action 最多生成的 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Student rollout 的采样温度。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Student rollout 的 nucleus sampling 概率阈值。",
    )
    parser.add_argument(
        "--teacher-concurrency",
        type=int,
        default=16,
        help="Teacher 并发计算整条轨迹 logprob 的最大请求数。",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="递归累计 Teacher 证据时的历史衰减系数。",
    )
    parser.add_argument(
        "--reshape-lambda",
        type=float,
        default=0.5,
        help=(
            "控制 Teacher 的逐轮证据对原始 GRPO advantage 的影响强度；"
            "0 表示所有轮次完全使用原始轨迹 advantage，1 表示完全使用"
            "轮级重塑结果，实际调整幅度仍受 --weight-bound 限制。"
        ),
    )
    parser.add_argument(
        "--weight-bound",
        type=float,
        default=0.2,
        help="每轮重塑权重相对 1.0 的最大偏移量。",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="优势标准化和 belief 裁剪使用的小常数。",
    )

    parser.add_argument(
        "--loss-mode",
        choices=["builtin_ppo"],
        default="builtin_ppo",
        help="训练损失实现，目前仅支持 PyTRIO 内置 PPO。",
    )
    parser.add_argument(
        "--ppo-clip-low",
        type=float,
        default=0.8,
        help="PPO importance ratio 的下裁剪阈值。",
    )
    parser.add_argument(
        "--ppo-clip-high",
        type=float,
        default=1.24,
        help="PPO importance ratio 的上裁剪阈值。",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 PyTRIO LoRA 训练客户端使用的基座模型。",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA 的秩。",
    )
    parser.add_argument(
        "--train-unembed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否训练输出词表映射层。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-6,
        help="Adam 优化器学习率。",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 优化器的一阶动量系数。",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Adam 优化器的二阶动量系数。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam 优化器的权重衰减系数。",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="梯度范数裁剪上限。",
    )

    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help="Teacher 使用的固定 ALFWorld SkillBank 目录。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="每隔多少个训练 step 保存状态和采样权重；0 表示只保存最终结果。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、环境和模型采样使用的随机种子。",
    )
    parser.add_argument(
        "--run-name",
        default="agentopsd-alfworld-qwen35-4b",
        help="checkpoint 和 SwanLab 实验使用的运行名称。",
    )
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-agentopsd",
        help="记录实验使用的 SwanLab 项目名称。",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 记录模式。",
    )
    args = parser.parse_args(argv)
    args.data_root = _path(args.data_root)
    args.skills_dir = _path(args.skills_dir)
    return args


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def population_std(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.std(np.asarray(items, dtype=np.float64), ddof=0))


def logical_step_count(trajectories: Sequence[Trajectory]) -> int:
    return sum(
        bool(step.completion_tokens)
        for trajectory in trajectories
        for step in trajectory.steps
    )


def rollout_metrics(
    batch: RolloutBatch,
    advantage_stats: AdvantageStats,
    teacher_stats: TeacherBatchStats,
    datums: Sequence[TrainingDatum],
) -> dict[str, float]:
    trajectories = batch.trajectories
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    credits = [credit for trajectory in trajectories for credit in trajectory.turn_credits]
    rewards = [float(trajectory.reward) for trajectory in trajectories]
    sequence_advantages = [
        float(trajectory.sequence_advantage) for trajectory in trajectories
    ]
    action_tokens_by_trajectory = [
        float(sum(len(step.completion_tokens) for step in trajectory.steps))
        for trajectory in trajectories
    ]
    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        grouped_rewards[trajectory.group_id].append(float(trajectory.reward))
    all_success = sum(all(value == 1.0 for value in values) for values in grouped_rewards.values())
    all_failure = sum(all(value == 0.0 for value in values) for values in grouped_rewards.values())
    group_count = len(grouped_rewards)
    multipliers = [
        credit.advantage / trajectory.sequence_advantage
        for trajectory in trajectories
        if abs(float(trajectory.sequence_advantage)) > 1e-12
        for credit in trajectory.turn_credits
    ]
    metrics = {
        "reward/mean": mean(rewards),
        "reward/std": population_std(rewards),
        "reward/success_rate": mean(float(trajectory.won) for trajectory in trajectories),
        "reward/all_success_group_rate": all_success / group_count,
        "reward/all_failure_group_rate": all_failure / group_count,
        "trajectory/turns_mean": mean(float(len(item.steps)) for item in trajectories),
        "trajectory/action_tokens_mean": mean(action_tokens_by_trajectory),
        "trajectory/truncated_rate": mean(
            float(item.truncated) for item in trajectories
        ),
        "trajectory/valid_tool_call_rate": mean(
            float(step.valid_format) for step in steps
        ),
        "trajectory/admissible_action_rate": mean(
            float(step.admissible) for step in steps
        ),
        "trajectory/valid_action_rate": mean(
            float(step.valid_action) for step in steps
        ),
        "trajectory/invalid_actions_mean": mean(
            float(item.invalid_action_count) for item in trajectories
        ),
        "advantage/sequence_std": population_std(sequence_advantages),
        "advantage/groups": float(advantage_stats.groups),
        "advantage/degenerate_group_rate": (
            advantage_stats.degenerate_groups / advantage_stats.groups
        ),
        "opsd/evidence_mean": mean(credit.evidence for credit in credits),
        "opsd/evidence_abs_mean": mean(abs(credit.evidence) for credit in credits),
        "opsd/delta_b_mean": mean(credit.delta_belief for credit in credits),
        "opsd/delta_b_abs_mean": mean(abs(credit.delta_belief) for credit in credits),
        "opsd/weight_mean": mean(credit.weight for credit in credits),
        "opsd/weight_min": min(credit.weight for credit in credits),
        "opsd/weight_max": max(credit.weight for credit in credits),
        "opsd/adv_multiplier_min": min(multipliers, default=1.0),
        "opsd/adv_multiplier_max": max(multipliers, default=1.0),
        "requests/teacher_count": float(teacher_stats.requests),
        "requests/reference_count": 0.0,
        "tokens/teacher_scored": float(teacher_stats.teacher_input_tokens),
        "tokens/teacher_action": float(teacher_stats.action_tokens),
        "train/logical_steps": float(logical_step_count(trajectories)),
        "train/nonzero_trajectory_datums": float(len(datums)),
        "train/batch_size": float(len(datums)),
        "train/tokens": float(sum(item.num_tokens for item in datums)),
        "train/action_tokens": float(sum(item.action_tokens for item in datums)),
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


def merge_result_metrics(result: Any, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{key}": float(value)
        for key, value in dict(result.metrics).items()
    }


def pick_mean_loss(metrics: dict[str, float]) -> float | None:
    return metrics.get("trainer/loss_mean")


def save_checkpoint(training_client: Any, name: str) -> None:
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(
        name=f"{name}-weights"
    ).result()
    tqdm.write(f"Saved state: {state.path}")
    tqdm.write(f"Saved sampler weights: {weights.path}")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main(args: argparse.Namespace) -> None:
    """Run snapshot -> rollout -> Teacher -> credit -> PPO -> optimizer."""

    games = shuffled_games(
        args.data_root,
        args.train_split,
        args.seed,
        max_games=args.max_train_games,
    )
    skill_provider = SkillProvider(args.skills_dir)
    agentopsd_config = AgentOPSDConfig(
        gamma=args.gamma,
        reshape_lambda=args.reshape_lambda,
        weight_bound=args.weight_bound,
        epsilon=args.epsilon,
    )
    ppo_config = BuiltinPPOConfig(args.ppo_clip_low, args.ppo_clip_high)
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_turns=args.max_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_action_tokens,
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

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=args.train_unembed,
    )
    tokenizer = training_client.get_tokenizer()

    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config={
            **serializable_args(args),
            "ppo_epochs": PPO_EPOCHS,
            "pytrio_version": importlib.metadata.version("pytrio"),
            "skill_source_commit": SKILL_SOURCE_COMMIT,
        },
        tags=["pytrio", "alfworld", "agentopsd", "builtin-ppo"],
        job_type="train",
    )
    try:
        with tqdm(
            total=args.max_steps,
            desc="AgentOPSD training",
            unit="step",
        ) as progress:
            for step_index in range(args.max_steps):
                step_started = perf_counter()
                games_batch = take_batch(
                    games,
                    step_index * args.tasks_per_update,
                    args.tasks_per_update,
                )

                progress.set_postfix(phase="snapshot", refresh=True)
                phase_started = perf_counter()
                sampling_client = (
                    training_client.save_weights_and_get_sampling_client()
                )
                snapshot_seconds = perf_counter() - phase_started
                snapshot_id = str(sampling_client.task_id)

                progress.set_postfix(phase="student-rollout", refresh=True)
                phase_started = perf_counter()
                with tqdm(
                    total=len(games_batch) * args.group_size,
                    desc=f"Step {step_index + 1} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    batch = rollout_batch(
                        sampling_client,
                        tokenizer,
                        games_batch,
                        rollout_config,
                        policy_snapshot_id=snapshot_id,
                        progress_callback=rollout_progress.update,
                    )
                rollout_seconds = perf_counter() - phase_started

                progress.set_postfix(phase="teacher", refresh=True)
                phase_started = perf_counter()
                _, teacher_stats = asyncio.run(
                    score_trajectories_async(
                        sampling_client,
                        tokenizer,
                        batch.trajectories,
                        skill_provider,
                        max_concurrency=args.teacher_concurrency,
                    )
                )
                teacher_seconds = perf_counter() - phase_started

                progress.set_postfix(phase="credit", refresh=True)
                phase_started = perf_counter()
                advantage_stats = assign_agentopsd_advantages(
                    batch.trajectories,
                    agentopsd_config,
                )
                datums = build_training_datums(batch.trajectories)
                credit_seconds = perf_counter() - phase_started

                backward_result = None
                optimizer_result = None
                backward_seconds = 0.0
                optimizer_seconds = 0.0
                if datums:
                    progress.set_postfix(phase="backward", refresh=True)
                    phase_started = perf_counter()
                    backward_result = training_client.forward_backward(
                        [item.datum for item in datums],
                        loss_fn="ppo",
                        loss_fn_config=ppo_config.as_loss_fn_config(),
                    ).result()
                    backward_seconds = perf_counter() - phase_started

                    progress.set_postfix(phase="optimizer", refresh=True)
                    phase_started = perf_counter()
                    optimizer_result = training_client.optim_step(
                        adam_params
                    ).result()
                    optimizer_seconds = perf_counter() - phase_started

                checkpoint_seconds = 0.0
                if (
                    args.save_every > 0
                    and (step_index + 1) % args.save_every == 0
                ):
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    phase_started = perf_counter()
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step_index + 1}",
                    )
                    checkpoint_seconds = perf_counter() - phase_started

                metrics = rollout_metrics(
                    batch,
                    advantage_stats,
                    teacher_stats,
                    datums,
                )
                if backward_result is not None:
                    metrics.update(
                        merge_result_metrics(backward_result, "trainer")
                    )
                if optimizer_result is not None:
                    metrics.update(
                        merge_result_metrics(optimizer_result, "optimizer")
                    )
                metrics.update(
                    {
                        "train/step_skipped": float(not datums),
                        "time/snapshot_seconds": snapshot_seconds,
                        "time/rollout_seconds": rollout_seconds,
                        "time/teacher_seconds": teacher_seconds,
                        "time/reference_seconds": 0.0,
                        "time/credit_seconds": credit_seconds,
                        "time/forward_backward_seconds": backward_seconds,
                        "time/optimizer_seconds": optimizer_seconds,
                        "time/checkpoint_seconds": checkpoint_seconds,
                        "time/step_seconds": perf_counter() - step_started,
                    }
                )
                swanlab.log(metrics, step=step_index)

                loss = pick_mean_loss(metrics)
                loss_text = "skipped" if not datums else (
                    f"{loss:.4f}" if loss is not None else "missing"
                )
                progress.update(1)
                progress.set_postfix(
                    success=f"{metrics['reward/success_rate']:.3f}",
                    loss=loss_text,
                    step_s=f"{metrics['time/step_seconds']:.1f}",
                    refresh=True,
                )
                tqdm.write(
                    f"Step {step_index + 1}/{args.max_steps} "
                    f"success={metrics['reward/success_rate']:.3f} "
                    f"datums={len(datums)} loss={loss_text}"
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
