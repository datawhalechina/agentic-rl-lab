"""GSPO 同步训练入口：DAPO-Math 数据 + 序列级重要性比率裁剪（arXiv:2507.18071）。

快速接口试跑（短 completion 可能使整组退化并跳过更新）：

uv run python train.py \
    --max-steps 100 \
    --groups-per-step 4 \
    --group-size 8 \
    --max-prompt-tokens 2048 \
    --max-tokens 4096 \
    --swanlab-mode disabled

"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from data import ExampleCursor, shuffled_examples
from loss import GSPOConfig, GSPOMeta, build_datum, make_gspo_loss_fn
from rollout import (
    RolloutBatch,
    RolloutConfig,
    RolloutGroup,
    collect_rollout_batch,
)


trio.configure(timeout=600, sampling_timeout=18000)

MAX_COMPLETIONS_PER_BATCH = 128


def is_trainable_group(group: RolloutGroup) -> bool:
    """组内 reward 全相同的退化组 advantage 全为 0，对 GSPO loss 没有贡献。"""
    return any(sample.advantage != 0.0 for sample in group.samples)


def build_training_data(
    groups: list[RolloutGroup],
) -> tuple[list[trio.Datum], list[GSPOMeta]]:
    """把所有有效组的 completion 转成 Datum 与本地元数据，顺序一一对应。"""
    datums: list[trio.Datum] = []
    metas: list[GSPOMeta] = []
    for group in groups:
        for sample in group.samples:
            # prompt token 整组共享，由 RolloutGroup 持有。
            datum, meta = build_datum(group.prompt_tokens, sample)
            datums.append(datum)
            metas.append(meta)
    return datums, metas


def mean(values: list[float]) -> float:
    """空列表安全的均值。"""
    return sum(values) / len(values) if values else 0.0


def rollout_metrics(
    rollout_batch: RolloutBatch,
    train_groups: list[RolloutGroup],
    *,
    data_cursor_consumed: int,
) -> dict[str, float]:
    """汇总 reward、采样成本、长度和多样性指标。

    GSPO 一次采齐、不做补采，rollout_batch.train_groups 即本 step 采到的
    全部组（退化组也保留在内）；reward 指标基于全部样本统计，避免把退化
    组的 reward 藏起来，train/* 指标只统计过滤后真正进入 loss 的样本。
    """
    all_groups = rollout_batch.train_groups
    samples = [sample for group in all_groups for sample in group.samples]
    train_samples = [
        sample for group in train_groups for sample in group.samples
    ]
    completion_lengths = [
        float(len(sample.completion_tokens)) for sample in samples
    ]
    all_logprobs = [
        logprob
        for sample in samples
        for logprob in sample.sampling_logprobs
    ]
    unique_rates = [
        len({sample.text for sample in group.samples})
        / max(len(group.samples), 1)
        for group in all_groups
    ]

    return {
        # reward 指标覆盖全部采样样本；GSPO 无长度惩罚，shaped 等于 base。
        "reward/base_mean": mean(
            [sample.reward_result.base_reward for sample in samples]
        ),
        "reward/shaped_mean": mean(
            [sample.reward_result.shaped_reward for sample in samples]
        ),
        "reward/accuracy": mean(
            [float(sample.reward_result.correct) for sample in samples]
        ),
        "reward/format_rate": mean(
            [float(sample.reward_result.valid_format) for sample in samples]
        ),
        # 采样组数与过滤退化组后真正参与训练的组数。
        "rollout/groups": float(len(all_groups)),
        "rollout/train_groups": float(len(train_groups)),
        "rollout/degenerate_groups": float(len(all_groups) - len(train_groups)),
        # 全部 rollout 成本与最终真正进入 GSPO loss 的数据量。
        "rollout/completions": float(len(samples)),
        "rollout/train_completions": float(len(train_samples)),
        "rollout/completion_tokens": float(sum(completion_lengths)),
        "rollout/mean_completion_tokens": mean(completion_lengths),
        "rollout/max_completion_tokens": max(completion_lengths, default=0.0),
        # 平均负 logprob 反映采样 token 的意外程度；文本去重率反映组内多样性。
        "rollout/mean_sampled_token_surprisal": (
            -mean(all_logprobs) if all_logprobs else 0.0
        ),
        "rollout/unique_completion_rate": mean(unique_rates),
        # 实际训练 Datum 数量，以及数据游标已消耗的题目数量。
        "train/datums": float(len(train_samples)),
        "train/data_cursor_consumed": float(data_cursor_consumed),
    }


def merge_trainer_metrics(result: Any) -> dict[str, float]:
    """保留 custom GSPO 指标，并为 PyTRIO 后端指标增加前缀。"""
    metrics: dict[str, float] = {}
    for key, value in dict(result.metrics).items():
        if not isinstance(value, (int, float, np.number)):
            continue
        output_key = key if key.startswith("gspo/") else f"trainer/{key}"
        metrics[output_key] = float(value)
    return metrics


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """将 Path 转成 SwanLab 可序列化配置。"""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(training_client: Any, name: str) -> dict[str, str]:
    """同时保存可续训 state 与可评测 sampler weights。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(
        name=f"{name}-sampler-weights"
    ).result()
    paths = {"state": state.path, "sampler_weights": weights.path}
    print(f"Saved state: {paths['state']}")
    print(f"Saved sampler weights: {paths['sampler_weights']}")
    return paths


def saved_path_fields(
    paths: dict[str, str],
    *,
    prefix: str,
) -> dict[str, Any]:
    """把 PyTRIO checkpoint 路径转换为 SwanLab 文本字段。"""
    return {
        f"{prefix}/state_path": swanlab.Text(paths["state"]),
        f"{prefix}/sampler_weights_path": swanlab.Text(
            paths["sampler_weights"]
        ),
    }


def parse_args() -> argparse.Namespace:
    """解析并校验 GSPO 训练入口参数。"""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="训练总 step 数。",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "datasets" / "train.jsonl",
        help="训练集 JSONL 文件路径。",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="最多使用的训练题数；0 表示全部使用。",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="PyTRIO 远端训练使用的基础模型。",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA 的 rank。",
    )
    parser.add_argument(
        "--groups-per-step",
        type=int,
        default=4,
        help=(
            "每个训练 step 采样的题目组数；与 --group-size 的乘积"
            f"不得超过 {MAX_COMPLETIONS_PER_BATCH}。"
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="每道题采样的 completion 数量。",
    )
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=16,
        help="同时采样的 prompt group 数量上限。",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1024,
        help="单条 prompt 的最大 token 数。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="单条 completion 的最大生成 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p 采样阈值。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k 采样阈值；-1 表示不限制。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="AdamW 学习率。",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW 的 beta1。",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW 的 beta2。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、采样和训练使用的随机种子。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="每隔多少个 step 保存 checkpoint；0 表示只保存最终结果。",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="运行名称；默认根据总 step 数生成。",
    )
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-gspo",
        help="SwanLab 项目名称。",
    )
    parser.add_argument(
        "--swanlab-workspace",
        default=None,
        help="SwanLab workspace；默认使用当前账号。",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 实验记录模式。",
    )
    args = parser.parse_args()

    positive_names = (
        "max_steps",
        "lora_rank",
        "groups_per_step",
        "group_size",
        "rollout_concurrency",
        "max_prompt_tokens",
        "max_tokens",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.group_size < 2:
        raise ValueError("--group-size must be >= 2")
    batch_completions = args.groups_per_step * args.group_size
    if batch_completions > MAX_COMPLETIONS_PER_BATCH:
        raise ValueError(
            "--groups-per-step * --group-size must be <= "
            f"{MAX_COMPLETIONS_PER_BATCH}, got {batch_completions}"
        )
    if args.max_train_samples < 0 or args.save_every < 0:
        raise ValueError("--max-train-samples and --save-every must be >= 0")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid sampling temperature/top-p")

    args.run_name = args.run_name or f"gspo-qwen35-4b-{args.max_steps}steps"
    return args


def run_training(args: argparse.Namespace) -> None:
    """执行同步训练循环；每个 step 内部并发 rollout。"""
    # 打乱训练题；GSPO 不做 Dynamic Sampling，游标每 step 固定取一批。
    examples = shuffled_examples(args.data, args.seed)
    if args.max_train_samples > 0:
        examples = examples[: args.max_train_samples]
    cursor = ExampleCursor(examples)

    # 创建远端 LoRA 训练客户端，并集中组装 rollout 与优化器配置。
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    gspo_config = GSPOConfig()
    adam = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    # 将完整超参数、GSPO 裁剪配置和依赖版本写入本次 SwanLab 实验。
    run = swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        name=args.run_name,
        mode=args.swanlab_mode,
        config={
            **serializable_config(args),
            "gspo": asdict(gspo_config),
            "dataset_size": len(examples),
            "pytrio_version": package_version("pytrio"),
            "swanlab_version": package_version("swanlab"),
        },
        tags=["PyTRIO", "GSPO", "DAPO-Math-17K"],
        log_dir=str(Path(__file__).resolve().parent / "swanlog"),
    )

    try:
        with tqdm(
            total=args.max_steps,
            desc="Training GSPO",
            unit="step",
        ) as progress:
            for step in range(args.max_steps):
                started = perf_counter()
                progress.set_postfix(phase="sampler", refresh=True)

                # 用当前最新训练权重创建采样客户端，保证 rollout 跟随当前策略。
                sampling_client = training_client.save_weights_and_get_sampling_client()

                progress.set_postfix(phase="rollout", refresh=True)
                # 普通组采样：每 step 固定取一批题，一次采齐、不做补采。
                step_rollout_config = replace(
                    rollout_config,
                    seed=args.seed + step * args.groups_per_step,
                )
                step_examples = cursor.take(args.groups_per_step)
                with tqdm(
                    total=len(step_examples),
                    desc=f"Step {step + 1}/{args.max_steps} rollout",
                    unit="group",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    rollout_batch = asyncio.run(
                        collect_rollout_batch(
                            sampling_client,
                            tokenizer,
                            step_examples,
                            step_rollout_config,
                            concurrency=args.rollout_concurrency,
                            progress_callback=rollout_progress.update,
                        )
                    )

                # advantage 全 0 的退化组不送去远端计算；它们的零目标仍通过
                # 原始 rollout completion 数保留在 GSPO loss 分母中。
                normalization_sequences = sum(
                    len(group.samples) for group in rollout_batch.train_groups
                )
                normalization_tokens = sum(
                    len(sample.completion_tokens)
                    for group in rollout_batch.train_groups
                    for sample in group.samples
                )
                train_groups = [
                    group
                    for group in rollout_batch.train_groups
                    if is_trainable_group(group)
                ]
                datums, metas = build_training_data(train_groups)

                # 只有存在有效训练样本时，才执行一次反向传播和参数更新。
                trainer_result = None
                if datums:
                    progress.set_postfix(phase="backward", refresh=True)
                    trainer_result = training_client.forward_backward_custom(
                        datums,
                        make_gspo_loss_fn(
                            metas,
                            gspo_config,
                            normalization_sequences=normalization_sequences,
                            normalization_tokens=normalization_tokens,
                        ),
                    ).result()
                    progress.set_postfix(phase="optimizer", refresh=True)
                    training_client.optim_step(adam).result()

                # 合并 rollout 与远端训练指标，统一记录到 SwanLab。
                metrics = rollout_metrics(
                    rollout_batch,
                    train_groups,
                    data_cursor_consumed=cursor.consumed,
                )
                if trainer_result is not None:
                    metrics.update(merge_trainer_metrics(trainer_result))
                metrics["train/update_skipped"] = float(not datums)
                metrics["train/learning_rate"] = args.learning_rate

                # 按配置周期性保存 checkpoint，并把远端路径记录到 SwanLab。
                checkpoint_paths: dict[str, str] | None = None
                if args.save_every > 0 and (step + 1) % args.save_every == 0:
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    checkpoint_paths = save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step + 1}",
                    )

                metrics["time/step_seconds"] = perf_counter() - started
                swanlab_record: dict[str, Any] = dict(metrics)
                if checkpoint_paths is not None:
                    swanlab_record.update(
                        saved_path_fields(
                            checkpoint_paths,
                            prefix="save/checkpoint",
                        )
                    )
                swanlab.log(swanlab_record, step=step)

                progress.update(1)
                progress.set_postfix(
                    reward=f"{metrics['reward/shaped_mean']:.3f}",
                    groups=(
                        f"{int(metrics['rollout/train_groups'])}/"
                        f"{int(metrics['rollout/groups'])}"
                    ),
                    step_s=f"{metrics['time/step_seconds']:.1f}",
                    refresh=True,
                )
                tqdm.write(
                    f"step={step + 1}/{args.max_steps} "
                    f"reward={metrics['reward/shaped_mean']:.3f} "
                    f"accuracy={metrics['reward/accuracy']:.3f} "
                    f"groups={int(metrics['rollout/train_groups'])}/"
                    f"{int(metrics['rollout/groups'])} "
                    f"tokens={int(metrics['rollout/completion_tokens'])}"
                )

        # 所有 step 完成后再保存一次最终 checkpoint，并记录远端路径。
        final_paths = save_checkpoint(
            training_client,
            f"{args.run_name}-final",
        )
        swanlab.log(
            saved_path_fields(final_paths, prefix="save"),
            step=args.max_steps,
        )
    except Exception as error:
        # 异常退出也显式标记实验状态，避免 SwanLab 中误显示为正常完成。
        run.finish(state="crashed", error=str(error))
        raise
    else:
        run.finish()


if __name__ == "__main__":
    run_training(parse_args())
