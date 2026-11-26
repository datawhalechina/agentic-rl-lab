"""判别决策模型的异步 LoRA masked SFT 训练：只在 marker 前一位置监督答案符号。

准备数据（仓库根目录）：
uv run python 00-prepare-data.py

小规模测试：
uv run python train.py \
    --max-samples 128 \
    --batch-size 16 \
    --epochs 1 \
    --save-every 0 \
    --swanlab-mode disabled

正式训练：
uv run python train.py \
    --epochs 2 \
    --batch-size 16 \
    --save-every 100 \
    --swanlab-mode online


换基座模型（experiment/weights 名字自动跟随）：
uv run python train.py --base-model Qwen/Qwen3.5-9B --epochs 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from importlib.metadata import version
from pathlib import Path

import pytrio as trio
import swanlab
from dotenv import load_dotenv

from data import Encoder, load_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "train.jsonl"
DEFAULT_DEV_DATA = SCRIPT_DIR / "datasets" / "dev.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="异步 LoRA masked SFT 训练")
    parser.add_argument(
        "--data", type=Path, default=Path(os.getenv("JEV_DATA", str(DEFAULT_DATA)))
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=Path(os.getenv("JEV_DEV_DATA", str(DEFAULT_DEV_DATA))),
        help="早停评估集，每 dev-eval-interval 跑一次 dev loss",
    )
    parser.add_argument(
        "--base-model", default=os.getenv("JEV_BASE_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--lora-rank", type=int, default=int(os.getenv("JEV_LORA_RANK", "32"))
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.getenv("JEV_EPOCHS", "2"))
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("JEV_BATCH_SIZE", "16"))
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("JEV_LEARNING_RATE", "1e-4")),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("JEV_MAX_SAMPLES", "0")),
        help="0 表示使用全部训练集",
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.getenv("JEV_SEED", "42"))
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=int(os.getenv("JEV_SAVE_EVERY", "60")),
        help="每隔多少个 step 保存一次推理权重；每个 epoch 结束总会保存，0 表示不做 step 级保存",
    )
    parser.add_argument(
        "--dev-eval-interval",
        type=float,

        default=float(os.getenv("JEV_DEV_EVAL_INTERVAL", "0.1")),
        help="dev 评估间隔，占总 step 的比例（0~1），例如 0.1 表示每 10% step 评估一次",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=int(os.getenv("JEV_EARLY_STOP_PATIENCE", "3")),
        help="dev loss 连续多少个评估点未创新低则早停；0 表示禁用早停",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default=os.getenv("JEV_SWANLAB_MODE", "online"),
    )
    parser.add_argument(
        "--swanlab-project",
        default=os.getenv("JEV_SWANLAB_PROJECT", "agentic-rl-lab-pytrio-jev"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("JEV_EXPERIMENT_NAME") or None,
        help="默认跟随 --base-model",
    )
    parser.add_argument(
        "--weights-name",
        default=os.getenv("JEV_WEIGHTS_NAME") or None,
        help="默认跟随 --base-model",
    )
    parser.add_argument(
        "--resume-from",
        default=os.getenv("JEV_RESUME_FROM") or None,
        help="续训：填写 save_state 保存的 trio://... checkpoint 路径；为空则从头训练",
    )
    args = parser.parse_args()
    model_slug = args.base_model.split("/")[-1].lower().replace(".", "")
    if args.experiment_name is None:
        args.experiment_name = f"pytrio-jev-{model_slug}"
    if args.weights_name is None:
        args.weights_name = f"pytrio-jev-{model_slug}"
    return args


async def save_weights(
    training_client: trio.TrainingClient, weights_name: str, step: int
) -> str:
    """每个存档点同时保存两份权重：

    1. 推理权重（save_weights_for_sampler_async）：LoRA 权重，用于采样/推理。
    2. 训练状态（save_state_async）：权重 + 优化器状态，用于断点续训。

    返回推理权重路径，训练状态路径仅打印。
    """
    name = f"{weights_name}-step-{step}"
    sampler_future = await training_client.save_weights_for_sampler_async(name=name)
    sampler_result = await sampler_future
    sampler_path = sampler_result.path
    print(f"保存推理权重 {name}: {sampler_path}")

    state_future = await training_client.save_state_async(name=name)
    state_result = await state_future
    state_path = state_result.path
    print(f"保存训练状态（断点续训）{name}: {state_path}")
    return sampler_path


def parse_step_from_checkpoint(path: str) -> int:
    """从 checkpoint 路径/名称里解析 step 数，用于续训时恢复 SwanLab 的 step 计数。

    命名约定为 `{weights_name}-step-{N}`，取末尾 `-step-N` 的 N；解析失败返回 0 并告警。
    """
    basename = path.rstrip("/").split("/")[-1]
    if "-step-" in basename:
        try:
            return int(basename.rsplit("-step-", 1)[-1])
        except ValueError:
            pass
    print(f"警告：无法从 checkpoint 路径解析 step（{path!r}），起始 step 按 0 处理")
    return 0


async def evaluate_dev(
    training_client: trio.TrainingClient,
    encoder: Encoder,
    dev_records: list[dict],
) -> tuple[float, int]:
    """在 dev 集上只前向计算 masked cross_entropy loss，不更新梯度。

    与训练 loss 同构：loss = loss_sum / supervised_tokens，只监督决策位置。
    forward_async 仅前向、不反向、不清/更新梯度，因此不会污染模型权重。
    """
    dev_datums = [encoder.build_sft_datum(record) for record in dev_records]
    supervised = sum(
        sum(d.loss_fn_inputs["weights"].data) for d in dev_datums
    )
    future = await training_client.forward_async(dev_datums, "cross_entropy")
    result = await future
    loss_sum = float(result.metrics["loss_sum"])
    return loss_sum / supervised, int(supervised)


async def main(args: argparse.Namespace) -> None:
    if not 0 < args.dev_eval_interval <= 1:
        raise ValueError("dev-eval-interval 必须在 (0, 1] 区间")
    if args.early_stop_patience < 0:
        raise ValueError("early-stop-patience 不能为负")
    records = load_jsonl(args.data)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    dev_records = load_jsonl(args.dev_data)
    if args.data.resolve() == args.dev_data.resolve():
        raise ValueError("训练集与 dev 集必须是不同文件")
    if not dev_records:
        raise ValueError("dev 集为空，无法评估 dev loss")
    train_ids = {record["id"] for record in records}
    dev_ids = {record["id"] for record in dev_records}
    if train_ids & dev_ids:
        raise ValueError("训练集与 dev 集存在重叠样本 ID")
    print(f"训练数据：{len(records)} 条 | PyTRIO：{version('pytrio')}")
    print(f"开发数据：{len(dev_records)} 条 | 每 {args.dev_eval_interval:.0%} step 评估 dev loss")

    service_client = trio.ServiceClient()
    start_step = 0
    if args.resume_from:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            path=args.resume_from
        )
        start_step = parse_step_from_checkpoint(args.resume_from)
        print(f"续训：从 {args.resume_from} 恢复权重与优化器状态，起始 step={start_step}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        print(f"从头训练：基座 {args.base_model} | LoRA rank={args.lora_rank} | seed={args.seed}")
    encoder = Encoder(training_client.get_tokenizer())
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)

    swanlab_run = swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "masked-sft",
            "dataset": "typed-decisions+toolace",
            "dataset_path": str(args.data),
            "dataset_size": len(records),
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "optimizer": "adam",
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
            "dev_data_path": str(args.dev_data),
            "dev_size": len(dev_records),
            "dev_eval_interval": args.dev_eval_interval,
            "early_stop_patience": args.early_stop_patience,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "swanlab_mode": args.swanlab_mode,
            "resume_from": args.resume_from,
            "start_step": start_step,
        },
    )

    global_step = start_step
    last_saved_step = start_step
    # 向上取整，保证 total_steps 与内层循环实际迭代的 batch 数一致，
    # 避免 len(records) 非 batch_size 整数倍时 step 数超出 total_steps（如 Step 276/270）。
    steps_per_epoch = (len(records) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    pending: list[asyncio.Task] = []
    dev_eval_interval_steps = max(1, int(total_steps * args.dev_eval_interval))
    next_dev_eval_step = start_step + dev_eval_interval_steps
    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    should_stop = False
    try:
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(records)
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]
                if len(batch) < args.batch_size:
                    # 末尾不足一个 batch 时跳过，保证每个 batch 大小一致；
                    # 步数仍按 steps_per_epoch（向上取整）计算，与 total_steps 对齐。
                    break
                datums = [encoder.build_sft_datum(record) for record in batch]
                supervised = sum(
                    sum(d.loss_fn_inputs["weights"].data) for d in datums
                )

                # 提交后不等待，loss 计算和 SwanLab 上报交给后台任务，训练流水线不断流。
                fwdbwd_future = await training_client.forward_backward_async(
                    datums, "cross_entropy"
                )
                optim_future = await training_client.optim_step_async(adam_params)
                global_step += 1

                async def log_step(fwdbwd_future, optim_future, supervised, epoch, step):
                    result = await fwdbwd_future
                    await optim_future
                    trainer_metrics = {
                        key: float(value) for key, value in result.metrics.items()
                    }
                    loss = trainer_metrics["loss_sum"] / supervised
                    swanlab.log(
                        {
                            "loss": loss,
                            "epoch": epoch,
                            "supervised_tokens": supervised,
                            **{
                                f"trainer/{key}": value
                                for key, value in trainer_metrics.items()
                            },
                        },
                        step=step,
                    )
                    print(
                        f"Step {step}/{total_steps} (epoch {epoch}) | "
                        f"loss={loss:.4f} | supervised_tokens={supervised:.0f}",
                        flush=True,
                    )

                pending.append(
                    asyncio.create_task(
                        log_step(fwdbwd_future, optim_future, supervised, epoch, global_step)
                    )
                )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    await asyncio.gather(*pending)
                    pending.clear()
                    await save_weights(training_client, args.weights_name, global_step)
                    last_saved_step = global_step

                # 到达评估点：收齐 pending loss 任务后，只前向算 dev loss 并做早停。
                if global_step >= next_dev_eval_step:
                    await asyncio.gather(*pending)
                    pending.clear()
                    dev_loss, dev_supervised = await evaluate_dev(
                        training_client, encoder, dev_records
                    )
                    swanlab.log(
                        {
                            "dev/loss": dev_loss,
                            "dev/supervised_tokens": dev_supervised,
                        },
                        step=global_step,
                    )
                    print(
                        f"Step {global_step}/{total_steps} | dev loss={dev_loss:.4f} "
                        f"(supervised_tokens={dev_supervised})"
                    )
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if (
                            args.early_stop_patience > 0
                            and epochs_without_improvement >= args.early_stop_patience
                        ):
                            print(
                                f"早停：dev loss 连续 {epochs_without_improvement} 个评估点未创新低 "
                                f"(best={best_dev_loss:.4f})"
                            )
                            should_stop = True
                            break
                    next_dev_eval_step += dev_eval_interval_steps

            # 每个 epoch 结束都存一份；与 step 存档点重合时跳过，避免重复保存。
            if last_saved_step != global_step:
                await asyncio.gather(*pending)
                pending.clear()
                await save_weights(training_client, args.weights_name, global_step)
                last_saved_step = global_step

            if should_stop:
                break

        await asyncio.gather(*pending)
        pending.clear()
        if last_saved_step != global_step:
            await save_weights(training_client, args.weights_name, global_step)
    finally:
        swanlab_run.finish()


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main(parse_args()))
    print(f"训练耗时：{time.perf_counter() - start_time:.2f}s")
    # records = load_jsonl(parse_args().data)
    # random.Random(parse_args().seed+1).shuffle(records)
    # print(records[:2])
