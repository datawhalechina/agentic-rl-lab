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
import random
import time
from importlib.metadata import version
from pathlib import Path

import pytrio as trio
import swanlab

from data import Encoder, load_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "train.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="异步 LoRA masked SFT 训练")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 表示使用全部训练集"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every",
        type=int,
        default=60,
        help="每隔多少个 step 保存一次推理权重；每个 epoch 结束总会保存，0 表示不做 step 级保存",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-pytrio-jev")
    parser.add_argument("--experiment-name", default=None, help="默认跟随 --base-model")
    parser.add_argument("--weights-name", default=None, help="默认跟随 --base-model")
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
    name = f"{weights_name}-step-{step}"
    future = await training_client.save_weights_for_sampler_async(name=name)
    path = (await future).path
    print(f"保存权重 {name}: {path}")
    return path


async def main(args: argparse.Namespace) -> None:
    records = load_jsonl(args.data)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"训练数据：{len(records)} 条 | PyTRIO：{version('pytrio')}")

    service_client = trio.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    encoder = Encoder(training_client.get_tokenizer())
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)

    swanlab_run = swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "masked-sft",
            "dataset": "typed-decisions+toolace",
            "dataset_size": len(records),
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
        },
    )

    global_step = 0
    last_saved_step = 0
    total_steps = args.epochs * (len(records) // args.batch_size)
    pending: list[asyncio.Task] = []
    try:
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(records)
            for start in range(0, len(records) - args.batch_size + 1, args.batch_size):
                batch = records[start : start + args.batch_size]
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

            # 每个 epoch 结束都存一份；与 step 存档点重合时跳过，避免重复保存。
            if last_saved_step != global_step:
                await asyncio.gather(*pending)
                pending.clear()
                await save_weights(training_client, args.weights_name, global_step)
                last_saved_step = global_step

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
