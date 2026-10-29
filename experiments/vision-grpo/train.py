"""使用 PyTRIO 在 GeoQA 上运行多模态 GRPO。

准备数据（仓库根目录）：
uv run python download-dataset.py

小规模测试：（5 元人民币）
uv run python train.py \
    --steps 20 \
    --batch-size 8 \
    --group-size 8 \
    --max-tokens 1024 \
    --save-every 10 \
    --swanlab-mode online
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoImageProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "datasets"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
IMAGE_PAD_TOKEN = "<|image_pad|>"
CHOICE_LETTERS = "ABCD"
BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)


@dataclass(frozen=True)
class RolloutSample:
    tokens: list[int]
    logprobs: list[float]
    text: str
    predicted_choice: str | None
    reward: float
    advantage: float


@dataclass(frozen=True)
class RolloutGroup:
    prompt_chunks: list[Any]
    prompt_length: int
    samples: list[RolloutSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoQA 多模态 GRPO")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 表示使用全部训练集"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-vision-grpo")
    parser.add_argument(
        "--experiment-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    parser.add_argument(
        "--weights-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每隔多少个 step 保存一次，0 表示只保存最终 checkpoint",
    )
    parser.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--show-samples", action="store_true")
    return parser.parse_args()


def load_geoqa_train(dataset_dir: Path, seed: int, max_samples: int) -> Dataset:
    """读取并打乱训练数据。"""
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_dir / "train.parquet"),
        split="train",
    ).shuffle(seed=seed)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def pick_batch(dataset: Dataset, step: int, batch_size: int) -> Dataset:
    """按 step 顺序取 batch，走完数据后从头继续。"""
    start = step * batch_size
    indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def encode_image(image: Image.Image, image_processor: Any) -> trio.ImageChunk:
    """将图片编码成 PyTRIO chunk，并计算视觉 token 数。"""
    chunk_format = "jpeg" if image.format in {"JPG", "JPEG"} else "png"
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, rgba).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format=chunk_format.upper())
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    expected_tokens = patches // int(image_processor.merge_size) ** 2
    return trio.ImageChunk(
        data=buffer.getvalue(),
        format=chunk_format,
        expected_tokens=expected_tokens,
    )


def format_question(subject: str, choices: list[str]) -> str:
    """将题目和四个选项整理成模型指令。"""
    choice_lines = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(CHOICE_LETTERS, choices, strict=True)
    )
    return (
        "请根据图片解答下面的几何选择题。\n"
        f"题目：{subject.strip()}\n"
        f"选项：\n{choice_lines}\n"
        "请先进行简单逻辑推理思考，再给出最终答案。"
        "最终选项格式必须是 \\boxed{A}、\\boxed{B}、\\boxed{C} 或 \\boxed{D}。"
    )


def build_prompt_chunks(
    tokenizer: Any,
    image_processor: Any,
    image: Image.Image,
    subject: str,
    choices: list[str],
) -> list[Any]:
    """先用 chat template 格式化 messages，再拆成图文 chunks。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": format_question(subject, choices)},
                {"type": "image", "image": "geoqa"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before_image, after_image = prompt.split(IMAGE_PAD_TOKEN)
    return [
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(before_image, add_special_tokens=False)
        ),
        encode_image(image, image_processor),
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(after_image, add_special_tokens=False)
        ),
    ]


def extract_choice(text: str) -> str | None:
    """提取回答中最后一个 boxed 选项。"""
    matches = BOXED_CHOICE_PATTERN.findall(text)
    return matches[-1].upper() if matches else None


async def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_chunks: list[Any],
    gold_choice: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> RolloutGroup:
    """异步采样同一道题的一组回答，并计算组内 advantage。"""
    prompt = trio.ModelInput(chunks=prompt_chunks)
    prompt_length = len(prompt)
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    )
    if response.input_tokens != prompt_length:
        raise ValueError(
            f"图文 prompt 长度不一致：local={prompt_length}, remote={response.input_tokens}"
        )

    raw_samples: list[tuple[list[int], list[float], str, str | None, float]] = []
    rewards: list[float] = []
    for sequence in response.sequences:
        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError("生成 token 与 logprob 长度不一致")
        text = sequence.text or tokenizer.decode(tokens, skip_special_tokens=True)
        predicted_choice = extract_choice(text)
        reward = float(predicted_choice == gold_choice)
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text, predicted_choice, reward))

    mean_reward = sum(rewards) / len(rewards)
    samples = [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            predicted_choice=predicted_choice,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for tokens, logprobs, text, predicted_choice, reward in raw_samples
    ]
    return RolloutGroup(prompt_chunks, prompt_length, samples)


def build_grpo_datum(group: RolloutGroup, sample: RolloutSample) -> trio.Datum:
    """把图文 prompt chunk 和 completion 拼成 GRPO Datum。"""
    model_input = trio.ModelInput(
        chunks=[
            *group.prompt_chunks,
            trio.types.EncodedTextChunk(tokens=sample.tokens[:-1]),
        ]
    )
    observation_length = group.prompt_length - 1
    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": np.asarray(
                [0] * observation_length + sample.tokens,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                [0.0] * observation_length + sample.logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                [0.0] * observation_length + [sample.advantage] * len(sample.tokens),
                dtype=np.float32,
            ),
        },
    )


def init_swanlab(args: argparse.Namespace, dataset_size: int) -> Any:
    """初始化训练日志。"""
    return swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "vision-grpo",
            "dataset": "hz2475/geoQA",
            "dataset_size": dataset_size,
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "enable_thinking": False,
            "lora_rank": args.lora_rank,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
        },
    )


async def save_checkpoint(
    training_client: trio.TrainingClient,
    weights_name: str,
    step: int,
) -> None:
    """同时保存推理权重和完整训练状态。"""
    prefix = f"{weights_name}-step-{step}"
    sampler_future = await training_client.save_weights_for_sampler_async(
        name=f"{prefix}-sampler"
    )
    state_future = await training_client.save_state_async(name=f"{prefix}-state")
    sampler_weights, training_state = await asyncio.gather(
        sampler_future,
        state_future,
    )
    print(f"Sampler 权重：{sampler_weights.path}")
    print(f"State 权重：{training_state.path}")


async def main(args: argparse.Namespace) -> None:
    train_data = load_geoqa_train(
        args.dataset_dir.expanduser().resolve(),
        args.seed,
        args.max_samples,
    )
    print(f"加载 GeoQA train 数据：{len(train_data)} 条")
    print(f"PyTRIO：{version('pytrio')}")

    service_client = trio.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()
    image_processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        backend="pil",
    )
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop="<|im_end|>",
    )
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)
    swanlab_run = init_swanlab(args, len(train_data))
    last_saved_step = 0

    try:
        for step in range(args.steps):
            batch_rows = list(pick_batch(train_data, step, args.batch_size))
            sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            datums: list[trio.Datum] = []
            all_samples: list[RolloutSample] = []
            prompt_rewards: list[float] = []
            degenerate_groups = 0

            # 一个 step 内的不同题目使用同一版 sampler 并发 rollout。
            rollout_groups = await tqdm_asyncio.gather(
                *(
                    run_rollout_group(
                        sampling_client,
                        tokenizer,
                        build_prompt_chunks(
                            tokenizer,
                            image_processor,
                            row["image"],
                            str(row["subject"]),
                            [str(choice) for choice in row["choices"]],
                        ),
                        CHOICE_LETTERS[int(row["label"])],
                        sampling_params,
                        args.group_size,
                    )
                    for row in batch_rows
                ),
                desc=f"Step {step + 1}/{args.steps} rollout",
                unit="题",
            )

            for row, group in zip(batch_rows, rollout_groups, strict=True):
                gold_choice = CHOICE_LETTERS[int(row["label"])]
                all_samples.extend(group.samples)
                rewards = [sample.reward for sample in group.samples]
                prompt_rewards.append(sum(rewards) / len(rewards))

                if args.show_samples:
                    print(f"\nGeoQA id={row['id']} gold={gold_choice}")
                    for index, sample in enumerate(group.samples):
                        print(
                            f"  sample={index} predicted={sample.predicted_choice} "
                            f"reward={sample.reward:.0f} text={sample.text!r}"
                        )

                # 整组 reward 相同时没有相对优势，不参与更新。
                if len(set(rewards)) == 1:
                    degenerate_groups += 1
                    continue
                datums.extend(
                    build_grpo_datum(group, sample)
                    for sample in group.samples
                    if sample.tokens
                )

            mean_output_tokens = sum(
                len(sample.tokens) for sample in all_samples
            ) / len(all_samples)
            tqdm_asyncio.write(
                f"本 batch 平均输出长度：{mean_output_tokens:.1f} tokens"
            )

            trainer_metrics: dict[str, float] = {}
            if datums:
                forward_backward = await training_client.forward_backward_async(
                    datums,
                    loss_fn="importance_sampling",
                )
                optim_step = await training_client.optim_step_async(adam_params)
                result = await forward_backward
                await optim_step
                trainer_metrics = {
                    key: float(value) for key, value in result.metrics.items()
                }

            mean_reward = sum(prompt_rewards) / len(prompt_rewards)
            format_rate = sum(
                sample.predicted_choice is not None for sample in all_samples
            ) / len(all_samples)
            degenerate_fraction = degenerate_groups / len(prompt_rewards)
            metrics = {
                "reward": mean_reward,
                "format_rate": format_rate,
                "degenerate_fraction": degenerate_fraction,
                "train_datums": len(datums),
                "rollout/completion_tokens_mean": mean_output_tokens,
                **{f"trainer/{key}": value for key, value in trainer_metrics.items()},
            }
            swanlab.log(metrics, step=step)

            loss_mean = trainer_metrics.get("loss_mean")
            loss_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            print(
                f"Step {step + 1}/{args.steps} | reward={mean_reward:.3f} | "
                f"format={format_rate:.1%} | degenerate={degenerate_fraction:.1%} | "
                f"datums={len(datums)} | loss_mean={loss_text}",
                flush=True,
            )

            current_step = step + 1
            if (
                args.save_weights
                and args.save_every > 0
                and current_step % args.save_every == 0
            ):
                await save_checkpoint(training_client, args.weights_name, current_step)
                last_saved_step = current_step

        if args.save_weights and last_saved_step != args.steps:
            await save_checkpoint(training_client, args.weights_name, args.steps)
    finally:
        swanlab_run.finish()


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main(parse_args()))
    print(f"训练耗时：{time.perf_counter() - start_time:.2f}s")
