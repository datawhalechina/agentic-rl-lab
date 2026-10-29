"""在固定的 100 条 GeoQA test 样本上评测单个模型。

评测 Base：
uv run python eval.py

评测训练后模型：
uv run python eval.py \
    --model-path trio://run_xxx/sampler_weights/xxx-step-100-sampler
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytrio as trio
from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoImageProcessor

from train import (
    CHOICE_LETTERS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL,
    build_prompt_chunks,
    extract_choice,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SEED = 42
EVAL_SIZE = 100
DEFAULT_OUTPUT = SCRIPT_DIR / "eval-results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 GeoQA 多模态 GRPO")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-path",
        help="训练脚本输出的 Sampler 权重路径，不传则评测 Base 模型",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        default=EVAL_SIZE,
        help="默认评测固定的 100 条；调试时可缩小",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_eval_data(dataset_dir: Path, limit: int) -> Dataset:
    """读取下载阶段固定的 100 条测试数据。"""
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_dir / "test.parquet"),
        split="train",
    )
    return dataset.select(range(min(limit, len(dataset))))


def parse_response(response: Any, tokenizer: Any) -> tuple[str, str | None]:
    """读取单条采样结果中的文本和选项。"""
    sequence = response.sequences[0]
    text = sequence.text or tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    return text, extract_choice(text)


async def main(args: argparse.Namespace) -> None:
    eval_data = load_eval_data(args.dataset_dir.expanduser().resolve(), args.limit)
    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()
    image_processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        backend="pil",
    )
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=EVAL_SEED,
        temperature=0.0,
        stop="<|im_end|>",
    )

    async def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
        choices = [str(choice) for choice in row["choices"]]
        gold_choice = CHOICE_LETTERS[int(row["label"])]
        prompt = trio.ModelInput(
            chunks=build_prompt_chunks(
                tokenizer,
                image_processor,
                row["image"],
                str(row["subject"]),
                choices,
            )
        )
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
            return_text=True,
        )
        text, predicted_choice = parse_response(response, tokenizer)
        return {
            "id": int(row["id"]),
            "gold": gold_choice,
            "prediction": predicted_choice,
            "text": text,
        }

    # 固定测试集共享同一个 sampler 并发评测。
    results = await tqdm_asyncio.gather(
        *(evaluate_row(row) for row in eval_data),
        desc="评测 GeoQA",
        unit="sample",
    )

    total = len(results)
    correct = sum(result["prediction"] == result["gold"] for result in results)
    formatted = sum(result["prediction"] is not None for result in results)
    metrics = {
        "accuracy": correct / total,
        "format_rate": formatted / total,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "model_path": args.model_path,
                "eval_seed": EVAL_SEED,
                "eval_size": total,
                "metrics": metrics,
                "samples": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"模型：{args.model_path or args.base_model}")
    print(f"Accuracy：{metrics['accuracy']:.1%}")
    print(f"Format rate：{metrics['format_rate']:.1%}")
    print(f"评测结果：{output}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main(parse_args()))
    print(f"评测耗时：{time.perf_counter() - start_time:.2f}s")
