"""准备 DAPO-Math-17K 训练集和 AIME25 评测集。

DAPO-Math 的官方 parquet 是约 100 倍重复版本，不能直接拿来训练。本脚本按
规范化后的 question 去重，丢弃同题但 ground truth 冲突的组，再用固定 seed
切出 50 条 dev smoke-test 数据，其余作为训练集。同时下载固定版本的 AIME25，
供 ``eval.py`` 使用。

运行：

    uv run python prepare_data.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DAPO_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
AIME25_DATASET_ID = "yentinglin/aime_2025"
AIME25_REVISION = "6f71d77b0b89b9dabe07ab466c51df33f514df7f"
AIME25_OUTPUT_NAME = "aime_2025"
AIME25_EXPECTED_ROWS = 30
AIME25_REQUIRED_COLUMNS = ("problem", "answer")
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


@dataclass(frozen=True)
class PreparationStats:
    """数据清洗统计。"""

    total_rows: int
    invalid_rows: int
    duplicate_rows: int
    conflicting_questions: int


def strip_dapo_template(question: str) -> str:
    """剥掉和 ``\\boxed{}`` 输出要求冲突的 DAPO 外层模板。"""
    if question.startswith(PROMPT_PREFIX):
        question = question[len(PROMPT_PREFIX) :]
    if question.endswith(PROMPT_SUFFIX):
        question = question[: -len(PROMPT_SUFFIX)]
    return question.strip()


def normalize_question_key(question: str) -> str:
    """生成只用于去重和切分的稳定题面 key。"""
    return " ".join(strip_dapo_template(question).split())


def normalize_row(index: int, row: dict[str, Any]) -> dict[str, str] | None:
    """把 verl 格式行转换为本项目的最小训练格式。"""
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], dict):
        return None

    question = strip_dapo_template(str(prompt[0].get("content") or ""))
    reward_model = row.get("reward_model")
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else None
    )
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None
    answer = str(ground_truth or "").strip()
    if not question or not answer:
        return None

    extra_info = row.get("extra_info")
    source_id = extra_info.get("index") if isinstance(extra_info, dict) else None
    return {
        "id": str(source_id or index),
        "question": question,
        "answer": answer,
        "data_source": str(row.get("data_source") or "math_dapo"),
    }


def deduplicate_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, str]], PreparationStats]:
    """按规范化 question 去重，并完整删除 ground truth 冲突的题组。"""
    unique: dict[str, dict[str, str]] = {}
    conflicts: set[str] = set()
    total_rows = 0
    invalid_rows = 0
    duplicate_rows = 0

    for index, row in enumerate(rows):
        total_rows += 1
        record = normalize_row(index, row)
        if record is None:
            invalid_rows += 1
            continue

        key = normalize_question_key(record["question"])
        if key in conflicts:
            duplicate_rows += 1
            continue

        previous = unique.get(key)
        if previous is None:
            unique[key] = record
            continue

        duplicate_rows += 1
        if previous["answer"] != record["answer"]:
            conflicts.add(key)
            del unique[key]
        elif record["id"] < previous["id"]:
            # 同题同答案时固定保留字典序最小的源 ID，避免依赖 parquet 行顺序。
            unique[key] = record

    records = sorted(
        unique.values(),
        key=lambda item: (normalize_question_key(item["question"]), item["id"]),
    )
    return records, PreparationStats(
        total_rows=total_rows,
        invalid_rows=invalid_rows,
        duplicate_rows=duplicate_rows,
        conflicting_questions=len(conflicts),
    )


def split_records(
    records: list[dict[str, str]],
    *,
    dev_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """固定 seed 打乱后切分 train/dev，两个 split 不共享题面。"""
    if dev_size < 1:
        raise ValueError("--dev-size must be >= 1")
    if dev_size >= len(records):
        raise ValueError("--dev-size must be smaller than the deduplicated dataset")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[dev_size:], shuffled[:dev_size]


def write_jsonl(records: Iterable[dict[str, str]], path: Path) -> None:
    """写入 UTF-8 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """解析数据准备参数。"""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="可选的本地 DAPO-Math parquet；默认从 Hugging Face Hub 读取",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "datasets",
        help="训练集、dev 集和 AIME25 的保存目录。",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=50,
        help="从去重后的训练题中切出的 dev 题目数。",
    )
    parser.add_argument("--seed", type=int, default=42, help="数据切分随机种子。")
    parser.add_argument(
        "--force-aime25",
        action="store_true",
        help="删除并重新下载已经存在的本地 AIME25。",
    )
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """流式读取本地 parquet 或官方 Hub 数据，避免展开百万行到内存。"""
    cache_dir = args.output_dir / ".hf-cache"
    if args.raw is not None:
        if not args.raw.is_file():
            raise FileNotFoundError(f"找不到本地 DAPO-Math parquet：{args.raw}")
        # datasets 的 IterableDataset 会初始化 torch shared memory；数据准备不需要
        # PyTorch，直接按 Arrow record batch 读取更轻量，也不会复制一份 parquet cache。
        import pyarrow.parquet as parquet

        def iter_parquet() -> Iterable[dict[str, Any]]:
            columns = [
                "data_source",
                "prompt",
                "reward_model",
                "extra_info",
            ]
            for batch in parquet.ParquetFile(args.raw).iter_batches(
                batch_size=65536,
                columns=columns,
            ):
                yield from batch.to_pylist()

        return iter_parquet()
    return load_dataset(
        DAPO_DATASET_ID,
        revision=DAPO_REVISION,
        split="train",
        cache_dir=str(cache_dir),
        streaming=True,
    )


def as_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    """把本地 AIME25 的 Dataset/单 split DatasetDict 统一为 Dataset。"""
    if isinstance(dataset, Dataset):
        return dataset
    if "train" not in dataset:
        raise ValueError(f"AIME25 缺少 train split：{list(dataset)}")
    return dataset["train"]


def validate_aime25(dataset: Dataset, location: str) -> None:
    """校验 AIME25 的行数、字段和首尾样本。"""
    missing = sorted(
        set(AIME25_REQUIRED_COLUMNS) - set(dataset.column_names)
    )
    if missing:
        raise ValueError(
            f"{AIME25_DATASET_ID} 在 {location} 缺少字段 {missing}"
        )
    if len(dataset) != AIME25_EXPECTED_ROWS:
        raise ValueError(
            f"{AIME25_DATASET_ID} 在 {location} 应有 "
            f"{AIME25_EXPECTED_ROWS} 题，实际为 {len(dataset)} 题"
        )
    for index in (0, len(dataset) - 1):
        empty = [
            column
            for column in AIME25_REQUIRED_COLUMNS
            if not str(dataset[index].get(column, "")).strip()
        ]
        if empty:
            raise ValueError(f"AIME25 第 {index} 题存在空字段：{empty}")


def prepare_aime25(
    output_dir: Path,
    *,
    force: bool,
) -> Path:
    """下载并校验本项目独立保存的固定版本 AIME25。"""
    output_path = output_dir / AIME25_OUTPUT_NAME
    if output_path.exists() and not force:
        local = as_dataset(load_from_disk(str(output_path)))
        validate_aime25(local, str(output_path))
        print(
            f"[skip] AIME25: 本地数据已存在且校验通过 "
            f"({len(local)} rows) -> {output_path}"
        )
        return output_path

    if output_path.exists():
        shutil.rmtree(output_path)

    cache_dir = output_dir / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[download] {AIME25_DATASET_ID}@{AIME25_REVISION} [train]"
    )
    dataset = load_dataset(
        AIME25_DATASET_ID,
        revision=AIME25_REVISION,
        split="train",
        cache_dir=str(cache_dir),
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    validate_aime25(dataset, "Hugging Face")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    saved = as_dataset(load_from_disk(str(output_path)))
    validate_aime25(saved, str(output_path))
    print(f"[done] AIME25: {len(saved)} rows -> {output_path}")
    return output_path


def main() -> None:
    """生成训练/dev JSONL，并下载本项目自己的 AIME25。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_aime25(args.output_dir, force=args.force_aime25)

    records, stats = deduplicate_rows(load_source(args))
    if not records:
        raise ValueError("DAPO-Math 清洗去重后为空")

    train_records, dev_records = split_records(
        records,
        dev_size=args.dev_size,
        seed=args.seed,
    )
    write_jsonl(train_records, args.output_dir / "train.jsonl")
    write_jsonl(dev_records, args.output_dir / "dev.jsonl")

    print(f"source rows: {stats.total_rows:,}")
    print(f"invalid rows: {stats.invalid_rows:,}")
    print(f"duplicate rows: {stats.duplicate_rows:,}")
    print(f"conflicting questions dropped: {stats.conflicting_questions:,}")
    print(f"deduplicated questions: {len(records):,}")
    print(f"train: {len(train_records):,}")
    print(f"dev: {len(dev_records):,}")


if __name__ == "__main__":
    main()
