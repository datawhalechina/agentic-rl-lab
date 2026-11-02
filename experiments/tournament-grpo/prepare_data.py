"""下载作者发布的 Tournament-GRPO 数据，并按作者的 4096-token 规则筛长。

运行（仓库根目录）：
uv run python experiments/tournament-grpo/prepare_data.py
"""

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from protocol import MAX_PROMPT_LENGTH, TOOL_SCHEMAS, render_chat


# 固定作者仓库 commit，保证数据与论文代码一致。
SOURCE_REVISION = "eea826cf66705cac9ec81823f746d0ccd6074228"
SOURCE_URL = f"https://raw.githubusercontent.com/PuffYang/Tournament_GRPO/{SOURCE_REVISION}"
SPLIT_FILES = {
    "train": "data/train_new_train_95.parquet",
    "val": "data/train_new_val_05.parquet",
}
# Deep Research Bench 评测资产，后续 eval.py 使用。
BENCH_FILES = {
    "query.jsonl": "deep_research_bench/data/prompt_data/query.jsonl",
    "criteria.jsonl": "deep_research_bench/data/criteria_data/criteria.jsonl",
    "reference.jsonl": "deep_research_bench/data/test_data/cleaned_data/reference.jsonl",
}


def download(remote: str, target: Path) -> Path:
    """下载固定 commit 的文件；已存在时直接复用。"""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{SOURCE_URL}/{remote}", timeout=60) as response:
            target.write_bytes(response.read())
    return target


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """只保留训练需要的字段；作者的 query/rubrics 原文和顺序不做修改。

    作者数据的 prompt 只有一条 user 消息，内容与 ground_truth.query 相同，所以只取 query。
    """
    ground_truth = row["reward_model"]["ground_truth"]
    return {
        "id": str(row["extra_info"]["index"]),
        "query": ground_truth["query"],
        "rubrics": ground_truth["rubrics"],
        "data_source": row["data_source"],
    }


def prompt_length(tokenizer: Any, query: str) -> int:
    """复现 verl filter_overlong_prompts：原 user 消息 + 原生工具 schema，不含研究 system。"""
    return len(render_chat(tokenizer, [{"role": "user", "content": query}], TOOL_SCHEMAS))


def prepare_split(path: Path, output_path: Path, tokenizer: Any) -> tuple[Counter[str], list[str]]:
    """按原顺序筛长并写成 JSONL；重复题目照原数据保留。"""
    counts: Counter[str] = Counter()
    dropped: list[str] = []
    with output_path.open("w", encoding="utf-8") as file:
        for row in pq.read_table(path).to_pylist():
            record = normalize_row(row)
            if prompt_length(tokenizer, record["query"]) > MAX_PROMPT_LENGTH:
                dropped.append(record["id"])
                continue
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["data_source"]] += 1
    return counts, dropped


def parse_args() -> argparse.Namespace:
    """解析数据准备命令行参数。"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets")
    # 筛长结果依赖 policy tokenizer，更换基座模型后需要重新运行。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    return parser.parse_args()


def main() -> None:
    """下载原始数据，生成 train/val JSONL 和 Deep Research Bench 评测文件。"""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    for split, remote in SPLIT_FILES.items():
        raw_path = download(remote, args.output_dir / "raw" / Path(remote).name)
        counts, dropped = prepare_split(raw_path, args.output_dir / f"{split}.jsonl", tokenizer)
        details = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
        print(f"{split}: total={sum(counts.values())}; {details}; overlong_dropped={dropped}")
    for name, remote in BENCH_FILES.items():
        download(remote, args.output_dir / "deep_research_bench" / name)
    print(f"deep_research_bench: {', '.join(BENCH_FILES)}")


if __name__ == "__main__":
    main()
