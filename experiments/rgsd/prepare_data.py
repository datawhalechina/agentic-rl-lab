"""下载 RubricHub（RuRL 子集）并整理成 RGSD 使用的 JSONL。

从仓库根目录运行：

    uv run python experiments/rgsd/prepare_data.py
    uv run python experiments/rgsd/prepare_data.py --domains medical

每个领域输出两份文件：

    experiments/rgsd/datasets/rubrichub_<domain>_train.jsonl
    experiments/rgsd/datasets/rubrichub_<domain>_eval.jsonl   # 固定 300 题

每行格式：

    {"id": "medical-000123", "domain": "medical", "question": "...",
     "rubrics": [{"criterion": "...", "weight": 9}, ...]}

注意：Hugging Face 公开的 RuRL parquet 没有区分 train / validation，行数（医学
29,681、科学 29,418）也与 RGSD 论文报告的训练集规模（12,519 / 19,806）不一致。
这里按固定 seed 打乱后自行切出 300 题评测集，其余作为训练集；因此数据划分与论文
不是同一份。数据 revision 被固定，避免上游更新导致划分静默变化。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "datasets"

REPO_ID = "sojuL/RubricHub_v1"
REVISION = "3837d55971473a872e84879c88f708b8da3ec2ef"
# 上游文件名里的 "rurbichub" 是原始拼写，不要修正。
DOMAIN_FILES = {
    "medical": "RuRL/rurbichub_v1_Medical.parquet",
    "science": "RuRL/rurbichub_v1_Science.parquet",
}
EXPECTED_ROWS = {"medical": 29_681, "science": 29_418}
EVAL_SIZE = 300
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    """解析领域选择、输出目录和强制重建参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_FILES),
        default=sorted(DOMAIN_FILES),
        help="要准备的领域；默认 medical 和 science",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face 下载缓存；默认使用 <data-root>/.cache",
    )
    parser.add_argument(
        "--force", action="store_true", help="覆盖已经存在的 JSONL 输出"
    )
    return parser.parse_args()


def convert_row(domain: str, index: int, row: dict) -> dict | None:
    """把一行 RuRL 记录转换为 RGSD 样本；不符合单轮问答假设的返回 None。"""
    # RGSD 的 Student 只看一条 user 消息，多轮或带 system 的样本不进入本实验。
    messages = row["prompt"] or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        return None
    question = str(messages[0].get("content") or "").strip()
    if not question:
        return None

    rubrics = []
    for item in row["Rubrics"] or []:
        criterion = str(item.get("criterion") or "").strip()
        weight = item.get("points")
        # 权重为 0 或负数的条目无法进入加权满足率，直接丢弃。
        if criterion and isinstance(weight, int) and weight > 0:
            rubrics.append({"criterion": criterion, "weight": weight})
    if not rubrics:
        return None

    return {
        "id": f"{domain}-{index:06d}",
        "domain": domain,
        "question": question,
        "rubrics": rubrics,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """把样本逐行写成 UTF-8 JSONL。"""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_domain(domain: str, data_root: Path, cache_dir: Path, force: bool) -> None:
    """下载一个领域的 parquet，过滤、划分并落盘。"""
    train_path = data_root / f"rubrichub_{domain}_train.jsonl"
    eval_path = data_root / f"rubrichub_{domain}_eval.jsonl"
    if train_path.exists() and eval_path.exists() and not force:
        print(f"[skip] {domain}: 输出已存在 -> {train_path.name}, {eval_path.name}")
        return

    print(f"[download] {REPO_ID}@{REVISION[:8]} {DOMAIN_FILES[domain]}")
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=DOMAIN_FILES[domain],
        repo_type="dataset",
        revision=REVISION,
        cache_dir=str(cache_dir),
    )
    # 只读需要的两列；extra_info 是 prompt / rubric 的冗余副本，体积占了大半。
    table = pq.read_table(parquet_path, columns=["prompt", "Rubrics"])
    if table.num_rows != EXPECTED_ROWS[domain]:
        raise ValueError(
            f"{domain} 应有 {EXPECTED_ROWS[domain]:,} 行，实际 {table.num_rows:,} 行"
        )

    rows = []
    for index, raw in enumerate(table.to_pylist()):
        converted = convert_row(domain, index, raw)
        if converted is not None:
            rows.append(converted)
    dropped = table.num_rows - len(rows)
    if len(rows) <= EVAL_SIZE:
        raise ValueError(f"{domain} 过滤后只剩 {len(rows)} 条，不足以切出评测集")

    # 先按固定 seed 打乱，再取前 300 条作评测集，保证两份划分互不重叠且可复现。
    random.Random(SPLIT_SEED).shuffle(rows)
    eval_rows, train_rows = rows[:EVAL_SIZE], rows[EVAL_SIZE:]

    data_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(train_path, train_rows)
    mean_k = sum(len(row["rubrics"]) for row in rows) / len(rows)
    print(
        f"[done] {domain}: train={len(train_rows):,} eval={len(eval_rows):,} "
        f"dropped={dropped:,} mean_criteria={mean_k:.1f}"
    )


def main(args: argparse.Namespace) -> None:
    """依次准备用户选择的领域。"""
    data_root = args.data_root.resolve()
    cache_dir = (args.cache_dir or data_root / ".cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        prepare_domain(domain, data_root, cache_dir, args.force)


if __name__ == "__main__":
    main(parse_args())
