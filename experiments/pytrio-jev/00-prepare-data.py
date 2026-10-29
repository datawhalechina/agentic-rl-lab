"""下载并准备训练/评测数据。

用法（仓库根目录）：
uv run python experiments/pytrio-jev/00-prepare-data.py

产物（experiments/pytrio-jev/datasets/）：
- train.jsonl       typed-decisions train 切出 960 案例 + ToolACE 1200 条
- dev.jsonl         typed-decisions 案例级切分 120 案例（选优用）
- calibration.jsonl typed-decisions 案例级切分 120 案例（温度拟合用）
- eval/*.jsonl      五个评测基准文件
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "datasets"
EVAL_DIR = DATASET_DIR / "eval"

TYPED_DECISIONS_REPO = "LocalLLaMA/typed-decisions"
TOOLACE_REPO = "Team-ACE/ToolACE"
TOOLACE_TRAIN_SIZE = 1200
SPLIT_SEED = "pytrio-jev"
SAMPLE_SEED = 42

EVAL_BASE = (
    "https://raw.githubusercontent.com/InternLM/Intern-Decision/"
    "main/benchmarks/accuracy-v1"
)
EVAL_FILES = {
    "jevbench_easy.jsonl": "jevbench/easy.jsonl",
    "jevbench_original.jsonl": "jevbench/original.jsonl",
    "jevbench_hard.jsonl": "jevbench/hard.jsonl",
    "typed_decisions.jsonl": "typed_decisions/test.jsonl",
    "toolace.jsonl": "toolace/test.jsonl",
}


def download_eval_files() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, remote in EVAL_FILES.items():
        target = EVAL_DIR / name
        if target.exists():
            continue
        with urllib.request.urlopen(f"{EVAL_BASE}/{remote}") as response:
            target.write_bytes(response.read())
        print(f"下载 {name}: {sum(1 for _ in open(target))} 行")


def convert_typed_decisions(row: dict) -> dict:
    """HF 行 → canonical record；gold 取硬标签（软分布在训练里不用）。"""
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    return {
        "id": row["id"],
        "state": json.loads(row["state"]),
        "questions": questions,
        "targets": {field: {"label": gold[field]["label"]} for field in questions},
    }


def split_typed_decisions() -> tuple[list[dict], list[dict], list[dict]]:
    """案例级确定性切分：每个 workflow 240 train / 30 dev / 30 calibration。"""
    dataset = load_dataset(TYPED_DECISIONS_REPO, "all", split="train")
    print(f"typed-decisions train: {len(dataset)} 案例")
    by_workflow: dict[str, list[dict]] = {}
    for row in dataset:
        by_workflow.setdefault(row["workflow"], []).append(row)

    train, dev, calibration = [], [], []
    for workflow, rows in sorted(by_workflow.items()):
        rows = sorted(
            rows,
            key=lambda r: hashlib.sha256(f"{SPLIT_SEED}:{r['id']}".encode()).hexdigest(),
        )
        train.extend(convert_typed_decisions(r) for r in rows[:240])
        dev.extend(convert_typed_decisions(r) for r in rows[240:270])
        calibration.extend(convert_typed_decisions(r) for r in rows[270:300])
        print(f"  {workflow}: {len(rows)} → 240 train / 30 dev / 30 calibration")
    return train, dev, calibration


def load_eval_request_keys() -> set[str]:
    """ToolACE 310 条 eval 行的请求文本指纹，训练采样要避开它们。"""
    keys = set()
    with open(EVAL_DIR / "toolace.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            keys.add(" ".join(row["state"]["request"].lower().split()))
    return keys


def convert_toolace(row: dict) -> dict | None:
    """raw ToolACE 行 → canonical record；只保留标准格式且单工具调用的样本。"""
    system = row["system"]
    marker = "Here is a list of functions in JSON format that you can invoke:\n"
    if marker not in system:
        return None
    tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].strip())
    tool_names = [tool["name"] for tool in tools]
    if len(tool_names) < 2:
        return None

    conversations = row["conversations"]
    for index, message in enumerate(conversations):
        if message["from"] != "user" or index + 1 >= len(conversations):
            continue
        reply = conversations[index + 1]
        if reply["from"] != "assistant":
            continue
        called = [
            name
            for name in tool_names
            if re.search(r"(?:^|\[|, )" + re.escape(name) + r"\(", reply["value"])
        ]
        if len(called) != 1:
            continue
        request = message["value"]
        return {
            "id": "",
            "state": {
                "request": request,
                "tools": [
                    {"name": t["name"], "description": t.get("description", "")}
                    for t in tools
                ],
            },
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Which single tool from the catalog should be called for this request?",
                    "criteria": {
                        t["name"]: t.get("description", "") for t in tools
                    },
                }
            },
            "targets": {"decision": {"label": called[0]}},
            "_request_key": " ".join(request.lower().split()),
        }
    return None


def sample_toolace() -> list[dict]:
    eval_keys = load_eval_request_keys()
    dataset = load_dataset(TOOLACE_REPO, split="train")
    print(f"ToolACE train: {len(dataset)} 行")
    pool = []
    for row in dataset:
        record = convert_toolace(row)
        if record is None or record["_request_key"] in eval_keys:
            continue
        pool.append(record)
    print(f"可转换且不与 eval 重叠: {len(pool)} 条")
    random.Random(SAMPLE_SEED).shuffle(pool)
    selected = pool[:TOOLACE_TRAIN_SIZE]
    for index, record in enumerate(selected):
        record["id"] = f"toolace-{index:05d}"
        del record["_request_key"]
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写出 {path.name}: {len(rows)} 行")


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    download_eval_files()

    train_td, dev, calibration = split_typed_decisions()
    train_toolace = sample_toolace()

    train = train_td + train_toolace
    random.Random(SAMPLE_SEED).shuffle(train)
    write_jsonl(DATASET_DIR / "train.jsonl", train)
    write_jsonl(DATASET_DIR / "dev.jsonl", dev)
    write_jsonl(DATASET_DIR / "calibration.jsonl", calibration)


if __name__ == "__main__":
    main()
