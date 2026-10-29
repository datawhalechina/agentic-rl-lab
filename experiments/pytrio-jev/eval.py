"""五项基准评测：Jevbench Easy/Original/Hard + Typed Decision + ToolACE。

先选优、先校准，最后才读 test（防污染红线：评测集一律不进训练）。

用法：
uv run python eval.py                          # base 模型
uv run python eval.py --weights trio://... --calibration outputs/calibration.json
uv run python eval.py --weights trio://... --suite typed_decisions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytrio as trio
from tqdm import tqdm

from data import Encoder, load_jsonl, normalize_label
from inference import decide_async

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "datasets" / "eval"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

SUITES = {
    "jevbench_easy": EVAL_DIR / "jevbench_easy.jsonl",
    "jevbench_original": EVAL_DIR / "jevbench_original.jsonl",
    "jevbench_hard": EVAL_DIR / "jevbench_hard.jsonl",
    "typed_decisions": EVAL_DIR / "typed_decisions.jsonl",
    "toolace": EVAL_DIR / "toolace.jsonl",
}


def adapt_jevbench(row: dict) -> dict:
    """Jevbench 上游单问题格式 → canonical record。

    choice 的选项顺序以 labels 数组为准（criteria 在文件里是字母序）；noul 的
    criteria 键是 true/false、score 的 criteria 是等级列表，二者由 question_options
    直接展开，顺序与 labels 天然一致。
    """
    question = row["question"]
    criteria = question["criteria"]
    if question["type"] == "choice":
        criteria = {label: criteria[label] for label in row["labels"]}
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            "decision": {
                "type": question["type"],
                "instructions": question["instructions"],
                "criteria": criteria,
            }
        },
        "targets": {"decision": {"label": row["expected"]}},
    }


def load_suite(name: str) -> list[dict]:
    rows = load_jsonl(SUITES[name])
    if name.startswith("jevbench"):
        return [adapt_jevbench(row) for row in rows]
    return rows


def ece10(confidences: np.ndarray, correct: np.ndarray) -> float:
    """10 个等宽 bin 的期望校准误差，置信度取 max(P)。"""
    value = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        mask = (confidences >= lo) & (confidences < lo + 0.1 if lo < 0.9 else confidences <= 1.0)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(value)


async def evaluate_suite(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    name: str,
    temperature: float,
    concurrency: int,
) -> dict:
    records = load_suite(name)
    semaphore = asyncio.Semaphore(concurrency)
    completed: list = [None] * len(records)

    with tqdm(total=len(records), desc=name, unit="条") as progress:

        async def run_one(index: int, record: dict) -> None:
            async with semaphore:
                result = await decide_async(
                    sampling_client, encoder, record, temperature
                )
            completed[index] = (record, result)
            progress.update(1)

        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records))
        )

    n = correct = 0
    confidences, correctness, briers = [], [], []
    for record, result in completed:
        for field, field_result in result.items():
            gold_label = normalize_label(
                record["questions"][field], record["targets"][field]["label"]
            )
            probs = field_result["probabilities"]
            hit = field_result["prediction"] == gold_label
            n += 1
            correct += hit
            confidences.append(max(probs.values()))
            correctness.append(hit)
            briers.append(
                sum((p - (1.0 if value == gold_label else 0.0)) ** 2 for value, p in probs.items())
            )
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    return {
        "suite": name,
        "records": len(records),
        "decisions": n,
        "accuracy": correct / n,
        "ece": ece10(confidences, correctness),
        "brier": float(np.mean(briers)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="五项决策基准评测")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--weights", default=None, help="trio:// LoRA 权重路径，不传则评 base")
    parser.add_argument("--calibration", type=Path, default=None, help="calibration.json")
    parser.add_argument("--suite", choices=list(SUITES) + ["all"], default="all")
    parser.add_argument("--tag", default=None, help="输出文件名标记")
    parser.add_argument("--concurrency", type=int, default=15, help="并发评测条数")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    temperature = 1.0
    if args.calibration:
        temperature = json.loads(args.calibration.read_text())["temperature"]

    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=args.weights
    )
    encoder = Encoder(sampling_client.get_tokenizer())

    suite_names = list(SUITES) if args.suite == "all" else [args.suite]
    results = [
        await evaluate_suite(
            sampling_client, encoder, name, temperature, args.concurrency
        )
        for name in suite_names
    ]

    print(f"\n{'suite':<20} {'decisions':>9} {'acc':>8} {'ece':>8} {'brier':>8}")
    for r in results:
        print(
            f"{r['suite']:<20} {r['decisions']:>9} "
            f"{r['accuracy']:>8.2%} {r['ece']:>8.4f} {r['brier']:>8.4f}"
        )
    if len(results) > 1:
        mean_acc = np.mean([r["accuracy"] for r in results])
        print(f"{'average':<20} {'':>9} {mean_acc:>8.2%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"trained-{int(time.time())}" if args.weights else "base")
    output = OUTPUT_DIR / f"eval-{tag}.json"
    output.write_text(
        json.dumps(
            {
                "weights": args.weights,
                "temperature": temperature,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"写出 {output}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
