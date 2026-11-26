"""温度校准（牛顿法版）：用一阶导 + 二阶导（曲率）求最优标量温度 T。

与 calibrate.py 的唯一区别：fit_temperature 不再用黄金分割，
改用牛顿法直接求解 NLL 关于 w = 1/T 的驻点（f'(w)=0），
并靠 f''(w) = Var(l) > 0 保证驻点是全局极小值（NLL 严格凸）。

p_T = softmax(logprob / T)，T 只改概率形状、不改 argmax。

推导（对每个字段，l_j = logprob_j，l_gold = 正确答案的 logprob）：
    NLL(w)   = log( Σ_j exp(w·l_j) ) - w·l_gold       ,  w = 1/T
    f'(w)    = Σ_j p_j(w)·l_j - l_gold                ,  p_j = softmax(w·l)_j
    f''(w)   = Σ_j p_j·l_j² - (Σ_j p_j·l_j)² = Var(l)

用法：
    uv run python 09-pytrio-jev/jev/calibrate_newton.py --weights trio://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv
from tqdm import tqdm

from data import Encoder, compile_record, load_jsonl
from inference import candidate_logprobs_async

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / "jev" / ".env")
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "calibration.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "calibration_newton.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


async def collect_instances(
    sampling_client: trio.SamplingClient,
    encoder: Encoder,
    records: list[dict],
    concurrency: int,
) -> list[tuple[np.ndarray, int]]:
    """逐字段收集 (候选 logprob 向量, gold 下标)。"""
    semaphore = asyncio.Semaphore(concurrency)
    per_record: list = [None] * len(records)

    with tqdm(total=len(records), desc="收集校准预测", unit="条") as progress:

        async def run_one(index: int, record: dict) -> None:
            targets = compile_record(record, include_targets=True)["targets"]
            async with semaphore:
                collected = await candidate_logprobs_async(
                    sampling_client, encoder, record
                )
            encoded, logprobs = collected["encoded"], collected["logprobs"]
            instances = []
            for field, options in encoded["options"].items():
                symbols = [symbol for symbol, _, _ in options]
                gold = symbols.index(targets[field])
                instances.append(
                    (np.array([logprobs[field][s] for s in symbols]), gold)
                )
            per_record[index] = instances
            progress.update(1)

        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records))
        )
    return [instance for instances in per_record for instance in instances]


def mean_nll(instances: list[tuple[np.ndarray, int]], temperature: float) -> float:
    total = 0.0
    for logits, gold in instances:
        z = logits / temperature
        total += float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[gold])
    return total / len(instances)


def _grad_and_hessian(
    instances: list[tuple[np.ndarray, int]], w: float
) -> tuple[float, float]:
    """返回平均 NLL 关于 w = 1/T 的一阶导 g 与二阶导 h。

    g = mean( Σ p_j·l_j - l_gold )
    h = mean( Σ p_j·l_j² - (Σ p_j·l_j)² )   # 即平均 Var(l)，恒 > 0
    """
    grad_sum = 0.0
    hess_sum = 0.0
    for logits, gold in instances:
        z = logits * w
        z = z - z.max()  # 数值稳定
        e = np.exp(z)
        p = e / e.sum()
        mean_l = float((p * logits).sum())
        grad_sum += mean_l - logits[gold]
        hess_sum += float((p * logits * logits).sum()) - mean_l * mean_l
    n = len(instances)
    return grad_sum / n, hess_sum / n


def fit_temperature(
    instances: list[tuple[np.ndarray, int]],
    max_iter: int = 50,
    tol: float = 1e-12,
) -> float:
    """用牛顿法在 w = 1/T 上求驻点 f'(w)=0（NLL 严格凸，驻点即全局最小）。

    w_new = w_old - g / h，其中 h = Var(l) > 0 保证每一步都朝谷底走。
    从 w=1（即 T=1）出发，二次收敛，通常几步即可锁定。
    """
    w = 1.0
    for step in range(max_iter):
        g, h = _grad_and_hessian(instances, w)
        if h <= 0:
            # 理论上 h = Var(l) > 0 恒成立；出现 <=0 说明数值退化，退回黄金分割兜底
            raise RuntimeError(f"二阶导 h={h:.6e} 非正，凸性假设被破坏")
        w_new = w - g / h
        if abs(w_new - w) <= tol * abs(w):
            w = w_new
            break
        w = w_new
    return 1.0 / w  # T = 1/w


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="温度校准（牛顿法）")
    parser.add_argument(
        "--base-model", default=os.getenv("JEV_BASE_MODEL", DEFAULT_MODEL)
    )
    weights = os.getenv("JEV_CALIBRATION_WEIGHTS") or None
    parser.add_argument(
        "--weights",
        default=weights,
        required=weights is None,
        help="trio:// LoRA 权重路径",
    )
    data_path = Path(os.getenv("JEV_CALIBRATION_DATA") or DEFAULT_DATA)
    if not data_path.is_absolute():
        data_path = SCRIPT_DIR / data_path
    output_path = Path(os.getenv("JEV_CALIBRATION_OUTPUT") or DEFAULT_OUTPUT)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--output", type=Path, default=output_path)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("JEV_CALIBRATION_CONCURRENCY", "15")),
        help="并发收集条数",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=args.weights
    )
    encoder = Encoder(sampling_client.get_tokenizer())

    records = load_jsonl(args.data)
    instances = await collect_instances(
        sampling_client, encoder, records, args.concurrency
    )
    print(f"校准实例：{len(instances)} 个字段（来自 {len(records)} 条记录）")

    nll_before = mean_nll(instances, 1.0)
    temperature = fit_temperature(instances)
    nll_after = mean_nll(instances, temperature)
    print(f"NLL: T=1 → {nll_before:.4f} | T={temperature:.6f} → {nll_after:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "weights": args.weights,
                "instances": len(instances),
                "nll_t1": nll_before,
                "nll_fitted": nll_after,
                "method": "newton",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"写出 {args.output}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
#T=1 → 0.4842 | T=0.978442 → 0.4841



#NLL: T=1 → 0.4837 | T=0.978816 → 0.4836