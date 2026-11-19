"""RGSD 结果分析：训练动态曲线 + 评测配对比较。

从仓库根目录运行：

    uv run python experiments/rgsd/analysis.py

读取：
- ``train-results/<run>/rollouts.jsonl``：train.py 每步写入的全部 rollout；
- ``eval-results/scores-*--<judge>.jsonl``：eval.py judge 的逐题打分。
裁判名称与评测共用 ``.env`` 中的 ``RGSD_JUDGE_MODEL_NAME``。

输出：
- ``images/training-dynamics.png``：逐步 KL(S‖m)、回答长度、截断率、Teacher 更认可的 token 比例；
- ``images/eval-comparison.png``：各模型得分（95% bootstrap CI）与平均回答长度；
- 终端打印得分表和配对差值（相对 base、相对提示词对照），readme 里的数字都来自这里。

配对差值的「等长校正」：对每道题的得分差 d 和长度比 ln(L_a/L_b) 做线性回归，
截距就是两边长度相同时的估计差值；它是外推，只在两边长度分布有重叠时可信。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "eval-results"
TRAIN_DIR = SCRIPT_DIR / "train-results"
IMAGE_DIR = SCRIPT_DIR / "images"
PREFIX = "scores-medical-eval-nothink-"
RUN = "rgsd-stageA-medical-lr4e-5-v3"

# (图例名, 文件标签)；按图里从左到右的顺序。
MODELS = [
    ("Base", "base"),
    ("Base + thorough prompt", "base-instr-thorough"),
    ("RGSD step 10", f"{RUN}-step000010"),
    ("RGSD step 20", f"{RUN}-step000020"),
    ("RGSD step 20 (4096 tok)", f"{RUN}-step000020-max4096"),
]
PAIRS = [
    ("RGSD step 10", "Base"),
    ("RGSD step 20", "Base"),
    ("RGSD step 20 (4096 tok)", "Base"),
    ("Base + thorough prompt", "Base"),
    ("RGSD step 20", "Base + thorough prompt"),
    ("RGSD step 20 (4096 tok)", "Base + thorough prompt"),
]
COLORS = ["#9CA3AF", "#F59E0B", "#93C5FD", "#2563EB", "#1E3A8A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=RUN, help="train-results 下的 run 目录名")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def configured_judge() -> str:
    """与 chat 评测读取同一个模型配置；分析只需模型名称，不需要 API Key。"""
    env_path = SCRIPT_DIR / ".env"
    load_dotenv(env_path)
    model = os.getenv("RGSD_JUDGE_MODEL_NAME", "")
    if not model.strip():
        raise ValueError(f"请在 {env_path} 中填写 RGSD_JUDGE_MODEL_NAME，与评测使用的裁判一致")
    return f"chat:{model}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def step_metrics(run: str) -> dict[str, np.ndarray]:
    """按 step 聚合 rollouts.jsonl；KL 与 Teacher 认可率按训练 token 加权，和训练日志一致。"""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(TRAIN_DIR / run / "rollouts.jsonl"):
        by_step[record["step"]].append(record)
    steps = np.array(sorted(by_step))
    out: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        rows = by_step[step]
        tokens = np.array([r["trained_tokens"] for r in rows], dtype=float)
        out["kl"].append(float(np.average([r["kl_student_mixture"] for r in rows], weights=tokens)))
        out["teacher_prefers"].append(
            float(np.average([r["teacher_prefers_rate"] for r in rows], weights=tokens))
        )
        out["length"].append(float(np.mean([r["completion_tokens"] for r in rows])))
        out["truncated"].append(float(np.mean([r["stop_reason"] == "length" for r in rows])))
    return {"step": steps, **{k: np.array(v) for k, v in out.items()}}


def plot_training(metrics: dict[str, np.ndarray], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = metrics["step"]
    axes[0].plot(x, metrics["kl"], marker="o", color="#2563EB")
    axes[0].set_title("KL(student || mixture), token-weighted")
    axes[1].plot(x, metrics["length"], marker="o", color="#DC2626", label="mean completion tokens")
    axes[1].set_title("Completion length (cap 2048)")
    twin = axes[1].twinx()
    twin.bar(x, metrics["truncated"] * 100, alpha=0.25, color="#DC2626", label="truncated %")
    twin.set_ylabel("truncated %")
    twin.set_ylim(0, 100)
    axes[2].plot(x, metrics["teacher_prefers"] * 100, marker="o", color="#16A34A")
    axes[2].axhline(50, color="#9CA3AF", linestyle="--", linewidth=1)
    axes[2].set_title("Tokens where teacher logprob > student (%)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load_scores(judge: str) -> dict[str, dict[str, dict[str, Any]]]:
    judge_slug = re.sub(r"[^a-zA-Z0-9]+", "-", judge).strip("-").lower()
    data = {}
    for name, tag in MODELS:
        path = RESULT_DIR / f"{PREFIX}{tag}--{judge_slug}.jsonl"
        if path.exists():
            data[name] = {r["id"]: r for r in read_jsonl(path)}
        else:
            print(f"[skip] 缺少 {path.name}")
    return data


def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def length_adjusted(diff: np.ndarray, log_ratio: np.ndarray, rng: np.random.Generator, n: int):
    """d = a + b·ln(L_a/L_b)；返回截距 a 及其 bootstrap CI。"""
    design = np.c_[np.ones_like(log_ratio), log_ratio]
    intercept = np.linalg.lstsq(design, diff, rcond=None)[0][0]
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(diff), len(diff))
        boots.append(np.linalg.lstsq(design[idx], diff[idx], rcond=None)[0][0])
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(intercept), float(low), float(high)


def report_eval(data, rng, n_boot, judge: str) -> list[dict[str, Any]]:
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"\n评测：{len(ids)} 道共同题目，judge = {judge}")
    print(f"{'model':26s} {'score':>6s} {'95% CI':>14s} {'tokens':>7s} {'trunc':>6s}")
    rows = []
    for name, records in data.items():
        score = np.array([records[i]["score"] for i in ids]) * 100
        tokens = np.array([records[i]["answer_tokens"] for i in ids])
        trunc = np.mean([records[i]["stop_reason"] == "length" for i in ids]) * 100
        low, high = bootstrap_mean_ci(score, rng, n_boot)
        rows.append({"name": name, "score": score.mean(), "low": low, "high": high, "tokens": tokens.mean()})
        print(f"{name:26s} {score.mean():6.1f} [{low:5.1f},{high:5.1f}] {tokens.mean():7.0f} {trunc:5.1f}%")

    print("\n配对差值（同一道题 a - b，单位 pp）")
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        sa = np.array([data[a][i]["score"] for i in ids]) * 100
        sb = np.array([data[b][i]["score"] for i in ids]) * 100
        la = np.array([data[a][i]["answer_tokens"] for i in ids], dtype=float)
        lb = np.array([data[b][i]["answer_tokens"] for i in ids], dtype=float)
        diff = sa - sb
        low, high = bootstrap_mean_ci(diff, rng, n_boot)
        log_ratio = np.log(np.maximum(la, 1) / np.maximum(lb, 1))
        adj, adj_low, adj_high = length_adjusted(diff, log_ratio, rng, n_boot // 2)
        near = np.abs(log_ratio) < np.log(1.25)
        near_text = ""
        if near.sum() > 5:
            n_low, n_high = bootstrap_mean_ci(diff[near], rng, n_boot)
            near_text = f" | 长度差 <25% 的 {near.sum()} 题 {diff[near].mean():+.1f} [{n_low:+.1f},{n_high:+.1f}]"
        print(
            f"{a} - {b}: {diff.mean():+.1f} [{low:+.1f},{high:+.1f}]"
            f" | 胜/负 {np.mean(diff > 0) * 100:.0f}%/{np.mean(diff < 0) * 100:.0f}%"
            f" | 长度比中位数 {np.exp(np.median(log_ratio)):.2f}"
            f" | 等长校正 {adj:+.1f} [{adj_low:+.1f},{adj_high:+.1f}]{near_text}"
        )
    return rows


def plot_eval(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))
    colors = COLORS[: len(rows)]
    scores = np.array([r["score"] for r in rows])
    errors = np.array([[r["score"] - r["low"], r["high"] - r["score"]] for r in rows]).T
    axes[0].barh(y, scores, xerr=errors, color=colors, capsize=3)
    for i, value in enumerate(scores):
        axes[0].text(value + errors[1][i] + 0.4, i, f"{value:.1f}", va="center", fontsize=9)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("weighted rubric satisfaction (%), 95% CI")
    axes[0].set_xlim(0, max(scores) + 8)
    axes[1].barh(y, [r["tokens"] for r in rows], color=colors)
    axes[1].set_yticks(y, [""] * len(rows))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean answer tokens")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("RubricHub medical eval (300 prompts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    judge = configured_judge()
    rng = np.random.default_rng(args.seed)

    metrics = step_metrics(args.run)
    print(f"训练：{args.run}，{len(metrics['step'])} 步")
    print(f"{'step':>4s} {'KL(S||m)':>9s} {'tokens':>7s} {'trunc':>6s} {'T>S':>6s}")
    for i, step in enumerate(metrics["step"]):
        print(
            f"{step:4d} {metrics['kl'][i]:9.4f} {metrics['length'][i]:7.0f}"
            f" {metrics['truncated'][i] * 100:5.1f}% {metrics['teacher_prefers'][i] * 100:5.1f}%"
        )
    plot_training(metrics, IMAGE_DIR / "training-dynamics.png", args.dpi)

    data = load_scores(judge)
    if len(data) >= 2:
        rows = report_eval(data, rng, args.bootstrap, judge)
        plot_eval(rows, IMAGE_DIR / "eval-comparison.png", args.dpi)
    print(f"\n图片已写入 {IMAGE_DIR}")


if __name__ == "__main__":
    main()
