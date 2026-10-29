"""绘制 Qwen3.5-4B / 9B 的五组开发集结果。

在 experiments/spec-o3/ 目录下运行：

uv run python analysis.py

数据固定为 readme.md 的结果表，无需加载模型或重新评测。
四个面板分别展示 Accuracy、Macro F1、严格格式合规率与答对题数。
输出 images/results_comparison.png（300 DPI）和同名矢量 PDF。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


# 固定结果快照：Base/SFT 使用当前答案提取规则重算，三组 RL 使用原 metrics.json。
# 来源依次为 outputs/{base-dev,sft-dev}/metrics-reparsed.json，以及
# outputs/{sft-rl,rl-4b-e3,rl-9b-e3}/metrics.json；百分比与 readme.md 一致。
MODELS = ["4B\nBase", "4B\nSFT epoch 2", "4B\nRL epoch 1", "4B\nRL epoch 3", "9B\nRL epoch 3"]
METRICS = ["Macro F1", "Accuracy", "Format"]
SCORES = np.array([
    [53.99, 38.67, 0.39],
    [63.05, 53.91, 85.94],
    [71.68, 67.97, 96.09],
    [70.93, 68.36, 98.05],
    [75.94, 74.61, 99.61],
])
CORRECT_ANSWERS = [99, 138, 174, 175, 191]
TOTAL_QUESTIONS = 256
BAR_COLORS = ["#D94A4A", "#3388B8"]
FORMAT_COLOR = "#F28E2B"
COUNT_COLOR = "#596A7D"
OUTPUT_DIR = Path(__file__).parent / "images"


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.labelcolor": "#202020",
        "text.color": "#202020",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "#202020",
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.81, hspace=0.55, wspace=0.18)
    positions = np.arange(len(MODELS))
    panels = [
        ("(a) Accuracy", SCORES[:, 1], BAR_COLORS[1]),
        ("(b) Macro F1", SCORES[:, 0], BAR_COLORS[0]),
        ("(c) Strict Format Compliance", SCORES[:, 2], FORMAT_COLOR),
        ("(d) Correct Answers", np.asarray(CORRECT_ANSWERS), COUNT_COLOR),
    ]

    # checkpoint 为离散类别；各面板使用独立柱形，不把 4B 与 9B 连成训练曲线。
    for panel_index, (ax, (title, values, color)) in enumerate(zip(axes.flat, panels)):
        is_count = panel_index == 3
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#C7C7C7", linestyle="-.", linewidth=0.8, alpha=0.75)
        bars = ax.bar(positions, values, width=0.62, color=color, edgecolor="#202020", linewidth=0.9, zorder=3)
        labels = [f"{int(value)} / {TOTAL_QUESTIONS}" if is_count else f"{value:.2f}%" for value in values]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=10, fontweight="bold")
        ax.set_xticks(positions, MODELS)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_title(title, pad=12)
        if is_count:
            ax.set_ylim(0, 280)
            ax.set_yticks([0, 64, 128, 192, 256])
            ax.set_ylabel("Correct answers (out of 256)")
        else:
            ax.set_ylim(0, 112)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.set_ylabel("Score")

    fig.suptitle("Spec-o3 · Qwen3.5-4B / 9B Evaluation", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.934,
        "Same 256 dev examples · 4B Base / SFT epoch 2 / RL epochs 1 & 3 · 9B RL epoch 3",
        ha="center", va="center", fontsize=11, color="#555555",
    )
    correct_gain = CORRECT_ANSWERS[-1] - CORRECT_ANSWERS[-2]
    accuracy_gain = SCORES[-1, 1] - SCORES[-2, 1]
    f1_gain = SCORES[-1, 0] - SCORES[-2, 0]
    fig.text(
        0.5, 0.89,
        f"9B vs 4B at RL epoch 3: +{correct_gain} correct · +{accuracy_gain:.2f} pp accuracy · +{f1_gain:.2f} pp Macro F1",
        ha="center", va="center", fontsize=11, color="#303030",
    )
    fig.text(
        0.5, 0.037,
        "Generation caps: 4B Base 6,132; 4B SFT 2,048; all RL checkpoints 6,144 tokens per turn.\n"
        "Base alone uses an extra format reminder. All runs use the same answer extraction and a 16,384-token context cap.",
        ha="center", va="center", fontsize=9, color="#666666", style="italic", linespacing=1.6,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"results_comparison.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.16)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
