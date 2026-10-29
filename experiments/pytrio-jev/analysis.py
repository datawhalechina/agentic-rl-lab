"""汇总评测结果，与参考基线对照。

用法：
uv run python experiments/pytrio-jev/analysis.py [outputs/eval-xxx.json ...]
不传参数时汇总 outputs/ 下全部 eval-*.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# 参考基线准确率（百分数）。
REFERENCE_4B = {
    "jevbench_easy": 100.00,
    "jevbench_original": 98.61,
    "jevbench_hard": 73.87,
    "typed_decisions": 80.55,
    "toolace": 96.45,
}


def render(path: Path) -> list[str]:
    report = json.loads(path.read_text())
    label = path.stem
    lines = [f"\n### {label}（T={report['temperature']:.4f}）", ""]
    lines.append("| suite | decisions | acc | reference | ece | brier |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["results"]:
        ref = REFERENCE_4B.get(r["suite"])
        ref_text = f"{ref:.2f}%" if ref is not None else "-"
        lines.append(
            f"| {r['suite']} | {r['decisions']} | {r['accuracy']:.2%} | "
            f"{ref_text} | {r['ece']:.4f} | {r['brier']:.4f} |"
        )
    accuracies = [r["accuracy"] for r in report["results"]]
    if len(accuracies) > 1:
        lines.append(f"| **average** | | **{sum(accuracies) / len(accuracies):.2%}** | | | |")
    return lines


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(OUTPUT_DIR.glob("eval-*.json"))
    if not paths:
        print(f"{OUTPUT_DIR} 下没有 eval-*.json，先跑 eval.py")
        return
    lines: list[str] = []
    for path in paths:
        lines.extend(render(path))
    text = "\n".join(lines)
    print(text)
    summary = OUTPUT_DIR / "summary.md"
    summary.write_text(text + "\n", encoding="utf-8")
    print(f"\n写出 {summary}")


if __name__ == "__main__":
    main()
