"""Plot AIME25 metrics across GSPO checkpoints.

Run from the repository root:

    uv run python experiments/gspo/analyse.py

The figure is written to ``experiments/gspo/images/aime25-gspo-progress.png``.
Metrics are read from the final ``type=summary`` record of each evaluation
JSONL in ``experiments/gspo/eval-results``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "aime25-gspo-progress.png"

COLOR_AVERAGE = "#D94A4A"
COLOR_PASS = "#3388B8"
COLOR_FORMAT = "#F28E2B"
COLOR_FORMAT_MARKER = "#F2B36B"
COLOR_TOKENS = "#596A7D"
COLOR_TOKENS_MARKER = "#A7B4C2"
COLOR_EDGE = "#202020"
COLOR_NOTE = "#555555"

STEP_PATTERN = re.compile(r"^aime25-gspo-step(\d+)\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing AIME25 evaluation JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; the extension selects the export format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview after saving",
    )
    args = parser.parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")
    return args


def load_summary(path: Path) -> dict[str, Any]:
    """Return the final summary record in an evaluation JSONL file."""
    summary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if record.get("type") == "summary":
                summary = record

    if summary is None:
        raise ValueError(f"No type=summary record found in {path}")
    return summary


def discover_checkpoints(result_dir: Path) -> list[tuple[int, str, Path]]:
    """Find the persisted Base run and numerically sorted GSPO checkpoints."""
    base_path = result_dir / "aime25-base.jsonl"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing evaluation result: {base_path}")

    checkpoints: list[tuple[int, str, Path]] = [(0, "Base", base_path)]
    for path in result_dir.glob("aime25-gspo-step*.jsonl"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        checkpoints.append((step, f"Step {step}", path))

    checkpoints.sort(key=lambda item: item[0])
    if len(checkpoints) == 1:
        raise FileNotFoundError(
            f"No aime25-gspo-step*.jsonl evaluation results found in {result_dir}"
        )
    return checkpoints


def load_metrics(
    result_dir: Path,
) -> tuple[
    list[str],
    int,
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Load evaluation metrics from persisted summaries."""
    labels: list[str] = []
    average_at_n: list[float] = []
    pass_at_n: list[float] = []
    format_rate: list[float] = []
    mean_completion_tokens: list[float] = []
    val_n: int | None = None

    for _, label, path in discover_checkpoints(result_dir):
        summary = load_summary(path)
        try:
            current_val_n = int(summary["val_n"])
            average_value = float(summary["average_at_n"])
            pass_value = float(summary["pass_at_n"])
            format_value = float(summary["format_rate"])
            token_value = float(summary["mean_completion_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Expected numeric val_n, average_at_n, pass_at_n, and "
                f"format_rate, and mean_completion_tokens metrics in {path}"
            ) from error

        if val_n is None:
            val_n = current_val_n
        elif current_val_n != val_n:
            raise ValueError(
                f"Inconsistent val_n in {path}: {current_val_n} != {val_n}"
            )

        for metric_name, value in (
            ("average_at_n", average_value),
            ("pass_at_n", pass_value),
            ("format_rate", format_value),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} in {path} is outside [0, 1]")
        if token_value <= 0.0:
            raise ValueError(f"mean_completion_tokens in {path} must be > 0")

        labels.append(label)
        average_at_n.append(average_value)
        pass_at_n.append(pass_value)
        format_rate.append(format_value)
        mean_completion_tokens.append(token_value)

    assert val_n is not None
    return (
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    )


def configure_style() -> None:
    """Apply the paper-style theme shared by the other subprojects."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": COLOR_EDGE,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_value_labels(
    axis: plt.Axes,
    x_positions: list[float] | list[int],
    values: list[float],
    *,
    offset: float,
    fontsize: int = 9,
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLOR_EDGE,
        )


def make_figure(
    labels: list[str],
    val_n: int,
    average_at_n: list[float],
    pass_at_n: list[float],
    format_rate: list[float],
    mean_completion_tokens: list[float],
) -> plt.Figure:
    configure_style()

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    accuracy_axis, format_axis = axes

    trained_indices = list(range(1, len(labels)))
    best_index = max(trained_indices, key=average_at_n.__getitem__)

    bar_width = 0.36
    average_positions = [x - bar_width / 2 for x in x_positions]
    pass_positions = [x + bar_width / 2 for x in x_positions]
    average_bars = accuracy_axis.bar(
        average_positions,
        average_at_n,
        width=bar_width,
        color=COLOR_AVERAGE,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Average@{val_n}",
    )
    accuracy_axis.bar(
        pass_positions,
        pass_at_n,
        width=bar_width,
        color=COLOR_PASS,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Pass@{val_n}",
    )
    average_bars[best_index].set_color(COLOR_FORMAT)
    average_bars[best_index].set_hatch("///")

    add_value_labels(
        accuracy_axis,
        average_positions,
        average_at_n,
        offset=0.02,
        fontsize=8,
    )
    add_value_labels(
        accuracy_axis,
        pass_positions,
        pass_at_n,
        offset=0.02,
        fontsize=8,
    )
    accuracy_axis.set_title(f"(a) Average@{val_n} & Pass@{val_n}", pad=14)
    accuracy_axis.set_ylabel("Score")
    accuracy_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.4,
    )

    format_line = format_axis.plot(
        x_positions,
        format_rate,
        color=COLOR_FORMAT,
        marker="s",
        markersize=7,
        markerfacecolor=COLOR_FORMAT_MARKER,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
        label="Format rate",
    )[0]
    add_value_labels(format_axis, x_positions, format_rate, offset=0.025)

    token_axis = format_axis.twinx()
    token_line = token_axis.plot(
        x_positions,
        mean_completion_tokens,
        color=COLOR_TOKENS,
        marker="o",
        markersize=6.5,
        markerfacecolor=COLOR_TOKENS_MARKER,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.0,
        linestyle="--",
        zorder=3,
        label="Mean completion tokens",
    )[0]
    for x_position, value in zip(
        x_positions,
        mean_completion_tokens,
        strict=True,
    ):
        token_axis.annotate(
            f"{value:,.0f}",
            (x_position, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLOR_TOKENS,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.0,
            },
            zorder=5,
        )

    token_ceiling = math.ceil(max(mean_completion_tokens) / 1000.0) * 1000.0
    token_axis.set_ylim(0.0, token_ceiling)
    token_axis.set_ylabel(
        "Mean completion tokens",
        color=COLOR_TOKENS,
        labelpad=10,
    )
    token_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000.0:.0f}k")
    )
    token_axis.tick_params(
        axis="y",
        which="major",
        direction="in",
        right=True,
        length=5,
        width=0.9,
        colors=COLOR_TOKENS,
    )
    token_axis.spines["right"].set_color(COLOR_EDGE)
    token_axis.spines["right"].set_linewidth(1.0)
    token_axis.grid(False)

    format_axis.legend(
        handles=[format_line, token_line],
        loc="lower left",
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
    )
    format_axis.set_title("(b) Format Rate & Completion Length", pad=14)

    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.set_ylim(0.0, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=axis is accuracy_axis,
            length=5,
            width=0.9,
        )
        axis.margins(x=0.06)

    figure.suptitle(
        "GSPO AIME25 Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        f"AIME 2025 (30 problems) · val-n {val_n} · temperature 1.0",
        ha="center",
        va="top",
        fontsize=10,
        color=COLOR_NOTE,
    )
    figure.text(
        0.5,
        0.015,
        "Orange hatching marks the highest observed trained "
        f"Average@{val_n} ({labels[best_index]}; descriptive only). "
        "The token axis starts at zero; Base is the persisted evaluation run.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLOR_NOTE,
    )
    figure.subplots_adjust(left=0.075, right=0.94, bottom=0.15, top=0.84)
    return figure


def main() -> None:
    args = parse_args()
    (
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    ) = load_metrics(args.result_dir)
    figure = make_figure(
        labels,
        val_n,
        average_at_n,
        pass_at_n,
        format_rate,
        mean_completion_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
