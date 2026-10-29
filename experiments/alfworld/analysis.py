"""Visualize ALFWorld base-model and checkpoint evaluation results.

Run from the repository root:

    uv run python experiments/alfworld/analysis.py

The figure is written to ``experiments/alfworld/images/alfworld_checkpoint_evaluation.png``
by default. Metrics are read from the final ``type=summary`` record in each
evaluation JSONL file.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval_results"
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "alfworld_checkpoint_evaluation.png"

CHECKPOINTS = (
    ("Base", "base-qwen35-4b-eval.jsonl"),
    ("Step 40", "checkpoint-40steps.jsonl"),
    ("Step 80", "checkpoint-80steps.jsonl"),
)
EVALUATION_SPLITS = ("valid_seen", "valid_unseen")
EVALUATION_TEMPERATURE = 0.01
EVALUATION_SEED = 42


@dataclass(frozen=True)
class CheckpointMetrics:
    """Metrics needed by the two figure panels."""

    label: str
    seen_success_rate: float
    unseen_success_rate: float
    overall_success_rate: float
    mean_invalid_actions: float
    seen_games: int
    unseen_games: int

    @property
    def total_games(self) -> int:
        return self.seen_games + self.unseen_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing the three ALFWorld evaluation JSONL files",
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
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    """Return the last summary record in an evaluation JSONL file."""
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


def numeric_metric(metrics: dict[str, Any], key: str, path: Path) -> float:
    """Read one finite numeric metric and provide a useful failure message."""
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or non-numeric metric {key!r} in {path}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric {key!r} in {path}: {value}")
    return value


def integer_metric(metrics: dict[str, Any], key: str, path: Path) -> int:
    value = numeric_metric(metrics, key, path)
    if value < 0 or not value.is_integer():
        raise ValueError(f"Expected a non-negative integer metric {key!r} in {path}")
    return int(value)


def load_metrics(
    result_dir: Path,
) -> tuple[str, list[CheckpointMetrics]]:
    """Load checkpoint metrics and verify that all runs used the same games."""
    base_model: str | None = None
    expected_game_counts: tuple[int, int] | None = None
    checkpoints: list[CheckpointMetrics] = []

    for label, filename in CHECKPOINTS:
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        summary = load_summary(path)
        model_name = summary.get("base_model")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"Missing base_model in {path}")
        if base_model is None:
            base_model = model_name
        elif model_name != base_model:
            raise ValueError(
                f"Inconsistent base_model in {path}: {model_name!r} != {base_model!r}"
            )

        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing metrics object in {path}")

        split_values: dict[str, tuple[int, float, float]] = {}
        for split in EVALUATION_SPLITS:
            games = integer_metric(metrics, f"eval/{split}/games", path)
            success_rate = numeric_metric(
                metrics, f"eval/{split}/success_rate", path
            )
            invalid_actions = numeric_metric(
                metrics, f"eval/{split}/invalid_actions_mean", path
            )
            if not 0.0 <= success_rate <= 1.0:
                raise ValueError(
                    f"Success rate for {split} is outside [0, 1] in {path}"
                )
            split_values[split] = (games, success_rate, invalid_actions)

        seen_games, seen_success, seen_invalid = split_values["valid_seen"]
        unseen_games, unseen_success, unseen_invalid = split_values["valid_unseen"]
        game_counts = (seen_games, unseen_games)
        if expected_game_counts is None:
            expected_game_counts = game_counts
        elif game_counts != expected_game_counts:
            raise ValueError(
                f"Inconsistent split sizes in {path}: {game_counts} "
                f"!= {expected_game_counts}"
            )

        total_games = seen_games + unseen_games
        if total_games == 0:
            raise ValueError(f"Evaluation contains no games: {path}")

        checkpoints.append(
            CheckpointMetrics(
                label=label,
                seen_success_rate=seen_success,
                unseen_success_rate=unseen_success,
                overall_success_rate=(
                    seen_success * seen_games + unseen_success * unseen_games
                )
                / total_games,
                mean_invalid_actions=(
                    seen_invalid * seen_games + unseen_invalid * unseen_games
                )
                / total_games,
                seen_games=seen_games,
                unseen_games=unseen_games,
            )
        )

    assert base_model is not None
    return base_model, checkpoints


def configure_style() -> None:
    """Apply the restrained paper-style theme used by nearby projects."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_percent_labels(
    axis: plt.Axes,
    x_positions: list[float],
    values: list[float],
    *,
    offset: float,
    color: str = "#202020",
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )


def make_figure(base_model: str, checkpoints: list[CheckpointMetrics]) -> plt.Figure:
    configure_style()

    labels = [checkpoint.label for checkpoint in checkpoints]
    x_positions = list(range(len(checkpoints)))
    seen_success = [checkpoint.seen_success_rate for checkpoint in checkpoints]
    unseen_success = [checkpoint.unseen_success_rate for checkpoint in checkpoints]
    overall_success = [checkpoint.overall_success_rate for checkpoint in checkpoints]
    invalid_actions = [checkpoint.mean_invalid_actions for checkpoint in checkpoints]

    figure, (split_axis, trend_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.5),
        gridspec_kw={"wspace": 0.18},
    )

    # Panel (a): success rate on both official validation splits.
    bar_width = 0.34
    seen_positions = [x - bar_width / 2 for x in x_positions]
    unseen_positions = [x + bar_width / 2 for x in x_positions]
    split_axis.bar(
        seen_positions,
        seen_success,
        width=bar_width,
        color="#D94A4A",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label="Valid Seen",
    )
    split_axis.bar(
        unseen_positions,
        unseen_success,
        width=bar_width,
        color="#3388B8",
        edgecolor="#202020",
        linewidth=1.0,
        hatch="///",
        alpha=0.92,
        zorder=3,
        label="Valid Unseen",
    )
    add_percent_labels(split_axis, seen_positions, seen_success, offset=0.012)
    add_percent_labels(split_axis, unseen_positions, unseen_success, offset=0.012)
    split_axis.set_title("(a) Success Rate by Evaluation Split", pad=14)
    split_axis.set_ylabel("Success rate")
    split_axis.set_xticks(x_positions, labels)
    split_axis.set_xlabel("Model / Checkpoint", labelpad=10)
    split_axis.set_ylim(0.0, 0.70)
    split_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    split_axis.grid(
        axis="y",
        color="#C7C7C7",
        linestyle="-.",
        linewidth=0.8,
        alpha=0.75,
        zorder=0,
    )
    split_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
    )

    # Panel (b): checkpoint trend, with focused axes for the two metrics.
    success_color = "#F28E2B"
    invalid_color = "#596A7D"
    success_line = trend_axis.plot(
        x_positions,
        overall_success,
        color=success_color,
        marker="s",
        markersize=7,
        markerfacecolor="#F2B36B",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.2,
        zorder=4,
        label="Overall success rate",
    )[0]
    invalid_axis = trend_axis.twinx()
    invalid_line = invalid_axis.plot(
        x_positions,
        invalid_actions,
        color=invalid_color,
        marker="o",
        markersize=7,
        markerfacecolor="#A7B4C2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.0,
        linestyle="--",
        zorder=4,
        label="Mean invalid actions",
    )[0]

    success_padding = max(max(overall_success) - min(overall_success), 0.01) * 0.55
    invalid_padding = max(max(invalid_actions) - min(invalid_actions), 0.20) * 0.28
    trend_axis.set_ylim(
        min(overall_success) - success_padding,
        max(overall_success) + success_padding,
    )
    invalid_axis.set_ylim(
        min(invalid_actions) - invalid_padding,
        max(invalid_actions) + invalid_padding,
    )

    for x_position, value in zip(x_positions, overall_success, strict=True):
        trend_axis.annotate(
            f"{value:.1%}",
            (x_position, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=success_color,
        )
    for x_position, value in zip(x_positions, invalid_actions, strict=True):
        invalid_axis.annotate(
            f"{value:.2f}",
            (x_position, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=invalid_color,
        )

    trend_axis.set_title("(b) Overall Success & Invalid Actions", pad=14)
    trend_axis.set_xticks(x_positions, labels)
    trend_axis.set_xlabel("Model / Checkpoint", labelpad=10)
    trend_axis.set_ylabel("Overall success rate", color=success_color)
    invalid_axis.set_ylabel("Mean invalid actions", color=invalid_color)
    trend_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    trend_axis.tick_params(axis="y", colors=success_color)
    invalid_axis.tick_params(axis="y", colors=invalid_color)
    trend_axis.spines["left"].set_color(success_color)
    invalid_axis.spines["right"].set_color(invalid_color)
    trend_axis.grid(
        axis="y",
        color="#C7C7C7",
        linestyle="-.",
        linewidth=0.8,
        alpha=0.65,
        zorder=0,
    )
    trend_axis.legend(
        [success_line, invalid_line],
        [success_line.get_label(), invalid_line.get_label()],
        loc="upper center",
        frameon=False,
        fontsize=9,
        handlelength=2.2,
    )

    for axis in (split_axis, trend_axis):
        axis.tick_params(axis="x", direction="in", top=True)

    first = checkpoints[0]
    model_display_name = base_model.rsplit("/", maxsplit=1)[-1]
    figure.suptitle(
        "ALFWorld Checkpoint Evaluation",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.915,
        (
            f"{model_display_name} · {first.total_games} games "
            f"({first.seen_games} seen + {first.unseen_games} unseen) · "
            f"temperature {EVALUATION_TEMPERATURE:g} · seed {EVALUATION_SEED}"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )
    figure.text(
        0.5,
        0.025,
        (
            "Panel (a) uses a zero baseline; panel (b) uses focused y-axes "
            "to show checkpoint movement."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        style="italic",
    )
    figure.subplots_adjust(left=0.075, right=0.925, bottom=0.16, top=0.81)
    return figure


def print_metrics(checkpoints: list[CheckpointMetrics]) -> None:
    print(
        f"{'Checkpoint':<12} {'Seen':>8} {'Unseen':>8} "
        f"{'Overall':>8} {'Invalid':>9}"
    )
    for checkpoint in checkpoints:
        print(
            f"{checkpoint.label:<12} "
            f"{checkpoint.seen_success_rate:>7.1%} "
            f"{checkpoint.unseen_success_rate:>7.1%} "
            f"{checkpoint.overall_success_rate:>7.1%} "
            f"{checkpoint.mean_invalid_actions:>9.2f}"
        )


def main(args: argparse.Namespace) -> None:
    base_model, checkpoints = load_metrics(args.result_dir)
    figure = make_figure(base_model, checkpoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print_metrics(checkpoints)
    print(f"Saved figure: {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main(parse_args())
