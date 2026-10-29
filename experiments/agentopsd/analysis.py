"""Plot AgentOPSD ALFWorld evaluation results for Base, Step 40, and Step 80.

Run from the repository root:

    uv run python experiments/agentopsd/analysis.py

The default output is ``experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png``.
The script reads the final ``type=summary`` record from each JSONL file and
also verifies that all three evaluations contain the same game IDs.
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
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "agentopsd_checkpoint_evaluation.png"

CHECKPOINTS = (
    ("Base", "base-qwen35-4b-eval.jsonl"),
    ("Step 40", "checkpoint-eval-step40.jsonl"),
    ("Step 80", "checkpoint-eval-step80.jsonl"),
)
EVALUATION_SPLITS = ("valid_seen", "valid_unseen")


@dataclass(frozen=True)
class CheckpointMetrics:
    """Metrics used in the two figure panels."""

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


@dataclass(frozen=True)
class EvaluationFile:
    """One parsed evaluation file plus its trajectory identity set."""

    summary: dict[str, Any]
    game_ids: frozenset[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing the three evaluation JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; its extension selects the export format",
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


def load_evaluation(path: Path) -> EvaluationFile:
    """Parse one JSONL file and collect its summary and trajectory IDs."""
    summary: dict[str, Any] | None = None
    game_ids: set[tuple[str, str]] = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error

            record_type = record.get("type")
            if record_type == "summary":
                if summary is not None:
                    raise ValueError(f"Multiple summary records found in {path}")
                summary = record
            elif record_type == "trajectory":
                split = record.get("evaluation_split", record.get("split"))
                game_id = record.get("game_id")
                if split not in EVALUATION_SPLITS:
                    raise ValueError(
                        f"Unexpected evaluation split in {path}:{line_number}: {split!r}"
                    )
                if not isinstance(game_id, str) or not game_id:
                    raise ValueError(f"Missing game_id in {path}:{line_number}")
                identity = (split, game_id)
                if identity in game_ids:
                    raise ValueError(f"Duplicate trajectory {identity!r} in {path}")
                game_ids.add(identity)

    if summary is None:
        raise ValueError(f"No type=summary record found in {path}")
    if not game_ids:
        raise ValueError(f"No trajectory records found in {path}")
    return EvaluationFile(summary=summary, game_ids=frozenset(game_ids))


def numeric_metric(metrics: dict[str, Any], key: str, path: Path) -> float:
    """Read one finite numeric metric with a useful failure message."""
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


def describe_game_set_difference(
    expected: frozenset[tuple[str, str]],
    actual: frozenset[tuple[str, str]],
) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = [f"missing={len(missing)}", f"extra={len(extra)}"]
    if missing:
        details.append(f"first missing={missing[0]!r}")
    if extra:
        details.append(f"first extra={extra[0]!r}")
    return ", ".join(details)


def load_metrics(
    result_dir: Path,
) -> tuple[str, bool, list[CheckpointMetrics]]:
    """Load metrics and enforce a like-for-like checkpoint comparison."""
    base_model: str | None = None
    skill_free: bool | None = None
    expected_game_ids: frozenset[tuple[str, str]] | None = None
    checkpoints: list[CheckpointMetrics] = []

    for label, filename in CHECKPOINTS:
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        evaluation = load_evaluation(path)
        summary = evaluation.summary

        model_name = summary.get("base_model")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"Missing base_model in {path}")
        if base_model is None:
            base_model = model_name
        elif model_name != base_model:
            raise ValueError(
                f"Inconsistent base_model in {path}: {model_name!r} != {base_model!r}"
            )

        run_skill_free = summary.get("skill_free")
        if not isinstance(run_skill_free, bool):
            raise ValueError(f"Missing boolean skill_free field in {path}")
        if skill_free is None:
            skill_free = run_skill_free
        elif run_skill_free != skill_free:
            raise ValueError(f"Inconsistent skill_free setting in {path}")

        if expected_game_ids is None:
            expected_game_ids = evaluation.game_ids
        elif evaluation.game_ids != expected_game_ids:
            difference = describe_game_set_difference(
                expected_game_ids, evaluation.game_ids
            )
            raise ValueError(f"Evaluation games differ in {path}: {difference}")

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
            if invalid_actions < 0.0:
                raise ValueError(f"Negative invalid-action mean for {split} in {path}")
            split_values[split] = (games, success_rate, invalid_actions)

        seen_games, seen_success, seen_invalid = split_values["valid_seen"]
        unseen_games, unseen_success, unseen_invalid = split_values["valid_unseen"]
        if seen_games + unseen_games != len(evaluation.game_ids):
            raise ValueError(
                f"Summary reports {seen_games + unseen_games} games but "
                f"{len(evaluation.game_ids)} trajectories were found in {path}"
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
    assert skill_free is not None
    return base_model, skill_free, checkpoints


def configure_style() -> None:
    """Reuse the restrained paper-style theme of nearby experiment figures."""
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
            color="#202020",
        )


def make_figure(
    base_model: str,
    skill_free: bool,
    checkpoints: list[CheckpointMetrics],
) -> plt.Figure:
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

    # Panel (a): absolute split success rates, always shown from zero.
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
    split_axis.set_xlabel("Evaluated checkpoint", labelpad=10)
    split_axis.set_ylim(0.0, 0.75)
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

    # Panel (b): three ordered, discrete evaluations on focused axes.
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

    success_span = max(overall_success) - min(overall_success)
    invalid_span = max(invalid_actions) - min(invalid_actions)
    success_padding = max(success_span, 0.01) * 0.55
    invalid_padding = max(invalid_span, 0.20) * 0.28
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

    success_gain = overall_success[-1] - overall_success[0]
    invalid_reduction = invalid_actions[0] - invalid_actions[-1]
    trend_axis.text(
        0.5,
        0.04,
        (
            f"Step 80 vs Base: +{success_gain * 100:.1f} pp success  ·  "
            f"{invalid_reduction:.2f} fewer invalid actions / game"
        ),
        transform=trend_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#303030",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#FFF8EE",
            "edgecolor": "#E1B678",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=5,
    )

    trend_axis.set_title("(b) Overall Success & Invalid Actions", pad=14)
    trend_axis.set_xticks(x_positions, labels)
    trend_axis.set_xlabel("Evaluated checkpoint", labelpad=10)
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
    evaluation_mode = "skill-free evaluation" if skill_free else "evaluation"
    figure.suptitle(
        "AgentOPSD · ALFWorld Checkpoint Evaluation",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.915,
        (
            f"{model_display_name} · same {first.total_games} games at every checkpoint "
            f"({first.seen_games} seen + {first.unseen_games} unseen) · "
            f"{evaluation_mode}"
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
            "Panel (a) starts at zero; panel (b) uses focused y-axes and "
            "connects only the three evaluated checkpoints."
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
    base_model, skill_free, checkpoints = load_metrics(args.result_dir)
    figure = make_figure(base_model, skill_free, checkpoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print_metrics(checkpoints)
    print(
        f"Verified identical game IDs across all checkpoints: "
        f"{checkpoints[0].total_games} games"
    )
    print(f"Saved figure: {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main(parse_args())
