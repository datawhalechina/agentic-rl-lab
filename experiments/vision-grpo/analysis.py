"""Plot the focused GeoQA evaluation comparison before and after GRPO."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


BASE_RESULTS = {
    "accuracy": 71.0,
    "format_rate": 75.0,
}
GRPO_RESULTS = {
    "accuracy": 87.0,
    "format_rate": 91.0,
}

BASE_COLOR = "#AEB4BE"
GRPO_COLOR = "#4A86E8"
INK = "#252525"
MUTED_INK = "#777777"
GRID_COLOR = "#D6D9DE"

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "images" / "eval-comparison.png"


def configure_style() -> None:
    """Configure the serif typography and restrained paper-figure palette."""
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "hatch.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: Axes) -> None:
    """Match the framed, inward-tick axes used in the reference figure."""
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle="-.",
        linewidth=0.65,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    ax.tick_params(top=True, right=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def plot_metric(
    ax: Axes,
    *,
    title: str,
    base_value: float,
    grpo_value: float,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    show_ylabel: bool,
) -> None:
    """Plot one Base-versus-checkpoint comparison on a focused y-axis."""
    x_positions = [0, 1]
    bars = ax.bar(
        x_positions,
        [base_value, grpo_value],
        width=0.48,
        color=[BASE_COLOR, GRPO_COLOR],
        edgecolor=INK,
        linewidth=0.85,
        zorder=3,
    )
    bars[1].set_hatch("///")

    value_offset = (y_limits[1] - y_limits[0]) * 0.04
    for x_position, value in zip(
        x_positions,
        [base_value, grpo_value],
        strict=True,
    ):
        ax.text(
            x_position,
            value + value_offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.0,
        )

    delta = grpo_value - base_value
    delta_y = (base_value + grpo_value) / 2 + 1.0
    ax.text(
        0.5,
        delta_y,
        f"+{delta:.1f} pp",
        ha="center",
        va="center",
        color="#2563C4",
        fontsize=8.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.23",
            "facecolor": "white",
            "edgecolor": "#7DB4FF",
            "linewidth": 0.9,
        },
        zorder=5,
    )

    ax.text(
        0.02,
        0.035,
        "focused y-axis",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#9A9A9A",
        fontsize=6.7,
        fontstyle="italic",
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Model / Checkpoint", labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Score")
    ax.set_xticks(x_positions, ["Base", "Step 100"])
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    style_axis(ax)


def build_figure() -> plt.Figure:
    """Build the two-panel publication-style comparison figure."""
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    plot_metric(
        axes[0],
        title="(a) Accuracy",
        base_value=BASE_RESULTS["accuracy"],
        grpo_value=GRPO_RESULTS["accuracy"],
        y_limits=(68, 92),
        y_ticks=[68, 72, 76, 80, 84, 88, 92],
        show_ylabel=True,
    )
    plot_metric(
        axes[1],
        title="(b) Format Rate",
        base_value=BASE_RESULTS["format_rate"],
        grpo_value=GRPO_RESULTS["format_rate"],
        y_limits=(72, 96),
        y_ticks=[72, 76, 80, 84, 88, 92, 96],
        show_ylabel=False,
    )

    figure.suptitle(
        "GeoQA · Vision GRPO Evaluation",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=12.0,
        fontweight="normal",
        color=INK,
    )
    figure.text(
        0.5,
        0.875,
        "Fixed 100-question evaluation set · Qwen3.5-4B · independent focused y-axes",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED_INK,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.72,
        wspace=0.22,
    )
    return figure


def save_figure(output: Path) -> Path:
    """Save the comparison as a 600 DPI PNG."""
    png_path = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    figure = build_figure()
    figure.savefig(png_path, dpi=600)
    plt.close(figure)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Base vs. GRPO step-100 GeoQA evaluation results."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path = save_figure(args.output)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
