"""从原始光谱重绘局部波段，作为下一轮的图片观察。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spectral_visualization_tool(
    wavelength,
    flux,
    redshift,
    wavelength_range,
    label,
    output_path,
):
    """在静止系波长上选取窗口，返回绘图结果及可直接回填的工具消息。"""
    # 1. 校正红移，再按模型指定的波长区间选取数据。
    wavelength = np.asarray(wavelength) / (1 + redshift)
    flux = np.asarray(flux)
    low, high = wavelength_range
    selected = (wavelength >= low) & (wavelength <= high)

    # 空窗口是工具动作的一种结果，返回文字让模型继续判断。
    if selected.sum() < 2:
        return {
            "content": [{
                "type": "text",
                "text": f"No spectrum data in wavelength range ({low:g}, {high:g}) Å.",
            }],
            "images": [],
            "ok": False,
        }

    # 2. 根据窗口宽度选择画布，绘制局部光谱。
    wave, values = wavelength[selected], flux[selected]
    width = high - low
    if width < 50:
        figsize = (6, 4)
    elif width < 200:
        figsize = (7, 4.5)
    elif width < 1000:
        figsize = (8, 5)
    else:
        figsize = (9, 5.5)

    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    ax.plot(wave, values, color="blue", linewidth=1, alpha=0.8)
    ax.set_xlabel("Wavelength (Å)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flux", fontsize=12, fontweight="bold")
    ax.set_title("Spectrum", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()

    # 3. 标注窗口内的数据范围，保存供模型继续观察的 PNG。
    ax.text(
        0.98,
        0.98,
        f"Points: {len(wave):,}\n"
        f"λ: {wave.min():.1f}–{wave.max():.1f} Å\n"
        f"Flux: {values.min():.2f}–{values.max():.2f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    # 4. 图片与文字说明一起作为下一轮的 tool observation。
    text = f"Generated spectral visualization for wavelength range ({low:g}, {high:g}) Å"
    if label:
        text += f" ({label})"
    return {
        "content": [
            {"type": "image", "image": str(output_path)},
            {"type": "text", "text": text},
        ],
        "images": [str(output_path)],
        "ok": True,
    }
