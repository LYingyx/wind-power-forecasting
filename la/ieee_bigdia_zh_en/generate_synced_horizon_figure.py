from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ieee-horizon")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
RESULT_PATH = Path(
    "/storage/luoao/GNN/DP_NEW/dayahead/dataset_3_years/result/"
    "dataset_3_years/all_result/horizon_curves.csv"
)
OUTPUT_DIR = ROOT / "figures_short"
FONT_FILES = (
    "/storage/luoao/fonts/windows-standard/Times.TTF",
    "/storage/luoao/fonts/windows-standard/Timesbd.TTF",
    "/storage/luoao/fonts/windows-standard/simsun.ttc",
)
OURS_COLOR = "#E76F51"
WEATHER_COLOR = "#6B8EAD"
EDGE_COLOR = "#263238"
GRID_COLOR = "#DCE3E8"


def load_curves() -> dict[tuple[str, str], np.ndarray]:
    with RESULT_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    curves: dict[tuple[str, str], np.ndarray] = {}
    for metric in ("MAE", "RMSE"):
        for model in ("Ours", "Weather-only"):
            selected = sorted(
                (row for row in rows if row["metric"] == metric and row["model"] == model),
                key=lambda row: int(row["horizon"]),
            )
            if len(selected) != 24:
                raise ValueError(f"Expected 24 {metric} values for {model}, got {len(selected)}")
            curves[(metric, model)] = np.asarray([float(row["mean"]) for row in selected])
    return curves


def configure_fonts() -> None:
    for font_path in FONT_FILES:
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": EDGE_COLOR,
            "xtick.color": EDGE_COLOR,
            "ytick.color": EDGE_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.55,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def draw(language: str, curves: dict[tuple[str, str], np.ndarray]) -> None:
    is_zh = language == "zh"
    horizons = np.arange(1, 25)
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 1.10), sharex=True)

    axes[0].plot(
        horizons,
        curves[("MAE", "Ours")],
        color=OURS_COLOR,
        linewidth=1.45,
        marker="o",
        markersize=2.5,
        markevery=2,
        label="本文方法" if is_zh else "Ours",
    )
    axes[0].plot(
        horizons,
        curves[("MAE", "Weather-only")],
        color=WEATHER_COLOR,
        linewidth=1.45,
        marker="s",
        markersize=2.4,
        markevery=2,
        label="Weather-only",
    )
    axes[0].set_title("(a) MAE", pad=2.0)
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.85)
    axes[0].spines[["top", "right"]].set_visible(False)

    gap = curves[("MAE", "Weather-only")] - curves[("MAE", "Ours")]
    axes[1].axhline(0.0, color=EDGE_COLOR, linewidth=0.7)
    axes[1].plot(horizons, gap, color="#4C78A8", linewidth=1.45, marker="o", markersize=2.3)
    axes[1].fill_between(horizons, 0.0, gap, color="#4C78A8", alpha=0.16)
    axes[1].set_title("(b) MAE 差值" if is_zh else "(b) MAE gap", pad=2.0)
    axes[1].set_ylabel("差值" if is_zh else "Gap")
    axes[1].grid(alpha=0.85)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="lower right", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.7)
    for axis in axes:
        axis.set_xlabel("预测步" if is_zh else "Forecast step")
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.90, wspace=0.22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / f"horizon_comparison_{language}"
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_fonts()
    curves = load_curves()
    draw("zh", curves)
    draw("en", curves)


if __name__ == "__main__":
    main()
