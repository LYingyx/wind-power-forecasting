from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


COLORS = ("#6b7280", "#2563eb", "#dc2626", "#059669", "#7c3aed")
_FIGURE_PROGRESS: tqdm | None = None


def progress_options(description: str, *, unit: str, leave: bool | None = None) -> dict[str, Any]:
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    environment_leave = os.environ.get("WIND_PROGRESS_LEAVE", "1").lower() not in {"0", "false", "no"}
    return {"desc": f"{prefix} {description}".strip(), "unit": unit, "dynamic_ncols": True, "position": int(position_value) if position_value else None, "leave": environment_leave if leave is None else leave, "disable": False}


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    if _FIGURE_PROGRESS is not None:
        _FIGURE_PROGRESS.set_postfix_str(path.name, refresh=False)
        _FIGURE_PROGRESS.update(1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def parse_group(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path).expanduser().resolve()


def load_exports(label: str, root: Path, seeds: list[int], split: str) -> dict[str, Any]:
    prediction = []
    target = cluster = origin = sample = None
    aggregation_error = []
    for seed in tqdm(seeds, total=len(seeds), **progress_options(f"load {label}", unit="seed", leave=False)):
        path = root / label / f"seed{seed}/predictions_{split}_station.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            pred = archive["station_prediction_mw"].astype(np.float32)
            current_target = archive["station_target_mw"].astype(np.float32)
            current_cluster = archive["cluster_prediction_mw"].astype(np.float32)
            current_origin = archive["origin_unix"].astype(np.int64)
            current_sample = archive["sample_index"].astype(np.int64)
        if target is None:
            target, cluster, origin, sample = current_target, current_cluster, current_origin, current_sample
        else:
            np.testing.assert_array_equal(origin, current_origin)
            np.testing.assert_array_equal(sample, current_sample)
            np.testing.assert_allclose(target, current_target, rtol=0.0, atol=1e-3)
        aggregation_error.append(float(np.max(np.abs(pred.sum(axis=1) - current_cluster))))
        prediction.append(pred)
    assert target is not None and origin is not None and sample is not None
    return {
        "label": label,
        "prediction": np.stack(prediction),
        "target": target,
        "origin": origin,
        "sample": sample,
        "max_aggregation_error_mw": max(aggregation_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate station-level evidence from exported predictions")
    parser.add_argument("--exports-root", default="artifacts/paper_evidence_test_v1/station_exports")
    parser.add_argument("--group", action="append", default=["Direct", "Normal", "KD"])
    parser.add_argument("--candidate", default="KD")
    parser.add_argument("--reference", default="Direct")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence_test_v1/station")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value]
    exports = Path(args.exports_root).resolve()
    output = Path(args.output_dir).resolve()
    cache = Path(args.cache_dir).resolve()
    groups = [load_exports(label, exports, seeds, args.split) for label in args.group]
    by_label = {group["label"]: group for group in groups}
    candidate, reference = by_label[args.candidate], by_label[args.reference]
    for group in groups[1:]:
        np.testing.assert_array_equal(groups[0]["origin"], group["origin"])
        np.testing.assert_array_equal(groups[0]["sample"], group["sample"])
    capacity = np.load(cache / "static/station_capacity_mw.npy").astype(np.float64)
    latitude = np.load(cache / "static/station_latitude.npy").astype(np.float64)
    longitude = np.load(cache / "static/station_longitude.npy").astype(np.float64)
    station_name = np.load(cache / "static/station_name.npy").astype(str)
    rows: list[dict[str, Any]] = []
    station_mae: dict[str, np.ndarray] = {}
    station_ramp_mae: dict[str, np.ndarray] = {}
    for group in groups:
        error = group["prediction"] - group["target"][None]
        station_mae[group["label"]] = np.abs(error).mean(axis=(0, 1, 3))
        prediction_ramp = np.diff(group["prediction"], axis=-1)
        target_ramp = np.diff(group["target"], axis=-1)
        station_ramp_mae[group["label"]] = np.abs(prediction_ramp - target_ramp[None]).mean(axis=(0, 1, 3))
        rows.append(
            {
                "group": group["label"],
                "seeds": len(seeds),
                "station_mae_mw": float(np.abs(error).mean()),
                "median_station_mae_mw": float(np.median(station_mae[group["label"]])),
                "p90_station_mae_mw": float(np.quantile(station_mae[group["label"]], 0.90)),
                "station_ramp_mae_mw": float(station_ramp_mae[group["label"]].mean()),
                "below_zero_prediction_fraction": float(np.mean(group["prediction"] < -1e-5)),
                "above_capacity_prediction_fraction": float(np.mean(group["prediction"] > capacity[None, None, :, None] + 1e-5)),
                "max_station_sum_vs_cluster_error_mw": group["max_aggregation_error_mw"],
            }
        )
    write_csv(output / "tables/station_summary.csv", rows)
    station_rows = []
    delta = station_mae[args.candidate] - station_mae[args.reference]
    ramp_delta = station_ramp_mae[args.candidate] - station_ramp_mae[args.reference]
    for index in range(len(capacity)):
        station_rows.append(
            {
                "station_index": index,
                "station_name": station_name[index],
                "capacity_mw": capacity[index],
                "latitude": latitude[index],
                "longitude": longitude[index],
                "candidate_mae_mw": station_mae[args.candidate][index],
                "reference_mae_mw": station_mae[args.reference][index],
                "candidate_minus_reference_mae_mw": delta[index],
                "candidate_minus_reference_ramp_mae_mw": ramp_delta[index],
            }
        )
    write_csv(output / "tables/per_station.csv", station_rows)

    colors = {group["label"]: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
    global _FIGURE_PROGRESS
    station_case_count = min(args.case_count, len(candidate["origin"]))
    with tqdm(total=3 + 6 * station_case_count, **progress_options("render station figures", unit="figure")) as progress:
        _FIGURE_PROGRESS = progress
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
        scale = max(abs(delta.min()), abs(delta.max()))
        scatter = axes[0].scatter(longitude, latitude, c=delta, s=25 + 1.2 * capacity, cmap="RdBu_r", vmin=-scale, vmax=scale, edgecolor="black", linewidth=0.3)
        axes[0].set(xlabel="Longitude", ylabel="Latitude", title="Station MAE delta (candidate − reference)")
        fig.colorbar(scatter, ax=axes[0], label="MW")
        order = np.argsort(delta)
        axes[1].bar(np.arange(len(delta)), delta[order], color=np.where(delta[order] < 0, "#059669", "#dc2626"))
        axes[1].axhline(0, color="black", lw=0.8); axes[1].set(xlabel="Stations ranked by improvement", ylabel="MAE delta (MW)", title="All 56 stations")
        save_figure(fig, output / "figures/station_map_and_rank")

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        axes[0].scatter(station_mae[args.reference], station_mae[args.candidate], s=20 + capacity, alpha=0.75)
        limit = max(station_mae[args.reference].max(), station_mae[args.candidate].max())
        axes[0].plot([0, limit], [0, limit], "k--", lw=0.8)
        axes[0].set(xlabel=f"{args.reference} station MAE (MW)", ylabel=f"{args.candidate} station MAE (MW)", title="Per-station accuracy")
        axes[1].scatter(capacity, -delta, c=-ramp_delta, cmap="viridis", s=40)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set(xlabel="Station capacity (MW)", ylabel="Candidate improvement (MW)", title="Capacity and station benefit")
        save_figure(fig, output / "figures/station_scatter_diagnostics")

        candidate_error = np.abs(candidate["prediction"] - candidate["target"][None]).mean(axis=(0, 1))
        reference_error = np.abs(reference["prediction"] - reference["target"][None]).mean(axis=(0, 1))
        matrix = candidate_error - reference_error
        fig, ax = plt.subplots(figsize=(10.0, 8.0))
        scale = np.max(np.abs(matrix))
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set(xlabel="Lead step (15 min)", ylabel="Station index", title="Station × horizon MAE delta")
        fig.colorbar(image, ax=ax, label="MW")
        save_figure(fig, output / "figures/station_horizon_delta_heatmap")

        fixed = 0
        target = candidate["target"]
        cand_pred = candidate["prediction"][fixed]
        ref_pred = reference["prediction"][fixed]
        station_order = np.concatenate((np.argsort(delta)[:3], np.argsort(delta)[-3:]))
        case_rows = []
        for station in station_order:
            ramp = np.abs(np.diff(target[:, station], axis=1)).max(axis=1)
            for rank, origin_index in enumerate(np.argsort(ramp)[-args.case_count:][::-1], start=1):
                timestamp = pd.to_datetime(candidate["origin"][origin_index], unit="s", utc=True)
                fig, ax = plt.subplots(figsize=(6.8, 3.8))
                hours = np.arange(1, 25) / 4.0
                ax.plot(hours, target[origin_index, station], color="black", lw=2, label="Observed")
                ax.plot(hours, ref_pred[origin_index, station], color=colors[args.reference], label=args.reference)
                ax.plot(hours, cand_pred[origin_index, station], color=colors[args.candidate], label=args.candidate)
                ax.set(xlabel="Lead time (h)", ylabel="Station power (MW)", title=f"Station {station} | high-ramp #{rank} | {timestamp:%Y-%m-%d %H:%M} UTC")
                ax.grid(alpha=0.2); ax.legend(fontsize=8)
                path = output / "figures/station_cases" / f"station_{station:02d}" / f"{rank:02d}_{timestamp:%Y%m%dT%H%M}"
                save_figure(fig, path)
                case_rows.append({"station_index": int(station), "rank": rank, "origin_unix": int(candidate["origin"][origin_index]), "png": str(path.with_suffix('.png').resolve())})
        _FIGURE_PROGRESS = None
    write_csv(output / "tables/station_case_manifest.csv", case_rows)
    summary = {
        "split": args.split,
        "groups": args.group,
        "seeds": seeds,
        "candidate": args.candidate,
        "reference": args.reference,
        "figure_png_count": len(list((output / "figures").rglob("*.png"))),
        "station_case_count": len(case_rows),
        "note": "Station cases use the first requested fixed seed and predefined high-ramp origins; they are visualization, not statistical evidence.",
    }
    (output / "station_evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
