from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts/farm_level_history_weather_v1"
CACHE = PROJECT / "data/model_cache_pad5_f8"
SEEDS = tuple(range(42, 52))
HELDOUT_SEEDS = (42, 46, 50)
HORIZONS = {"all": slice(0, 24), "0_1h": slice(0, 4), "1_3h": slice(4, 12), "3_6h": slice(12, 24)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("")
        return
    fields = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def prediction_path(group: str, seed: int, split: str) -> Path:
    return ROOT / "reused" / group / f"seed{seed}" / f"predictions_{split}.npz"


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def aligned_cache(split: str, sample_index: np.ndarray, key: str) -> np.ndarray:
    cache_index = np.load(CACHE / split / "sample_index.npy")
    values = np.load(CACHE / split / f"{key}.npy", mmap_mode="r")
    if np.array_equal(cache_index, sample_index):
        return np.asarray(values)
    lookup = {int(value): index for index, value in enumerate(cache_index)}
    rows = np.asarray([lookup[int(value)] for value in sample_index], dtype=np.int64)
    return np.asarray(values[rows])


def station_metrics(
    prediction_mw: np.ndarray,
    target_mw: np.ndarray,
    current_cf: np.ndarray,
    capacity: np.ndarray,
    station_indices: list[int] | None = None,
) -> dict[str, Any]:
    indices = np.arange(len(capacity)) if station_indices is None else np.asarray(station_indices)
    pred_cf = prediction_mw[:, indices] / capacity[indices][None, :, None]
    target_cf = target_mw[:, indices] / capacity[indices][None, :, None]
    error = pred_cf - target_cf
    per_farm_mae = np.mean(np.abs(error), axis=(0, 2))
    per_farm_rmse = np.sqrt(np.mean(np.square(error), axis=(0, 2)))
    lead = np.mean(np.abs(error), axis=(0, 1))
    current = current_cf[:, indices]
    pred_path = np.concatenate((current[:, :, None], pred_cf), axis=2)
    target_path = np.concatenate((current[:, :, None], target_cf), axis=2)
    ramp = np.diff(pred_path, axis=2) - np.diff(target_path, axis=2)
    persistence = current[:, :, None] - target_cf
    cluster_pred = prediction_mw[:, indices].sum(axis=1)
    cluster_target = target_mw[:, indices].sum(axis=1)
    return {
        "station_mae_cf": float(per_farm_mae.mean()),
        "station_rmse_cf": float(per_farm_rmse.mean()),
        "station_ramp_mae_cf": float(np.mean(np.abs(ramp), axis=(0, 2)).mean()),
        "station_persistence_mae_cf": float(np.mean(np.abs(persistence), axis=(0, 2)).mean()),
        "station_skill_vs_persistence": float(1.0 - per_farm_mae.mean() / np.mean(np.abs(persistence), axis=(0, 2)).mean()),
        "lead_station_mae_cf": lead.tolist(),
        "horizon_0_1h_station_mae_cf": float(lead[:4].mean()),
        "horizon_1_3h_station_mae_cf": float(lead[4:12].mean()),
        "horizon_3_6h_station_mae_cf": float(lead[12:24].mean()),
        "physical_sum_cluster_mae_mw": float(np.abs(cluster_pred - cluster_target).mean()),
        "per_farm_mae_cf": per_farm_mae.tolist(),
        "per_farm_rmse_cf": per_farm_rmse.tolist(),
    }


def covariates(split: str) -> dict[str, np.ndarray]:
    output = ROOT / "analysis/cache" / f"regime_covariates_{split}.npz"
    if output.is_file():
        return load_archive(output)
    history = np.load(CACHE / split / "station_history.npy", mmap_mode="r")
    current_cf = np.asarray(history[:, :, -1], dtype=np.float32)
    recent = np.asarray(history[:, :, -12:], dtype=np.float32)
    x = np.arange(recent.shape[-1], dtype=np.float32) * 0.25
    centered = x - x.mean()
    slope = np.sum((recent - recent.mean(axis=-1, keepdims=True)) * centered, axis=-1) / np.sum(centered**2)

    checkpoint = torch.load(
        ROOT / "reused/hw_int/seed42/best_model.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["model_state"]
    patch = state["spatial_station_query.patch_index"].numpy()
    weights = state["spatial_station_query.fixed_spatial_weights"].numpy()
    weather = np.load(CACHE / split / "weather.npy", mmap_mode="r")
    cycle = np.load(CACHE / split / "weather_cycle_index.npy", mmap_mode="r")
    offsets = np.load(CACHE / split / "weather_time_offset_hours.npy", mmap_mode="r")
    channel_names = np.load(CACHE / "static/nwp_channel_name.npy").tolist()
    u_index, v_index = channel_names.index("u10"), channel_names.index("v10")
    mean = np.load(CACHE / "static/nwp_mean.npy")
    std = np.load(CACHE / "static/nwp_std.npy")
    node0 = np.abs(offsets - 0.0).argmin(axis=1)
    node6 = np.abs(offsets - 6.0).argmin(axis=1)
    speed_nodes = np.empty((len(cycle), patch.shape[0], weather.shape[1]), dtype=np.float32)
    flat_patch = patch.reshape(-1)
    flat_weights = weights.reshape(-1)
    for start in range(0, len(cycle), 256):
        stop = min(start + 256, len(cycle))
        cycles = np.asarray(cycle[start:stop])
        for node in range(weather.shape[1]):
            for channel, destination in ((u_index, 0), (v_index, 1)):
                raw = np.asarray(weather[cycles, node, channel]).reshape(stop - start, -1)
                gathered = raw[:, flat_patch].reshape(stop - start, patch.shape[0], 4)
                interpolated = (gathered * weights[None]).sum(axis=-1)
                physical = interpolated * std[channel] + mean[channel]
                if destination == 0:
                    u = physical
                else:
                    v = physical
            speed_nodes[start:stop, :, node] = np.sqrt(u * u + v * v)
    rows = np.arange(len(cycle))
    wind0 = speed_nodes[rows, :, node0]
    wind6 = speed_nodes[rows, :, node6]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        current_cf=current_cf,
        history_slope_cf_per_hour=slope.astype(np.float32),
        abs_history_slope_cf_per_hour=np.abs(slope).astype(np.float32),
        future_wind_change_ms=(wind6 - wind0).astype(np.float32),
        future_wind_std_ms=np.std(speed_nodes, axis=2).astype(np.float32),
        origin_unix=np.load(CACHE / split / "origin_unix.npy"),
        sample_index=np.load(CACHE / split / "sample_index.npy"),
    )
    return load_archive(output)


def quantile_edges(values: np.ndarray, quantiles: int) -> list[float]:
    return np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0, 1, quantiles + 1))).tolist()


def freeze_regime_protocol() -> dict[str, Any]:
    path = ROOT / "selection/regime_protocol.json"
    if path.is_file():
        return json.loads(path.read_text())
    train = covariates("train")
    edges = {
        "current_cf": [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        "history_slope_cf_per_hour": quantile_edges(train["history_slope_cf_per_hour"], 5),
        "abs_history_slope_cf_per_hour": quantile_edges(train["abs_history_slope_cf_per_hour"], 5),
        "future_wind_change_ms": quantile_edges(train["future_wind_change_ms"], 5),
        "future_wind_std_ms": quantile_edges(train["future_wind_std_ms"], 5),
        "ramp_tercile": quantile_edges(train["history_slope_cf_per_hour"], 3),
        "wind_change_tercile": quantile_edges(train["future_wind_change_ms"], 3),
    }
    protocol = {
        "edges": edges,
        "edge_source": "2022 training cache only; CF bins fixed a priori",
        "gain": "abs_error_IAdd - abs_error_Int (positive favors Int)",
        "estimand": "equal-farm mean",
        "uncertainty": "10000 shared UTC-day block bootstrap replicates",
        "horizons": {key: [value.start or 0, value.stop or 24] for key, value in HORIZONS.items()},
        "test_results_read": False,
    }
    atomic_json(path, protocol)
    return protocol


def equal_farm_day_bootstrap(
    gain: np.ndarray, mask: np.ndarray, origin_unix: np.ndarray, seed: int
) -> tuple[float, float, float, int, int]:
    # gain is [sample, farm] after averaging the selected leads.
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], gain.shape)
    valid = mask & np.isfinite(gain)
    days, inverse = np.unique(origin_unix // 86400, return_inverse=True)
    sums = np.zeros((len(days), gain.shape[1]), dtype=np.float64)
    counts = np.zeros_like(sums)
    for farm in range(gain.shape[1]):
        np.add.at(sums[:, farm], inverse[valid[:, farm]], gain[:, farm][valid[:, farm]])
        np.add.at(counts[:, farm], inverse[valid[:, farm]], 1.0)
    farm_total = sums.sum(0)
    farm_count = counts.sum(0)
    estimate = float(np.nanmean(np.divide(farm_total, farm_count, out=np.full_like(farm_total, np.nan), where=farm_count > 0)))
    rng = np.random.default_rng(seed)
    boot = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 200):
        size = min(200, 10000 - start)
        sampled = rng.integers(0, len(days), size=(size, len(days)))
        selected_sums = sums[sampled].sum(axis=1)
        selected_counts = counts[sampled].sum(axis=1)
        farm_means = np.divide(selected_sums, selected_counts, out=np.full_like(selected_sums, np.nan), where=selected_counts > 0)
        boot[start:start + size] = np.nanmean(farm_means, axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high), int(valid.sum()), int((farm_count > 0).sum())


def regime_rows(split: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    reference = load_archive(prediction_path("hw_int", SEEDS[0], split))
    int_mean = np.zeros_like(reference["station_prediction_mw"], dtype=np.float32)
    add_mean = np.zeros_like(int_mean)
    for seed in SEEDS:
        int_mean += load_archive(prediction_path("hw_int", seed, split))["station_prediction_mw"] / len(SEEDS)
        add_mean += load_archive(prediction_path("hw_iadd", seed, split))["station_prediction_mw"] / len(SEEDS)
    capacity = np.load(CACHE / "static/station_capacity_mw.npy")
    target_cf = reference["station_target_mw"] / capacity[None, :, None]
    int_cf = int_mean / capacity[None, :, None]
    add_cf = add_mean / capacity[None, :, None]
    gain = np.abs(add_cf - target_cf) - np.abs(int_cf - target_cf)
    cov = covariates(split)
    if not np.array_equal(cov["sample_index"], reference["sample_index"]):
        lookup = {int(value): index for index, value in enumerate(cov["sample_index"])}
        order = np.asarray([lookup[int(value)] for value in reference["sample_index"]])
        cov = {key: value[order] for key, value in cov.items()}
    rows: list[dict[str, Any]] = []
    for variable in (
        "current_cf", "history_slope_cf_per_hour", "abs_history_slope_cf_per_hour",
        "future_wind_change_ms", "future_wind_std_ms",
    ):
        values = cov[variable]
        edges = protocol["edges"][variable]
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (values >= lower) & (values < upper if index < len(edges) - 2 else values <= upper)
            for horizon, lead_slice in HORIZONS.items():
                estimate, low, high, count, farms = equal_farm_day_bootstrap(
                    gain[:, :, lead_slice].mean(axis=2), mask, cov["origin_unix"],
                    seed=20260907 + index + 17 * len(rows),
                )
                rows.append({
                    "split": split, "variable": variable, "bin": index,
                    "lower": lower, "upper": upper, "horizon": horizon,
                    "gain_cf": estimate, "ci95_low": low, "ci95_high": high,
                    "observations": count, "farms": farms,
                })
    ramp_edges = protocol["edges"]["ramp_tercile"]
    wind_edges = protocol["edges"]["wind_change_tercile"]
    ramp = cov["history_slope_cf_per_hour"]
    wind = cov["future_wind_change_ms"]
    for ri, (rlo, rhi) in enumerate(zip(ramp_edges[:-1], ramp_edges[1:])):
        for wi, (wlo, whi) in enumerate(zip(wind_edges[:-1], wind_edges[1:])):
            mask = (ramp >= rlo) & (ramp <= rhi) & (wind >= wlo) & (wind <= whi)
            for horizon, lead_slice in HORIZONS.items():
                estimate, low, high, count, farms = equal_farm_day_bootstrap(
                    gain[:, :, lead_slice].mean(axis=2), mask, cov["origin_unix"],
                    seed=20261907 + ri * 100 + wi * 10 + len(rows),
                )
                rows.append({
                    "split": split, "variable": "ramp_x_wind_change",
                    "bin": f"{ri}x{wi}", "lower": f"{rlo}|{wlo}",
                    "upper": f"{rhi}|{whi}", "horizon": horizon,
                    "gain_cf": estimate, "ci95_low": low, "ci95_high": high,
                    "observations": count, "farms": farms,
                })
    return rows


def baseline_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selection = json.loads((ROOT / "selection/history_validation_freeze.json").read_text())
    winner = int(selection["selected_history_visible_steps"])
    strong_group = {4: "h_strong_4", 12: "h_matched", 24: "h_strong_24"}[winner]
    groups = {
        "H-Matched": ROOT / "baseline/h_matched",
        "H-Strong": ROOT / f"baseline/{strong_group}",
        "W": ROOT / "baseline/w_anchor",
        "HW-None": ROOT / "reused/hw_none",
        "HW-SAdd": ROOT / "reused/hw_sadd",
        "HW-IAdd": ROOT / "reused/hw_iadd",
        "HW-Int": ROOT / "reused/hw_int",
    }
    capacity = np.load(CACHE / "static/station_capacity_mw.npy")
    rows: list[dict[str, Any]] = []
    farm_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        reference = load_archive(prediction_path("hw_int", 42, split))
        history = aligned_cache(split, reference["sample_index"], "station_history")
        for seed in SEEDS:
            for label, directory in groups.items():
                archive = load_archive(directory / f"seed{seed}" / f"predictions_{split}.npz")
                metrics = station_metrics(
                    archive["station_prediction_mw"], archive["station_target_mw"],
                    history[:, :, -1], capacity,
                )
                metric_file = json.loads((directory / f"seed{seed}" / f"metrics_{split}.json").read_text())
                history_file = directory / f"seed{seed}" / "history.json"
                train_seconds = None
                if history_file.is_file():
                    train_seconds = float(sum(row.get("seconds", 0.0) for row in json.loads(history_file.read_text()).get("train", [])))
                rows.append({
                    "split": split, "model": label, "seed": seed,
                    **{k: v for k, v in metrics.items() if not isinstance(v, list)},
                    "parameter_count": metric_file.get("parameter_count"),
                    "training_seconds": train_seconds,
                })
                for farm, (mae, rmse) in enumerate(zip(metrics["per_farm_mae_cf"], metrics["per_farm_rmse_cf"])):
                    farm_rows.append({"split": split, "model": label, "seed": seed, "farm": farm, "mae_cf": mae, "rmse_cf": rmse})
                for lead, mae in enumerate(metrics["lead_station_mae_cf"], 1):
                    lead_rows.append({"split": split, "model": label, "seed": seed, "lead_minutes": lead * 15, "mae_cf": mae})
        persistence_mw = history[:, :, -1, None] * capacity[None, :, None]
        persistence_mw = np.broadcast_to(persistence_mw, reference["station_target_mw"].shape)
        persistence = station_metrics(
            persistence_mw, reference["station_target_mw"], history[:, :, -1], capacity
        )
        rows.append({"split": split, "model": "P", "seed": "deterministic", **{k: v for k, v in persistence.items() if not isinstance(v, list)}, "parameter_count": 0, "training_seconds": 0.0})
        for farm, (mae, rmse) in enumerate(zip(persistence["per_farm_mae_cf"], persistence["per_farm_rmse_cf"])):
            farm_rows.append({"split": split, "model": "P", "seed": "deterministic", "farm": farm, "mae_cf": mae, "rmse_cf": rmse})
        for lead, mae in enumerate(persistence["lead_station_mae_cf"], 1):
            lead_rows.append({"split": split, "model": "P", "seed": "deterministic", "lead_minutes": lead * 15, "mae_cf": mae})
    return rows, farm_rows, lead_rows


def heldout_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((ROOT / "selection/fold_manifest.json").read_text())
    capacity = np.load(CACHE / "static/station_capacity_mw.npy")
    rows: list[dict[str, Any]] = []
    fold_delta: dict[str, list[float]] = {"val": [], "test": []}
    for split in ("val", "test"):
        for fold_index, heldout in enumerate(manifest["folds"]):
            by_model: dict[str, list[float]] = {"IAdd": [], "Int": []}
            for label, suffix in (("IAdd", "iadd"), ("Int", "int")):
                for seed in HELDOUT_SEEDS:
                    path = ROOT / f"heldout/fold{fold_index}_{suffix}/seed{seed}/predictions_{split}.npz"
                    archive = load_archive(path)
                    history = aligned_cache(split, archive["sample_index"], "station_history")
                    metrics = station_metrics(
                        archive["station_prediction_mw"], archive["station_target_mw"],
                        history[:, :, -1], capacity, station_indices=heldout,
                    )
                    by_model[label].append(metrics["station_mae_cf"])
                    rows.append({"split": split, "fold": fold_index, "model": label, "seed": seed, **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
            fold_delta[split].append(float(np.mean(by_model["Int"]) - np.mean(by_model["IAdd"])))
    summary: dict[str, Any] = {}
    for split, delta in fold_delta.items():
        values = np.asarray(delta)
        mean = float(values.mean())
        half = 2.365 * float(values.std(ddof=1)) / math.sqrt(8)
        summary[split] = {
            "int_minus_iadd_cf": mean, "ci95": [mean - half, mean + half],
            "improved_folds": int(np.sum(values < 0)), "fold_delta": values.tolist(),
            "success": bool(mean + half < 0 and np.sum(values < 0) >= 6),
        }
    return rows, summary


def trajectory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, group in (("IAdd", "b_independent_add"), ("Int", "b_int")):
        for seed in SEEDS:
            station_metric = json.loads((ROOT / f"reused/hw_{model.lower()}/seed{seed}/metrics_val.json").read_text())
            rows.append({"model": model, "seed": seed, "epoch": 0, "station_mae_cf": station_metric["station_mae_cf"], "cluster_mae_mw": station_metric["cluster_mae_mw"], "cluster_ramp_mae_mw": station_metric["cluster_ramp_mae_mw"]})
            history = json.loads((PROJECT / f"artifacts/local_state_cluster_transfer_v2/transfer/{group}/seed{seed}/history.json").read_text())
            for epoch, values in enumerate(history["val"], 1):
                rows.append({"model": model, "seed": seed, "epoch": epoch, "station_mae_cf": values["station_mae_cf"], "cluster_mae_mw": values["cluster_mae_mw"], "cluster_ramp_mae_mw": values["cluster_ramp_mae_mw"]})
    return rows


def make_figures(
    baseline: list[dict[str, Any]], regimes: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "analysis/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = ["P", "H-Matched", "H-Strong", "W", "HW-None", "HW-SAdd", "HW-IAdd", "HW-Int"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, split, title in zip(axes, ("val", "test"), ("2023 validation", "2024 frozen test")):
        values = [np.mean([row["station_mae_cf"] for row in baseline if row["split"] == split and row["model"] == model]) for model in models]
        axis.bar(np.arange(len(models)), values)
        axis.set_xticks(np.arange(len(models)), models, rotation=45, ha="right")
        axis.set_ylabel("Macro station CF MAE")
        axis.set_title(title)
    fig.savefig(figure_dir / "baseline_station_mae.png", dpi=180)
    plt.close(fig)

    for split in ("val", "test"):
        selected = [row for row in regimes if row["split"] == split and row["variable"] == "ramp_x_wind_change" and row["horizon"] == "all"]
        matrix = np.full((3, 3), np.nan)
        for row in selected:
            ri, wi = (int(value) for value in str(row["bin"]).split("x"))
            matrix[ri, wi] = row["gain_cf"]
        fig, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        image = axis.imshow(matrix, origin="lower", cmap="RdBu_r", aspect="auto")
        axis.set_xlabel("Future Pangu wind-change tercile")
        axis.set_ylabel("Historical power-slope tercile")
        axis.set_title(f"Int gain (IAdd MAE - Int MAE), {split}")
        fig.colorbar(image, ax=axis, label="CF MAE gain")
        fig.savefig(figure_dir / f"regime_ramp_wind_{split}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for model in ("IAdd", "Int"):
        epochs = sorted({row["epoch"] for row in trajectories if row["model"] == model})
        station = [np.mean([row["station_mae_cf"] for row in trajectories if row["model"] == model and row["epoch"] == epoch]) for epoch in epochs]
        cluster = [np.mean([row["cluster_mae_mw"] for row in trajectories if row["model"] == model and row["epoch"] == epoch]) for epoch in epochs]
        axes[0].plot(epochs, station, marker="o", markersize=2, label=model)
        axes[1].plot(epochs, cluster, marker="o", markersize=2, label=model)
    axes[0].set(xlabel="Cluster fine-tune epoch (0 = station checkpoint)", ylabel="Station CF MAE")
    axes[1].set(xlabel="Cluster fine-tune epoch (0 = station checkpoint)", ylabel="Cluster MAE (MW)")
    for axis in axes: axis.legend()
    fig.savefig(figure_dir / "cluster_finetune_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-protocol", action="store_true")
    args = parser.parse_args()
    protocol = freeze_regime_protocol()
    if args.freeze_protocol:
        print(ROOT / "selection/regime_protocol.json")
        return
    analysis = ROOT / "analysis"
    baseline, baseline_farms, baseline_leads = baseline_rows()
    heldout, heldout_summary = heldout_rows()
    regimes = regime_rows("val", protocol) + regime_rows("test", protocol)
    trajectories = trajectory_rows()
    write_csv(analysis / "baseline_seed_metrics.csv", baseline)
    write_csv(analysis / "baseline_per_farm.csv", baseline_farms)
    write_csv(analysis / "baseline_per_lead.csv", baseline_leads)
    write_csv(analysis / "heldout_fold_metrics.csv", heldout)
    write_csv(analysis / "interaction_regimes.csv", regimes)
    write_csv(analysis / "cluster_finetune_trajectory.csv", trajectories)
    make_figures(baseline, regimes, trajectories)
    summary = {
        "history_selection": json.loads((ROOT / "selection/history_validation_freeze.json").read_text()),
        "heldout_generalization": heldout_summary,
        "baseline_mean": {
            split: {
                model: {
                    key: float(np.mean([row[key] for row in baseline if row["split"] == split and row["model"] == model]))
                    for key in ("station_mae_cf", "station_rmse_cf", "station_ramp_mae_cf", "station_skill_vs_persistence", "physical_sum_cluster_mae_mw")
                }
                for model in sorted({row["model"] for row in baseline if row["split"] == split})
            }
            for split in ("val", "test")
        },
        "regime_protocol": protocol,
        "route_rule": "Retain farm-level interaction as a transferable mechanism only if held-out Int-IAdd passes on both years; cluster is a secondary scale-boundary result.",
    }
    atomic_json(analysis / "summary.json", summary)
    report = f"""# Farm-level history–weather study v1

This document is generated only after the 2023 validation protocol was frozen and the 2024 outputs were evaluated.

## Scientific estimand

The primary test is whether future local Pangu forcing and realized local generation history are predictively non-separable: `HW-Int` versus the equal-parameter independent `HW-IAdd`. History is not claimed to identify a causal turbine state.

## Baseline completeness

The machine-readable complete table is `analysis/baseline_seed_metrics.csv`. It contains persistence, matched/strong history-only, anchor+weather, no-extra-state, shared additive, independent additive, and interaction models for 10 seeds on both years, with station MAE/RMSE/ramp/lead-derived summaries, persistence skill, and physical-sum cluster MAE.

## Held-out-farm result

- 2023: Int-IAdd = {heldout_summary['val']['int_minus_iadd_cf']:.6f} CF, CI {heldout_summary['val']['ci95']}, improved folds {heldout_summary['val']['improved_folds']}/8, pass={heldout_summary['val']['success']}.
- 2024: Int-IAdd = {heldout_summary['test']['int_minus_iadd_cf']:.6f} CF, CI {heldout_summary['test']['ci95']}, improved folds {heldout_summary['test']['improved_folds']}/8, pass={heldout_summary['test']['success']}.

## Interpretation boundary

Positive regime gain means the joint model reduces absolute CF error relative to the independent additive control. Conditional results and their pre-frozen day-block intervals are in `analysis/interaction_regimes.csv`. They identify when non-separability is predictively useful; they do not identify turbine availability, curtailment, or a causal physical state without corresponding observations.

## Cluster extension

`analysis/cluster_finetune_trajectory.csv` records station-only epoch zero and every cluster-aware fine-tuning epoch. This is a scale-boundary diagnostic, not a new cluster model search.
"""
    (PROJECT / "exp_docs/farm_level_history_weather_v1_results.md").write_text(report)
    print(analysis / "summary.json")


if __name__ == "__main__":
    main()
