from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_GROUPS = (
    "Direct=artifacts/paper_evidence_test_v1/cluster_exports/Direct",
    "Normal=artifacts/paper_evidence_test_v1/cluster_exports/Normal",
    "KD=artifacts/paper_evidence_test_v1/cluster_exports/KD",
)
COLORS = ("#6b7280", "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
HORIZONS = np.arange(1, 25, dtype=np.float64) / 4.0
_FIGURE_PROGRESS: tqdm | None = None


def progress_options(description: str, *, unit: str, leave: bool | None = None) -> dict[str, Any]:
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    environment_leave = os.environ.get("WIND_PROGRESS_LEAVE", "1").lower() not in {"0", "false", "no"}
    return {
        "desc": f"{prefix} {description}".strip(),
        "unit": unit,
        "dynamic_ncols": True,
        "position": int(position_value) if position_value else None,
        "leave": environment_leave if leave is None else leave,
        "disable": False,
    }


@dataclass
class Group:
    label: str
    root: Path
    seeds: list[int]
    prediction: np.ndarray
    target: np.ndarray
    origin: np.ndarray
    sample_index: np.ndarray
    history_last: np.ndarray
    optional: dict[str, np.ndarray]

    @property
    def ensemble(self) -> np.ndarray:
        return self.prediction.mean(axis=0)


def parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--group must be LABEL=GROUP_ROOT")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--group must be LABEL=GROUP_ROOT")
    return label.strip(), Path(path).expanduser().resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    if _FIGURE_PROGRESS is not None:
        _FIGURE_PROGRESS.set_postfix_str(output.name, refresh=False)
        _FIGURE_PROGRESS.update(1)


def load_group(label: str, root: Path, seeds: list[int], split: str) -> Group:
    predictions: list[np.ndarray] = []
    optional_lists: dict[str, list[np.ndarray]] = {}
    target = origin = sample_index = history_last = None
    common_optional: set[str] | None = None
    for seed in tqdm(seeds, total=len(seeds), **progress_options(f"load {label}", unit="seed", leave=False)):
        path = root / f"seed{seed}" / f"predictions_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["cluster_prediction_mw"].astype(np.float64)
            current_target = archive["cluster_target_mw"].astype(np.float64)
            current_origin = archive["origin_unix"].astype(np.int64)
            current_sample = archive["sample_index"].astype(np.int64)
            current_history = archive["cluster_history_last_mw"].astype(np.float64)
            keys = {
                key
                for key in archive.files
                if key
                in {
                    "baseline_cluster_prediction_mw",
                    "anchor_baseline_cluster_prediction_mw",
                    "realized_cluster_correction_mw",
                    "cluster_raw_correction_mw",
                }
            }
            common_optional = keys if common_optional is None else common_optional & keys
            local_optional = {key: archive[key].astype(np.float64) for key in keys}
        if target is None:
            target, origin, sample_index, history_last = (
                current_target,
                current_origin,
                current_sample,
                current_history,
            )
        else:
            np.testing.assert_array_equal(origin, current_origin)
            np.testing.assert_array_equal(sample_index, current_sample)
            np.testing.assert_allclose(target, current_target, rtol=0.0, atol=1e-4)
        predictions.append(prediction)
        for key, value in local_optional.items():
            optional_lists.setdefault(key, []).append(value)
    assert target is not None and origin is not None and sample_index is not None
    assert history_last is not None
    common_optional = common_optional or set()
    optional = {
        key: np.stack(optional_lists[key])
        for key in sorted(common_optional)
        if len(optional_lists.get(key, ())) == len(seeds)
    }
    return Group(
        label=label,
        root=root,
        seeds=seeds,
        prediction=np.stack(predictions),
        target=target,
        origin=origin,
        sample_index=sample_index,
        history_last=history_last,
        optional=optional,
    )


def align_cache(cache_dir: Path, sample_index: np.ndarray, split: str) -> dict[str, np.ndarray]:
    cache_ids = np.load(cache_dir / f"{split}/sample_index.npy", mmap_mode="r")
    order = np.argsort(cache_ids, kind="stable")
    sorted_ids = np.asarray(cache_ids[order])
    positions = np.searchsorted(sorted_ids, sample_index)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], sample_index
    ):
        raise ValueError("Prediction sample indices do not align with validation cache")
    rows = order[positions]
    capacity = np.load(cache_dir / "static/station_capacity_mw.npy").astype(np.float64)
    total_capacity = float(capacity.sum())
    return {
        "cluster_history_mw": np.load(
            cache_dir / f"{split}/cluster_history.npy", mmap_mode="r"
        )[rows].astype(np.float64)
        * total_capacity,
        "nwp_age_minutes": np.load(
            cache_dir / f"{split}/nwp_age_minutes.npy", mmap_mode="r"
        )[rows].astype(np.int64),
        "origin_phase": np.load(cache_dir / f"{split}/origin_phase.npy", mmap_mode="r")[
            rows
        ].astype(np.int64),
        "capacity_mw": capacity,
        "total_capacity_mw": np.asarray(total_capacity),
    }


def training_ramp_thresholds(cache_dir: Path, total_capacity: float) -> dict[str, float]:
    future = np.load(cache_dir / "train/cluster_future.npy", mmap_mode="r").astype(
        np.float64
    )
    history = np.load(cache_dir / "train/cluster_history.npy", mmap_mode="r").astype(
        np.float64
    )
    path = np.concatenate((history[:, -1:], future), axis=1) * total_capacity
    absolute_ramp = np.abs(np.diff(path, axis=1)).reshape(-1)
    return {
        "p90_mw_per_15min": float(np.quantile(absolute_ramp, 0.90)),
        "p95_mw_per_15min": float(np.quantile(absolute_ramp, 0.95)),
    }


def extended_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    history_last: np.ndarray,
    ramp_threshold: float,
) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    target_path = np.concatenate((history_last[:, None], target), axis=1)
    prediction_path = np.concatenate((history_last[:, None], prediction), axis=1)
    target_ramp = np.diff(target_path, axis=1)
    prediction_ramp = np.diff(prediction_path, axis=1)
    true_event = np.abs(target_ramp) >= ramp_threshold
    predicted_event = np.abs(prediction_ramp) >= ramp_threshold
    true_positive = float(np.sum(true_event & predicted_event))
    false_positive = float(np.sum(~true_event & predicted_event))
    false_negative = float(np.sum(true_event & ~predicted_event))
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    energy_error = error.sum(axis=1) * 0.25
    flat_prediction = prediction.reshape(-1)
    flat_target = target.reshape(-1)
    slope, intercept = np.polyfit(flat_prediction, flat_target, 1)
    return {
        "bias_mw": float(error.mean()),
        "median_absolute_error_mw": float(np.median(absolute)),
        "p90_absolute_error_mw": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_mw": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error_mw": float(np.quantile(absolute, 0.99)),
        "error_std_mw": float(error.std()),
        "pearson_prediction_target": float(np.corrcoef(flat_prediction, flat_target)[0, 1]),
        "within_50mw_fraction": float(np.mean(absolute <= 50.0)),
        "within_100mw_fraction": float(np.mean(absolute <= 100.0)),
        "within_200mw_fraction": float(np.mean(absolute <= 200.0)),
        "overprediction_fraction": float(np.mean(error > 0.0)),
        "calibration_slope_target_on_prediction": float(slope),
        "calibration_intercept_mw": float(intercept),
        "energy_bias_mwh": float(energy_error.mean()),
        "energy_mae_mwh": float(np.abs(energy_error).mean()),
        "energy_p95_absolute_error_mwh": float(np.quantile(np.abs(energy_error), 0.95)),
        "ramp_mae_mw_per_15min": float(np.abs(prediction_ramp - target_ramp).mean()),
        "ramp_direction_accuracy": float(np.mean(np.sign(prediction_ramp) == np.sign(target_ramp))),
        "extreme_ramp_precision": precision,
        "extreme_ramp_recall": recall,
        "extreme_ramp_f1": f1,
    }


def bootstrap_daily_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    delta = np.abs(candidate - target).mean(axis=1) - np.abs(reference - target).mean(axis=1)
    day = origin // 86400
    unique = np.unique(day)
    sums = np.asarray([delta[day == value].sum() for value in unique])
    counts = np.asarray([np.sum(day == value) for value in unique])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in tqdm(range(repeats), total=repeats, **progress_options("daily bootstrap", unit="sample", leave=False)):
        selected = rng.integers(0, len(unique), len(unique))
        samples[index] = sums[selected].sum() / counts[selected].sum()
    return {
        "candidate_minus_reference_mae_mw": float(delta.mean()),
        "ci95_low_mw": float(np.quantile(samples, 0.025)),
        "ci95_high_mw": float(np.quantile(samples, 0.975)),
        "daily_win_fraction": float(
            np.mean(
                [
                    delta[day == value].mean() < 0.0
                    for value in unique
                ]
            )
        ),
        "utc_days": int(len(unique)),
    }


def make_summary_tables(
    groups: list[Group],
    cache: dict[str, np.ndarray],
    threshold: float,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for group in groups:
        for index, seed in enumerate(group.seeds):
            metric_rows.append(
                {
                    "group": group.label,
                    "representation": f"seed{seed}",
                    **extended_metrics(
                        group.prediction[index], group.target, group.history_last, threshold
                    ),
                }
            )
        metric_rows.append(
            {
                "group": group.label,
                "representation": "ensemble",
                **extended_metrics(group.ensemble, group.target, group.history_last, threshold),
            }
        )
        error = group.prediction - group.target[None]
        persistence = group.history_last[:, None] - group.target
        for lead in range(24):
            seed_mae = np.abs(error[:, :, lead]).mean(axis=1)
            lead_rows.append(
                {
                    "group": group.label,
                    "lead_hours": float(HORIZONS[lead]),
                    "seed_mean_mae_mw": float(seed_mae.mean()),
                    "seed_std_mae_mw": float(seed_mae.std(ddof=1)),
                    "ensemble_mae_mw": float(np.abs(group.ensemble[:, lead] - group.target[:, lead]).mean()),
                    "ensemble_bias_mw": float((group.ensemble[:, lead] - group.target[:, lead]).mean()),
                    "persistence_skill": float(
                        1.0
                        - np.abs(group.ensemble[:, lead] - group.target[:, lead]).mean()
                        / np.abs(persistence[:, lead]).mean()
                    ),
                    "ensemble_disagreement_mw": float(group.prediction[:, :, lead].std(axis=0, ddof=1).mean()),
                }
            )
    atomic_csv(output / "tables/extended_metrics.csv", metric_rows)
    atomic_csv(output / "tables/per_lead.csv", lead_rows)
    return metric_rows, lead_rows


def regime_rows(
    group: Group,
    cache: dict[str, np.ndarray],
    output: Path,
) -> list[dict[str, Any]]:
    prediction = group.ensemble
    target = group.target
    history = cache["cluster_history_mw"]
    timestamp = pd.to_datetime(group.origin, unit="s", utc=True)
    current = history[:, -1]
    current_ramp = history[:, -1] - history[:, -2]
    target_ramp = np.max(
        np.abs(np.diff(np.concatenate((current[:, None], target), axis=1), axis=1)),
        axis=1,
    )
    variables: dict[str, np.ndarray] = {
        "month": timestamp.month.to_numpy(),
        "season": np.asarray(["DJF", "MAM", "JJA", "SON"])[
            ((timestamp.month.to_numpy() % 12) // 3)
        ],
        "nwp_age_minutes": cache["nwp_age_minutes"],
        "origin_phase": cache["origin_phase"],
        "current_power_decile": pd.qcut(current, 10, labels=False, duplicates="drop") + 1,
        "current_ramp_decile": pd.qcut(current_ramp, 10, labels=False, duplicates="drop") + 1,
        "future_max_ramp_decile": pd.qcut(target_ramp, 10, labels=False, duplicates="drop") + 1,
    }
    rows: list[dict[str, Any]] = []
    for name, values in variables.items():
        for value in np.unique(values):
            mask = values == value
            rows.append(
                {
                    "group": group.label,
                    "regime": name,
                    "value": str(value),
                    "origins": int(mask.sum()),
                    "ensemble_mae_mw": float(np.abs(prediction[mask] - target[mask]).mean()),
                    "ensemble_bias_mw": float((prediction[mask] - target[mask]).mean()),
                    "energy_mae_mwh": float(
                        np.abs((prediction[mask] - target[mask]).sum(axis=1) * 0.25).mean()
                    ),
                }
            )
    return rows


def plot_core(
    groups: list[Group],
    cache: dict[str, np.ndarray],
    candidate: Group,
    reference: Group,
    output: Path,
) -> None:
    figures = output / "figures/aggregate"
    color = {group.label: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for group in groups:
        seed_mae = np.abs(group.prediction - group.target[None]).mean(axis=1)
        mean, std = seed_mae.mean(axis=0), seed_mae.std(axis=0, ddof=1)
        ax.plot(HORIZONS, mean, label=group.label, color=color[group.label], lw=2)
        ax.fill_between(HORIZONS, mean - std, mean + std, color=color[group.label], alpha=0.12)
    ax.set(xlabel="Lead time (h)", ylabel="Cluster MAE (MW)", title="Lead-time error (mean ± seed SD)")
    ax.grid(alpha=0.25); ax.legend()
    save_figure(fig, figures / "lead_mae_seed_mean")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for group in groups:
        error = group.ensemble - group.target
        axes[0].plot(HORIZONS, error.mean(axis=0), label=group.label, color=color[group.label])
        persistence_mae = np.abs(group.history_last[:, None] - group.target).mean(axis=0)
        skill = 1.0 - np.abs(error).mean(axis=0) / persistence_mae
        axes[1].plot(HORIZONS, skill, label=group.label, color=color[group.label])
        axes[2].plot(
            HORIZONS,
            group.prediction.std(axis=0, ddof=1).mean(axis=0),
            label=group.label,
            color=color[group.label],
        )
    axes[0].axhline(0, color="black", lw=0.7); axes[0].set_title("Signed bias")
    axes[1].axhline(0, color="black", lw=0.7); axes[1].set_title("Skill vs persistence")
    axes[2].set_title("Inter-seed disagreement")
    for ax in axes: ax.set_xlabel("Lead time (h)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("MW"); axes[2].set_ylabel("MW"); axes[1].legend(fontsize=8)
    save_figure(fig, figures / "lead_bias_skill_disagreement")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for group in groups:
        absolute = np.sort(np.abs(group.ensemble - group.target).reshape(-1))
        probability = np.arange(1, len(absolute) + 1) / len(absolute)
        axes[0].plot(absolute, probability, label=group.label, color=color[group.label])
        survival = 1.0 - probability + 1.0 / len(probability)
        axes[1].semilogy(absolute, survival, label=group.label, color=color[group.label])
    axes[0].set(xlabel="Absolute error (MW)", ylabel="ECDF", xlim=(0, 1000), title="Absolute-error distribution")
    axes[1].set(xlabel="Absolute error (MW)", ylabel="Exceedance probability", xlim=(0, 1500), title="Extreme-error survival")
    for ax in axes: ax.grid(alpha=0.25); ax.legend(fontsize=8)
    save_figure(fig, figures / "absolute_error_ecdf_and_survival")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for group in groups:
        prediction = group.ensemble.reshape(-1)
        target = group.target.reshape(-1)
        bins = pd.qcut(prediction, 20, labels=False, duplicates="drop")
        x = np.asarray([prediction[bins == value].mean() for value in np.unique(bins)])
        y = np.asarray([target[bins == value].mean() for value in np.unique(bins)])
        axes[0].plot(x, y, marker="o", ms=3, label=group.label, color=color[group.label])
    limit = float(max(group.target.max() for group in groups))
    axes[0].plot([0, limit], [0, limit], "k--", lw=0.8)
    axes[0].set(xlabel="Mean prediction (MW)", ylabel="Mean observation (MW)", title="Deterministic reliability")
    error = candidate.ensemble - candidate.target
    axes[1].hexbin(candidate.ensemble.reshape(-1), error.reshape(-1), gridsize=60, bins="log", cmap="viridis")
    axes[1].axhline(0, color="white", lw=0.8)
    axes[1].set(xlabel=f"{candidate.label} prediction (MW)", ylabel="Signed error (MW)", title="Residual structure")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    save_figure(fig, figures / "reliability_and_residual_hexbin")

    for lead in (3, 11, 23):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        mesh = ax.hexbin(
            candidate.target[:, lead], candidate.ensemble[:, lead], gridsize=55, bins="log", cmap="magma"
        )
        limit = float(candidate.target[:, lead].max())
        ax.plot([0, limit], [0, limit], "w--", lw=0.9)
        ax.set(
            xlabel="Observed cluster power (MW)",
            ylabel=f"{candidate.label} prediction (MW)",
            title=f"Observed vs predicted at {HORIZONS[lead]:g} h",
        )
        fig.colorbar(mesh, ax=ax, label="log count")
        save_figure(fig, figures / f"observed_predicted_hexbin_lead_{HORIZONS[lead]:g}h")

    timestamp = pd.to_datetime(candidate.origin, unit="s", utc=True)
    daily_rows = []
    for day in np.unique(timestamp.date):
        mask = timestamp.date == day
        daily_rows.append(
            {
                "date": pd.Timestamp(day),
                candidate.label: np.abs(candidate.ensemble[mask] - candidate.target[mask]).mean(),
                reference.label: np.abs(reference.ensemble[mask] - reference.target[mask]).mean(),
            }
        )
    daily = pd.DataFrame(daily_rows).set_index("date")
    delta = daily[candidate.label] - daily[reference.label]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True)
    axes[0].plot(daily.index, daily[candidate.label].rolling(14, min_periods=1).mean(), label=candidate.label, color=color[candidate.label])
    axes[0].plot(daily.index, daily[reference.label].rolling(14, min_periods=1).mean(), label=reference.label, color=color[reference.label])
    axes[0].set_ylabel("14-day MAE (MW)"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].bar(daily.index, delta, width=1.0, color=np.where(delta < 0, "#059669", "#dc2626"))
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylabel("Candidate − reference (MW)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    save_figure(fig, figures / "daily_rolling_mae_and_delta")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].hist(delta, bins=32, color="#2563eb", alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set(xlabel="Daily candidate − reference MAE (MW)", ylabel="Days", title="Daily paired improvement")
    energy = {
        group.label: (group.ensemble - group.target).sum(axis=1) * 0.25 for group in groups
    }
    axes[1].boxplot([energy[group.label] for group in groups], tick_labels=[group.label for group in groups], showfliers=False)
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set(ylabel="6-h energy error (MWh)", title="Trajectory energy error")
    save_figure(fig, figures / "daily_delta_and_energy_error")

    history = cache["cluster_history_mw"]
    current = history[:, -1]
    current_ramp = history[:, -1] - history[:, -2]
    power_bin = pd.qcut(current, 10, labels=False, duplicates="drop")
    ramp_bin = pd.qcut(current_ramp, 10, labels=False, duplicates="drop")
    origin_delta = np.abs(candidate.ensemble - candidate.target).mean(axis=1) - np.abs(reference.ensemble - reference.target).mean(axis=1)
    matrix = np.full((10, 10), np.nan)
    counts = np.zeros((10, 10), dtype=int)
    for row in range(10):
        for column in range(10):
            mask = (ramp_bin == row) & (power_bin == column)
            if mask.any():
                matrix[row, column] = origin_delta[mask].mean(); counts[row, column] = mask.sum()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    scale = np.nanmax(np.abs(matrix))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set(xlabel="Current-power decile", ylabel="Recent-ramp decile", title="Conditional MAE delta (candidate − reference)")
    ax.set_xticks(range(10), range(1, 11)); ax.set_yticks(range(10), range(1, 11))
    fig.colorbar(image, ax=ax, label="MW")
    save_figure(fig, figures / "power_ramp_regime_delta_heatmap")
    atomic_csv(
        output / "tables/power_ramp_regime_delta.csv",
        [
            {"power_decile": column + 1, "ramp_decile": row + 1, "delta_mw": matrix[row, column], "origins": counts[row, column]}
            for row in range(10)
            for column in range(10)
        ],
    )

    months = timestamp.month.to_numpy()
    month_matrix = np.zeros((12, 24), dtype=np.float64)
    for month in range(1, 13):
        mask = months == month
        month_matrix[month - 1] = (
            np.abs(candidate.ensemble[mask] - candidate.target[mask]).mean(axis=0)
            - np.abs(reference.ensemble[mask] - reference.target[mask]).mean(axis=0)
        )
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    scale = np.max(np.abs(month_matrix))
    image = ax.imshow(month_matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set(xlabel="Lead step (15 min)", ylabel="Month", title="Monthly lead-wise MAE delta")
    ax.set_yticks(range(12), range(1, 13)); fig.colorbar(image, ax=ax, label="MW")
    save_figure(fig, figures / "month_lead_delta_heatmap")

    true_path = np.concatenate((candidate.history_last[:, None], candidate.target), axis=1)
    true_ramp = np.diff(true_path, axis=1).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    thresholds = np.quantile(np.abs(true_ramp), np.linspace(0.5, 0.99, 60))
    for group in groups:
        predicted_ramp = np.diff(
            np.concatenate((group.history_last[:, None], group.ensemble), axis=1), axis=1
        ).reshape(-1)
        precision, recall = [], []
        for threshold in thresholds:
            truth = np.abs(true_ramp) >= threshold
            predicted = np.abs(predicted_ramp) >= threshold
            tp = np.sum(truth & predicted)
            precision.append(tp / max(np.sum(predicted), 1))
            recall.append(tp / max(np.sum(truth), 1))
        axes[0].plot(recall, precision, label=group.label, color=color[group.label])
        axes[1].scatter(true_ramp[::100], predicted_ramp[::100], s=2, alpha=0.12, label=group.label, color=color[group.label])
    axes[0].set(xlabel="Ramp recall", ylabel="Ramp precision", title="Large-ramp detection curve")
    axes[1].set(xlabel="Observed 15-min ramp (MW)", ylabel="Predicted ramp (MW)", title="Ramp fidelity (1/100 points)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=0.25)
    save_figure(fig, figures / "ramp_precision_recall_and_scatter")

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for group in groups:
        values = []
        rng = np.random.default_rng(700 + len(group.label))
        diagnostic_origins = np.sort(
            rng.choice(
                group.target.shape[0],
                size=min(4096, group.target.shape[0]),
                replace=False,
            )
        )
        diagnostic_target = group.target[diagnostic_origins]
        for size in range(1, len(group.seeds) + 1):
            trials = min(50, math.comb(len(group.seeds), size))
            scores = []
            for _ in range(trials):
                subset = rng.choice(len(group.seeds), size=size, replace=False)
                prediction = group.prediction[subset][:, diagnostic_origins].mean(axis=0)
                scores.append(np.abs(prediction - diagnostic_target).mean())
            values.append((np.mean(scores), np.quantile(scores, 0.1), np.quantile(scores, 0.9)))
        values = np.asarray(values)
        ax.plot(range(1, len(group.seeds) + 1), values[:, 0], marker="o", label=group.label, color=color[group.label])
        ax.fill_between(range(1, len(group.seeds) + 1), values[:, 1], values[:, 2], color=color[group.label], alpha=0.12)
    ax.set(xlabel="Ensemble members", ylabel="Cluster MAE (MW)", title="Ensemble-size saturation (fixed 4,096-origin diagnostic)")
    ax.grid(alpha=0.25); ax.legend()
    save_figure(fig, figures / "ensemble_size_saturation")

    if "baseline_cluster_prediction_mw" in candidate.optional:
        baseline = candidate.optional["baseline_cluster_prediction_mw"].mean(axis=0)
        correction = candidate.ensemble - baseline
        baseline_mae = np.abs(baseline - candidate.target).mean(axis=0)
        final_mae = np.abs(candidate.ensemble - candidate.target).mean(axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
        axes[0].plot(HORIZONS, baseline_mae, label="Baseline", color="#6b7280")
        axes[0].plot(HORIZONS, final_mae, label="Baseline + correction", color=color[candidate.label])
        axes[0].set(xlabel="Lead time (h)", ylabel="MAE (MW)", title="Correction benefit by horizon")
        axes[1].plot(HORIZONS, np.mean(np.abs(correction), axis=0), color=color[candidate.label])
        axes[1].set(xlabel="Lead time (h)", ylabel="Mean |correction| (MW)", title="Realized correction magnitude")
        for ax in axes: ax.grid(alpha=0.25); ax.legend() if ax is axes[0] else None
        save_figure(fig, figures / "baseline_correction_horizon")


def case_indices(
    predictions: dict[str, np.ndarray],
    candidate: str,
    reference: str,
    target: np.ndarray,
    history: np.ndarray,
    origin: np.ndarray,
    count: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    candidate_error = np.abs(predictions[candidate] - target).mean(axis=1)
    reference_error = np.abs(predictions[reference] - target).mean(axis=1)
    improvement = reference_error - candidate_error
    target_path = np.concatenate((history[:, -1:], target), axis=1)
    max_ramp = np.abs(np.diff(target_path, axis=1)).max(axis=1)
    current_ramp = np.abs(history[:, -1] - history[:, -2])
    rng = np.random.default_rng(random_seed)
    result = {
        "largest_improvement": np.argsort(improvement)[-count:][::-1],
        "largest_degradation": np.argsort(improvement)[:count],
        "largest_future_ramp": np.argsort(max_ramp)[-count:][::-1],
        "largest_current_ramp": np.argsort(current_ramp)[-count:][::-1],
        "fixed_random": np.sort(rng.choice(len(target), size=min(count, len(target)), replace=False)),
    }
    month = pd.to_datetime(origin, unit="s", utc=True).month.to_numpy()
    representatives = []
    for value in range(1, 13):
        candidates = np.flatnonzero(month == value)
        order = np.argsort(candidate_error[candidates])
        representatives.append(candidates[order[len(order) // 2]])
    result["monthly_median_error"] = np.asarray(representatives)
    return result


def plot_cases(
    groups: list[Group],
    candidate: Group,
    reference: Group,
    cache: dict[str, np.ndarray],
    output: Path,
    count: int,
    fixed_seed: int,
) -> list[dict[str, Any]]:
    history = cache["cluster_history_mw"]
    time_history = np.arange(-23, 1) / 4.0
    time_future = np.arange(1, 25) / 4.0
    colors = {group.label: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
    records: list[dict[str, Any]] = []
    for representation in ("ensemble", f"seed{fixed_seed}"):
        predictions: dict[str, np.ndarray] = {}
        for group in groups:
            if representation == "ensemble":
                predictions[group.label] = group.ensemble
            else:
                if fixed_seed not in group.seeds:
                    raise ValueError(f"{group.label} does not contain fixed seed {fixed_seed}")
                predictions[group.label] = group.prediction[group.seeds.index(fixed_seed)]
        categories = case_indices(
            predictions,
            candidate.label,
            reference.label,
            candidate.target,
            history,
            candidate.origin,
            count,
            20260831,
        )
        for category, indices in categories.items():
            for rank, index in enumerate(indices, start=1):
                timestamp = pd.to_datetime(candidate.origin[index], unit="s", utc=True)
                fig, ax = plt.subplots(figsize=(7.5, 4.2))
                ax.plot(time_history, history[index], color="black", lw=1.5, label="Observed history")
                ax.plot(time_future, candidate.target[index], color="black", lw=2.2, label="Observed future")
                ax.plot([0, 0.25], [history[index, -1], candidate.target[index, 0]], color="black", lw=2.2)
                for group in groups:
                    ax.plot(time_future, predictions[group.label][index], color=colors[group.label], lw=1.5, label=group.label)
                ax.axvline(0, color="#6b7280", ls="--", lw=0.8)
                ax.set(
                    xlabel="Hours relative to forecast origin",
                    ylabel="Cluster power (MW)",
                    title=f"{category} #{rank} | {timestamp:%Y-%m-%d %H:%M} UTC | {representation}",
                )
                ax.grid(alpha=0.2); ax.legend(ncol=2, fontsize=8)
                filename = f"{rank:02d}_{timestamp:%Y%m%dT%H%M}_{index:05d}"
                destination = output / "figures/cases" / representation / category / filename
                save_figure(fig, destination)
                candidate_mae = float(np.abs(predictions[candidate.label][index] - candidate.target[index]).mean())
                reference_mae = float(np.abs(predictions[reference.label][index] - candidate.target[index]).mean())
                records.append(
                    {
                        "representation": representation,
                        "category": category,
                        "rank": rank,
                        "row_index": int(index),
                        "origin_unix": int(candidate.origin[index]),
                        "origin_utc": timestamp.isoformat(),
                        "candidate_mae_mw": candidate_mae,
                        "reference_mae_mw": reference_mae,
                        "reference_minus_candidate_mw": reference_mae - candidate_mae,
                        "png": str(destination.with_suffix(".png").resolve()),
                    }
                )
    atomic_csv(output / "tables/case_manifest.csv", records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-headline forecast evidence and many traceable figures")
    parser.add_argument("--group", action="append", default=[], help="LABEL=GROUP_ROOT; repeat for each model")
    parser.add_argument("--candidate", default="KD")
    parser.add_argument("--reference", default="Direct")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence_test_v1/cluster")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    specs = args.group or list(DEFAULT_GROUPS)
    groups = [load_group(label, path, seeds, args.split) for label, path in map(parse_group, specs)]
    by_label = {group.label: group for group in groups}
    if args.candidate not in by_label or args.reference not in by_label:
        raise ValueError("Candidate and reference labels must be supplied as groups")
    first = groups[0]
    for group in groups[1:]:
        np.testing.assert_array_equal(first.origin, group.origin)
        np.testing.assert_array_equal(first.sample_index, group.sample_index)
        np.testing.assert_allclose(first.target, group.target, rtol=0.0, atol=1e-4)
    output = Path(args.output_dir).resolve()
    cache = align_cache(Path(args.cache_dir).resolve(), first.sample_index, args.split)
    thresholds = training_ramp_thresholds(Path(args.cache_dir).resolve(), float(cache["total_capacity_mw"]))
    metric_rows, lead_rows = make_summary_tables(groups, cache, thresholds["p90_mw_per_15min"], output)
    regimes: list[dict[str, Any]] = []
    for group in groups:
        regimes.extend(regime_rows(group, cache, output))
    atomic_csv(output / "tables/regime_metrics.csv", regimes)
    candidate, reference = by_label[args.candidate], by_label[args.reference]
    comparison = bootstrap_daily_delta(
        candidate.ensemble,
        reference.ensemble,
        candidate.target,
        candidate.origin,
        args.bootstrap_repeats,
        20260831,
    )
    aggregate_figures = 13 + int("baseline_cluster_prediction_mw" in candidate.optional)
    case_figures = 2 * (5 * min(args.case_count, len(candidate.origin)) + 12)
    global _FIGURE_PROGRESS
    with tqdm(total=aggregate_figures + case_figures, **progress_options("render figures", unit="figure")) as progress:
        _FIGURE_PROGRESS = progress
        plot_core(groups, cache, candidate, reference, output)
        case_records = plot_cases(groups, candidate, reference, cache, output, args.case_count, args.fixed_seed)
        _FIGURE_PROGRESS = None
    payload = {
        "protocol": {
            "split": args.split,
            "groups": {group.label: str(group.root) for group in groups},
            "seeds": seeds,
            "candidate": args.candidate,
            "reference": args.reference,
            "fixed_single_model_seed": args.fixed_seed,
            "case_selection": (
                "objective predefined categories; use figures for readability, not selective performance claims"
            ),
            "ramp_thresholds_derived_from_training_split": thresholds,
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "ensemble_comparison": comparison,
        "figure_png_count": len(list((output / "figures").rglob("*.png"))),
        "case_png_count": len(case_records),
        "extended_metric_rows": len(metric_rows),
        "lead_rows": len(lead_rows),
    }
    atomic_json(output / "evidence_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
