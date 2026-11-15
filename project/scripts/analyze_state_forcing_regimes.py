from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_incremental_mechanism_v2 import group_predictions
except ModuleNotFoundError:
    from analyze_incremental_mechanism_v2 import group_predictions  # type: ignore[no-redef]


SEEDS = list(range(42, 52))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def quantile_bins(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1))
    if len(np.unique(edges)) != len(edges):
        raise ValueError("Quantile edges are not unique")
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return bins.astype(np.int64), edges


def utc_day_bootstrap_counts(
    origins: np.ndarray, repeats: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days, inverse = np.unique(np.asarray(origins, dtype=np.int64) // 86400, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(days), np.full(len(days), 1.0 / len(days)), size=repeats
    )
    return days, inverse, counts


def grouped_bootstrap(
    values: np.ndarray,
    mask: np.ndarray,
    day_inverse: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    selected = np.asarray(mask, dtype=bool)
    if selected.sum() == 0:
        raise ValueError("Cannot bootstrap an empty group")
    day_count = bootstrap_counts.shape[1]
    day_sums = np.zeros((day_count, matrix.shape[1]), dtype=np.float64)
    day_samples = np.zeros(day_count, dtype=np.int64)
    np.add.at(day_sums, day_inverse[selected], matrix[selected])
    np.add.at(day_samples, day_inverse[selected], 1)
    denominator = bootstrap_counts @ day_samples
    valid = denominator > 0
    if valid.sum() < int(0.95 * len(valid)):
        raise RuntimeError("Fewer than 95% of bootstrap samples contain the group")
    samples = (bootstrap_counts[valid] @ day_sums) / denominator[valid, None]
    return (
        matrix[selected].mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
        int(selected.sum()),
        int(np.sum(day_samples > 0)),
        int(valid.sum()),
    )


def grouped_contrast_bootstrap(
    values: np.ndarray,
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
    day_inverse: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]

    def sufficient(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(mask, dtype=bool)
        sums = np.zeros((bootstrap_counts.shape[1], matrix.shape[1]), dtype=np.float64)
        counts = np.zeros(bootstrap_counts.shape[1], dtype=np.int64)
        np.add.at(sums, day_inverse[selected], matrix[selected])
        np.add.at(counts, day_inverse[selected], 1)
        return sums, counts

    candidate_sums, candidate_counts = sufficient(candidate_mask)
    reference_sums, reference_counts = sufficient(reference_mask)
    candidate_denominator = bootstrap_counts @ candidate_counts
    reference_denominator = bootstrap_counts @ reference_counts
    valid = (candidate_denominator > 0) & (reference_denominator > 0)
    if valid.sum() < int(0.95 * len(valid)):
        raise RuntimeError("Fewer than 95% of bootstrap samples contain both groups")
    candidate_samples = (bootstrap_counts[valid] @ candidate_sums) / candidate_denominator[
        valid, None
    ]
    reference_samples = (bootstrap_counts[valid] @ reference_sums) / reference_denominator[
        valid, None
    ]
    difference = candidate_samples - reference_samples
    point = matrix[candidate_mask].mean(axis=0) - matrix[reference_mask].mean(axis=0)
    return (
        point,
        np.quantile(difference, 0.025, axis=0),
        np.quantile(difference, 0.975, axis=0),
        int(valid.sum()),
    )


def physical_wind_features(cache: Path) -> dict[str, np.ndarray]:
    split = cache / "val"
    weather = np.load(split / "weather.npy", mmap_mode="r")
    names = np.load(cache / "static" / "nwp_channel_name.npy").tolist()
    mean = np.load(cache / "static" / "nwp_mean.npy").astype(np.float64)
    std = np.load(cache / "static" / "nwp_std.npy").astype(np.float64)
    pairs = (("u10", "v10"), ("u1000", "v1000"), ("u925", "v925"))
    indices = [(names.index(u), names.index(v)) for u, v in pairs]
    cycle_count, nodes = weather.shape[:2]
    regional_speed = np.empty((cycle_count, nodes), dtype=np.float32)
    spatial_speed_std = np.empty((cycle_count, nodes), dtype=np.float32)
    regional_u = np.empty((cycle_count, nodes), dtype=np.float32)
    regional_v = np.empty((cycle_count, nodes), dtype=np.float32)
    for start in range(0, cycle_count, 128):
        stop = min(start + 128, cycle_count)
        speed_sum = np.zeros((stop - start, nodes, *weather.shape[-2:]), dtype=np.float32)
        u_sum = np.zeros_like(speed_sum)
        v_sum = np.zeros_like(speed_sum)
        for u_index, v_index in indices:
            u = np.asarray(weather[start:stop, :, u_index], dtype=np.float32)
            v = np.asarray(weather[start:stop, :, v_index], dtype=np.float32)
            u = u * np.float32(std[u_index]) + np.float32(mean[u_index])
            v = v * np.float32(std[v_index]) + np.float32(mean[v_index])
            speed_sum += np.sqrt(np.square(u) + np.square(v))
            u_sum += u
            v_sum += v
        speed = speed_sum / len(indices)
        regional_speed[start:stop] = speed.mean(axis=(-2, -1))
        spatial_speed_std[start:stop] = speed.std(axis=(-2, -1))
        regional_u[start:stop] = (u_sum / len(indices)).mean(axis=(-2, -1))
        regional_v[start:stop] = (v_sum / len(indices)).mean(axis=(-2, -1))

    cycle_index = np.load(split / "weather_cycle_index.npy").astype(np.int64)
    offsets = np.load(split / "weather_time_offset_hours.npy").astype(np.float64)
    first = np.zeros(len(cycle_index), dtype=np.int64)
    near_six = np.argmin(np.abs(offsets - 6.0), axis=1)
    row = np.arange(len(cycle_index))
    speed = regional_speed[cycle_index]
    u = regional_u[cycle_index]
    v = regional_v[cycle_index]
    direction_first = np.arctan2(v[row, first], u[row, first])
    direction_six = np.arctan2(v[row, near_six], u[row, near_six])
    direction_delta = np.arctan2(
        np.sin(direction_six - direction_first),
        np.cos(direction_six - direction_first),
    )
    return {
        "wind_trend_6h_ms": speed[row, near_six] - speed[row, first],
        "wind_temporal_std_ms": speed[:, :7].std(axis=1),
        "wind_spatial_std_ms": spatial_speed_std[cycle_index, :7].mean(axis=1),
        "wind_direction_change_abs_rad": np.abs(direction_delta),
    }


def ensemble_components(group: Path, seeds: list[int]) -> dict[str, np.ndarray]:
    archives = [load_npz(group / f"seed{seed}" / "predictions_val.npz") for seed in seeds]
    origins = archives[0]["origin_unix"]
    target = archives[0]["cluster_target_mw"].astype(np.float64)
    for archive in archives[1:]:
        np.testing.assert_array_equal(archive["origin_unix"], origins)
        np.testing.assert_allclose(archive["cluster_target_mw"], target)
    return {
        "origin_unix": origins,
        "target": target,
        "baseline": np.mean(
            [archive["baseline_cluster_prediction_mw"] for archive in archives], axis=0
        ).astype(np.float64),
        "correction": np.mean(
            [archive["realized_cluster_correction_mw"] for archive in archives], axis=0
        ).astype(np.float64),
        "prediction": np.mean(
            [archive["cluster_prediction_mw"] for archive in archives], axis=0
        ).astype(np.float64),
    }


def metric_rows_for_bins(
    *,
    analysis: str,
    feature: str,
    values: np.ndarray,
    bin_count: int,
    metrics: dict[str, np.ndarray],
    day_inverse: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    bins, edges = quantile_bins(values, bin_count)
    rows = []
    for bin_index in range(bin_count):
        mask = bins == bin_index
        row: dict[str, Any] = {
            "analysis": analysis,
            "feature": feature,
            "bin": bin_index + 1,
            "bin_count": bin_count,
            "lower_edge": float(edges[bin_index]),
            "upper_edge": float(edges[bin_index + 1]),
            "samples": int(mask.sum()),
        }
        for label, matrix in metrics.items():
            point, low, high, _, days, effective_repeats = grouped_bootstrap(
                matrix, mask, day_inverse, bootstrap_counts
            )
            if len(point) != 1:
                raise ValueError("Marginal metrics must be scalar per sample")
            row[f"{label}_mean_mw"] = float(point[0])
            row[f"{label}_ci95_low_mw"] = float(low[0])
            row[f"{label}_ci95_high_mw"] = float(high[0])
            row["utc_days"] = days
            row["effective_bootstrap_repeats"] = effective_repeats
        rows.append(row)
    return rows, bins, edges


def horizon_rows_for_bins(
    *,
    feature: str,
    values: np.ndarray,
    bin_count: int,
    gain: np.ndarray,
    day_inverse: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> list[dict[str, Any]]:
    bins, edges = quantile_bins(values, bin_count)
    rows = []
    for bin_index in range(bin_count):
        point, low, high, count, days, effective_repeats = grouped_bootstrap(
            gain, bins == bin_index, day_inverse, bootstrap_counts
        )
        for lead in range(gain.shape[1]):
            rows.append(
                {
                    "feature": feature,
                    "bin": bin_index + 1,
                    "lower_edge": float(edges[bin_index]),
                    "upper_edge": float(edges[bin_index + 1]),
                    "samples": count,
                    "utc_days": days,
                    "effective_bootstrap_repeats": effective_repeats,
                    "lead": lead + 1,
                    "lead_hours": (lead + 1) / 4.0,
                    "conditioning_gain_mw": float(point[lead]),
                    "ci95_low_mw": float(low[lead]),
                    "ci95_high_mw": float(high[lead]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze state-forcing regimes")
    parser.add_argument(
        "--conditional-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--weather-group")
    parser.add_argument("--full-group")
    parser.add_argument("--component-group")
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--output-dir", default="artifacts/state_forcing_diagnostics_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.conditional_root).resolve()
    cache = Path(args.cache_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    weather_group = (
        Path(args.weather_group).resolve()
        if args.weather_group
        else root / "groups" / "factor_weather_only"
    )
    full_group = (
        Path(args.full_group).resolve()
        if args.full_group
        else root / "groups" / "factor_history_weather"
    )
    component_group = (
        Path(args.component_group).resolve()
        if args.component_group
        else root / "groups" / "anchor_020"
    )

    _, weather_ensemble = group_predictions(weather_group, SEEDS)
    _, full_ensemble = group_predictions(full_group, SEEDS)
    origins = full_ensemble["origin_unix"].astype(np.int64)
    np.testing.assert_array_equal(weather_ensemble["origin_unix"], origins)
    target = full_ensemble["cluster_target_mw"].astype(np.float64)
    np.testing.assert_allclose(weather_ensemble["cluster_target_mw"], target)
    gain = (
        np.abs(weather_ensemble["cluster_prediction_mw"].astype(np.float64) - target)
        - np.abs(full_ensemble["cluster_prediction_mw"].astype(np.float64) - target)
    )

    component = ensemble_components(component_group, SEEDS)
    np.testing.assert_array_equal(component["origin_unix"], origins)
    usefulness = np.abs(target - component["baseline"]) - np.abs(
        target - component["prediction"]
    )
    correction = component["correction"]

    split = cache / "val"
    cache_origins = np.load(split / "origin_unix.npy").astype(np.int64)
    np.testing.assert_array_equal(cache_origins, origins)
    cluster_history = np.load(split / "cluster_history.npy").astype(np.float64)
    station_history = np.load(split / "station_history.npy", mmap_mode="r")
    state = {
        "current_power_factor": cluster_history[:, -1],
        "ramp_1h_factor": cluster_history[:, -1] - cluster_history[:, -5],
        "recent_1h_volatility": cluster_history[:, -5:].std(axis=1),
        "station_current_dispersion": np.asarray(
            station_history[:, :, -1], dtype=np.float64
        ).std(axis=1),
    }
    weather = physical_wind_features(cache)
    features = {**state, **weather}

    _, day_inverse, bootstrap_counts = utc_day_bootstrap_counts(
        origins, args.bootstrap_repeats, 8100
    )
    horizon_slices = {
        "gain_all": slice(0, 24),
        "gain_0_1h": slice(0, 4),
        "gain_1_3h": slice(4, 12),
        "gain_3_6h": slice(12, 24),
    }
    scalar_metrics = {
        label: gain[:, lead_slice].mean(axis=1)
        for label, lead_slice in horizon_slices.items()
    }
    scalar_metrics["component_usefulness_all"] = usefulness.mean(axis=1)

    marginal_rows = []
    contrast_rows = []
    feature_bins: dict[str, np.ndarray] = {}
    feature_edges: dict[str, np.ndarray] = {}
    for feature, values in features.items():
        count = 10 if feature == "current_power_factor" else 5
        rows, bins, edges = metric_rows_for_bins(
            analysis="marginal",
            feature=feature,
            values=values,
            bin_count=count,
            metrics=scalar_metrics,
            day_inverse=day_inverse,
            bootstrap_counts=bootstrap_counts,
        )
        marginal_rows.extend(rows)
        feature_bins[feature] = bins
        feature_edges[feature] = edges
        middle = count // 2
        for contrast, candidate_bin, reference_bin in (
            ("upper_minus_lower", count - 1, 0),
            ("lower_minus_middle", 0, middle),
            ("upper_minus_middle", count - 1, middle),
        ):
            for label, values_matrix in scalar_metrics.items():
                point, low, high, effective = grouped_contrast_bootstrap(
                    values_matrix,
                    bins == candidate_bin,
                    bins == reference_bin,
                    day_inverse,
                    bootstrap_counts,
                )
                contrast_rows.append(
                    {
                        "analysis": "marginal_quantile_contrast",
                        "feature": feature,
                        "contrast": contrast,
                        "candidate_bin": candidate_bin + 1,
                        "reference_bin": reference_bin + 1,
                        "metric": label,
                        "candidate_minus_reference_mw": float(point[0]),
                        "ci95_low_mw": float(low[0]),
                        "ci95_high_mw": float(high[0]),
                        "effective_bootstrap_repeats": effective,
                    }
                )

    horizon_rows = []
    for feature in (
        "current_power_factor",
        "ramp_1h_factor",
        "station_current_dispersion",
        "wind_trend_6h_ms",
    ):
        horizon_rows.extend(
            horizon_rows_for_bins(
                feature=feature,
                values=features[feature],
                bin_count=5,
                gain=gain,
                day_inverse=day_inverse,
                bootstrap_counts=bootstrap_counts,
            )
        )

    heatmap_rows = []
    heatmaps = (
        ("power_x_wind_trend", "current_power_factor", "wind_trend_6h_ms"),
        ("ramp_x_wind_trend", "ramp_1h_factor", "wind_trend_6h_ms"),
        (
            "spatial_dispersion_x_wind_trend",
            "station_current_dispersion",
            "wind_trend_6h_ms",
        ),
        (
            "power_x_spatial_dispersion",
            "current_power_factor",
            "station_current_dispersion",
        ),
    )
    quintile_cache = {
        feature: quantile_bins(values, 5)
        for feature, values in features.items()
    }
    for analysis, row_feature, column_feature in heatmaps:
        row_bin, row_edges = quintile_cache[row_feature]
        if analysis == "power_x_spatial_dispersion":
            column_bin = np.empty(len(origins), dtype=np.int64)
            conditional_column_edges = []
            for row_index in range(5):
                row_mask = row_bin == row_index
                local_bin, local_edges = quantile_bins(
                    features[column_feature][row_mask], 5
                )
                column_bin[row_mask] = local_bin
                conditional_column_edges.append(local_edges)
        else:
            column_bin, column_edges = quintile_cache[column_feature]
            conditional_column_edges = [column_edges] * 5
        for row_index in range(5):
            for column_index in range(5):
                mask = (row_bin == row_index) & (column_bin == column_index)
                point, low, high, count, days, effective_repeats = grouped_bootstrap(
                    np.stack(
                        (
                            gain.mean(axis=1),
                            gain[:, 12:].mean(axis=1),
                            usefulness.mean(axis=1),
                        ),
                        axis=1,
                    ),
                    mask,
                    day_inverse,
                    bootstrap_counts,
                )
                heatmap_rows.append(
                    {
                        "analysis": analysis,
                        "row_feature": row_feature,
                        "row_bin": row_index + 1,
                        "row_lower_edge": float(row_edges[row_index]),
                        "row_upper_edge": float(row_edges[row_index + 1]),
                        "column_feature": column_feature,
                        "column_bin": column_index + 1,
                        "column_lower_edge": float(
                            conditional_column_edges[row_index][column_index]
                        ),
                        "column_upper_edge": float(
                            conditional_column_edges[row_index][column_index + 1]
                        ),
                        "samples": count,
                        "utc_days": days,
                        "effective_bootstrap_repeats": effective_repeats,
                        "conditioning_gain_all_mw": float(point[0]),
                        "conditioning_gain_all_ci95_low_mw": float(low[0]),
                        "conditioning_gain_all_ci95_high_mw": float(high[0]),
                        "conditioning_gain_3_6h_mw": float(point[1]),
                        "conditioning_gain_3_6h_ci95_low_mw": float(low[1]),
                        "conditioning_gain_3_6h_ci95_high_mw": float(high[1]),
                        "component_usefulness_all_mw": float(point[2]),
                    }
                )

    correction_point, correction_low, correction_high, _, _, _ = grouped_bootstrap(
        np.concatenate(
            (
                np.abs(correction),
                (correction > 0.0).astype(np.float64),
                usefulness,
            ),
            axis=1,
        ),
        np.ones(len(origins), dtype=bool),
        day_inverse,
        bootstrap_counts,
    )
    correction_rows = []
    for lead in range(24):
        correction_rows.append(
            {
                "lead": lead + 1,
                "lead_hours": (lead + 1) / 4.0,
                "correction_abs_mean_mw": float(correction_point[lead]),
                "correction_abs_ci95_low_mw": float(correction_low[lead]),
                "correction_abs_ci95_high_mw": float(correction_high[lead]),
                "correction_positive_probability": float(correction_point[24 + lead]),
                "correction_positive_ci95_low": float(correction_low[24 + lead]),
                "correction_positive_ci95_high": float(correction_high[24 + lead]),
                "correction_usefulness_mw": float(correction_point[48 + lead]),
                "correction_usefulness_ci95_low_mw": float(correction_low[48 + lead]),
                "correction_usefulness_ci95_high_mw": float(correction_high[48 + lead]),
            }
        )

    trend_bins, trend_edges = quantile_bins(weather["wind_trend_6h_ms"], 5)
    correction_regime_rows = []
    for bin_index in range(5):
        mask = trend_bins == bin_index
        regime_values = np.stack(
            (
                correction.mean(axis=1),
                (correction > 0.0).mean(axis=1),
                np.abs(correction).mean(axis=1),
                usefulness.mean(axis=1),
            ),
            axis=1,
        )
        point, low, high, count, days, effective_repeats = grouped_bootstrap(
            regime_values, mask, day_inverse, bootstrap_counts
        )
        correction_regime_rows.append(
            {
                "wind_trend_bin": bin_index + 1,
                "lower_edge_ms": float(trend_edges[bin_index]),
                "upper_edge_ms": float(trend_edges[bin_index + 1]),
                "samples": count,
                "utc_days": days,
                "effective_bootstrap_repeats": effective_repeats,
                "correction_signed_mean_mw": float(point[0]),
                "correction_signed_ci95_low_mw": float(low[0]),
                "correction_signed_ci95_high_mw": float(high[0]),
                "correction_positive_probability": float(point[1]),
                "correction_positive_ci95_low": float(low[1]),
                "correction_positive_ci95_high": float(high[1]),
                "correction_abs_mean_mw": float(point[2]),
                "correction_usefulness_mean_mw": float(point[3]),
                "correction_usefulness_ci95_low_mw": float(low[3]),
                "correction_usefulness_ci95_high_mw": float(high[3]),
            }
        )

    correlations = {}
    gain_all = gain.mean(axis=1)
    gain_long = gain[:, 12:].mean(axis=1)
    for feature, values in features.items():
        correlations[feature] = {
            "conditioning_gain_all_pearson": float(np.corrcoef(values, gain_all)[0, 1]),
            "conditioning_gain_3_6h_pearson": float(np.corrcoef(values, gain_long)[0, 1]),
            "component_usefulness_all_pearson": float(
                np.corrcoef(values, usefulness.mean(axis=1))[0, 1]
            ),
        }
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "model_seeds": SEEDS,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
            "origins": len(origins),
            "utc_day_blocks": bootstrap_counts.shape[1],
            "weather_wind_definition": (
                "physical-unit mean of 10m, 1000hPa, and 925hPa wind speed "
                "over the cropped regional grid"
            ),
            "weather_group": str(weather_group),
            "full_group": str(full_group),
            "component_group": str(component_group),
        },
        "overall": {
            "conditioning_gain_ensemble_mw": float(gain.mean()),
            "conditioning_gain_0_1h_mw": float(gain[:, :4].mean()),
            "conditioning_gain_1_3h_mw": float(gain[:, 4:12].mean()),
            "conditioning_gain_3_6h_mw": float(gain[:, 12:].mean()),
            "component_correction_abs_mean_mw": float(np.abs(correction).mean()),
            "component_correction_usefulness_mw": float(usefulness.mean()),
            "component_correction_positive_probability": float((correction > 0.0).mean()),
            "correction_abs_horizon_pearson": float(
                np.corrcoef(np.arange(1, 25), np.abs(correction).mean(axis=0))[0, 1]
            ),
            "correction_usefulness_horizon_pearson": float(
                np.corrcoef(np.arange(1, 25), usefulness.mean(axis=0))[0, 1]
            ),
        },
        "feature_edges": {
            feature: edges.tolist() for feature, edges in feature_edges.items()
        },
        "correlations": correlations,
    }
    write_csv(output / "state_weather_marginals.csv", marginal_rows)
    write_csv(output / "state_weather_contrasts.csv", contrast_rows)
    write_csv(output / "horizon_state.csv", horizon_rows)
    write_csv(output / "state_weather_heatmaps.csv", heatmap_rows)
    write_csv(output / "correction_horizon.csv", correction_rows)
    write_csv(output / "correction_wind_regimes.csv", correction_regime_rows)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps({"output": str(output), "rows": {
        "marginals": len(marginal_rows),
        "contrasts": len(contrast_rows),
        "horizon_state": len(horizon_rows),
        "heatmaps": len(heatmap_rows),
        "correction_horizon": len(correction_rows),
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
