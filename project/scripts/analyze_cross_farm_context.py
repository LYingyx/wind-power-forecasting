from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import yaml

try:
    from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix
    from scripts.analyze_incremental_mechanism_v2 import (
        compare_groups,
        group_predictions,
        summarize_group,
    )
except ModuleNotFoundError:
    from analyze_conditional_forcing_validation import block_bootstrap_matrix  # type: ignore[no-redef]
    from analyze_incremental_mechanism_v2 import (  # type: ignore[no-redef]
        compare_groups,
        group_predictions,
        summarize_group,
    )


SEEDS = list(range(42, 52))
GROUPS = {
    "G1-self-duplicate": "g1_self_duplicate",
    "G2-geographic-K4": "g2_geographic_k4",
    "G3-global-other": "g3_global_other",
    "G4-weather-coherent-K4": "g4_weather_coherent_k4",
}
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def horizon_bands(
    candidate: Path,
    reference: Path,
    comparison: str,
    repeats: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    _, candidate_prediction = group_predictions(candidate, SEEDS)
    _, reference_prediction = group_predictions(reference, SEEDS)
    np.testing.assert_array_equal(
        candidate_prediction["origin_unix"], reference_prediction["origin_unix"]
    )
    np.testing.assert_allclose(
        candidate_prediction["cluster_target_mw"],
        reference_prediction["cluster_target_mw"],
    )
    candidate_error = np.abs(
        candidate_prediction["cluster_prediction_mw"]
        - candidate_prediction["cluster_target_mw"]
    ).astype(np.float64)
    reference_error = np.abs(
        reference_prediction["cluster_prediction_mw"]
        - reference_prediction["cluster_target_mw"]
    ).astype(np.float64)
    delta = candidate_error - reference_error
    rows = []
    for index, (label, selection) in enumerate(HORIZON_BANDS):
        point, low, high, days = block_bootstrap_matrix(
            delta[:, selection].mean(axis=1, keepdims=True),
            candidate_prediction["origin_unix"],
            repeats,
            bootstrap_seed + index,
        )
        rows.append(
            {
                "comparison": comparison,
                "horizon_band": label,
                "candidate_minus_reference_mw": float(point[0]),
                "ci95_low_mw": float(low[0]),
                "ci95_high_mw": float(high[0]),
                "utc_day_blocks": days,
            }
        )
    return rows


def two_sided_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    statistic = array.mean() / max(array.std(ddof=1) / np.sqrt(len(array)), 1e-12)
    return float(2.0 * student_t.sf(abs(statistic), len(array) - 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * values[index])
        running = max(running, current)
        adjusted[index] = running
    return adjusted.tolist()


def flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def audit_configs(groups_root: Path) -> dict[str, Any]:
    allowed = {"model.cross_farm_context_mode", "experiment.id"}
    unexpected: dict[str, list[str]] = {}
    for seed in SEEDS:
        reference = flatten(
            yaml.safe_load(
                (
                    groups_root
                    / GROUPS["G1-self-duplicate"]
                    / f"seed{seed}"
                    / "config.yaml"
                ).read_text(encoding="utf-8")
            )
        )
        for label, group in GROUPS.items():
            if label == "G1-self-duplicate":
                continue
            candidate = flatten(
                yaml.safe_load(
                    (groups_root / group / f"seed{seed}" / "config.yaml").read_text(
                        encoding="utf-8"
                    )
                )
            )
            differences = sorted(
                key
                for key in reference.keys() | candidate.keys()
                if reference.get(key) != candidate.get(key) and key not in allowed
            )
            if differences:
                unexpected[f"{label}/seed{seed}"] = differences
    return {
        "allowed_differences": sorted(allowed),
        "unexpected_differences": unexpected,
        "differ_only_by_context_mode_and_experiment_id": not unexpected,
    }


def input_context_diagnostics(cache_dir: Path) -> dict[str, Any]:
    station = np.load(cache_dir / "val/station_history.npy", mmap_mode="r")[
        :, :, -4:
    ].astype(np.float64)
    cluster = np.load(cache_dir / "val/cluster_history.npy", mmap_mode="r")[
        :, -4:
    ].astype(np.float64)
    latitude = np.load(cache_dir / "static/station_latitude.npy").astype(np.float64)
    longitude = np.load(cache_dir / "static/station_longitude.npy").astype(np.float64)
    dx = (
        (longitude[:, None] - longitude[None, :])
        * np.cos(np.deg2rad(latitude.mean()))
        * 111.32
    )
    dy = (latitude[:, None] - latitude[None, :]) * 111.32
    distance = np.hypot(dx, dy)
    np.fill_diagonal(distance, np.inf)
    neighbor_index = np.argsort(distance, axis=1)[:, :4]
    neighbor_distance = np.take_along_axis(distance, neighbor_index, axis=1)
    geographic = station[:, neighbor_index, :].mean(axis=2)
    global_other = (
        station.sum(axis=1, keepdims=True) - station
    ) / float(station.shape[1] - 1)
    aggregate = np.broadcast_to(cluster[:, None, :], station.shape)

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.corrcoef(first.reshape(-1), second.reshape(-1))[0, 1])

    def conditional_r2(target: np.ndarray) -> float:
        y = target.reshape(-1)
        x1 = station.reshape(-1)
        x2 = aggregate.reshape(-1)
        y = y - y.mean()
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        covariance = np.asarray(
            [[x1 @ x1, x1 @ x2], [x1 @ x2, x2 @ x2]], dtype=np.float64
        )
        cross = np.asarray([x1 @ y, x2 @ y], dtype=np.float64)
        coefficients = np.linalg.solve(covariance, cross)
        explained = float(cross @ coefficients)
        return explained / float(y @ y)

    return {
        "geographic_k4_distance_km": {
            "mean": float(neighbor_distance.mean()),
            "median": float(np.median(neighbor_distance)),
            "p95": float(np.quantile(neighbor_distance, 0.95)),
            "max": float(neighbor_distance.max()),
        },
        "correlations": {
            "self_vs_geographic_k4": correlation(station, geographic),
            "self_vs_global_other": correlation(station, global_other),
            "aggregate_vs_global_other": correlation(aggregate, global_other),
            "geographic_k4_vs_global_other": correlation(
                geographic, global_other
            ),
        },
        "variance_explained_by_self_plus_aggregate": {
            "geographic_k4_r2": conditional_r2(geographic),
            "global_other_r2": conditional_r2(global_other),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cross-farm context matrix")
    parser.add_argument(
        "--experiment-root", default="artifacts/cross_farm_context_v1"
    )
    parser.add_argument(
        "--aggregate-reference",
        default="artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    aggregate_reference = Path(args.aggregate_reference).resolve()
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in GROUPS.values()
        for seed in SEEDS
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing formal runs:\n" + "\n".join(missing))
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    aggregate_summary = summarize_group(aggregate_reference, SEEDS)
    aggregate_summary["group"] = "G0-aggregate-recent"
    summary_rows.append(aggregate_summary)
    for label, group in GROUPS.items():
        row = summarize_group(groups_root / group, SEEDS)
        row["group"] = label
        summary_rows.append(row)

    paths = {label: groups_root / group for label, group in GROUPS.items()}
    paths["G0-aggregate-recent"] = aggregate_reference
    pair_specs = (
        ("G1-self-duplicate", "G0-aggregate-recent"),
        ("G2-geographic-K4", "G1-self-duplicate"),
        ("G3-global-other", "G1-self-duplicate"),
        ("G4-weather-coherent-K4", "G1-self-duplicate"),
        ("G2-geographic-K4", "G3-global-other"),
        ("G4-weather-coherent-K4", "G3-global-other"),
        ("G4-weather-coherent-K4", "G2-geographic-K4"),
    )
    comparisons = []
    band_rows = []
    for index, (candidate, reference) in enumerate(pair_specs):
        row = compare_groups(
            paths[candidate],
            paths[reference],
            SEEDS,
            args.bootstrap_repeats,
            16300 + index,
        )
        row["candidate"] = candidate
        row["reference"] = reference
        row["comparison"] = f"{candidate}_vs_{reference}"
        row["seedwise_p_two_sided"] = two_sided_p(
            row["seedwise_differences_mw"]
        )
        comparisons.append(row)
        band_rows.extend(
            horizon_bands(
                paths[candidate],
                paths[reference],
                row["comparison"],
                args.bootstrap_repeats,
                17300 + 10 * index,
            )
        )

    primary_indices = [1, 2, 3]
    adjusted = holm_adjust(
        [comparisons[index]["seedwise_p_two_sided"] for index in primary_indices]
    )
    for index, value in zip(primary_indices, adjusted, strict=True):
        comparisons[index]["holm_p_cross_farm_family"] = value

    primary_decisions = []
    band_by_comparison: dict[str, list[dict[str, Any]]] = {}
    for row in band_rows:
        band_by_comparison.setdefault(row["comparison"], []).append(row)
    for index in primary_indices:
        row = comparisons[index]
        bands = band_by_comparison[row["comparison"]]
        stable_evidence = bool(
            row["seedwise_ci95_high_mw"] < 0.0
            and row["ensemble_ci95_high_mw"] < 0.0
            and row["holm_p_cross_farm_family"] < 0.05
        )
        no_severe_band_degradation = not any(
            band["candidate_minus_reference_mw"] > 2.0
            and band["ci95_low_mw"] > 0.0
            for band in bands
        )
        main_model_threshold = bool(
            -row["seedwise_difference_mean_mw"] >= 2.0
            and -row["ensemble_candidate_minus_reference_mw"] >= 2.0
        )
        primary_decisions.append(
            {
                "candidate": row["candidate"],
                "stable_evidence": stable_evidence,
                "no_severe_band_degradation": no_severe_band_degradation,
                "meets_2mw_main_model_threshold": main_model_threshold,
                "eligible_for_main_model": (
                    stable_evidence
                    and no_severe_band_degradation
                    and main_model_threshold
                ),
            }
        )
    eligible = [
        decision["candidate"]
        for decision in primary_decisions
        if decision["eligible_for_main_model"]
    ]
    selected = None
    if eligible:
        comparison_by_candidate = {
            row["candidate"]: row for row in comparisons[1:4]
        }
        selected = min(
            eligible,
            key=lambda label: comparison_by_candidate[label][
                "seedwise_difference_mean_mw"
            ],
        )
    decision = {
        "selected_cross_farm_context": selected,
        "action": (
            "rebuild_teacher_then_run_KD_for_selected_structure"
            if selected is not None
            else "stop_cross_farm_modeling_and_keep_aggregate_plus_self_state"
        ),
        "primary_candidates": primary_decisions,
        "main_model_threshold": (
            "paired seed and ensemble evidence plus at least 2 MW improvement in both centers"
        ),
    }
    parameter_counts = {
        row["group"]: row["parameter_count"] for row in summary_rows
    }
    cross_counts = [parameter_counts[label] for label in GROUPS]
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "seeds": SEEDS,
            "training": "Normal (no KD), jointly trained B with anchor weight 0.2",
            "weather_frontend": "fixed learned 4x4 with wind geometry",
            "neighbors": 4,
            "bootstrap": "paired UTC-day blocks on ten-model ensemble predictions",
            "bootstrap_repeats": args.bootstrap_repeats,
            "multiple_testing": "Holm correction over G2/G3/G4 versus G1",
        },
        "capacity_audit": {
            "cross_farm_parameter_counts": {
                label: parameter_counts[label] for label in GROUPS
            },
            "all_cross_farm_groups_exactly_matched": len(set(cross_counts)) == 1,
            "self_duplicate_is_information_redundant": True,
            "config_audit": audit_configs(groups_root),
        },
        "input_context_diagnostics": input_context_diagnostics(
            Path(args.cache_dir).resolve()
        ),
        "summaries": summary_rows,
        "comparisons": comparisons,
        "horizon_bands": band_rows,
        "decision": decision,
    }
    write_csv(output / "summaries.csv", summary_rows)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "horizon_bands.csv", band_rows)
    write_csv(output / "decisions.csv", primary_decisions)
    write_json(output / "analysis.json", payload)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
