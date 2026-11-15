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
    "R0-matched-control": "r0_matched_control",
    "R1-handcrafted-statistics": "r1_handcrafted_statistics",
    "R2-tiny-DeepSets": "r2_tiny_deepsets",
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
                    / GROUPS["R0-matched-control"]
                    / f"seed{seed}"
                    / "config.yaml"
                ).read_text(encoding="utf-8")
            )
        )
        for label, group in GROUPS.items():
            if label == "R0-matched-control":
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
        "differ_only_by_information_mode_and_experiment_id": not unexpected,
    }


def input_diagnostics(cache_dir: Path) -> dict[str, Any]:
    station = np.load(cache_dir / "val/station_history.npy", mmap_mode="r")[
        :, :, -4:
    ].astype(np.float64)
    cluster = np.load(cache_dir / "val/cluster_history.npy", mmap_mode="r")[
        :, -4:
    ].astype(np.float64)
    dispersion = station.std(axis=1)
    interquartile_range = np.quantile(station, 0.75, axis=1) - np.quantile(
        station, 0.25, axis=1
    )
    ramp = station[:, :, -1] - station[:, :, -2]
    ramp_dispersion = ramp.std(axis=1)
    unweighted_mean = station.mean(axis=1)
    return {
        "station_histories_are_capacity_normalized_factors": True,
        "aggregate_vs_unweighted_station_mean_correlation": float(
            np.corrcoef(cluster.reshape(-1), unweighted_mean.reshape(-1))[0, 1]
        ),
        "cross_station_std": {
            "mean": float(dispersion.mean()),
            "median": float(np.median(dispersion)),
            "p95": float(np.quantile(dispersion, 0.95)),
        },
        "cross_station_iqr": {
            "mean": float(interquartile_range.mean()),
            "median": float(np.median(interquartile_range)),
            "p95": float(np.quantile(interquartile_range, 0.95)),
        },
        "last_ramp_cross_station_std": {
            "mean": float(ramp_dispersion.mean()),
            "median": float(np.median(ramp_dispersion)),
            "p95": float(np.quantile(ramp_dispersion, 0.95)),
        },
        "aggregate_current_vs_current_dispersion_correlation": float(
            np.corrcoef(cluster[:, -1], dispersion[:, -1])[0, 1]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze regional composition ladder")
    parser.add_argument(
        "--experiment-root", default="artifacts/regional_composition_v1"
    )
    parser.add_argument(
        "--g1-reference",
        default="artifacts/cross_farm_context_v1/groups/g1_self_duplicate",
    )
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    g1_reference = Path(args.g1_reference).resolve()
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
    g1_summary = summarize_group(g1_reference, SEEDS)
    g1_summary["group"] = "G1-previous-self-duplicate"
    summary_rows.append(g1_summary)
    paths = {label: groups_root / group for label, group in GROUPS.items()}
    paths["G1-previous-self-duplicate"] = g1_reference
    for label, group in GROUPS.items():
        row = summarize_group(groups_root / group, SEEDS)
        row["group"] = label
        summary_rows.append(row)

    pair_specs = (
        ("R0-matched-control", "G1-previous-self-duplicate"),
        ("R1-handcrafted-statistics", "R0-matched-control"),
        ("R2-tiny-DeepSets", "R0-matched-control"),
        ("R2-tiny-DeepSets", "R1-handcrafted-statistics"),
    )
    comparisons = []
    band_rows = []
    for index, (candidate, reference) in enumerate(pair_specs):
        row = compare_groups(
            paths[candidate],
            paths[reference],
            SEEDS,
            args.bootstrap_repeats,
            18300 + index,
        )
        row["candidate"] = candidate
        row["reference"] = reference
        row["comparison"] = f"{candidate}_vs_{reference}"
        row["seedwise_p_two_sided"] = two_sided_p(row["seedwise_differences_mw"])
        comparisons.append(row)
        band_rows.extend(
            horizon_bands(
                paths[candidate],
                paths[reference],
                row["comparison"],
                args.bootstrap_repeats,
                19300 + 10 * index,
            )
        )

    primary_indices = [1, 2]
    adjusted = holm_adjust(
        [comparisons[index]["seedwise_p_two_sided"] for index in primary_indices]
    )
    for index, value in zip(primary_indices, adjusted, strict=True):
        comparisons[index]["holm_p_composition_family"] = value

    band_by_comparison: dict[str, list[dict[str, Any]]] = {}
    for row in band_rows:
        band_by_comparison.setdefault(row["comparison"], []).append(row)
    decisions = []
    for index in primary_indices:
        row = comparisons[index]
        stable_evidence = bool(
            row["seedwise_ci95_high_mw"] < 0.0
            and row["ensemble_ci95_high_mw"] < 0.0
            and row["holm_p_composition_family"] < 0.05
        )
        no_severe_band_degradation = not any(
            band["candidate_minus_reference_mw"] > 2.0
            and band["ci95_low_mw"] > 0.0
            for band in band_by_comparison[row["comparison"]]
        )
        main_model_threshold = bool(
            -row["seedwise_difference_mean_mw"] >= 2.0
            and -row["ensemble_candidate_minus_reference_mw"] >= 2.0
        )
        decisions.append(
            {
                "candidate": row["candidate"],
                "stable_evidence": stable_evidence,
                "no_severe_band_degradation": no_severe_band_degradation,
                "meets_2mw_main_model_threshold": main_model_threshold,
                "eligible_for_main_model": stable_evidence
                and no_severe_band_degradation
                and main_model_threshold,
            }
        )

    eligible = [row["candidate"] for row in decisions if row["eligible_for_main_model"]]
    selected = None
    if eligible:
        comparison_by_candidate = {
            row["candidate"]: row for row in comparisons[1:3]
        }
        selected = min(
            eligible,
            key=lambda label: comparison_by_candidate[label][
                "seedwise_difference_mean_mw"
            ],
        )
    decision = {
        "selected_regional_composition": selected,
        "action": (
            "rebuild_teacher_then_confirm_KD_for_selected_composition"
            if selected is not None
            else "freeze_architecture_and_stop_spatial_state_search"
        ),
        "primary_candidates": decisions,
        "main_model_threshold": (
            "paired seed and ensemble evidence plus at least 2 MW improvement "
            "in both centers versus the matched control"
        ),
    }

    parameter_counts = {row["group"]: row["parameter_count"] for row in summary_rows}
    composition_counts = [parameter_counts[label] for label in GROUPS]
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "seeds": SEEDS,
            "training": "Normal (no KD), jointly trained B with anchor weight 0.2",
            "weather_frontend": "fixed learned 4x4 with wind geometry",
            "information_ladder": (
                "R0 redundant aggregate branches; R1 opens handcrafted statistics; "
                "R2 additionally opens the station set"
            ),
            "bootstrap": "paired UTC-day blocks on ten-model ensemble predictions",
            "bootstrap_repeats": args.bootstrap_repeats,
            "multiple_testing": "Holm correction over R1 and R2 versus R0",
        },
        "capacity_audit": {
            "composition_parameter_counts": {
                label: parameter_counts[label] for label in GROUPS
            },
            "all_composition_groups_exactly_matched": len(set(composition_counts)) == 1,
            "config_audit": audit_configs(groups_root),
        },
        "input_diagnostics": input_diagnostics(Path(args.cache_dir).resolve()),
        "summaries": summary_rows,
        "comparisons": comparisons,
        "horizon_bands": band_rows,
        "decision": decision,
    }
    write_csv(output / "summaries.csv", summary_rows)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "horizon_bands.csv", band_rows)
    write_csv(output / "decisions.csv", decisions)
    write_json(output / "analysis.json", payload)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
