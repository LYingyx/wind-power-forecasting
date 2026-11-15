from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from scipy.stats import t as student_t

try:
    from scripts.analyze_paper_results import align_predictions
except ModuleNotFoundError:
    from analyze_paper_results import align_predictions  # type: ignore[no-redef]


DEFAULT_GROUPS = (
    "direct_d64",
    "direct_d72_capacity",
    "joint_full",
    "joint_dct4",
    "joint_dct7",
    "joint_dct8",
    "joint_dct12",
    "joint_weather_only",
    "joint_no_baseline_aux",
)


def parse_int_csv(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if len(result) != len(set(result)):
        raise ValueError("Seeds must be unique")
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def paired_utc_day_block_bootstrap(
    candidate_error: np.ndarray,
    reference_error: np.ndarray,
    origins: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Bootstrap a sample-weighted paired MAE difference by whole UTC days."""
    candidate_error = np.asarray(candidate_error, dtype=np.float64)
    reference_error = np.asarray(reference_error, dtype=np.float64)
    origins = np.asarray(origins)
    if candidate_error.shape != reference_error.shape:
        raise ValueError("Candidate and reference error shapes differ")
    if candidate_error.ndim != 1 or origins.shape != candidate_error.shape:
        raise ValueError("Errors and origins must be aligned one-dimensional arrays")
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive")

    paired_delta = candidate_error - reference_error
    block_id = origins // 86400
    blocks = np.unique(block_id)
    block_sums = np.asarray(
        [paired_delta[block_id == block].sum() for block in blocks],
        dtype=np.float64,
    )
    block_counts = np.asarray(
        [(block_id == block).sum() for block in blocks],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = block_sums[selected].sum() / block_counts[selected].sum()
    return (
        float(paired_delta.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        int(len(blocks)),
    )


def group_predictions(
    group: Path, seeds: list[int], split: str = "val"
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    predictions = [
        load_prediction(group / f"seed{seed}" / f"predictions_{split}.npz")
        for seed in seeds
    ]
    base = predictions[0]
    for prediction in predictions[1:]:
        np.testing.assert_array_equal(prediction["origin_unix"], base["origin_unix"])
        np.testing.assert_allclose(
            prediction["cluster_target_mw"], base["cluster_target_mw"]
        )
    ensemble = {
        "origin_unix": base["origin_unix"],
        "cluster_target_mw": base["cluster_target_mw"],
        "cluster_history_last_mw": base["cluster_history_last_mw"],
        "cluster_prediction_mw": np.mean(
            [prediction["cluster_prediction_mw"] for prediction in predictions],
            axis=0,
        ),
    }
    return predictions, ensemble


def summarize_group(group: Path, seeds: list[int], split: str = "val") -> dict[str, Any]:
    metrics = [
        load_json(group / f"seed{seed}" / f"metrics_{split}.json") for seed in seeds
    ]
    cluster = [float(metric["cluster_mae_mw"]) for metric in metrics]
    station = [float(metric["station_mae_mw"]) for metric in metrics]
    _, ensemble = group_predictions(group, seeds, split)
    ensemble_error = np.abs(
        ensemble["cluster_prediction_mw"] - ensemble["cluster_target_mw"]
    ).astype(np.float64)
    return {
        "group": group.name,
        "seeds": len(seeds),
        "seed_values": seeds,
        "cluster_mae_mw_mean": statistics.mean(cluster),
        "cluster_mae_mw_sample_std": statistics.stdev(cluster),
        "cluster_mae_mw_min": min(cluster),
        "cluster_mae_mw_max": max(cluster),
        "station_mae_mw_mean": statistics.mean(station),
        "ensemble_cluster_mae_mw": float(ensemble_error.mean()),
        "ensemble_gain_vs_mean_single_mw": statistics.mean(cluster)
        - float(ensemble_error.mean()),
        "parameter_count": int(metrics[0]["parameter_count"]),
    }


def compare_groups(
    candidate: Path,
    reference: Path,
    seeds: list[int],
    repeats: int,
    bootstrap_seed: int,
    split: str = "val",
) -> dict[str, Any]:
    candidate_individual, candidate_ensemble = group_predictions(candidate, seeds, split)
    reference_individual, reference_ensemble = group_predictions(reference, seeds, split)
    candidate_ensemble, reference_ensemble = align_predictions(
        candidate_ensemble, reference_ensemble
    )
    candidate_error = np.abs(
        candidate_ensemble["cluster_prediction_mw"]
        - candidate_ensemble["cluster_target_mw"]
    ).astype(np.float64)
    reference_error = np.abs(
        reference_ensemble["cluster_prediction_mw"]
        - reference_ensemble["cluster_target_mw"]
    ).astype(np.float64)
    delta, low, high, day_blocks = paired_utc_day_block_bootstrap(
        candidate_error.mean(axis=1),
        reference_error.mean(axis=1),
        candidate_ensemble["origin_unix"],
        repeats,
        bootstrap_seed,
    )

    seed_differences = []
    for candidate_seed, reference_seed in zip(
        candidate_individual, reference_individual, strict=True
    ):
        candidate_seed, reference_seed = align_predictions(
            candidate_seed, reference_seed
        )
        candidate_seed_mae = float(
            np.abs(
                candidate_seed["cluster_prediction_mw"]
                - candidate_seed["cluster_target_mw"]
            )
            .astype(np.float64)
            .mean()
        )
        reference_seed_mae = float(
            np.abs(
                reference_seed["cluster_prediction_mw"]
                - reference_seed["cluster_target_mw"]
            )
            .astype(np.float64)
            .mean()
        )
        seed_differences.append(candidate_seed_mae - reference_seed_mae)

    lead_difference = candidate_error.mean(axis=0) - reference_error.mean(axis=0)
    seedwise_mean = statistics.mean(seed_differences)
    seedwise_std = statistics.stdev(seed_differences)
    seedwise_half_width = float(
        student_t.ppf(0.975, df=len(seed_differences) - 1)
        * seedwise_std
        / np.sqrt(len(seed_differences))
    )
    return {
        "candidate": candidate.name,
        "reference": reference.name,
        "seeds": len(seeds),
        "ensemble_candidate_mae_mw": float(candidate_error.mean()),
        "ensemble_reference_mae_mw": float(reference_error.mean()),
        "ensemble_candidate_minus_reference_mw": float(delta),
        "ensemble_ci95_low_mw": float(low),
        "ensemble_ci95_high_mw": float(high),
        "seedwise_difference_mean_mw": seedwise_mean,
        "seedwise_difference_sample_std_mw": seedwise_std,
        "seedwise_ci95_low_mw": seedwise_mean - seedwise_half_width,
        "seedwise_ci95_high_mw": seedwise_mean + seedwise_half_width,
        "seedwise_candidate_wins": sum(value < 0.0 for value in seed_differences),
        "seedwise_differences_mw": seed_differences,
        "lead_difference_mw": lead_difference.tolist(),
        "block": "UTC_day",
        "paired_forecast_origins": int(candidate_error.shape[0]),
        "utc_day_blocks": day_blocks,
        "bootstrap_repeats": repeats,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ten-seed incremental-mechanism experiments"
    )
    parser.add_argument(
        "--experiment-root", default="artifacts/incremental_mechanism_v2"
    )
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--reference", default="joint_full")
    parser.add_argument("--direct-reference", default="direct_d64")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds)
    if len(seeds) != 10:
        raise ValueError("Formal analysis requires exactly ten seeds")
    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    group_names = [item for item in args.groups.split(",") if item]
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in group_names
        for seed in seeds
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Formal analysis requires all group/seed completions; missing:\n"
            + "\n".join(missing)
        )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_group(groups_root / group, seeds) for group in group_names
    ]
    reference = groups_root / args.reference
    direct_reference = groups_root / args.direct_reference
    contrasts = []
    for index, group in enumerate(group_names):
        candidate = groups_root / group
        if group != args.reference:
            contrasts.append(
                compare_groups(
                    candidate,
                    reference,
                    seeds,
                    args.bootstrap_repeats,
                    1200 + index,
                )
            )
        if group.startswith("joint_"):
            contrasts.append(
                compare_groups(
                    candidate,
                    direct_reference,
                    seeds,
                    args.bootstrap_repeats,
                    2200 + index,
                )
            )

    capacity_group = "direct_d72_capacity"
    if capacity_group in group_names and args.direct_reference in group_names:
        contrasts.append(
            compare_groups(
                groups_root / capacity_group,
                direct_reference,
                seeds,
                args.bootstrap_repeats,
                3200,
            )
        )

    payload = {
        "protocol": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "selection_split": "2023 validation",
            "test_split_used": False,
            "bootstrap": "paired UTC-day blocks on ten-seed ensemble predictions",
            "bootstrap_estand": "sample-weighted MAE difference",
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "summaries": summaries,
        "contrasts": contrasts,
    }
    write_csv(
        output / "summaries.csv",
        summaries,
        [
            "group",
            "seeds",
            "cluster_mae_mw_mean",
            "cluster_mae_mw_sample_std",
            "cluster_mae_mw_min",
            "cluster_mae_mw_max",
            "station_mae_mw_mean",
            "ensemble_cluster_mae_mw",
            "ensemble_gain_vs_mean_single_mw",
            "parameter_count",
        ],
    )
    write_csv(
        output / "contrasts.csv",
        contrasts,
        [
            "candidate",
            "reference",
            "seeds",
            "ensemble_candidate_mae_mw",
            "ensemble_reference_mae_mw",
            "ensemble_candidate_minus_reference_mw",
            "ensemble_ci95_low_mw",
            "ensemble_ci95_high_mw",
            "seedwise_difference_mean_mw",
            "seedwise_difference_sample_std_mw",
            "seedwise_ci95_low_mw",
            "seedwise_ci95_high_mw",
            "seedwise_candidate_wins",
            "block",
            "paired_forecast_origins",
            "utc_day_blocks",
            "bootstrap_repeats",
        ],
    )
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
