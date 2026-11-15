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
import torch

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
    "aggregate_snapshot": "frozen_cluster_current",
    "station_snapshot": "frozen_station_current",
    "aggregate_recent": "frozen_cluster_recent_1h",
    "station_recent": "frozen_recent_1h",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def ensemble_error(group: Path) -> tuple[np.ndarray, np.ndarray]:
    _, prediction = group_predictions(group, SEEDS)
    error = np.abs(
        prediction["cluster_prediction_mw"].astype(np.float64)
        - prediction["cluster_target_mw"].astype(np.float64)
    )
    return error, prediction["origin_unix"].astype(np.int64)


def seed_mae(group: Path, seed: int) -> float:
    with np.load(group / f"seed{seed}" / "predictions_val.npz") as archive:
        return float(
            np.abs(
                archive["cluster_prediction_mw"].astype(np.float64)
                - archive["cluster_target_mw"].astype(np.float64)
            ).mean()
        )


def interaction_statistics(
    groups_root: Path, repeats: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: dict[str, np.ndarray] = {}
    origins = None
    for label, group in GROUPS.items():
        errors[label], group_origins = ensemble_error(groups_root / group)
        if origins is None:
            origins = group_origins
        else:
            np.testing.assert_array_equal(origins, group_origins)
    assert origins is not None

    contrasts = {
        "aggregate_trajectory_benefit": (
            errors["aggregate_snapshot"] - errors["aggregate_recent"]
        ),
        "station_trajectory_benefit": (
            errors["station_snapshot"] - errors["station_recent"]
        ),
        "snapshot_spatial_benefit": (
            errors["aggregate_snapshot"] - errors["station_snapshot"]
        ),
        "recent_spatial_trajectory_benefit": (
            errors["aggregate_recent"] - errors["station_recent"]
        ),
    }
    contrasts["spatial_x_trajectory_interaction"] = (
        contrasts["recent_spatial_trajectory_benefit"]
        - contrasts["snapshot_spatial_benefit"]
    )

    lead_statistics = {}
    for index, (label, matrix) in enumerate(contrasts.items()):
        point, low, high, days = block_bootstrap_matrix(
            matrix, origins, repeats, 9600 + index
        )
        lead_statistics[label] = (point, low, high, days)

    seed_interactions = []
    for seed in SEEDS:
        value = (
            seed_mae(groups_root / GROUPS["aggregate_recent"], seed)
            - seed_mae(groups_root / GROUPS["station_recent"], seed)
            - seed_mae(groups_root / GROUPS["aggregate_snapshot"], seed)
            + seed_mae(groups_root / GROUPS["station_snapshot"], seed)
        )
        seed_interactions.append(value)
    seed_mean = statistics.mean(seed_interactions)
    seed_std = statistics.stdev(seed_interactions)
    half_width = float(
        student_t.ppf(0.975, df=len(seed_interactions) - 1)
        * seed_std
        / np.sqrt(len(seed_interactions))
    )

    scalar = {}
    for index, (label, matrix) in enumerate(contrasts.items()):
        point, low, high, days = block_bootstrap_matrix(
            matrix.mean(axis=1, keepdims=True), origins, repeats, 9700 + index
        )
        scalar[label] = {
            "benefit_mw": float(point[0]),
            "ci95_low_mw": float(low[0]),
            "ci95_high_mw": float(high[0]),
            "utc_day_blocks": days,
        }
    scalar["spatial_x_trajectory_interaction"].update(
        {
            "seedwise_mean_mw": seed_mean,
            "seedwise_ci95_low_mw": seed_mean - half_width,
            "seedwise_ci95_high_mw": seed_mean + half_width,
            "seedwise_positive": sum(value > 0.0 for value in seed_interactions),
            "seedwise_values_mw": seed_interactions,
        }
    )

    horizon_rows = []
    for lead in range(24):
        row: dict[str, Any] = {
            "lead": lead + 1,
            "lead_hours": (lead + 1) / 4.0,
        }
        for label, (point, low, high, days) in lead_statistics.items():
            row[f"{label}_mw"] = float(point[lead])
            row[f"{label}_ci95_low_mw"] = float(low[lead])
            row[f"{label}_ci95_high_mw"] = float(high[lead])
            row["utc_day_blocks"] = days
        horizon_rows.append(row)
    return scalar, horizon_rows


def audit_new_group(
    groups_root: Path, conditional_root: Path
) -> dict[str, Any]:
    exact = []
    replay = []
    for seed in SEEDS:
        checkpoint = torch.load(
            groups_root / GROUPS["aggregate_recent"] / f"seed{seed}" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        prior_state = torch.load(
            conditional_root
            / "groups"
            / "history_baseline"
            / f"seed{seed}"
            / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        exact.append(
            all(
                f"baseline_model.{name}" in checkpoint
                and torch.equal(checkpoint[f"baseline_model.{name}"], value)
                for name, value in prior_state.items()
            )
        )
        with np.load(
            groups_root
            / GROUPS["aggregate_recent"]
            / f"seed{seed}"
            / "predictions_val.npz"
        ) as candidate, np.load(
            conditional_root
            / "groups"
            / "history_baseline"
            / f"seed{seed}"
            / "predictions_val.npz"
        ) as prior:
            np.testing.assert_array_equal(candidate["origin_unix"], prior["origin_unix"])
            replay.append(
                float(
                    np.abs(
                        candidate["baseline_cluster_prediction_mw"].astype(np.float64)
                        - prior["cluster_prediction_mw"].astype(np.float64)
                    ).mean()
                )
            )
    return {
        "checkpoint_state_exact_all_seeds": all(exact),
        "checkpoint_state_exact_seed_values": exact,
        "prediction_replay_abs_difference_mean_mw": float(np.mean(replay)),
        "prediction_replay_abs_difference_max_seed_mw": float(np.max(replay)),
        "prediction_replay_seed_values_mw": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aggregate/station x snapshot/recent 2x2")
    parser.add_argument("--experiment-root", default="artifacts/state_forcing_ladder_v1")
    parser.add_argument(
        "--conditional-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    conditional = Path(args.conditional_root).resolve()
    if not (root / "SPATIAL_TRAJECTORY_COMPLETE").is_file():
        raise FileNotFoundError(root / "SPATIAL_TRAJECTORY_COMPLETE")
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in GROUPS.values()
        for seed in SEEDS
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed cells:\n" + "\n".join(missing))

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "spatial_trajectory_analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_group(groups_root / group, SEEDS) for group in GROUPS.values()
    ]
    pairs = (
        ("frozen_cluster_recent_1h", "frozen_cluster_current"),
        ("frozen_recent_1h", "frozen_station_current"),
        ("frozen_station_current", "frozen_cluster_current"),
        ("frozen_recent_1h", "frozen_cluster_recent_1h"),
        ("frozen_cluster_recent_1h", "frozen_cluster_ramp15m"),
    )
    comparisons = [
        compare_groups(
            groups_root / candidate,
            groups_root / reference,
            SEEDS,
            args.bootstrap_repeats,
            9800 + index,
        )
        for index, (candidate, reference) in enumerate(pairs)
    ]
    factorial, horizon = interaction_statistics(groups_root, args.bootstrap_repeats)
    audit = audit_new_group(groups_root, conditional)
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "model_seeds": SEEDS,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
            "factorial_cells": GROUPS,
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "factorial": factorial,
        "new_group_frozen_baseline_audit": audit,
    }
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(
        output / "factorial.csv",
        [{"contrast": label, **values} for label, values in factorial.items()],
    )
    write_csv(output / "horizon.csv", horizon)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps({"output": str(output), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
