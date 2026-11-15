from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
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
LADDER_GROUPS = (
    "frozen_weather_only",
    "frozen_cluster_current",
    "frozen_station_current",
    "frozen_cluster_ramp15m",
    "frozen_recent_1h",
    "frozen_full_history",
)


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


def ensemble_error(group: Path, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    _, prediction = group_predictions(group, seeds)
    return (
        np.abs(
            prediction["cluster_prediction_mw"].astype(np.float64)
            - prediction["cluster_target_mw"].astype(np.float64)
        ),
        prediction["origin_unix"].astype(np.int64),
    )


def horizon_analysis(
    groups_root: Path, seeds: list[int], repeats: int
) -> list[dict[str, Any]]:
    errors = {}
    origins = None
    for group in LADDER_GROUPS:
        errors[group], group_origins = ensemble_error(groups_root / group, seeds)
        if origins is None:
            origins = group_origins
        else:
            np.testing.assert_array_equal(origins, group_origins)
    assert origins is not None
    weather = errors["frozen_weather_only"]
    contrasts = {
        "cluster_current_benefit_vs_weather": weather - errors["frozen_cluster_current"],
        "station_current_benefit_vs_weather": weather - errors["frozen_station_current"],
        "cluster_ramp_benefit_vs_weather": weather - errors["frozen_cluster_ramp15m"],
        "recent_1h_benefit_vs_weather": weather - errors["frozen_recent_1h"],
        "full_history_benefit_vs_weather": weather - errors["frozen_full_history"],
        "spatial_snapshot_benefit": (
            errors["frozen_cluster_current"] - errors["frozen_station_current"]
        ),
        "aggregate_ramp_benefit": (
            errors["frozen_cluster_current"] - errors["frozen_cluster_ramp15m"]
        ),
        "recent_trajectory_benefit": (
            errors["frozen_station_current"] - errors["frozen_recent_1h"]
        ),
        "recent_state_benefit_over_aggregate_ramp": (
            errors["frozen_cluster_ramp15m"] - errors["frozen_recent_1h"]
        ),
        "long_memory_benefit": (
            errors["frozen_recent_1h"] - errors["frozen_full_history"]
        ),
    }
    statistics = {}
    for index, (label, values) in enumerate(contrasts.items()):
        mean, low, high, days = block_bootstrap_matrix(
            values, origins, repeats, 9100 + index
        )
        statistics[label] = (mean, low, high, days)
    rows = []
    for lead in range(weather.shape[1]):
        row: dict[str, Any] = {
            "lead": lead + 1,
            "lead_hours": (lead + 1) / 4.0,
        }
        for group, values in errors.items():
            row[f"{group}_ensemble_mae_mw"] = float(values[:, lead].mean())
        for label, (mean, low, high, days) in statistics.items():
            row[f"{label}_mw"] = float(mean[lead])
            row[f"{label}_ci95_low_mw"] = float(low[lead])
            row[f"{label}_ci95_high_mw"] = float(high[lead])
            row["utc_day_blocks"] = days
        rows.append(row)
    return rows


def frozen_baseline_audit(
    ladder_root: Path,
    conditional_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    replay_by_group: dict[str, list[float]] = {
        group: [] for group in LADDER_GROUPS
    }
    exact_state_by_group: dict[str, list[bool]] = {
        group: [] for group in LADDER_GROUPS
    }
    for seed in seeds:
        prior = load_npz(
            conditional_root
            / "groups"
            / "history_baseline"
            / f"seed{seed}"
            / "predictions_val.npz"
        )
        prior_state = torch.load(
            conditional_root
            / "groups"
            / "history_baseline"
            / f"seed{seed}"
            / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        for group in LADDER_GROUPS:
            candidate = load_npz(
                ladder_root
                / "groups"
                / group
                / f"seed{seed}"
                / "predictions_val.npz"
            )
            np.testing.assert_array_equal(
                candidate["origin_unix"], prior["origin_unix"]
            )
            replay_by_group[group].append(
                float(
                    np.abs(
                        candidate["baseline_cluster_prediction_mw"].astype(
                            np.float64
                        )
                        - prior["cluster_prediction_mw"].astype(np.float64)
                    ).mean()
                )
            )
            checkpoint = torch.load(
                ladder_root
                / "groups"
                / group
                / f"seed{seed}"
                / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )["model_state"]
            exact_state_by_group[group].append(
                all(
                    f"baseline_model.{name}" in checkpoint
                    and torch.equal(checkpoint[f"baseline_model.{name}"], value)
                    for name, value in prior_state.items()
                )
            )
    replay = [
        value for group_values in replay_by_group.values() for value in group_values
    ]
    exact_state = [
        value
        for group_values in exact_state_by_group.values()
        for value in group_values
    ]
    return {
        "audited_checkpoints": len(exact_state),
        "checkpoint_state_exact_all_runs": all(exact_state),
        "checkpoint_state_exact_by_group": {
            group: all(values) for group, values in exact_state_by_group.items()
        },
        "checkpoint_state_exact_run_values_by_group": exact_state_by_group,
        "prediction_replay_abs_difference_mean_mw": float(np.mean(replay)),
        "prediction_replay_abs_difference_max_run_mw": float(np.max(replay)),
        "prediction_replay_abs_difference_by_group_mw": {
            group: {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "seed_values": values,
            }
            for group, values in replay_by_group.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen-B0 state ladder")
    parser.add_argument("--experiment-root", default="artifacts/state_forcing_ladder_v1")
    parser.add_argument(
        "--conditional-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    conditional = Path(args.conditional_root).resolve()
    if not (root / "ALL_COMPLETE").is_file():
        raise FileNotFoundError(f"Experiment is not complete: {root / 'ALL_COMPLETE'}")
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    groups_root = root / "groups"
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in LADDER_GROUPS
        for seed in SEEDS
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed seeds:\n" + "\n".join(missing))

    summaries = [summarize_group(groups_root / group, SEEDS) for group in LADDER_GROUPS]
    local_pairs = (
        ("frozen_cluster_current", "frozen_weather_only"),
        ("frozen_station_current", "frozen_weather_only"),
        ("frozen_cluster_ramp15m", "frozen_weather_only"),
        ("frozen_recent_1h", "frozen_weather_only"),
        ("frozen_full_history", "frozen_weather_only"),
        ("frozen_station_current", "frozen_cluster_current"),
        ("frozen_cluster_ramp15m", "frozen_cluster_current"),
        ("frozen_recent_1h", "frozen_station_current"),
        ("frozen_recent_1h", "frozen_cluster_ramp15m"),
        ("frozen_full_history", "frozen_recent_1h"),
    )
    comparisons = []
    for index, (candidate, reference) in enumerate(local_pairs):
        comparisons.append(
            compare_groups(
                groups_root / candidate,
                groups_root / reference,
                SEEDS,
                args.bootstrap_repeats,
                9200 + index,
            )
        )
    external_pairs = (
        ("reference_anchor_020", conditional / "groups" / "anchor_020"),
        (
            "reference_pretrained_full",
            conditional / "groups" / "factor_history_weather",
        ),
        (
            "reference_random_init_full",
            conditional / "groups" / "initialization_random",
        ),
    )
    for index, (reference_label, reference_path) in enumerate(external_pairs):
        result = compare_groups(
            groups_root / "frozen_full_history",
            reference_path,
            SEEDS,
            args.bootstrap_repeats,
            9300 + index,
        )
        result["reference"] = reference_label
        comparisons.append(result)

    horizon = horizon_analysis(groups_root, SEEDS, args.bootstrap_repeats)
    audit = frozen_baseline_audit(root, conditional, SEEDS)
    payload = {
        "protocol": {
            "model_seeds": SEEDS,
            "model_seed_count": len(SEEDS),
            "split": "2023 validation",
            "test_split_used": False,
            "history_baseline": "pretrained B0(H), exactly frozen in every group",
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "state_definitions": {
            "frozen_weather_only": "no correction-state input",
            "frozen_cluster_current": "current aggregate power only",
            "frozen_station_current": "current 56-station vector plus aggregate power",
            "frozen_cluster_ramp15m": "current aggregate power plus last 15-minute aggregate ramp",
            "frozen_recent_1h": "latest four 15-minute station and aggregate points",
            "frozen_full_history": "all 24 15-minute station and aggregate history points",
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "frozen_baseline_audit": audit,
    }
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "horizon.csv", horizon)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps({"output": str(output), "groups": len(summaries), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
