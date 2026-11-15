from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np

try:
    from scripts.analyze_paper_results import (
        align_predictions,
        block_bootstrap_difference,
        mean_prediction,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from analyze_paper_results import (  # type: ignore[no-redef]
        align_predictions,
        block_bootstrap_difference,
        mean_prediction,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize_group(name: str, path: Path, seeds: list[int]) -> dict[str, Any]:
    values: list[float] = []
    ramp_values: list[float] = []
    parameter_counts: list[int] = []
    for seed in seeds:
        seed_dir = path / f"seed{seed}"
        for required in ("COMPLETE", "metrics_val.json", "predictions_val.npz"):
            if not (seed_dir / required).is_file():
                raise FileNotFoundError(f"Incomplete group {name}: {seed_dir / required}")
        metrics = load_json(seed_dir / "metrics_val.json")
        values.append(float(metrics["cluster_mae_mw"]))
        if "cluster_ramp_mae_mw" in metrics:
            ramp_values.append(float(metrics["cluster_ramp_mae_mw"]))
        if "parameter_count" in metrics:
            parameter_counts.append(int(metrics["parameter_count"]))
    row: dict[str, Any] = {
        "group": name,
        "directory": str(path),
        "seeds": len(values),
        "cluster_mae_mw_mean": statistics.mean(values),
        "cluster_mae_mw_sample_std": statistics.stdev(values),
    }
    if parameter_counts:
        row["parameter_count_mean"] = statistics.mean(parameter_counts)
    if len(ramp_values) == len(values):
        row["cluster_ramp_mae_mw_mean"] = statistics.mean(ramp_values)
    return row


def compare_groups(
    candidate_name: str,
    candidate_path: Path,
    reference_name: str,
    reference_path: Path,
    *,
    scientific_question: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate = mean_prediction(candidate_path, "val")
    reference = mean_prediction(reference_path, "val")
    candidate, reference = align_predictions(candidate, reference)
    candidate_error = np.abs(
        candidate["cluster_prediction_mw"] - candidate["cluster_target_mw"]
    ).mean(axis=1)
    reference_error = np.abs(
        reference["cluster_prediction_mw"] - reference["cluster_target_mw"]
    ).mean(axis=1)
    delta, low, high = block_bootstrap_difference(
        candidate_error,
        reference_error,
        candidate["origin_unix"],
        repeats,
        seed,
    )
    return {
        "scientific_question": scientific_question,
        "candidate": candidate_name,
        "reference": reference_name,
        "candidate_mae_mw": float(candidate_error.mean()),
        "reference_mae_mw": float(reference_error.mean()),
        "candidate_minus_reference_mae_mw": delta,
        "ci95_low_mw": low,
        "ci95_high_mw": high,
        "candidate_better": bool(delta < 0.0),
        "ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
        "block": "UTC_day",
        "bootstrap_repeats": repeats,
        "paired_samples": int(candidate_error.size),
    }


def analyze(
    final_root: Path,
    paper_root: Path,
    transition_root: Path,
    output: Path,
    *,
    seeds: list[int],
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = {
        "controlled_full": final_root / "candidates" / "m0_controlled_full",
        "parallel_aux": paper_root / "controls" / "scratch_aux",
        "controlled_noaux": transition_root / "transition" / "r1_controlled_gru",
        "no_wind_geometry": final_root / "candidates" / "s0_no_wind_geometry",
        "fixed_bilinear": final_root / "candidates" / "s1_fixed_bilinear",
        "independent_weather": final_root / "candidates" / "t0_independent_weather",
        "nearest_hour": final_root / "candidates" / "t1_nearest_hour",
    }
    contrasts = (
        ("controlled_full", "parallel_aux", "state_recurrence_vs_parallel_decoder"),
        ("controlled_full", "controlled_noaux", "auxiliary_weather_constraint"),
        ("controlled_full", "no_wind_geometry", "wind_direction_geometry"),
        (
            "no_wind_geometry",
            "fixed_bilinear",
            "learned_station_mapping_without_wind_geometry",
        ),
        ("controlled_full", "fixed_bilinear", "complete_spatial_conditioning"),
        ("controlled_full", "independent_weather", "hourly_weather_change_path"),
        ("controlled_full", "nearest_hour", "full_future_weather_path"),
    )
    output.mkdir(parents=True, exist_ok=True)
    group_rows = [summarize_group(name, path, seeds) for name, path in groups.items()]
    comparison_rows = [
        compare_groups(
            candidate,
            groups[candidate],
            reference,
            groups[reference],
            scientific_question=question,
            repeats=bootstrap_repeats,
            seed=42 + index,
        )
        for index, (candidate, reference, question) in enumerate(contrasts)
    ]
    write_csv(output / "validation_groups.csv", group_rows)
    write_csv(output / "paired_day_bootstrap.csv", comparison_rows)
    payload = {
        "selection_split": "val",
        "note": "Negative candidate_minus_reference_mae_mw favors the candidate.",
        "groups": group_rows,
        "comparisons": comparison_rows,
    }
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output / "analysis.json")
    return group_rows, comparison_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only analysis for the final wind-forecast route"
    )
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--transition-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 2:
        raise ValueError("Analysis requires at least two seeds")
    groups, comparisons = analyze(
        Path(args.final_root).resolve(),
        Path(args.paper_root).resolve(),
        Path(args.transition_root).resolve(),
        Path(args.output_dir).resolve(),
        seeds=seeds,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(
        f"[final-route-analysis] groups={len(groups)} comparisons={len(comparisons)} "
        f"output={Path(args.output_dir).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
