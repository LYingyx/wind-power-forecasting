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
except ModuleNotFoundError:
    from analyze_paper_results import (  # type: ignore[no-redef]
        align_predictions,
        block_bootstrap_difference,
        mean_prediction,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def summarize(path: Path, seeds: list[int]) -> dict[str, Any]:
    metrics = [load_json(path / f"seed{seed}" / "metrics_val.json") for seed in seeds]
    values = [float(item["cluster_mae_mw"]) for item in metrics]
    return {
        "directory": str(path),
        "seeds": len(seeds),
        "cluster_mae_mw_mean": statistics.mean(values),
        "cluster_mae_mw_sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "station_mae_mw_mean": statistics.mean(float(item["station_mae_mw"]) for item in metrics),
        "parameter_count": int(metrics[0]["parameter_count"]),
    }


def mean_prediction_for_seeds(
    group: Path, split: str, seeds: list[int]
) -> dict[str, np.ndarray]:
    if seeds == [42, 43, 44]:
        return mean_prediction(group, split)
    archives = []
    for seed in seeds:
        path = group / f"seed{seed}" / f"predictions_{split}.npz"
        with np.load(path) as values:
            archives.append({key: values[key] for key in values.files})
    for archive in archives[1:]:
        np.testing.assert_array_equal(archive["origin_unix"], archives[0]["origin_unix"])
        np.testing.assert_allclose(
            archive["cluster_target_mw"], archives[0]["cluster_target_mw"]
        )
    return {
        "origin_unix": archives[0]["origin_unix"],
        "cluster_target_mw": archives[0]["cluster_target_mw"],
        "cluster_history_last_mw": archives[0]["cluster_history_last_mw"],
        "cluster_prediction_mw": np.mean(
            [archive["cluster_prediction_mw"] for archive in archives], axis=0
        ),
    }


def compare(
    candidate_path: Path,
    reference_path: Path,
    seeds: list[int],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate = mean_prediction_for_seeds(candidate_path, "val", seeds)
    reference = mean_prediction_for_seeds(reference_path, "val", seeds)
    candidate, reference = align_predictions(candidate, reference)
    candidate_error = np.abs(
        candidate["cluster_prediction_mw"] - candidate["cluster_target_mw"]
    ).mean(axis=1)
    reference_error = np.abs(
        reference["cluster_prediction_mw"] - reference["cluster_target_mw"]
    ).mean(axis=1)
    delta, low, high = block_bootstrap_difference(
        candidate_error, reference_error, candidate["origin_unix"], repeats, seed
    )
    return {
        "candidate_mae_mw": float(candidate_error.mean()),
        "reference_mae_mw": float(reference_error.mean()),
        "candidate_minus_reference_mae_mw": delta,
        "ci95_low_mw": low,
        "ci95_high_mw": high,
        "paired_samples": int(len(candidate_error)),
        "block": "UTC_day",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze incremental-correction experiments")
    parser.add_argument("--screening-root", required=True)
    parser.add_argument("--confirmation-root")
    parser.add_argument("--full-group", default="joint_full")
    parser.add_argument("--selected-group", default="joint_temporal4")
    parser.add_argument("--frozen-group", default="temporal4")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()
    screening_root = Path(args.screening_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    screening_rows = []
    for candidate in sorted(path for path in screening_root.iterdir() if path.is_dir()):
        if not (candidate / "seed42" / "COMPLETE").is_file():
            continue
        row = {"candidate": candidate.name, **summarize(candidate, [42])}
        row.update(
            {
                f"vs_baseline_{key}": value
                for key, value in compare(
                    candidate, baseline_root, [42], args.bootstrap_repeats, 42
                ).items()
            }
        )
        screening_rows.append(row)
    write_csv(output / "screening.csv", screening_rows)
    payload: dict[str, Any] = {"screening": screening_rows}

    if args.confirmation_root:
        confirmation_root = Path(args.confirmation_root).resolve()
        full = confirmation_root / args.full_group
        selected = confirmation_root / args.selected_group
        frozen = confirmation_root / args.frozen_group
        seeds = [42, 43, 44]
        confirmation_rows = [
            {"group": "full", **summarize(full, seeds)},
            {"group": "selected", **summarize(selected, seeds)},
            {"group": "frozen_selected_structure", **summarize(frozen, seeds)},
            {"group": "matched_parallel_baseline", **summarize(baseline_root, seeds)},
        ]
        contrasts = {
            "selected_minus_full": compare(
                selected, full, seeds, args.bootstrap_repeats, 101
            ),
            "selected_minus_matched_parallel": compare(
                selected, baseline_root, seeds, args.bootstrap_repeats, 102
            ),
            "full_minus_matched_parallel": compare(
                full, baseline_root, seeds, args.bootstrap_repeats, 103
            ),
            "selected_minus_frozen_selected_structure": compare(
                selected, frozen, seeds, args.bootstrap_repeats, 104
            ),
        }
        write_csv(output / "confirmation.csv", confirmation_rows)
        payload.update(confirmation=confirmation_rows, contrasts=contrasts)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
