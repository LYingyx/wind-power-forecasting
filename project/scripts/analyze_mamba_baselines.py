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
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metric_summary(name: str, group: Path, split: str) -> dict[str, Any]:
    metrics = [
        load_json(path)
        for path in sorted(group.glob(f"seed*/metrics_{split}.json"))
    ]
    if len(metrics) != 3:
        raise FileNotFoundError(f"Expected three {split} metric files under {group}")
    mae = [float(item["cluster_mae_mw"]) for item in metrics]
    parameters = [int(item["parameter_count"]) for item in metrics]
    return {
        "group": name,
        "split": split,
        "cluster_mae_mw_mean": statistics.mean(mae),
        "cluster_mae_mw_sample_std": statistics.stdev(mae),
        "parameter_count_mean": statistics.mean(parameters),
        "seeds": len(metrics),
        "directory": str(group),
    }


def compare(
    candidate_name: str,
    candidate_path: Path,
    reference_name: str,
    reference_path: Path,
    split: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate, reference = align_predictions(
        mean_prediction(candidate_path, split),
        mean_prediction(reference_path, split),
    )
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
        "candidate": candidate_name,
        "reference": reference_name,
        "split": split,
        "candidate_mae_mw": float(candidate_error.mean()),
        "reference_mae_mw": float(reference_error.mean()),
        "candidate_minus_reference_mae_mw": delta,
        "ci95_low_mw": low,
        "ci95_high_mw": high,
        "ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
        "block": "UTC_day",
        "bootstrap_repeats": repeats,
        "paired_samples": int(candidate_error.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze official Mamba baselines")
    parser.add_argument("--root", required=True)
    parser.add_argument("--final-route-root", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    final_root = Path(args.final_route_root).resolve()
    paper_root = Path(args.paper_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = {
        "mamba": root / "mamba" / "m0_official_mamba",
        "mamba2": root / "mamba" / "m1_official_mamba2",
        "controlled_gru": final_root / "candidates" / "m0_controlled_full",
        "parallel_decoder": paper_root / "controls" / "scratch_aux",
        "validation_selected_final": final_root / "selected" / "model",
    }
    summaries = [
        metric_summary(name, groups[name], split)
        for name in ("mamba", "mamba2")
        for split in ("val", "test")
    ]
    comparisons = []
    pairs = (
        ("mamba", "controlled_gru", "val"),
        ("mamba2", "controlled_gru", "val"),
        ("mamba2", "mamba", "val"),
        ("mamba", "parallel_decoder", "val"),
        ("mamba2", "parallel_decoder", "val"),
        ("mamba", "validation_selected_final", "test"),
        ("mamba2", "validation_selected_final", "test"),
    )
    for index, (candidate, reference, split) in enumerate(pairs):
        comparisons.append(
            compare(
                candidate,
                groups[candidate],
                reference,
                groups[reference],
                split,
                args.bootstrap_repeats,
                142 + index,
            )
        )
    write_csv(output / "metrics.csv", summaries)
    write_csv(output / "paired_day_bootstrap.csv", comparisons)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {"metrics": summaries, "comparisons": comparisons},
            stream,
            ensure_ascii=True,
            indent=2,
        )
    os.replace(temporary, output / "analysis.json")
    print(f"[mamba-analysis] comparisons={len(comparisons)} output={output}")


if __name__ == "__main__":
    main()
