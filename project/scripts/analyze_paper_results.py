from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np


SEED_PATTERN = re.compile(r"/seed\d+$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_name(relative: str) -> str:
    return SEED_PATTERN.sub("", relative)


def mean_prediction(group: Path, split: str) -> dict[str, np.ndarray]:
    archives = []
    for seed_dir in sorted(group.glob("seed*")):
        path = seed_dir / f"predictions_{split}.npz"
        if path.is_file():
            with np.load(path) as values:
                archives.append({key: values[key] for key in values.files})
    if not archives:
        raise FileNotFoundError(f"No {split} prediction archives under {group}")
    for archive in archives[1:]:
        np.testing.assert_array_equal(archive["origin_unix"], archives[0]["origin_unix"])
        np.testing.assert_allclose(
            archive["cluster_target_mw"], archives[0]["cluster_target_mw"]
        )
    prediction = np.mean(
        [archive["cluster_prediction_mw"] for archive in archives], axis=0
    )
    return {
        "origin_unix": archives[0]["origin_unix"],
        "cluster_target_mw": archives[0]["cluster_target_mw"],
        "cluster_history_last_mw": archives[0]["cluster_history_last_mw"],
        "cluster_prediction_mw": prediction,
    }


def block_bootstrap_difference(
    model_error: np.ndarray,
    baseline_error: np.ndarray,
    origins: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    block_id = origins // 86400
    blocks = np.unique(block_id)
    block_delta = np.asarray(
        [
            (model_error[block_id == block] - baseline_error[block_id == block]).mean()
            for block in blocks
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(block_delta), size=len(block_delta))
        samples[index] = block_delta[selected].mean()
    return (
        float(block_delta.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def align_predictions(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference_origins = np.asarray(reference["origin_unix"])
    candidate_origins = np.asarray(candidate["origin_unix"])
    if np.unique(reference_origins).size != reference_origins.size:
        raise ValueError("Reference prediction archive contains duplicate origins")
    if np.unique(candidate_origins).size != candidate_origins.size:
        raise ValueError("Candidate prediction archive contains duplicate origins")
    common, reference_index, candidate_index = np.intersect1d(
        reference_origins,
        candidate_origins,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        raise ValueError("Prediction archives have no common forecast origins")
    aligned_reference = {
        key: value[reference_index]
        for key, value in reference.items()
    }
    aligned_candidate = {
        key: value[candidate_index]
        for key, value in candidate.items()
    }
    np.testing.assert_array_equal(
        aligned_reference["origin_unix"], aligned_candidate["origin_unix"]
    )
    np.testing.assert_allclose(
        aligned_reference["cluster_target_mw"],
        aligned_candidate["cluster_target_mw"],
    )
    return aligned_reference, aligned_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the complete paper experiment suite")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("**/metrics_*.json")):
        directory = metrics_path.parent
        if "selected" in directory.parts:
            continue
        split = metrics_path.stem.removeprefix("metrics_")
        metrics = load_json(metrics_path)
        row: dict[str, Any] = {
            "experiment": str(directory.relative_to(root)),
            "group": group_name(str(directory.relative_to(root))),
            "split": split,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value
        history_path = directory / "history.json"
        if history_path.is_file():
            history = load_json(history_path)
            row["training_seconds"] = sum(
                float(record.get("seconds", 0.0)) for record in history.get("train", [])
            )
            row["trained_epochs"] = len(history.get("val", []))
        runs.append(row)
    write_csv(output / "runs.csv", runs)

    grouped: list[dict[str, Any]] = []
    identities = sorted({(row["group"], row["split"]) for row in runs})
    for group, split in identities:
        members = [row for row in runs if row["group"] == group and row["split"] == split]
        record: dict[str, Any] = {"group": group, "split": split, "seeds": len(members)}
        numeric_keys = sorted(
            {
                key
                for row in members
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in {"samples"}
            }
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in members if isinstance(row.get(key), (int, float))]
            if values:
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        grouped.append(record)
    write_csv(output / "groups.csv", grouped)
    with (output / "groups.json").open("w", encoding="utf-8") as stream:
        json.dump(grouped, stream, ensure_ascii=True, indent=2)

    comparisons: list[dict[str, Any]] = []
    final = mean_prediction(root / "final", "test")
    baseline_roots = [
        path
        for category in ("baselines_history", "baselines_weather", "baselines_fusion")
        for path in sorted((root / category).glob("*"))
        if path.is_dir()
    ]
    for baseline_root in baseline_roots:
        baseline = mean_prediction(baseline_root, "test")
        matched_final, matched_baseline = align_predictions(final, baseline)
        final_error = np.abs(
            matched_final["cluster_prediction_mw"]
            - matched_final["cluster_target_mw"]
        ).mean(axis=1)
        baseline_error = np.abs(
            matched_baseline["cluster_prediction_mw"]
            - matched_baseline["cluster_target_mw"]
        ).mean(axis=1)
        delta, low, high = block_bootstrap_difference(
            final_error,
            baseline_error,
            matched_final["origin_unix"],
            args.bootstrap_repeats,
            42,
        )
        comparisons.append(
            {
                "baseline": str(baseline_root.relative_to(root)),
                "final_mae_mw": float(final_error.mean()),
                "baseline_mae_mw": float(baseline_error.mean()),
                "final_minus_baseline_mae_mw": delta,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "block": "UTC_day",
                "bootstrap_repeats": args.bootstrap_repeats,
                "paired_samples": int(final_error.size),
            }
        )
    write_csv(output / "paired_block_bootstrap.csv", comparisons)

    final_error = np.abs(
        final["cluster_prediction_mw"] - final["cluster_target_mw"]
    ).mean(axis=1)
    target_change = np.max(
        np.abs(np.diff(final["cluster_target_mw"], axis=1)), axis=1
    )
    threshold = float(np.quantile(target_change, 0.9))
    rapid = target_change >= threshold
    diagnostic = {
        "definition": "top_10_percent_by_maximum_future_15min_absolute_change",
        "threshold_mw": threshold,
        "samples": int(rapid.sum()),
        "final_mae_mw": float(final_error[rapid].mean()),
        "all_period_final_mae_mw": float(final_error.mean()),
    }
    with (output / "rapid_change_diagnostic.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostic, stream, ensure_ascii=True, indent=2)
    print(
        f"[analysis] runs={len(runs)} groups={len(grouped)} "
        f"comparisons={len(comparisons)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
