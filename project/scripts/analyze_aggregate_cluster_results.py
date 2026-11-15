from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REGIMES = ("current_power", "recent_1h_change", "history_variability", "future_wind_variation")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def wind_variation(cache: Path, split: str, density: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    weather = np.load(cache / split / "weather.npy", mmap_mode="r", allow_pickle=False)
    cycle_values = np.empty(len(weather), dtype=np.float32)
    for start in range(0, len(weather), 256):
        packed = np.asarray(weather[start : start + 256, :, [1, 2]], dtype=np.float32)
        u = packed[:, :, 0] * std[1] + mean[1]
        v = packed[:, :, 1] * std[2] + mean[2]
        speed = np.sqrt(u * u + v * v)
        regional = np.einsum("cthw,hw->ct", speed, density, optimize=True)
        cycle_values[start : start + len(packed)] = np.ptp(regional, axis=1)
    sample_cycle = np.load(cache / split / "weather_cycle_index.npy", mmap_mode="r", allow_pickle=False)
    return cycle_values[np.asarray(sample_cycle, dtype=np.int64)]


def features(cache: Path, split: str, density: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    history = np.asarray(
        np.load(cache / split / "cluster_history_mw.npy", mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
    )
    return {
        "current_power": history[:, -1],
        "recent_1h_change": np.abs(history[:, -1] - history[:, -5]),
        "history_variability": np.std(np.diff(history, axis=1), axis=1),
        "future_wind_variation": wind_variation(cache, split, density, mean, std),
    }


def labels(values: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    return np.digitize(values, np.asarray(cuts, dtype=np.float64), right=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate-only regime analysis")
    parser.add_argument("--root", default="artifacts/aggregate_cluster_v1")
    parser.add_argument("--cache", default="data/aggregate_cluster_v1")
    parser.add_argument("--manifest", default="artifacts/aggregate_cluster_v1/frozen_test_manifest.json")
    parser.add_argument("--output", default="artifacts/aggregate_cluster_v1/analysis/regimes")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cache = Path(args.cache).resolve()
    output = Path(args.output).resolve()
    frozen = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    methods = [tuple(item) for item in frozen["methods"]]
    seeds = [int(seed) for seed in frozen["seeds"]]
    density = np.asarray(np.load(cache / "static" / "capacity_density.npy"), dtype=np.float32)
    mean = np.asarray(np.load(cache / "static" / "nwp_mean.npy"), dtype=np.float32)
    std = np.asarray(np.load(cache / "static" / "nwp_std.npy"), dtype=np.float32)
    train = features(cache, "train", density, mean, std)
    thresholds = {
        name: [float(item) for item in np.quantile(train[name], (1 / 3, 2 / 3))]
        for name in REGIMES
    }
    atomic_json(
        output / "thresholds.json",
        {
            "source_split": "train",
            "definitions": {
                "current_power": "latest aggregate power (MW)",
                "recent_1h_change": "absolute latest-minus-one-hour-ago power change (MW)",
                "history_variability": "SD of 15-min increments over the full 6-h history (MW)",
                "future_wind_variation": "range across 8 NWP nodes of capacity-density pooled 10-m wind speed (m/s)",
            },
            "tertile_cuts": thresholds,
        },
    )
    cache_index = {
        split: np.asarray(np.load(cache / split / "sample_index.npy"), dtype=np.int64)
        for split in ("val", "test")
    }
    split_features = {
        split: features(cache, split, density, mean, std)
        for split in ("val", "test")
    }
    rows: list[dict[str, Any]] = []
    names = ("low", "medium", "high")
    for split in ("val", "test"):
        index_lookup = {int(value): index for index, value in enumerate(cache_index[split])}
        regime_labels = {
            name: labels(split_features[split][name], thresholds[name]) for name in REGIMES
        }
        for stage, group in methods:
            for seed in seeds:
                archive_path = root / stage / group / f"seed{seed}" / f"predictions_{split}.npz"
                with np.load(archive_path, allow_pickle=False) as archive:
                    prediction = np.asarray(archive["cluster_prediction_mw"], dtype=np.float64)
                    target = np.asarray(archive["cluster_target_mw"], dtype=np.float64)
                    indices = np.asarray(archive["sample_index"], dtype=np.int64)
                positions = np.asarray([index_lookup[int(value)] for value in indices], dtype=np.int64)
                sample_mae = np.abs(prediction - target).mean(axis=1)
                for regime in REGIMES:
                    assigned = regime_labels[regime][positions]
                    for level, level_name in enumerate(names):
                        selected = assigned == level
                        rows.append(
                            {
                                "split": split, "stage": stage, "group": group, "seed": seed,
                                "regime": regime, "level": level_name,
                                "samples": int(selected.sum()),
                                "cluster_mae_mw": float(sample_mae[selected].mean()),
                            }
                        )
    write_csv(output / "seedwise.csv", rows)
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["split"], row["stage"], row["group"], row["regime"], row["level"])
        grouped.setdefault(key, []).append(float(row["cluster_mae_mw"]))
    summaries = []
    for key, observed in sorted(grouped.items()):
        split, stage, group, regime, level = key
        summaries.append(
            {
                "split": split, "stage": stage, "group": group,
                "regime": regime, "level": level, "seeds": len(observed),
                "cluster_mae_mw_mean": float(np.mean(observed)),
                "cluster_mae_mw_sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
            }
        )
    write_csv(output / "summary.csv", summaries)
    print(f"[aggregate-regimes] rows={len(rows)} summaries={len(summaries)} output={output}", flush=True)


if __name__ == "__main__":
    main()
