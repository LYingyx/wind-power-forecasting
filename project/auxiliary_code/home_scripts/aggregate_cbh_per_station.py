"""Extract complete 56-station x 10-seed results from existing audited runs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ART = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts")
ROOT = ART / "cbh_per_station_independent_20260918_v1"
CACHE = Path("/storage/liying/new_try/wind_cluster_forced_evolution/data/model_cache_pad5_f8")
SEEDS = list(range(42, 52))
SPLITS = ["val", "test"]

SOURCES = {
    "trajectory": {
        "root": ART / "trajectory_attribution_20260914_v1",
        "models": ["history", "W", "L", "A", "J", "H", "T", "LP", "G", "JM"],
    },
    "p0i_transfer": {
        "root": ART / "p0i_condition_transfer_20260915_v1",
        "models": ["P0", "PB", "PA", "PJ"],
    },
    "cbh_condition": {
        "root": ART / "p0i_attribution_closure_20260915_v1",
        "models": ["CBH", "CB", "CH", "CNEUTRAL"],
    },
    "summary_fixed": {
        "root": ART / "p0i_summary_calibration_20260916_v1",
        "models": ["SLAST", "SRECENT", "STREND"],
    },
    "summary_learned": {
        "root": ART / "p0i_mlp_final_20260917_v1",
        "models": ["MFLAT", "MPOOL", "HMLP", "BMLP"],
    },
}

COMPARISONS = {
    "trajectory": [
        ("W", "history"), ("L", "W"), ("J", "A"), ("J", "H"),
        ("J", "T"), ("LP", "L"), ("G", "J"), ("JM", "G"),
    ],
    "p0i_transfer": [("PB", "P0"), ("PA", "P0"), ("PJ", "P0"), ("PJ", "PA")],
    "cbh_condition": [("CBH", "CB"), ("CBH", "CH"), ("CBH", "CNEUTRAL")],
    "summary": [
        ("SLAST", "CBH"), ("SRECENT", "CBH"), ("STREND", "CBH"),
        ("MFLAT", "CBH"), ("MPOOL", "CBH"), ("HMLP", "CBH"), ("BMLP", "CBH"),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result.tolist()


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    station_ids = np.load(CACHE / "static/station_id.npy", allow_pickle=True).astype(str).tolist()
    station_names = np.load(CACHE / "static/station_name.npy", allow_pickle=True).astype(str).tolist()
    capacities = np.load(CACHE / "static/station_capacity_mw.npy").astype(float).tolist()
    if len(station_ids) != 56:
        raise RuntimeError(f"Expected 56 stations, found {len(station_ids)}")

    records: list[dict] = []
    source_hashes: dict[str, str] = {}
    arrays: dict[tuple[str, str], np.ndarray] = {}
    group_by_model: dict[str, str] = {}
    overall_checks: list[dict] = []
    for group, specification in SOURCES.items():
        source = specification["root"]
        for model in specification["models"]:
            group_by_model[model] = group
            for split in SPLITS:
                values = np.empty((10, 56), dtype=np.float64)
                values_mw = np.empty_like(values)
                for seed_index, seed in enumerate(SEEDS):
                    path = source / "analysis" / split / f"{model}_seed{seed}" / "metrics.json"
                    if not path.exists():
                        raise RuntimeError(f"Missing {path}")
                    source_hashes[str(path)] = sha256(path)
                    metric = json.loads(path.read_text())
                    cf = np.asarray(metric["per_station_mae_cf"], dtype=np.float64)
                    mw = np.asarray(metric["metrics"]["per_station_mae_mw"], dtype=np.float64)
                    if cf.shape != (56,) or mw.shape != (56,) or not np.isfinite(cf).all():
                        raise RuntimeError(f"Invalid per-station values in {path}")
                    if abs(float(cf.mean()) - float(metric["station_mae_cf_recomputed"])) > 2e-9:
                        raise RuntimeError(f"Station mean mismatch in {path}")
                    if not np.allclose(mw, cf * np.asarray(capacities), atol=3e-5, rtol=1e-7):
                        raise RuntimeError(f"CF/MW mismatch in {path}")
                    values[seed_index] = cf
                    values_mw[seed_index] = mw
                    for station_index, station_id in enumerate(station_ids):
                        records.append({
                            "split": split,
                            "family_group": group,
                            "model": model,
                            "seed": seed,
                            "station_index": station_index,
                            "station_id": station_id,
                            "station_name": station_names[station_index],
                            "capacity_mw": capacities[station_index],
                            "mae_cf": float(cf[station_index]),
                            "mae_mw": float(mw[station_index]),
                            "source_metrics": str(path),
                            "source_metrics_sha256": source_hashes[str(path)],
                        })
                arrays[(split, model)] = values
                overall_checks.append({
                    "split": split,
                    "family_group": group,
                    "model": model,
                    "station_seed_mean_cf": float(values.mean()),
                    "seed_equal_mean_cf": float(values.mean(axis=1).mean()),
                    "station_equal_mean_cf": float(values.mean(axis=0).mean()),
                    "seed_sd_of_station_equal_cf": float(values.mean(axis=1).std(ddof=1)),
                })

    station_summary: list[dict] = []
    for (split, model), values in arrays.items():
        for station_index, station_id in enumerate(station_ids):
            item = values[:, station_index]
            low, high = int(item.argmin()), int(item.argmax())
            ci = bootstrap_ci(item, 20260918 + 1000 * SPLITS.index(split) + 31 * station_index + len(model))
            station_summary.append({
                "split": split,
                "family_group": group_by_model[model],
                "model": model,
                "station_index": station_index,
                "station_id": station_id,
                "station_name": station_names[station_index],
                "capacity_mw": capacities[station_index],
                "seed_mean_mae_cf": float(item.mean()),
                "seed_sd_mae_cf": float(item.std(ddof=1)),
                "seed_ci95_low": ci[0],
                "seed_ci95_high": ci[1],
                "min_mae_cf": float(item[low]),
                "min_seed": SEEDS[low],
                "max_mae_cf": float(item[high]),
                "max_seed": SEEDS[high],
                "seed_mean_mae_mw": float((item * capacities[station_index]).mean()),
            })

    comparison_rows: list[dict] = []
    for comparison_group, pairs in COMPARISONS.items():
        for split in SPLITS:
            for model, control in pairs:
                difference = arrays[(split, model)] - arrays[(split, control)]
                block: list[dict] = []
                p_values: list[float] = []
                for station_index, station_id in enumerate(station_ids):
                    item = difference[:, station_index]
                    try:
                        p_value = float(wilcoxon(item, method="exact").pvalue)
                    except ValueError:
                        p_value = 1.0
                    ci = bootstrap_ci(
                        item,
                        20260918 + 10000 * SPLITS.index(split) + 101 * station_index + len(model + control),
                    )
                    block.append({
                        "split": split,
                        "comparison_group": comparison_group,
                        "model": model,
                        "control": control,
                        "station_index": station_index,
                        "station_id": station_id,
                        "station_name": station_names[station_index],
                        "capacity_mw": capacities[station_index],
                        "delta_mae_cf": float(item.mean()),
                        "relative_percent": float(100 * item.mean() / arrays[(split, control)][:, station_index].mean()),
                        "seed_ci95_low": ci[0],
                        "seed_ci95_high": ci[1],
                        "seed_wins": int((item < 0).sum()),
                        "wilcoxon_p": p_value,
                    })
                    p_values.append(p_value)
                adjusted = holm(p_values)
                for row, value in zip(block, adjusted):
                    row["holm_p_across_56_stations_within_comparison"] = value
                comparison_rows.extend(block)

    # Wide tables are convenient for plotting station heterogeneity.
    wide_rows: list[dict] = []
    models = [model for specification in SOURCES.values() for model in specification["models"]]
    for split in SPLITS:
        for station_index, station_id in enumerate(station_ids):
            row = {
                "split": split,
                "station_index": station_index,
                "station_id": station_id,
                "station_name": station_names[station_index],
                "capacity_mw": capacities[station_index],
            }
            for model in models:
                row[f"{model}_mean_mae_cf"] = float(arrays[(split, model)][:, station_index].mean())
                row[f"{model}_sd_mae_cf"] = float(arrays[(split, model)][:, station_index].std(ddof=1))
            wide_rows.append(row)

    write_csv(ROOT / "analysis/shared_per_station_seed_long.csv", records)
    write_csv(ROOT / "analysis/shared_per_station_summary.csv", station_summary)
    write_csv(ROOT / "analysis/shared_per_station_comparisons.csv", comparison_rows)
    write_csv(ROOT / "analysis/shared_per_station_wide.csv", wide_rows)
    write_csv(ROOT / "analysis/shared_overall_checks.csv", overall_checks)
    audit = {
        "passed": True,
        "splits": SPLITS,
        "stations": 56,
        "seeds": SEEDS,
        "model_families": len(arrays) // len(SPLITS),
        "per_station_seed_records": len(records),
        "expected_records": 2 * 56 * 10 * sum(len(item["models"]) for item in SOURCES.values()),
        "station_summary_records": len(station_summary),
        "comparison_records": len(comparison_rows),
        "source_metric_hashes": source_hashes,
    }
    if audit["per_station_seed_records"] != audit["expected_records"]:
        raise RuntimeError(str(audit))
    dump(ROOT / "status/SHARED_EXTRACTION_AUDIT.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "source_metric_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
