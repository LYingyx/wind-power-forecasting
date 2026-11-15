from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SEEDS = tuple(range(42, 52))
BANDS = {
    "0.25-1h": np.arange(0, 4),
    "1.25-3h": np.arange(4, 12),
    "3.25-6h": np.arange(12, 24),
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def interval(values: np.ndarray, quantile: float = 0.975) -> tuple[float, float]:
    half = stats.t.ppf(quantile, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze final validation decision before reading 2024")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = project / "artifacts/cross_resolution_temporal_v2/final"
    current_root = project / "artifacts/horizon_specialist_v1/groups/kd_l1_anchor_taper"
    output = root / "analysis_val"
    output.mkdir(parents=True, exist_ok=True)

    archives: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    sources = {
        "final_normal": root / "groups/normal",
        "final_kd": root / "groups/kd",
        "current_kd": current_root,
    }
    rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for label, group_root in sources.items():
        archives[label] = {}
        values = []
        rmse_values = []
        r2_values = []
        station_values = []
        for seed in SEEDS:
            directory = group_root / f"seed{seed}"
            for required in ("COMPLETE", "metrics_val.json", "predictions_val.npz"):
                if not (directory / required).is_file():
                    raise SystemExit(f"Incomplete {label}: {directory / required}")
            run = load(directory / "predictions_val.npz")
            metrics = json.loads((directory / "metrics_val.json").read_text(encoding="utf-8"))
            archives[label][seed] = run
            error = np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"])
            value = float(error.mean())
            values.append(value)
            rmse_values.append(float(metrics["cluster_rmse_mw"]))
            r2_values.append(float(metrics["cluster_r2"]))
            station_values.append(float(metrics["station_mae_mw"]))
            rows.append({
                "group": label, "seed": seed, "cluster_mae_mw": value,
                "cluster_rmse_mw": rmse_values[-1], "cluster_r2": r2_values[-1],
                "station_mae_mw": station_values[-1],
            })
            for band, indices in BANDS.items():
                band_rows.append(
                    {"group": label, "seed": seed, "horizon_band": band, "cluster_mae_mw": float(error[:, indices].mean())}
                )
        array = np.asarray(values)
        low, high = interval(array)
        rows.append(
            {
                "group": label,
                "seed": "mean",
                "cluster_mae_mw": float(array.mean()),
                "sd_mw": float(array.std(ddof=1)),
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "cluster_rmse_mw": float(np.mean(rmse_values)),
                "cluster_r2": float(np.mean(r2_values)),
                "station_mae_mw": float(np.mean(station_values)),
            }
        )

    comparisons = (
        ("final_kd_vs_normal", "final_kd", "final_normal"),
        ("final_kd_vs_current_kd", "final_kd", "current_kd"),
    )
    comparison_rows = []
    paired: dict[str, np.ndarray] = {}
    for name, candidate, reference in comparisons:
        values = []
        for seed in SEEDS:
            cand, ref = archives[candidate][seed], archives[reference][seed]
            np.testing.assert_array_equal(cand["origin_unix"], ref["origin_unix"])
            np.testing.assert_array_equal(cand["cluster_target_mw"], ref["cluster_target_mw"])
            cand_error = np.abs(cand["cluster_prediction_mw"] - cand["cluster_target_mw"])
            ref_error = np.abs(ref["cluster_prediction_mw"] - ref["cluster_target_mw"])
            values.append(float((cand_error - ref_error).mean()))
        array = np.asarray(values)
        paired[name] = array
        low, high = interval(array)
        comparison_rows.append(
            {
                "comparison": name,
                "candidate_minus_reference_mean_mw": float(array.mean()),
                "sd_mw": float(array.std(ddof=1)),
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "candidate_wins": int(np.sum(array < 0)),
                "p_value_two_sided": float(stats.ttest_1samp(array, 0.0).pvalue),
                "seed_differences_mw": json.dumps(values),
            }
        )
    current_values = paired["final_kd_vs_current_kd"]
    noninferior_upper = float(
        current_values.mean()
        + stats.t.ppf(0.95, len(current_values) - 1)
        * current_values.std(ddof=1)
        / np.sqrt(len(current_values))
    )
    kd_values = np.asarray(
        [
            np.abs(archives["final_kd"][seed]["cluster_prediction_mw"] - archives["final_kd"][seed]["cluster_target_mw"]).mean()
            for seed in SEEDS
        ]
    )
    median = float(np.median(kd_values))
    representative_seed = int(SEEDS[int(np.argmin(np.abs(kd_values - median)))])
    decision = {
        "selection_split": "2023_validation",
        "test_2024_read": False,
        "single_checkpoint_statistics": True,
        "ensemble_role": "training_teacher_only",
        "final_kd_vs_current_kd_margin_mw": 2.0,
        "final_kd_vs_current_kd_one_sided_ci95_upper_mw": noninferior_upper,
        "final_kd_noninferior_current_kd": bool(noninferior_upper < 2.0),
        "representative_rule": "validation_mae_closest_to_10seed_median",
        "representative_seed": representative_seed,
        "representative_validation_mae_mw": float(kd_values[representative_seed - 42]),
        "median_validation_mae_mw": median,
        "architecture_and_representative_frozen_for_2024": True,
    }
    write_csv(output / "single_checkpoint_summary.csv", rows)
    write_csv(output / "paired_comparisons.csv", comparison_rows)
    write_csv(output / "per_horizon_band.csv", band_rows)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "VALIDATION_SELECTION_FROZEN").touch()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
