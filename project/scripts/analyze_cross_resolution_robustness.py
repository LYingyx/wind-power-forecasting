from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


GROUPS = ("T0_raw_linear", "T1_latent_linear", "T2_final", "T4_native_query")
SEEDS = tuple(range(42, 52))
DROPS = tuple(f"drop{index}" for index in range(1, 7))


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def mae(run: dict[str, np.ndarray]) -> float:
    return float(np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"]).mean())


def ci(values: np.ndarray) -> tuple[float, float]:
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen-checkpoint node robustness")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = project / "artifacts/cross_resolution_temporal_v2/robustness"
    manifest = json.loads((project / "artifacts/cross_resolution_temporal_v2/formal/run_manifest.json").read_text(encoding="utf-8"))
    native_paths = {(r["group"], int(r["seed"])): Path(r["directory"]) for r in manifest["runs"]}
    selection = json.loads(
        (project / "artifacts/cross_resolution_temporal_v2/temporal_selection.json").read_text(encoding="utf-8")
    )
    for record in selection["runs"]:
        native_paths[("T2_final", int(record["seed"]))] = Path(record["directory"])
    rows = []
    degradation: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUPS:
        degradation[group] = {}
        native = np.asarray([mae(load(native_paths[(group, seed)] / "predictions_val.npz")) for seed in SEEDS])
        r1 = np.asarray([mae(load(root / "groups" / group / f"seed{seed}/r1_2hour/predictions_val.npz")) for seed in SEEDS])
        r2 = np.asarray([
            np.mean([mae(load(root / "groups" / group / f"seed{seed}/{drop}/predictions_val.npz")) for drop in DROPS])
            for seed in SEEDS
        ])
        for scenario, values in (("R1_2hour", r1 - native), ("R2_drop_one", r2 - native)):
            degradation[group][scenario] = values
            low, high = ci(values)
            rows.append({"group": group, "scenario": scenario, "delta_mae_mean_mw": float(values.mean()), "delta_mae_sd_mw": float(values.std(ddof=1)), "ci95_low_mw": low, "ci95_high_mw": high, "seed_deltas_mw": json.dumps(values.tolist())})
    comparison_rows = []
    robust = True
    for scenario in ("R1_2hour", "R2_drop_one"):
        for reference in ("T0_raw_linear", "T1_latent_linear"):
            values = degradation["T2_final"][scenario] - degradation[reference][scenario]
            low, high = ci(values)
            passed = values.mean() <= -1 and high < 0
            robust = robust and bool(passed)
            comparison_rows.append({"scenario": scenario, "comparison": f"T2_minus_{reference}", "degradation_difference_mean_mw": float(values.mean()), "ci95_low_mw": low, "ci95_high_mw": high, "passed": bool(passed), "seed_differences_mw": json.dumps(values.tolist())})
    decision = {"data_split": "2023_validation", "test_2024_read": False, "resolution_robustness_claim_supported": robust, "time_normalized_difference_followup": not robust}
    output = root / "analysis_val"; output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "degradation_summary.csv", rows); write_csv(output / "paired_degradation_comparisons.csv", comparison_rows)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch(); print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
