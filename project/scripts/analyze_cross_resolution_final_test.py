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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def ci(values: np.ndarray) -> tuple[float, float]:
    half = stats.t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()-half), float(values.mean()+half)


def one_sided_upper(values: np.ndarray, confidence: float = 0.95) -> float:
    return float(
        values.mean()
        + stats.t.ppf(confidence, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the single frozen 2024 evaluation")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = project / "artifacts/cross_resolution_temporal_v2/final"
    validation = json.loads((root / "analysis_val/decision.json").read_text(encoding="utf-8"))
    if not validation.get("architecture_and_representative_frozen_for_2024"):
        raise SystemExit("Validation selection was not frozen")
    representative = int(validation["representative_seed"])
    sources = {
        "final_normal": root / "groups/normal",
        "final_kd": root / "groups/kd",
        "current_kd": project / "artifacts/horizon_specialist_v1/groups/kd_l1_anchor_taper",
    }
    archives: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    metric_records: dict[str, dict[int, dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for label, group_root in sources.items():
        archives[label] = {}
        metric_records[label] = {}
        values_by_metric = {name: [] for name in ("cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "station_mae_mw")}
        for seed in SEEDS:
            directory=group_root/f"seed{seed}"
            for name in ("TEST_COMPLETE", "metrics_test.json", "predictions_test.npz"):
                if not (directory/name).is_file(): raise SystemExit(f"Incomplete test artifact: {directory/name}")
            metrics=json.loads((directory/"metrics_test.json").read_text())
            run=load(directory/"predictions_test.npz"); archives[label][seed]=run; metric_records[label][seed]=metrics
            error=np.abs(run["cluster_prediction_mw"]-run["cluster_target_mw"])
            mae=float(error.mean())
            values_by_metric["cluster_mae_mw"].append(mae)
            for metric_name in ("cluster_rmse_mw", "cluster_r2", "station_mae_mw"):
                values_by_metric[metric_name].append(float(metrics[metric_name]))
            summary_rows.append({
                "group":label,"seed":seed,"cluster_mae_mw":mae,
                "cluster_rmse_mw":float(metrics["cluster_rmse_mw"]),
                "cluster_r2":float(metrics["cluster_r2"]),
                "station_mae_mw":float(metrics["station_mae_mw"]),
            })
            for band,indices in BANDS.items():
                band_rows.append({"group":label,"seed":seed,"horizon_band":band,"cluster_mae_mw":float(error[:,indices].mean())})
        mae_values=np.asarray(values_by_metric["cluster_mae_mw"]); low,high=ci(mae_values)
        summary_rows.append({
            "group":label,"seed":"mean","cluster_mae_mw":float(mae_values.mean()),
            "cluster_mae_sd_mw":float(mae_values.std(ddof=1)),"cluster_mae_ci95_low_mw":low,"cluster_mae_ci95_high_mw":high,
            "cluster_rmse_mw":float(np.mean(values_by_metric["cluster_rmse_mw"])),
            "cluster_rmse_sd_mw":float(np.std(values_by_metric["cluster_rmse_mw"],ddof=1)),
            "cluster_r2":float(np.mean(values_by_metric["cluster_r2"])),
            "cluster_r2_sd":float(np.std(values_by_metric["cluster_r2"],ddof=1)),
            "station_mae_mw":float(np.mean(values_by_metric["station_mae_mw"])),
            "station_mae_sd_mw":float(np.std(values_by_metric["station_mae_mw"],ddof=1)),
        })
    comparisons=[]
    paired: dict[str, np.ndarray] = {}
    for name,candidate,reference in (
        ("final_kd_vs_normal","final_kd","final_normal"),
        ("final_kd_vs_current_kd","final_kd","current_kd"),
    ):
        deltas=[]
        for seed in SEEDS:
            c,r=archives[candidate][seed],archives[reference][seed]
            np.testing.assert_array_equal(c["origin_unix"],r["origin_unix"])
            np.testing.assert_array_equal(c["cluster_target_mw"],r["cluster_target_mw"])
            ce=np.abs(c["cluster_prediction_mw"]-c["cluster_target_mw"])
            re=np.abs(r["cluster_prediction_mw"]-r["cluster_target_mw"])
            deltas.append(float((ce-re).mean()))
        values=np.asarray(deltas); low,high=ci(values)
        paired[name] = values
        comparisons.append({"comparison":name,"candidate_minus_reference_mean_mw":float(values.mean()),"sd_mw":float(values.std(ddof=1)),"ci95_low_mw":low,"ci95_high_mw":high,"candidate_wins":int(np.sum(values<0)),"p_value_two_sided":float(stats.ttest_1samp(values,0).pvalue),"seed_differences_mw":json.dumps(deltas)})
    representative_rows=[]
    for label in sources:
        run=archives[label][representative]
        metrics=metric_records[label][representative]
        representative_rows.append({
            "group":label,"representative_seed":representative,
            "cluster_mae_mw":float(np.abs(run["cluster_prediction_mw"]-run["cluster_target_mw"]).mean()),
            "cluster_rmse_mw":float(metrics["cluster_rmse_mw"]),
            "cluster_r2":float(metrics["cluster_r2"]),
            "station_mae_mw":float(metrics["station_mae_mw"]),
        })
    output=root/"analysis_test"; output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"single_checkpoint_summary.csv",summary_rows)
    write_csv(output/"paired_comparisons.csv",comparisons)
    write_csv(output/"per_horizon_band.csv",band_rows)
    write_csv(output/"representative_checkpoint.csv",representative_rows)
    margin_mw = 2.0
    current_kd_upper = one_sided_upper(paired["final_kd_vs_current_kd"])
    decision={
        "test_split":"2024_internal_test",
        "single_frozen_evaluation":True,
        "single_checkpoint_statistics":True,
        "ensemble_performance_reported":False,
        "representative_seed":representative,
        "final_kd_vs_current_kd_margin_mw":margin_mw,
        "final_kd_vs_current_kd_one_sided_ci95_upper_mw":current_kd_upper,
        "final_kd_noninferior_current_kd":bool(current_kd_upper < margin_mw),
    }
    (output/"decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"ANALYSIS_COMPLETE").touch(); print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
