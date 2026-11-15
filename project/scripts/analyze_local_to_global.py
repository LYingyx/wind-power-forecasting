from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_to_global_raw_bilinear_v1"
A0 = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
SEEDS = tuple(range(42, 52))
SCREEN = (42, 46, 50)
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
    "station_mae_mw", "samples_per_second", "parameter_count",
)


def run_metric(group: str, seed: int, split: str) -> dict[str, Any]:
    return json.loads(
        (ROOT / "formal" / group / f"seed{seed}" / f"metrics_{split}.json").read_text(
            encoding="utf-8"
        )
    )


def a0_metric(seed: int, split: str) -> dict[str, Any]:
    stage = "screen_joint" if seed in SCREEN else "formal_joint"
    return json.loads(
        (A0 / stage / "a0_ft" / f"seed{seed}" / f"metrics_{split}.json").read_text(
            encoding="utf-8"
        )
    )


def summarize(group: str, split: str) -> dict[str, Any]:
    rows = [a0_metric(seed, split) if group == "a0_ft" else run_metric(group, seed, split) for seed in SEEDS]
    result: dict[str, Any] = {"group": group, "split": split, "seeds": list(SEEDS)}
    for key in METRICS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def paired(candidate: str, reference: str, split: str) -> dict[str, Any]:
    differences = []
    for seed in SEEDS:
        c = run_metric(candidate, seed, split)
        r = a0_metric(seed, split) if reference == "a0_ft" else run_metric(reference, seed, split)
        differences.append(float(c["cluster_mae_mw"]) - float(r["cluster_mae_mw"]))
    values = np.asarray(differences)
    return {
        "candidate": candidate, "reference": reference, "split": split,
        "mean_delta_mw": float(values.mean()),
        "sd_delta_mw": float(values.std(ddof=1)),
        "wins": int((values < 0).sum()),
        "differences_mw": values.tolist(),
    }


def intervention(group: str, name: str, split: str) -> dict[str, Any]:
    differences = []
    for seed in SEEDS:
        base = run_metric(group, seed, split)
        changed = json.loads(
            (ROOT / "interventions" / group / name / f"seed{seed}" / f"metrics_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        differences.append(float(changed["cluster_mae_mw"]) - float(base["cluster_mae_mw"]))
    return {
        "intervention": name, "split": split,
        "mean_degradation_mw": statistics.mean(differences),
        "sd_mw": statistics.stdev(differences), "values_mw": differences,
    }


def fmt(row: dict[str, Any], key: str) -> str:
    return f"{row[key + '_mean']:.3f} ± {row[key + '_sd']:.3f}"


def main() -> None:
    freeze = json.loads((ROOT / "selection" / "validation_freeze.json").read_text(encoding="utf-8"))
    screening = json.loads((ROOT / "selection" / "screening.json").read_text(encoding="utf-8"))
    groups = list(freeze["formal_means"])
    local_best = str(screening["local_best"])
    val = [summarize("a0_ft", "val"), *[summarize(group, "val") for group in groups]]
    test = [summarize("a0_ft", "test"), *[summarize(group, "test") for group in groups]]
    comparisons = []
    for split in ("val", "test"):
        comparisons.extend(paired(group, "a0_ft", split) for group in groups)
        comparisons.append(paired(local_best, "l1_mft", split))
        comparisons.append(paired(local_best, "l2_agg", split))
    interventions = [
        intervention(local_best, name, split)
        for split in ("val", "test")
        for name in ("station_roll", "batch_roll", "reference")
    ]
    summary = {
        "screening": screening, "validation_freeze": freeze,
        "validation": val, "test": test,
        "comparisons": comparisons, "interventions": interventions,
    }
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    labels = {
        "a0_ft": "A0-FT aggregate", "l1_mft": "L1 local no-state",
        "l2_agg": "L2 aggregate-state local", "l3_local": "L3 local-state shared",
        "l4_id16": "L4 local-state + ID16", "l5_affine": "L5 local-state + affine",
    }
    def table(rows: Sequence[dict[str, Any]]) -> str:
        lines = [
            "| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {labels[row['group']]} | {fmt(row, 'cluster_mae_mw')} | "
                f"{fmt(row, 'cluster_rmse_mw')} | {fmt(row, 'cluster_r2')} | "
                f"{fmt(row, 'cluster_ramp_mae_mw')} | {fmt(row, 'horizon_0_1h_mae_mw')} | "
                f"{fmt(row, 'horizon_1_3h_mae_mw')} | {fmt(row, 'horizon_3_6h_mae_mw')} |"
            )
        return "\n".join(lines)
    document = f"""# Raw-Bilinear Local-to-Global 实验结果

状态：全部训练完成；模型按2023 validation的10-seed独立checkpoint平均MAE冻结为 **{labels[str(freeze['selected'])]}**，随后一次性读取2024。

## 2023 validation

{table(val)}

## 冻结后的2024

{table(test)}

## 机制比较

正式比较和状态错配干预的机器可读结果位于 `artifacts/local_to_global_raw_bilinear_v1/analysis/summary.json`。负的candidate-reference差值表示候选更好；正的intervention degradation说明模型使用了正确的局地对应关系。

本实验使用原始14变量先做固定2×2双线性插值，再由共享point encoder进行局地天气—功率转换；没有learned interpolation、GNN、KD或预测ensemble。
"""
    (PROJECT / "exp_docs" / "local_to_global_raw_bilinear_experiment_results.md").write_text(
        document, encoding="utf-8"
    )
    print(f"[analysis] selected={freeze['selected']}", flush=True)


if __name__ == "__main__":
    main()
