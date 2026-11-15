from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_state_operator_v1"
PRIOR = PROJECT / "artifacts" / "local_to_global_raw_bilinear_v1"
SEEDS = tuple(range(42, 52))
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
    "station_mae_mw", "samples_per_second", "parameter_count",
)
LABELS = {
    "l1": "L1 no-state",
    "g_rmul": "G-RMul (existing L2)",
    "l_rmul": "L-RMul (existing L3)",
    "g_int": "G-Int",
    "l_int": "L-Int",
    "g_cat": "G-Cat",
    "l_cat": "L-Cat",
}
CONTROL_PATHS = {
    "l1": "l1_mft",
    "g_rmul": "l2_agg",
    "l_rmul": "l3_local",
}


def run_path(group: str, seed: int) -> Path:
    if group in CONTROL_PATHS:
        return PRIOR / "formal" / CONTROL_PATHS[group] / f"seed{seed}"
    return ROOT / "formal" / group / f"seed{seed}"


def metric(group: str, seed: int, split: str) -> dict[str, Any]:
    return json.loads(
        (run_path(group, seed) / f"metrics_{split}.json").read_text(encoding="utf-8")
    )


def intervention_names(group: str) -> tuple[str, ...]:
    return (
        ("batch_roll", "reference")
        if group.startswith("g_")
        else ("station_roll", "batch_roll", "reference")
        if group.startswith("l_") and group != "l1"
        else ()
    )


def intervention_metric(group: str, name: str, seed: int, split: str) -> dict[str, Any]:
    path = (
        PRIOR / "interventions" / "l3_local" / name / f"seed{seed}" / f"metrics_{split}.json"
        if group == "l_rmul"
        else ROOT / "interventions" / "formal" / group / name / f"seed{seed}" / f"metrics_{split}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(group: str, split: str) -> dict[str, Any]:
    rows = [metric(group, seed, split) for seed in SEEDS]
    result: dict[str, Any] = {"group": group, "split": split, "seeds": list(SEEDS)}
    for key in METRICS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def paired(candidate: str, reference: str, split: str) -> dict[str, Any]:
    differences = [
        float(metric(candidate, seed, split)["cluster_mae_mw"])
        - float(metric(reference, seed, split)["cluster_mae_mw"])
        for seed in SEEDS
    ]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = 2.262 * sd / math.sqrt(len(differences))
    horizon = {}
    for key in (
        "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw"
    ):
        horizon[key] = statistics.mean(
            float(metric(candidate, seed, split)[key])
            - float(metric(reference, seed, split)[key])
            for seed in SEEDS
        )
    return {
        "candidate": candidate,
        "reference": reference,
        "split": split,
        "mean_delta_mw": mean,
        "sd_delta_mw": sd,
        "ci_low_mw": mean - half,
        "ci_high_mw": mean + half,
        "wins": sum(value < 0.0 for value in differences),
        "differences_mw": differences,
        "horizon_delta_mw": horizon,
    }


def intervention(group: str, name: str, split: str) -> dict[str, Any]:
    differences = [
        float(intervention_metric(group, name, seed, split)["cluster_mae_mw"])
        - float(metric(group, seed, split)["cluster_mae_mw"])
        for seed in SEEDS
    ]
    return {
        "group": group,
        "intervention": name,
        "split": split,
        "mean_degradation_mw": statistics.mean(differences),
        "sd_mw": statistics.stdev(differences),
        "positive_seeds": sum(value > 0.0 for value in differences),
        "values_mw": differences,
    }


def fmt(row: dict[str, Any], key: str) -> str:
    return f"{row[key + '_mean']:.3f} ± {row[key + '_sd']:.3f}"


def result_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Model | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {LABELS[row['group']]} | {fmt(row, 'cluster_mae_mw')} | "
            f"{fmt(row, 'cluster_rmse_mw')} | {fmt(row, 'cluster_r2')} | "
            f"{fmt(row, 'cluster_ramp_mae_mw')} | {fmt(row, 'horizon_0_1h_mae_mw')} | "
            f"{fmt(row, 'horizon_1_3h_mae_mw')} | {fmt(row, 'horizon_3_6h_mae_mw')} |"
        )
    return "\n".join(lines)


def main() -> None:
    screening = json.loads((ROOT / "selection" / "screening.json").read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "selection" / "validation_freeze.json").read_text(encoding="utf-8"))
    groups = list(freeze["formal_means"])
    validation = [summarize(group, "val") for group in groups]
    test = [summarize(group, "test") for group in groups]
    comparisons = []
    for split in ("val", "test"):
        for group in groups:
            if group in {"l1", "g_rmul", "l_rmul"}:
                continue
            reference = "g_rmul" if group.startswith("g_") else "l1"
            comparisons.append(paired(group, reference, split))
        comparisons.append(paired("g_rmul", "l1", split))
        comparisons.append(paired("l_rmul", "l1", split))
        comparisons.append(paired("l_rmul", "g_rmul", split))
    state_groups = [group for group in groups if group != "l1"]
    interventions = [
        intervention(group, name, split)
        for split in ("val", "test")
        for group in state_groups
        for name in intervention_names(group)
    ]
    summary = {
        "screening": screening,
        "validation_freeze": freeze,
        "validation": validation,
        "test": test,
        "comparisons": comparisons,
        "interventions": interventions,
        "zero_weather_invariant_by_design": {
            group: group not in {"g_cat", "l_cat"} for group in groups
        },
    }
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    screen_lines = [
        "| Candidate | Reference | ΔMAE / MW | wins | state-used | local correspondence | safety | promoted |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("g_int", "l_int", "g_cat", "l_cat"):
        row = screening["candidates"][group]
        screen_lines.append(
            f"| {LABELS[group]} | {LABELS[row['reference']]} | "
            f"{row['mean_mae_delta_mw']:+.3f} | {row['mae_wins']}/3 | "
            f"{row['sample_state_used']} | {row['local_correspondence']} | "
            f"{row['safety_passed']} | {row['promoted']} |"
        )

    comparison_lines = [
        "| Split | Candidate − reference | ΔMAE / MW | 95% paired CI | wins |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        comparison_lines.append(
            f"| {row['split']} | {LABELS[row['candidate']]} − {LABELS[row['reference']]} | "
            f"{row['mean_delta_mw']:+.3f} | [{row['ci_low_mw']:+.3f}, {row['ci_high_mw']:+.3f}] | "
            f"{row['wins']}/10 |"
        )

    intervention_lines = [
        "| Split | Model | Intervention | degradation / MW | positive seeds |",
        "|---|---|---|---:|---:|",
    ]
    for row in interventions:
        intervention_lines.append(
            f"| {row['split']} | {LABELS[row['group']]} | {row['intervention']} | "
            f"{row['mean_degradation_mw']:+.3f} ± {row['sd_mw']:.3f} | "
            f"{row['positive_seeds']}/10 |"
        )

    promoted = screening["promoted"]
    document = f"""# Local State × Interaction Operator 实验结果

状态：全部完成。现有 L2/L3 被正确归类为 residual multiplicative；本轮没有训练重复的纯乘法 gate，也没有加入 station embedding、adapter 或 GNN。

2023 validation 冻结的 zero-invariant 模型为 **{LABELS[freeze['selected_zero_invariant']]}**。晋级的新候选：`{promoted}`。Cat 诊断选择：`{freeze['selected_cat_diagnostic']}`。候选的 2024 只在冻结以后评估；该年份此前已用于旧实验报告，因此不表述为从未查看的纯盲测试。

## 3-seed screening

{chr(10).join(screen_lines)}

门槛固定为：ΔMAE≤−0.5 MW且至少2/3获胜；batch/reference至少一种 degradation≥0.5 MW且至少2/3为正；RMSE恶化≤3 MW，任一时效恶化≤5 MW。local correspondence另要求station-roll degradation≥0.25 MW且至少2/3为正。

## 2023 validation（10 seeds）

{result_table(validation)}

## 冻结后的2024（10 seeds）

{result_table(test)}

## 配对比较

负的candidate-reference差值表示候选更好。

{chr(10).join(comparison_lines)}

## 状态干预

正 degradation 表示正确状态对应关系被模型利用。

{chr(10).join(intervention_lines)}

## 解释规则

- L-Int 优于L1且状态干预稳定为正：支持“此前失败来自operator mismatch”。
- L-Cat有效而L-Int无效：收益更可能来自state-only predictive path，不能称为weather-response modulation。
- L-Int/L-Cat均无效或干预接近零：停止local-state modulation。
- G-Int未超过G-RMul：保留简单的aggregate residual gain。
- Cat明确不是zero-weather-invariant主模型；即使数值更低也只作为表达能力上界。

完整机器结果：`artifacts/local_state_operator_v1/analysis/summary.json`。
"""
    (PROJECT / "exp_docs" / "local_state_operator_experiment_results.md").write_text(
        document, encoding="utf-8"
    )
    print(
        f"[analysis] selected={freeze['selected_zero_invariant']} promoted={promoted}",
        flush=True,
    )


if __name__ == "__main__":
    main()
