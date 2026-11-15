from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_weather_dynamics_v1"
STRICT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
RESULT = PROJECT / "exp_docs" / "aggregate_weather_driven_dynamics_experiment_results.md"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
GROUPS = ("m1", "m2", "m3", "m4a", "m4b", "m5", "m6a", "m6b")
LABELS = {
    "a0_ft": "A0-FT", "m1": "M1 Baseline Headroom",
    "m2": "M2 Anomaly Headroom", "m3": "M3 Recursive Headroom",
    "m4a": "M4a Fixed Equilibrium", "m4b": "M4b Adaptive Equilibrium",
    "m5": "M5 Envelope × Realization", "m6a": "M6a Fixed Error Decay",
    "m6b": "M6b Adaptive Error Decay", "m4_no_recurrence": "M4b without recurrence",
}
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
)


def a0(seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN else "formal_joint"
    return STRICT / stage / "a0_ft" / f"seed{seed}"


def run(group: str, seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN else "formal_joint"
    return ROOT / stage / group / f"seed{seed}"


def load_metrics(path: Path, split: str) -> dict:
    return json.loads((path / f"metrics_{split}.json").read_text(encoding="utf-8"))


def summarize(group: str, seeds, split: str) -> dict:
    paths = [a0(seed) if group == "a0_ft" else run(group, seed) for seed in seeds]
    rows = [load_metrics(path, split) for path in paths]
    result = {"group": group, "split": split, "seeds": list(seeds)}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    result["parameter_count_mean"] = statistics.mean(float(row["parameter_count"]) for row in rows)
    result["latency_ms_per_origin_mean"] = statistics.mean(
        float(row["latency_ms_per_origin"]) for row in rows
    )
    return result


def paired(group: str, seeds, split: str) -> dict:
    values = np.asarray([
        load_metrics(run(group, seed), split)["cluster_mae_mw"]
        - load_metrics(a0(seed), split)["cluster_mae_mw"] for seed in seeds
    ], dtype=np.float64)
    critical = 4.303 if len(values) == 3 else 2.262
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta_mw": float(values.mean()), "wins": int((values < 0).sum()),
        "ci_low_mw": float(values.mean() - half), "ci_high_mw": float(values.mean() + half),
        "differences_mw": values.tolist(),
    }


def discrepancy_diagnostic(seed: int, split: str = "val") -> dict:
    stage = "screen_aux_joint" if seed in SCREEN else "formal_aux_joint"
    path = ROOT / stage / "weather_mean" / f"seed{seed}" / "predictions_val.npz"
    with np.load(path, allow_pickle=False) as archive:
        target = archive["cluster_target_mw"].astype(np.float64)
        future = archive["cluster_prediction_mw"].astype(np.float64)
        current = archive["cluster_history_last_mw"].astype(np.float64)
        current_weather = archive["current_weather_prediction_factor"].reshape(-1) * 6827.01
    e0 = current - current_weather
    residual = target - future
    correlations = []
    for lead in range(24):
        if np.std(e0) < 1e-12 or np.std(residual[:, lead]) < 1e-12:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(e0, residual[:, lead])[0, 1]))
    previous = np.concatenate((current[:, None], target[:, :-1]), axis=1)
    realized_change = target - previous
    target_gap = future - previous
    sign = float(np.mean(np.sign(realized_change) == np.sign(target_gap)))
    magnitude = float(np.corrcoef(np.abs(realized_change).reshape(-1), np.abs(target_gap).reshape(-1))[0, 1])
    return {
        "seed": seed, "lead_correlation": correlations,
        "weather_target_change_sign_agreement": sign,
        "absolute_change_vs_target_gap_correlation": magnitude,
    }


def envelope_diagnostic(seed: int) -> dict:
    path = ROOT / "screen_aux_joint" / "weather_envelope" / f"seed{seed}" / "predictions_val.npz"
    with np.load(path, allow_pickle=False) as archive:
        coverage = float(np.mean(archive["cluster_target_mw"] <= archive["cluster_prediction_mw"]))
    eta_path = ROOT / "realization_cache" / f"seed{seed}" / "train" / "past_realization_factor.npy"
    eta = np.load(eta_path, mmap_mode="r", allow_pickle=False)
    acf = {}
    for lag in (1, 2, 4):
        left = np.asarray(eta[:, :-lag], dtype=np.float64).reshape(-1)
        right = np.asarray(eta[:, lag:], dtype=np.float64).reshape(-1)
        acf[f"lag_{lag * 15}_minutes"] = float(np.corrcoef(left, right)[0, 1])
    return {"seed": seed, "validation_coverage": coverage, "realization_acf": acf}


def table(rows: list[dict], delta: dict[str, dict] | None = None) -> str:
    lines = [
        "| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h | Δ vs A0 / MW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        group = row["group"]
        d = "—" if delta is None or group == "a0_ft" else f"{delta[group]['mean_delta_mw']:+.3f}"
        lines.append(
            f"| {LABELS.get(group, group)} | {row['cluster_mae_mw_mean']:.3f} ± {row['cluster_mae_mw_sd']:.3f} "
            f"| {row['cluster_rmse_mw_mean']:.3f} ± {row['cluster_rmse_mw_sd']:.3f} "
            f"| {row['cluster_r2_mean']:.3f} | {row['cluster_ramp_mae_mw_mean']:.3f} "
            f"| {row['horizon_0_1h_mae_mw_mean']:.3f} | {row['horizon_1_3h_mae_mw_mean']:.3f} "
            f"| {row['horizon_3_6h_mae_mw_mean']:.3f} | {d} |"
        )
    return "\n".join(lines)


def main() -> None:
    selection = json.loads((ROOT / "selection" / "screening_selection.json").read_text())
    freeze = json.loads((ROOT / "selection" / "validation_freeze.json").read_text())
    formal = tuple(selection["formal_candidates"])
    ablations = (
        ("m4_no_recurrence",)
        if all((run("m4_no_recurrence", seed) / "metrics_val.json").is_file() for seed in FORMAL)
        else ()
    )
    screen_rows = [summarize("a0_ft", SCREEN, "val")] + [
        summarize(group, SCREEN, "val") for group in GROUPS
    ]
    screen_delta = {group: paired(group, SCREEN, "val") for group in GROUPS}
    formal_rows = [summarize("a0_ft", FORMAL, "val")] + [
        summarize(group, FORMAL, "val") for group in formal
    ] + [summarize(group, FORMAL, "val") for group in ablations]
    formal_delta = {group: paired(group, FORMAL, "val") for group in (*formal, *ablations)}
    test_rows = [summarize("a0_ft", FORMAL, "test")] + [
        summarize(group, FORMAL, "test") for group in formal
    ] + [summarize(group, FORMAL, "test") for group in ablations]
    test_delta = {group: paired(group, FORMAL, "test") for group in (*formal, *ablations)}
    discrepancy = [discrepancy_diagnostic(seed) for seed in SCREEN]
    envelope = [envelope_diagnostic(seed) for seed in SCREEN]
    summary = {
        "screening": screen_rows, "screening_paired": screen_delta,
        "formal_validation": formal_rows, "formal_paired": formal_delta,
        "frozen_test": test_rows, "frozen_test_paired": test_delta,
        "selection": selection, "freeze": freeze,
        "discrepancy_diagnostic": discrepancy, "envelope_diagnostic": envelope,
    }
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = freeze["selected_final_model"]
    selected_text = LABELS.get(selected, selected)
    doc = f"""# Aggregate 天气驱动功率机制实验结果

## 结论

- 全部结构只在2023 validation上选择，冻结后才统一读取2024。
- 10-seed最终选择为 **{selected_text}**；主表汇报独立checkpoint的mean ± SD，不使用ensemble或最好seed。
- 所有差值均为候选减A0-FT，负值表示结构化模型更好；置信区间只作描述，不是选型门槛。

## 3-seed完整筛选

{table(screen_rows, screen_delta)}

## 10-seed validation正式结果

{table(formal_rows, formal_delta)}

## 冻结后的2024结果

{table(test_rows, test_delta)}

## 机制和数据诊断

- Weather-only当前误差与未来误差逐lead相关系数保存在 `artifacts/aggregate_weather_dynamics_v1/analysis/summary.json`。
- Q0.9 envelope覆盖率及realization factor的15/30/60 min ACF保存在同一文件。
- M3/M4/M5/M6的逐样本signal、equilibrium、rate、envelope和decay诊断随预测归档保存，不把可视化后验替代总体MAE结论。

## 选择记录

- 3-seed正式候选：{', '.join(formal)}。
- 冻结模型：`{selected}`。
- 旧R2-W、旧状态调制结果和242 MW ensemble均未作为本轮性能参照；唯一主参照为同预算A0-FT。
"""
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(doc, encoding="utf-8")
    print(f"[dynamics-analysis] selected={selected} output={RESULT}", flush=True)


if __name__ == "__main__":
    main()
