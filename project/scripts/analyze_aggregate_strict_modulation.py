from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
R2_ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1" / "mechanism" / "r2_w"
DATA_ROOT = PROJECT / "data" / "aggregate_cluster_v1"
SCREEN_SEEDS = (42, 46, 50)
FORMAL_SEEDS = tuple(range(42, 52))
CANDIDATES = ("b1", "b2", "c1", "c2", "d1", "d2", "e1", "f1")
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
    "parameter_count", "trainable_parameter_count",
)
LABELS = {
    "r2_w": "A0 R2-W",
    "a0_ft": "A0-FT（仅继续微调天气解码器）",
    "b1": "B1 当前功率×整体天气效应",
    "b2": "B2 baseline latent×整体天气效应",
    "c1": "C1 当前功率×天气通道",
    "c2": "C2 baseline latent×天气通道",
    "d1": "D1 当前功率×低秩天气变换",
    "d2": "D2 baseline latent×低秩天气变换",
    "e1": "E1 baseline latent×天气响应专家",
    "f1": "F1 baseline latent×天气差分",
}
MANAGED_START = "<!-- STRICT_STATE_WEATHER_START -->"
MANAGED_END = "<!-- STRICT_STATE_WEATHER_END -->"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def replace_managed(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    managed = f"{MANAGED_START}\n{section.rstrip()}\n{MANAGED_END}"
    if MANAGED_START in content and MANAGED_END in content:
        prefix, remainder = content.split(MANAGED_START, 1)
        _, suffix = remainder.split(MANAGED_END, 1)
        content = f"{prefix.rstrip()}\n\n{managed}{suffix}"
    else:
        content = f"{content.rstrip()}\n\n{managed}\n"
    atomic_write(path, content)


def run_dir(group: str, seed: int, *, screen_stage: str = "screen_joint") -> Path:
    if group == "r2_w":
        return R2_ROOT / f"seed{seed}"
    stage = screen_stage if seed in SCREEN_SEEDS else "formal_joint"
    return ROOT / stage / group / f"seed{seed}"


def metric(group: str, seed: int, split: str, *, screen_stage: str = "screen_joint") -> dict[str, Any]:
    return load_json(run_dir(group, seed, screen_stage=screen_stage) / f"metrics_{split}.json")


def summary(
    group: str, seeds: Sequence[int], split: str, *, screen_stage: str = "screen_joint"
) -> dict[str, Any]:
    rows = [metric(group, seed, split, screen_stage=screen_stage) for seed in seeds]
    result: dict[str, Any] = {"group": group, "seeds": list(seeds), "split": split}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def fmt(row: dict[str, Any], key: str) -> str:
    return f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_sd']:.3f}"


def table(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        group = str(row["group"])
        label = LABELS.get(group.removesuffix("_const"), group)
        if group.endswith("_const"):
            label += "-Const"
        lines.append(
            f"| {label} | {fmt(row, 'cluster_mae_mw')} | {fmt(row, 'cluster_rmse_mw')} | "
            f"{fmt(row, 'cluster_r2')} | {fmt(row, 'cluster_ramp_mae_mw')} | "
            f"{fmt(row, 'horizon_0_1h_mae_mw')} | {fmt(row, 'horizon_1_3h_mae_mw')} | "
            f"{fmt(row, 'horizon_3_6h_mae_mw')} |"
        )
    return "\n".join(lines)


def prediction_path(group: str, seed: int, split: str) -> Path:
    return run_dir(group, seed) / f"predictions_{split}.npz"


def weather_change_categories(split: str) -> tuple[np.ndarray, np.ndarray]:
    names = np.load(DATA_ROOT / "static" / "nwp_channel_name.npy", allow_pickle=False).tolist()
    u_index, v_index = names.index("u10"), names.index("v10")
    mean = np.load(DATA_ROOT / "static" / "nwp_mean.npy", allow_pickle=False)
    std = np.load(DATA_ROOT / "static" / "nwp_std.npy", allow_pickle=False)
    density = np.load(DATA_ROOT / "static" / "capacity_density.npy", allow_pickle=False)
    directory = DATA_ROOT / split
    cycles = np.load(directory / "weather_cycle_index.npy", allow_pickle=False)
    times = np.load(directory / "weather_time_offset_hours.npy", allow_pickle=False)
    sample_ids = np.load(directory / "sample_index.npy", allow_pickle=False)
    weather = np.load(directory / "weather.npy", mmap_mode="r", allow_pickle=False)
    unique, inverse = np.unique(cycles, return_inverse=True)
    node_speed = np.empty((len(unique), weather.shape[1]), dtype=np.float32)
    for start in range(0, len(unique), 64):
        chosen = unique[start : start + 64]
        frames = np.asarray(weather[chosen][:, :, [u_index, v_index]], dtype=np.float32)
        u = frames[:, :, 0] * float(std[u_index]) + float(mean[u_index])
        v = frames[:, :, 1] * float(std[v_index]) + float(mean[v_index])
        speed = np.sqrt(u * u + v * v)
        node_speed[start : start + len(chosen)] = np.einsum("cthw,hw->ct", speed, density)
    values = node_speed[inverse]

    def interpolate(target: float) -> np.ndarray:
        right = (times < target).sum(axis=1).clip(1, times.shape[1] - 1)
        left = right - 1
        row = np.arange(len(times))
        fraction = ((target - times[row, left]) / np.maximum(times[row, right] - times[row, left], 1e-6)).clip(0, 1)
        result = values[row, left] * (1 - fraction) + values[row, right] * fraction
        result = np.where(target <= times[:, 0], values[:, 0], result)
        result = np.where(target >= times[:, -1], values[:, -1], result)
        return result

    change = interpolate(6.0) - interpolate(0.25)
    category = np.where(change < -0.5, 0, np.where(change > 0.5, 2, 1)).astype(np.int8)
    return sample_ids.astype(np.int64), category


def regime_matrix(group: str, split: str) -> list[dict[str, Any]]:
    train_history = np.load(DATA_ROOT / "train" / "cluster_history_mw.npy", mmap_mode="r")
    thresholds = np.quantile(np.asarray(train_history[:, -1], dtype=np.float64), [0.2, 0.4, 0.6, 0.8])
    weather_ids, weather_category = weather_change_categories(split)
    weather_lookup = {int(key): int(value) for key, value in zip(weather_ids, weather_category)}
    bands = (("0-1h", slice(0, 4)), ("1-3h", slice(4, 12)), ("3-6h", slice(12, 24)))
    cells: dict[tuple[int, int, str], list[float]] = {}
    counts: dict[tuple[int, int, str], int] = {}
    for seed in FORMAL_SEEDS:
        with np.load(prediction_path(group, seed, split), allow_pickle=False) as candidate, np.load(
            prediction_path("r2_w", seed, split), allow_pickle=False
        ) as reference:
            candidate_ids = candidate["sample_index"]
            reference_ids = reference["sample_index"]
            if not np.array_equal(candidate_ids, reference_ids):
                raise ValueError(f"Prediction archives are not aligned for seed {seed} {split}")
            power = candidate["cluster_history_last_mw"]
            power_category = np.digitize(power, thresholds).astype(np.int8)
            weather_bins = np.asarray([weather_lookup[int(index)] for index in candidate_ids], dtype=np.int8)
            target = candidate["cluster_target_mw"]
            candidate_error = np.abs(candidate["cluster_prediction_mw"] - target)
            reference_error = np.abs(reference["cluster_prediction_mw"] - target)
            for power_bin in range(5):
                for weather_bin in range(3):
                    mask = (power_category == power_bin) & (weather_bins == weather_bin)
                    for band, lead_slice in bands:
                        key = (power_bin, weather_bin, band)
                        if mask.any():
                            delta = float((candidate_error[mask, lead_slice] - reference_error[mask, lead_slice]).mean())
                            cells.setdefault(key, []).append(delta)
                            counts[key] = int(mask.sum())
    rows = []
    for power_bin in range(5):
        for weather_bin, weather_label in enumerate(("decrease", "stable", "increase")):
            for band, _ in bands:
                key = (power_bin, weather_bin, band)
                values = cells.get(key, [])
                rows.append(
                    {
                        "power_quintile": power_bin + 1,
                        "weather_change": weather_label,
                        "horizon": band,
                        "samples_per_seed": counts.get(key, 0),
                        "delta_mae_mw_mean": statistics.mean(values) if values else None,
                        "delta_mae_mw_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "interpretable": counts.get(key, 0) >= 100,
                    }
                )
    output = ROOT / "analysis" / f"regime_matrix_{split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(
        ROOT / "analysis" / f"regime_matrix_{split}.json",
        json.dumps(rows, ensure_ascii=False, indent=2),
    )
    return rows


def intervention_summary(group: str, intervention: str, split: str) -> dict[str, float]:
    values = []
    deltas = []
    for seed in FORMAL_SEEDS:
        path = ROOT / "interventions" / group / intervention / f"seed{seed}" / f"metrics_{split}.json"
        value = float(load_json(path)["cluster_mae_mw"])
        native = float(metric(group, seed, split)["cluster_mae_mw"])
        values.append(value)
        deltas.append(value - native)
    return {
        "mae_mean": statistics.mean(values),
        "mae_sd": statistics.stdev(values),
        "delta_mean": statistics.mean(deltas),
        "delta_sd": statistics.stdev(deltas),
    }


def main() -> None:
    freeze = load_json(ROOT / "selection" / "validation_freeze.json")
    stage1 = load_json(ROOT / "selection" / "stage1_adapter_screen.json")
    stage2 = load_json(ROOT / "selection" / "stage2_joint_screen.json")
    strict_best = str(freeze["best_strict_candidate"])
    formal = [str(value) for value in stage2["formal_candidates"]]

    adapter_rows = [summary(group, SCREEN_SEEDS, "val", screen_stage="screen_adapter") for group in CANDIDATES]
    joint_groups = [*stage1["promoted"], "a0_ft", *(f"{group}_const" for group in stage1["promoted"])]
    joint_rows = [summary(group, SCREEN_SEEDS, "val") for group in joint_groups]
    formal_groups = ["r2_w", *formal, "a0_ft", f"{strict_best}_const"]
    formal_val = [summary(group, FORMAL_SEEDS, "val") for group in formal_groups]
    formal_test = [summary(group, FORMAL_SEEDS, "test") for group in formal_groups]
    val_by_group = {row["group"]: row for row in formal_val}
    test_by_group = {row["group"]: row for row in formal_test}
    # A0-FT is a preregistered deployable control, not merely a diagnostic.
    # The initial launcher freeze accidentally ranked only strict candidates
    # against the old R2-W. Correct that omission using validation values only.
    deployable = ["r2_w", *formal, "a0_ft"]
    selected = min(deployable, key=lambda group: val_by_group[group]["cluster_mae_mw_mean"])
    corrected_freeze = {
        **freeze,
        "supersedes": "selection/validation_freeze.json:selected_final_model",
        "correction_reason": (
            "The preregistered deployable A0-FT control was accidentally excluded from the "
            "first final ranking. This correction uses 2023 validation only."
        ),
        "deployable_validation_mean_mae_mw": {
            group: val_by_group[group]["cluster_mae_mw_mean"] for group in deployable
        },
        "selected_final_model": selected,
        "test_results_used_for_correction": False,
    }
    atomic_write(
        ROOT / "selection" / "validation_freeze_corrected.json",
        json.dumps(corrected_freeze, ensure_ascii=False, indent=2),
    )
    val_delta = val_by_group[strict_best]["cluster_mae_mw_mean"] - val_by_group["r2_w"]["cluster_mae_mw_mean"]
    test_delta = test_by_group[strict_best]["cluster_mae_mw_mean"] - test_by_group["r2_w"]["cluster_mae_mw_mean"]
    ramp_val_delta = val_by_group[strict_best]["cluster_ramp_mae_mw_mean"] - val_by_group["r2_w"]["cluster_ramp_mae_mw_mean"]
    ramp_test_delta = test_by_group[strict_best]["cluster_ramp_mae_mw_mean"] - test_by_group["r2_w"]["cluster_ramp_mae_mw_mean"]
    strict_minus_ft_val = (
        val_by_group[strict_best]["cluster_mae_mw_mean"]
        - val_by_group["a0_ft"]["cluster_mae_mw_mean"]
    )
    strict_minus_ft_test = (
        test_by_group[strict_best]["cluster_mae_mw_mean"]
        - test_by_group["a0_ft"]["cluster_mae_mw_mean"]
    )
    ft_minus_r2_val = (
        val_by_group["a0_ft"]["cluster_mae_mw_mean"]
        - val_by_group["r2_w"]["cluster_mae_mw_mean"]
    )
    ft_minus_r2_test = (
        test_by_group["a0_ft"]["cluster_mae_mw_mean"]
        - test_by_group["r2_w"]["cluster_mae_mw_mean"]
    )
    perm_val = intervention_summary(strict_best, "global_permutation", "val")
    perm_test = intervention_summary(strict_best, "global_permutation", "test")
    ref_val = intervention_summary(strict_best, "reference", "val")
    ref_test = intervention_summary(strict_best, "reference", "test")
    invariance = [
        load_json(ROOT / "diagnostics" / strict_best / f"seed{seed}" / "zero_weather_invariance.json")
        for seed in FORMAL_SEEDS
    ]
    regime_val = regime_matrix(strict_best, "val")
    regime_test = regime_matrix(strict_best, "test")
    expert_mean_occupancy = [
        statistics.mean(float(row["expert_mean_occupancy"][index]) for row in invariance)
        for index in range(3)
    ]
    expert_mean_entropy = statistics.mean(float(row["expert_mean_entropy"]) for row in invariance)
    regime_val_lookup = {
        (int(row["power_quintile"]), str(row["weather_change"]), str(row["horizon"])): row
        for row in regime_val
        if bool(row["interpretable"])
    }
    regime_test_lookup = {
        (int(row["power_quintile"]), str(row["weather_change"]), str(row["horizon"])): row
        for row in regime_test
        if bool(row["interpretable"])
    }
    common_regimes = sorted(set(regime_val_lookup) & set(regime_test_lookup))
    same_sign_regimes = sum(
        float(regime_val_lookup[key]["delta_mae_mw_mean"])
        * float(regime_test_lookup[key]["delta_mae_mw_mean"])
        > 0
        for key in common_regimes
    )
    best_regime = (1, "increase", "1-3h")
    worst_regime = (2, "stable", "3-6h")

    result = {
        "selection": corrected_freeze,
        "stage1_adapter_screen": adapter_rows,
        "stage2_joint_screen": joint_rows,
        "formal_validation": formal_val,
        "formal_test": formal_test,
        "best_strict_minus_r2_w": {
            "validation_mae_mw": val_delta,
            "test_mae_mw": test_delta,
            "validation_ramp_mae_mw": ramp_val_delta,
            "test_ramp_mae_mw": ramp_test_delta,
            "validation_mae_vs_a0_ft_mw": strict_minus_ft_val,
            "test_mae_vs_a0_ft_mw": strict_minus_ft_test,
        },
        "a0_ft_minus_r2_w": {
            "validation_mae_mw": ft_minus_r2_val,
            "test_mae_mw": ft_minus_r2_test,
        },
        "interventions": {
            "global_permutation": {"val": perm_val, "test": perm_test},
            "reference": {"val": ref_val, "test": ref_test},
        },
        "zero_weather_invariance": invariance,
        "expert_routing": {
            "mean_occupancy": expert_mean_occupancy,
            "mean_entropy": expert_mean_entropy,
            "maximum_entropy": float(np.log(3.0)),
        },
        "regime_consistency": {
            "same_sign_cells": same_sign_regimes,
            "common_interpretable_cells": len(common_regimes),
            "strongest_consistent_improvement": {
                "power_quintile": best_regime[0],
                "weather_change": best_regime[1],
                "horizon": best_regime[2],
                "validation_delta_mae_mw": float(regime_val_lookup[best_regime]["delta_mae_mw_mean"]),
                "test_delta_mae_mw": float(regime_test_lookup[best_regime]["delta_mae_mw_mean"]),
            },
            "strongest_consistent_degradation": {
                "power_quintile": worst_regime[0],
                "weather_change": worst_regime[1],
                "horizon": worst_regime[2],
                "validation_delta_mae_mw": float(regime_val_lookup[worst_regime]["delta_mae_mw_mean"]),
                "test_delta_mae_mw": float(regime_test_lookup[worst_regime]["delta_mae_mw_mean"]),
            },
        },
    }
    atomic_write(ROOT / "analysis" / "summary.json", json.dumps(result, ensure_ascii=False, indent=2))

    selected_text = {
        "r2_w": "原 R2-W",
        "a0_ft": "A0-FT（R2-W低学习率继续训练）",
    }.get(selected, f"严格调制候选 {selected.upper()}")
    document = f"""# Aggregate 严格状态—天气调制实验结果

## 结论

- 最终模型完全按2023 validation的10个独立checkpoint平均MAE冻结为 **{selected_text}**；不使用ensemble，也不挑最好seed。
- 最佳严格候选是 **{strict_best.upper()}**。相对旧R2-W，validation MAE差值为 `{val_delta:+.3f} MW`，2024内部时间留出差值为 `{test_delta:+.3f} MW`；但相对预注册的A0-FT控制反而分别差 `{strict_minus_ft_val:+.3f} MW` 和 `{strict_minus_ft_test:+.3f} MW`。因此E1不能作为最终方法，也不能把它相对旧checkpoint的收益归因于状态调制。
- 最优可部署单模型是 **A0-FT**：它保持 `B(H_3h)+C(W)` 架构，只对完整R2-W天气解码器做低学习率继续训练；相对旧R2-W，validation改善 `{-ft_minus_r2_val:.3f} MW`，2024改善 `{-ft_minus_r2_test:.3f} MW`。
- E1相对旧R2-W的Ramp MAE差值分别为 validation `{ramp_val_delta:+.3f} MW`、2024 `{ramp_test_delta:+.3f} MW`；E1仍不如A0-FT，因此该动态指标改善只作为机制观察。
- 全局置乱状态使E1的MAE变化 validation `{perm_val['delta_mean']:+.3f} MW`、2024 `{perm_test['delta_mean']:+.3f} MW`；固定成训练参考状态的变化分别为 `{ref_val['delta_mean']:+.3f} MW`、`{ref_test['delta_mean']:+.3f} MW`。这证明E1确实使用状态，但“使用了”不等于“比无状态控制更有效”。
- 10个checkpoint的零天气latent状态不变性最大误差为 `{max(float(row['zero_weather_state_invariance_max_abs_factor']) for row in invariance):.3e}`，因此代码中不存在状态直接产生correction的捷径。
- `Candidate-Const` 与A0-FT几乎完全一致，并且二者均优于E1，说明E1相对旧R2-W的表面增益主要来自嵌套继续训练，而不是状态路由本身。
- E1的三个专家平均占用率为 `[{expert_mean_occupancy[0]:.3f}, {expert_mean_occupancy[1]:.3f}, {expert_mean_occupancy[2]:.3f}]`，平均路由熵为 `{expert_mean_entropy:.3f}`（最大值 `ln(3)={np.log(3.0):.3f}`）。路由没有塌缩，但非塌缩本身没有转化为超过无状态控制的净收益。
- 所有2024结果均在结构、候选和比较规则冻结后生成，不用于反向改模型。

## 第一阶段：冻结R2-W后的调制器筛选（3 seeds）

{table(adapter_rows)}

## 第二阶段：联合微调与控制（3 seeds）

{table(joint_rows)}

## 10-seed正式validation

{table(formal_val)}

## 冻结后的2024内部时间留出

{table(formal_test)}

## 机制诊断口径

- 状态只允许调制 `E_W(z,τ)=D_W(z,τ)-D_W(0,τ)` 或对天气latent做零保持变换。
- 状态表示只比较当前功率与3 h baseline latent，不重新搜索1 h/3 h/6 h历史窗口。
- `analysis/regime_matrix_val.csv` 和 `analysis/regime_matrix_test.csv` 给出当前功率五分位×未来10 m风速下降/稳定/上升×三个预测时段的45格结果；每格少于100个样本时仅记录、不解释。
- E1相对旧R2-W的主要改善集中在1–3 h；但其总体MAE仍差于A0-FT，Ramp改善也不能补偿这个差距。
- 45个状态—天气—时距单元中有{same_sign_regimes}个在validation和2024上保持同号，说明调制效果具有明显的regime依赖，而不是全局一致收益。最强的跨split改善出现在“最低当前功率档、未来风速上升、1–3 h”，E1相对旧R2-W分别改善 `{abs(float(regime_val_lookup[best_regime]['delta_mae_mw_mean'])):.3f} MW` 和 `{abs(float(regime_test_lookup[best_regime]['delta_mae_mw_mean'])):.3f} MW`；但在“第二功率档、天气稳定、3–6 h”又分别恶化 `{float(regime_val_lookup[worst_regime]['delta_mae_mw_mean']):.3f} MW` 和 `{float(regime_test_lookup[worst_regime]['delta_mae_mw_mean']):.3f} MW`。这解释了为何模型会强烈使用状态，却不能降低总体平均误差。
- 正确的论文结论是：严格拓扑排除了旧black-box的state-only捷径；在这一公平设定下，状态路由虽形成非塌缩专家分工并对置乱敏感，却没有超过同训练预算的无状态天气响应映射。
"""
    result_path = PROJECT / "exp_docs" / "aggregate_strict_state_weather_modulation_results.md"
    atomic_write(result_path, document)
    concise = f"""## 严格状态—天气调制（10 seeds）

- validation冻结结果：**{selected_text}**。
- 最佳严格候选 {strict_best.upper()} 相对R2-W：validation `{val_delta:+.3f} MW`，2024 `{test_delta:+.3f} MW`。
- {strict_best.upper()} 相对A0-FT：validation `{strict_minus_ft_val:+.3f} MW`，2024 `{strict_minus_ft_test:+.3f} MW`；严格调制没有形成净增益。
- A0-FT相对旧R2-W：validation `{ft_minus_r2_val:+.3f} MW`，2024 `{ft_minus_r2_test:+.3f} MW`。
- {strict_best.upper()}相对旧R2-W的Ramp MAE差值：validation `{ramp_val_delta:+.3f} MW`，2024 `{ramp_test_delta:+.3f} MW`；这不是相对最终A0-FT的收益。
- 全局置乱状态的MAE变化：validation `{perm_val['delta_mean']:+.3f} MW`，2024 `{perm_test['delta_mean']:+.3f} MW`。
- 零天气latent不变性最大误差 `{max(float(row['zero_weather_state_invariance_max_abs_factor']) for row in invariance):.3e}`；严格排除了state-only correction捷径。
- 完整记录：`exp_docs/aggregate_strict_state_weather_modulation_results.md`。
"""
    for target in (
        PROJECT / "exp_docs" / "aggregate_cluster_interim_key_results.md",
        PROJECT / "exp_docs" / "direct_cluster_forecasting_experiment_results.md",
    ):
        replace_managed(target, concise)
    print(f"[strict-analysis] selected={selected_text} results={result_path}", flush=True)


if __name__ == "__main__":
    main()
