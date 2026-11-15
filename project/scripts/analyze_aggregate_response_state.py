from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_aggregate_strict_modulation import weather_change_categories


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_response_state_v1"
STRICT_ROOT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
DATA_ROOT = PROJECT / "data" / "aggregate_cluster_v1"
SCREEN_SEEDS = (42, 46, 50)
FORMAL_SEEDS = tuple(range(42, 52))
NEW_GROUPS = ("c2", "c3", "r1", "r2", "r3")
REFERENCE_GROUPS = ("c0_a0_ft", "c1_d2")
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
    "parameter_count", "trainable_parameter_count",
)
LABELS = {
    "c0_a0_ft": "C0 A0-FT（无响应状态）",
    "c1_d2": "C1 功率历史 latent",
    "c2": "C2 仅历史功率 token",
    "c3": "C3 仅历史天气 token",
    "r1": "R1 天气—功率 level pair",
    "r2": "R2 天气—功率 transition pair",
    "r3": "R3 transition pair + identification loss",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run_dir(group: str, seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    if group in REFERENCE_GROUPS:
        strict_group = {"c0_a0_ft": "a0_ft", "c1_d2": "d2"}[group]
        return STRICT_ROOT / stage / strict_group / f"seed{seed}"
    return ROOT / stage / group / f"seed{seed}"


def metric(group: str, seed: int, split: str) -> dict[str, Any]:
    return load_json(run_dir(group, seed) / f"metrics_{split}.json")


def summarize(group: str, seeds: Sequence[int], split: str) -> dict[str, Any]:
    records = [metric(group, seed, split) for seed in seeds]
    result: dict[str, Any] = {"group": group, "seeds": list(seeds), "split": split}
    for key in METRICS:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    identification = [
        float(record["identification_delta_mae_mw"])
        for record in records if record.get("identification_delta_mae_mw") is not None
    ]
    if identification:
        result["identification_delta_mae_mw_mean"] = statistics.mean(identification)
        result["identification_delta_mae_mw_sd"] = (
            statistics.stdev(identification) if len(identification) > 1 else 0.0
        )
    return result


def fmt(row: dict[str, Any], key: str) -> str:
    return f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_sd']:.3f}"


def result_table(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {LABELS[str(row['group'])]} | {fmt(row, 'cluster_mae_mw')} | "
            f"{fmt(row, 'cluster_rmse_mw')} | {fmt(row, 'cluster_r2')} | "
            f"{fmt(row, 'cluster_ramp_mae_mw')} | {fmt(row, 'horizon_0_1h_mae_mw')} | "
            f"{fmt(row, 'horizon_1_3h_mae_mw')} | {fmt(row, 'horizon_3_6h_mae_mw')} |"
        )
    return "\n".join(lines)


def screening_table(rows: Iterable[dict[str, Any]], baseline: float) -> str:
    lines = [
        "| 模型 | 3-seed validation MAE / MW | 相对 A0-FT / MW |",
        "|---|---:|---:|",
    ]
    for row in rows:
        mae = float(row["cluster_mae_mw_mean"])
        lines.append(
            f"| {LABELS[str(row['group'])]} | {fmt(row, 'cluster_mae_mw')} | {mae - baseline:+.3f} |"
        )
    return "\n".join(lines)


def intervention_summary(group: str, intervention: str, split: str) -> dict[str, float]:
    values: list[float] = []
    deltas: list[float] = []
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


def prediction_path(group: str, seed: int, split: str) -> Path:
    return run_dir(group, seed) / f"predictions_{split}.npz"


def regime_matrix(group: str, split: str) -> list[dict[str, Any]]:
    train_history = np.load(DATA_ROOT / "train" / "cluster_history_mw.npy", mmap_mode="r")
    thresholds = np.quantile(
        np.asarray(train_history[:, -1], dtype=np.float64), [0.2, 0.4, 0.6, 0.8]
    )
    weather_ids, weather_category = weather_change_categories(split)
    weather_lookup = {int(key): int(value) for key, value in zip(weather_ids, weather_category)}
    bands = (("0-1h", slice(0, 4)), ("1-3h", slice(4, 12)), ("3-6h", slice(12, 24)))
    cells: dict[tuple[int, int, str], list[float]] = {}
    counts: dict[tuple[int, int, str], int] = {}
    for seed in FORMAL_SEEDS:
        with np.load(prediction_path(group, seed, split), allow_pickle=False) as candidate, np.load(
            prediction_path("c0_a0_ft", seed, split), allow_pickle=False
        ) as reference:
            if not np.array_equal(candidate["sample_index"], reference["sample_index"]):
                raise ValueError(f"Prediction archives are not aligned for seed {seed} {split}")
            sample_ids = candidate["sample_index"]
            power_category = np.digitize(candidate["cluster_history_last_mw"], thresholds).astype(np.int8)
            weather_bins = np.asarray(
                [weather_lookup[int(index)] for index in sample_ids], dtype=np.int8
            )
            target = candidate["cluster_target_mw"]
            candidate_error = np.abs(candidate["cluster_prediction_mw"] - target)
            reference_error = np.abs(reference["cluster_prediction_mw"] - target)
            for power_bin in range(5):
                for weather_bin in range(3):
                    mask = (power_category == power_bin) & (weather_bins == weather_bin)
                    for band, lead_slice in bands:
                        key = (power_bin, weather_bin, band)
                        if mask.any():
                            cells.setdefault(key, []).append(
                                float(
                                    (
                                        candidate_error[mask, lead_slice]
                                        - reference_error[mask, lead_slice]
                                    ).mean()
                                )
                            )
                            counts[key] = int(mask.sum())
    rows: list[dict[str, Any]] = []
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
                        "delta_mae_vs_a0_ft_mw_mean": statistics.mean(values) if values else None,
                        "delta_mae_vs_a0_ft_mw_sd": (
                            statistics.stdev(values) if len(values) > 1 else 0.0
                        ),
                        "interpretable": counts.get(key, 0) >= 100,
                    }
                )
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / f"regime_matrix_{split}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(
        analysis / f"regime_matrix_{split}.json",
        json.dumps(rows, ensure_ascii=False, indent=2),
    )
    plot_regime(rows, split)
    return rows


def plot_regime(rows: Sequence[dict[str, Any]], split: str) -> None:
    horizons = ("0-1h", "1-3h", "3-6h")
    arrays = []
    for horizon in horizons:
        values = np.full((5, 3), np.nan)
        for row in rows:
            if row["horizon"] == horizon:
                values[int(row["power_quintile"]) - 1, ("decrease", "stable", "increase").index(str(row["weather_change"]))] = float(
                    row["delta_mae_vs_a0_ft_mw_mean"]
                )
        arrays.append(values)
    limit = max(float(np.nanmax(np.abs(values))) for values in arrays)
    limit = max(limit, 1e-6)
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), constrained_layout=True)
    image = None
    for axis, horizon, values in zip(axes, horizons, arrays):
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(horizon)
        axis.set_xticks(range(3), ("Down", "Stable", "Up"))
        axis.set_yticks(range(5), ("Q1", "Q2", "Q3", "Q4", "Q5"))
        axis.set_xlabel("Future wind-speed change")
        for row in range(5):
            for column in range(3):
                axis.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Current-power quintile")
    assert image is not None
    figure.colorbar(image, ax=axes, label="ΔMAE vs A0-FT (MW); negative is better")
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / f"response_regime_{split}.png", dpi=240)
    figure.savefig(figures / f"response_regime_{split}.pdf")
    plt.close(figure)


def best_and_worst_common(
    validation: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    def lookup(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
        return {
            (int(row["power_quintile"]), str(row["weather_change"]), str(row["horizon"])): row
            for row in rows if bool(row["interpretable"])
        }

    val = lookup(validation)
    tst = lookup(test)
    keys = sorted(set(val) & set(tst))
    same_sign = sum(
        float(val[key]["delta_mae_vs_a0_ft_mw_mean"])
        * float(tst[key]["delta_mae_vs_a0_ft_mw_mean"]) > 0
        for key in keys
    )
    consistent_better = [
        key for key in keys
        if float(val[key]["delta_mae_vs_a0_ft_mw_mean"]) < 0
        and float(tst[key]["delta_mae_vs_a0_ft_mw_mean"]) < 0
    ]
    consistent_worse = [
        key for key in keys
        if float(val[key]["delta_mae_vs_a0_ft_mw_mean"]) > 0
        and float(tst[key]["delta_mae_vs_a0_ft_mw_mean"]) > 0
    ]

    def item(key: tuple[int, str, str] | None) -> dict[str, Any] | None:
        if key is None:
            return None
        return {
            "power_quintile": key[0], "weather_change": key[1], "horizon": key[2],
            "validation_delta_mae_mw": val[key]["delta_mae_vs_a0_ft_mw_mean"],
            "test_delta_mae_mw": tst[key]["delta_mae_vs_a0_ft_mw_mean"],
        }

    best = min(
        consistent_better,
        key=lambda key: float(val[key]["delta_mae_vs_a0_ft_mw_mean"])
        + float(tst[key]["delta_mae_vs_a0_ft_mw_mean"]),
        default=None,
    )
    worst = max(
        consistent_worse,
        key=lambda key: float(val[key]["delta_mae_vs_a0_ft_mw_mean"])
        + float(tst[key]["delta_mae_vs_a0_ft_mw_mean"]),
        default=None,
    )
    return {
        "same_sign_cells": same_sign,
        "common_interpretable_cells": len(keys),
        "strongest_consistent_improvement": item(best),
        "strongest_consistent_degradation": item(worst),
    }


def main() -> None:
    screening = load_json(ROOT / "selection" / "screening_selection.json")
    freeze = load_json(ROOT / "selection" / "validation_freeze.json")
    formal = [str(value) for value in screening["formal_candidates"]]
    best_response = str(freeze["best_response_candidate"])
    selected = str(freeze["selected_final_model"])
    screen_adapter = [
        summarize(group, SCREEN_SEEDS, "val")
        for group in NEW_GROUPS
    ]
    # The regular helper points at joint runs. Adapter values are loaded explicitly.
    for row in screen_adapter:
        group = str(row["group"])
        records = [
            load_json(ROOT / "screen_adapter" / group / f"seed{seed}" / "metrics_val.json")
            for seed in SCREEN_SEEDS
        ]
        for key in METRICS:
            values = [float(record[key]) for record in records if record.get(key) is not None]
            if values:
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_sd"] = statistics.stdev(values)
    screen_joint = [summarize(group, SCREEN_SEEDS, "val") for group in NEW_GROUPS]
    formal_groups = [*REFERENCE_GROUPS, *formal]
    formal_val = [summarize(group, FORMAL_SEEDS, "val") for group in formal_groups]
    formal_test = [summarize(group, FORMAL_SEEDS, "test") for group in formal_groups]
    val_lookup = {str(row["group"]): row for row in formal_val}
    test_lookup = {str(row["group"]): row for row in formal_test}
    interventions = {
        intervention: {
            split: intervention_summary(best_response, intervention, split)
            for split in ("val", "test")
        }
        for intervention in ("power_shuffle", "weather_shuffle", "pair_shift", "response_mean", "support_only")
    }
    invariance = [
        load_json(ROOT / "diagnostics" / best_response / f"seed{seed}" / "zero_weather_invariance.json")
        for seed in FORMAL_SEEDS
    ]
    regime_val = regime_matrix(best_response, "val")
    regime_test = regime_matrix(best_response, "test")
    regime = best_and_worst_common(regime_val, regime_test)
    selected_val = float(val_lookup[selected]["cluster_mae_mw_mean"])
    selected_test = float(test_lookup[selected]["cluster_mae_mw_mean"])
    response_val = float(val_lookup[best_response]["cluster_mae_mw_mean"])
    response_test = float(test_lookup[best_response]["cluster_mae_mw_mean"])
    baseline_val = float(val_lookup["c0_a0_ft"]["cluster_mae_mw_mean"])
    baseline_test = float(test_lookup["c0_a0_ft"]["cluster_mae_mw_mean"])
    summary = {
        "selection": freeze,
        "screen_adapter": screen_adapter,
        "screen_joint": screen_joint,
        "formal_validation": formal_val,
        "formal_test": formal_test,
        "best_response_minus_a0_ft": {
            "validation_mae_mw": response_val - baseline_val,
            "test_mae_mw": response_test - baseline_test,
        },
        "interventions": interventions,
        "zero_weather_invariance": {
            "all_pass": all(bool(row["zero_weather_state_invariance_pass"]) for row in invariance),
            "maximum_abs_factor": max(
                float(row["zero_weather_state_invariance_max_abs_factor"]) for row in invariance
            ),
            "records": invariance,
        },
        "regime_consistency": regime,
    }
    atomic_text(ROOT / "analysis" / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    response_delta_val = response_val - baseline_val
    response_delta_test = response_test - baseline_test
    lines = [
        "# 气象—功率响应状态辨识实验结果",
        "",
        "## 结论",
        "",
        f"- 全部结构由2023 validation冻结；最终按10个独立checkpoint平均MAE选择 **{LABELS[selected]}**。",
        f"- 最佳响应状态候选是 **{LABELS[best_response]}**。相对无响应状态A0-FT，validation差值为 `{response_delta_val:+.3f} MW`，2024内部时间留出差值为 `{response_delta_test:+.3f} MW`。",
        f"- 最终模型validation / 2024平均MAE分别为 `{selected_val:.3f} / {selected_test:.3f} MW`；不使用ensemble，也不挑最好seed。",
        f"- 零未来天气不变量在10个seed上{'全部通过' if summary['zero_weather_invariance']['all_pass'] else '未全部通过'}，最大状态相关差异为 `{summary['zero_weather_invariance']['maximum_abs_factor']:.3e}`（归一化功率）。",
        "- 本实验回答的是response state的信息来源是否足够，而不是继续搜索调制算子；历史baseline仍固定为已经验证的3 h TCN。",
        "",
        "## 3-seed筛选",
        "",
        "### Adapter-only",
        "",
        screening_table(screen_adapter, float(screening["a0_ft_mean_mae_mw"])),
        "",
        "### Joint fine-tuning",
        "",
        screening_table(screen_joint, float(screening["a0_ft_mean_mae_mw"])),
        "",
        f"按预注册规则补齐10 seeds：`{', '.join(formal)}`。",
        "",
        "## 10-seed正式结果：2023 validation",
        "",
        result_table(formal_val),
        "",
        "## 冻结后的2024内部时间留出",
        "",
        result_table(formal_test),
        "",
        "## 状态干预",
        "",
        "以下差值均为干预后MAE减原模型MAE，正值表示破坏配对后变差。",
        "",
        "| 干预 | validation ΔMAE / MW | 2024 ΔMAE / MW |",
        "|---|---:|---:|",
    ]
    for intervention, values in interventions.items():
        lines.append(
            f"| {intervention} | {values['val']['delta_mean']:+.3f} ± {values['val']['delta_sd']:.3f} | "
            f"{values['test']['delta_mean']:+.3f} ± {values['test']['delta_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Regime诊断",
            "",
            f"当前功率五分位×未来10 m风速变化×预测时段共45格，其中 `{regime['same_sign_cells']}/{regime['common_interpretable_cells']}` 个可解释单元跨年份同号。负值表示响应状态优于A0-FT。",
            "",
            "可视化：",
            "",
            "- `artifacts/aggregate_response_state_v1/figures/response_regime_val.png`",
            "- `artifacts/aggregate_response_state_v1/figures/response_regime_test.png`",
            "",
            "## 产物",
            "",
            "- 原始训练与预测：`artifacts/aggregate_response_state_v1/{screen_adapter,screen_joint,formal_adapter,formal_joint}/`",
            "- 干预与架构诊断：`artifacts/aggregate_response_state_v1/{interventions,diagnostics}/`",
            "- 机器可读汇总：`artifacts/aggregate_response_state_v1/analysis/summary.json`",
            "",
        ]
    )
    atomic_text(
        PROJECT / "exp_docs" / "weather_power_response_state_experiment_results.md",
        "\n".join(lines),
    )
    print(
        f"[response-analysis] selected={selected} best_response={best_response} "
        f"val_delta={response_delta_val:+.3f}MW test_delta={response_delta_test:+.3f}MW",
        flush=True,
    )


if __name__ == "__main__":
    main()
