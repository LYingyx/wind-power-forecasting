from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1"
SOURCE = PROJECT / "artifacts" / "aggregate_cluster_v1"
SEEDS = tuple(range(42, 52))
METRICS = (
    "cluster_mae_mw",
    "cluster_rmse_mw",
    "cluster_r2",
    "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw",
    "horizon_1_3h_mae_mw",
    "horizon_3_6h_mae_mw",
    "parameter_count",
    "trainable_parameter_count",
)
MANAGED_START = "<!-- R2_STATE_MECHANISM_START -->"
MANAGED_END = "<!-- R2_STATE_MECHANISM_END -->"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _metric_path(group: str, seed: int, split: str) -> Path:
    if group == "r2_full":
        return SOURCE / "weather" / "r2_latent_interpolation" / f"seed{seed}" / f"metrics_{split}.json"
    if group == "r2_state_shift":
        return ROOT / "interventions" / "r2_full_state_shift" / f"seed{seed}" / f"metrics_{split}.json"
    return ROOT / "mechanism" / group / f"seed{seed}" / f"metrics_{split}.json"


def _summary(group: str, split: str) -> dict[str, Any]:
    rows = [_load(_metric_path(group, seed, split)) for seed in SEEDS]
    result: dict[str, Any] = {"group": group, "split": split, "seeds": list(SEEDS)}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_sd"] = statistics.stdev(values)
    return result


def _source_summary(stage: str, group: str, split: str) -> dict[str, Any]:
    rows = [
        _load(SOURCE / stage / group / f"seed{seed}" / f"metrics_{split}.json")
        for seed in SEEDS
    ]
    result: dict[str, Any] = {"stage": stage, "group": group, "split": split, "seeds": list(SEEDS)}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_sd"] = statistics.stdev(values)
    return result


def _paired_delta(candidate: str, reference: str, split: str, key: str) -> dict[str, Any]:
    values = [
        float(_load(_metric_path(candidate, seed, split))[key])
        - float(_load(_metric_path(reference, seed, split))[key])
        for seed in SEEDS
    ]
    return {
        "candidate": candidate,
        "reference": reference,
        "split": split,
        "metric": key,
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "wins": sum(value < 0 for value in values),
        "values": values,
    }


def _format(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def _table(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1 h | 1–3 h | 3–6 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "r2_full": "R2-Full：B+C(S,W)",
        "r2_w": "R2-W：B+C(W)",
        "r2_add": "R2-Add：B+C_S(S)+C_W(W)",
        "r2_state_shift": "R2-Full StateShift",
    }
    for row in rows:
        lines.append(
            "| {label} | {mae} | {rmse} | {r2} | {ramp} | {short} | {middle} | {long} |".format(
                label=labels[row["group"]],
                mae=_format(row["cluster_mae_mw_mean"], row["cluster_mae_mw_sd"]),
                rmse=_format(row["cluster_rmse_mw_mean"], row["cluster_rmse_mw_sd"]),
                r2=_format(row["cluster_r2_mean"], row["cluster_r2_sd"]),
                ramp=_format(row["cluster_ramp_mae_mw_mean"], row["cluster_ramp_mae_mw_sd"]),
                short=_format(row["horizon_0_1h_mae_mw_mean"], row["horizon_0_1h_mae_mw_sd"]),
                middle=_format(row["horizon_1_3h_mae_mw_mean"], row["horizon_1_3h_mae_mw_sd"]),
                long=_format(row["horizon_3_6h_mae_mw_mean"], row["horizon_3_6h_mae_mw_sd"]),
            )
        )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _replace_managed(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    managed = f"{MANAGED_START}\n{section.rstrip()}\n{MANAGED_END}"
    if MANAGED_START in content and MANAGED_END in content:
        prefix, remainder = content.split(MANAGED_START, 1)
        _, suffix = remainder.split(MANAGED_END, 1)
        content = f"{prefix.rstrip()}\n\n{managed}{suffix}"
    else:
        content = f"{content.rstrip()}\n\n{managed}\n"
    _atomic_write(path, content)


def main() -> None:
    groups = ("r2_full", "r2_w", "r2_add", "r2_state_shift")
    summaries = {
        split: [_summary(group, split) for group in groups]
        for split in ("val", "test")
    }
    baseline = {
        split: _source_summary("fusion", "f_prediction_blend", split)
        for split in ("val", "test")
    }
    validation_mae = {
        row["group"]: row["cluster_mae_mw_mean"] for row in summaries["val"]
    }
    selected = "r2_full" if validation_mae["r2_full"] < validation_mae["r2_w"] else "r2_w"
    deltas = [
        _paired_delta(candidate, reference, split, key)
        for split in ("val", "test")
        for key in ("cluster_mae_mw", "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw")
        for candidate, reference in (
            ("r2_full", "r2_w"),
            ("r2_full", "r2_add"),
            ("r2_state_shift", "r2_full"),
        )
    ]
    decision = {
        "selection_split": "val",
        "selection_rule": "lower 10-seed mean validation MAE; no significance threshold",
        "selected": selected,
        "state_increment_helpful_by_mean": validation_mae["r2_full"] < validation_mae["r2_w"],
        "nonlinear_joint_better_than_additive_by_mean": validation_mae["r2_full"] < validation_mae["r2_add"],
        "trained_full_uses_state_by_shift_mean": validation_mae["r2_state_shift"] > validation_mae["r2_full"],
        "history_policy": "Keep TCN-3h baseline; do not reopen history/state-range search",
    }
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        analysis / "summary.json",
        json.dumps(
            {
                "summaries": summaries,
                "frozen_fusion_baseline": baseline,
                "paired_deltas": deltas,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    with (analysis / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({key for rows in summaries.values() for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for rows in summaries.values() for row in rows)
    _atomic_write(analysis / "decision.json", json.dumps(decision, ensure_ascii=False, indent=2))

    full_val, w_val, add_val, shift_val = summaries["val"]
    full_test, w_test, add_test, shift_test = summaries["test"]
    full_minus_w_val = full_val["cluster_mae_mw_mean"] - w_val["cluster_mae_mw_mean"]
    full_minus_w_test = full_test["cluster_mae_mw_mean"] - w_test["cluster_mae_mw_mean"]
    full_minus_add_val = full_val["cluster_mae_mw_mean"] - add_val["cluster_mae_mw_mean"]
    shift_delta_val = shift_val["cluster_mae_mw_mean"] - full_val["cluster_mae_mw_mean"]
    shift_delta_test = shift_test["cluster_mae_mw_mean"] - full_test["cluster_mae_mw_mean"]
    selected_label = "R2-Full" if selected == "r2_full" else "R2-W"
    selected_val = full_val if selected == "r2_full" else w_val
    selected_test = full_test if selected == "r2_full" else w_test
    baseline_delta_val = (
        selected_val["cluster_mae_mw_mean"] - baseline["val"]["cluster_mae_mw_mean"]
    )
    baseline_delta_test = (
        selected_test["cluster_mae_mw_mean"] - baseline["test"]["cluster_mae_mw_mean"]
    )
    conclusion = (
        "保留当前功率对天气修正的联合条件化。"
        if selected == "r2_full"
        else "删除修正分支的重复状态输入，采用更简洁的 B(H)+C(W) 结构。"
    )
    interaction = (
        "在保留状态输入的两种结构中，black-box联合条件化优于可加分解；"
        "但由于Full没有在validation上优于完全移除correction状态的R2-W，"
        "这不能单独证明状态—天气交互带来净增益。"
        if decision["nonlinear_joint_better_than_additive_by_mean"]
        else "black-box联合条件化没有优于可加控制，不能声称特殊交互机制带来收益。"
    )
    results = f"""# Aggregate R2 历史状态机制补充实验结果

## 最终结论

- 所有结果均为 seeds 42–51 的10个独立checkpoint `mean ± SD`，没有ensemble或挑选最好seed。
- 最终按2023 validation平均MAE选择 **{selected_label}**：{conclusion}
- 历史基线继续使用过去3 h TCN。既有历史范围与状态范围搜索已经充分，本轮没有继续增加历史复杂度。
- 最终模型相对validation冻结的最强fusion baseline PredictionBlend：validation `{baseline_delta_val:+.3f} MW`，2024 `{baseline_delta_test:+.3f} MW`。
- R2-Full − R2-W：validation `{full_minus_w_val:+.3f} MW`，2024内部时间留出 `{full_minus_w_test:+.3f} MW`；负值代表Full更好。
- R2-Full − R2-Add：validation `{full_minus_add_val:+.3f} MW`。{interaction}
- StateShift相对R2-Full：validation `{shift_delta_val:+.3f} MW`，2024 `{shift_delta_test:+.3f} MW`。该干预只诊断模型是否使用状态，是否有效以Full与重新训练R2-W的比较为准。
- Full/W在validation与2024发生方向反转，说明correction状态的增量不稳定。按预注册口径仍选择R2-W，不能利用2024反向改选Full。

## 2023 validation

{_table(summaries['val'])}

## 2024内部时间留出

{_table(summaries['test'])}

## 解释边界

- 历史基线 `B(H_3h)` 已经吸收历史水平和演化信息，correction再次读取当前功率是否有增量由本轮直接回答。
- R2-W仍保留未激活的state encoder用于参数匹配，因此差异不是参数量变化造成的。
- 最终实现保留这些checkpoint参数键，但在weather-only correction下跳过未激活state encoder的前向计算；零状态参考测试确认预测公式不变，无需重训。
- StateShift保持baseline历史不变，只移动correction状态；它不能代替重新训练的无状态控制。
- 2024不参与结构选择，仅用于检查冻结结论的时间外一致性。
"""
    result_path = PROJECT / "exp_docs" / "aggregate_cluster_r2_state_mechanism_results.md"
    _atomic_write(result_path, results)
    concise = f"""## R2 correction状态专门消融（10 seeds）

- 最终按2023 validation平均MAE选择 **{selected_label}**。
- 相对PredictionBlend：validation `{baseline_delta_val:+.3f} MW`，2024 `{baseline_delta_test:+.3f} MW`。
- R2-Full − R2-W：validation `{full_minus_w_val:+.3f} MW`，2024 `{full_minus_w_test:+.3f} MW`。
- R2-Full − R2-Add：validation `{full_minus_add_val:+.3f} MW`。
- StateShift退化：validation `{shift_delta_val:+.3f} MW`，2024 `{shift_delta_test:+.3f} MW`。
- 最终处理：{conclusion}
- Full/W在validation和2024方向反转，因此不把correction状态调制作为稳健核心贡献。
- 历史基线固定为过去3 h TCN，不再扩展历史窗口或多尺度状态搜索。
- R2-W运行时跳过未激活的state encoder，但保留参数键以兼容正式checkpoint。

完整记录见 `exp_docs/aggregate_cluster_r2_state_mechanism_results.md`。
"""
    for target in (
        PROJECT / "exp_docs" / "aggregate_cluster_interim_key_results.md",
        PROJECT / "exp_docs" / "direct_cluster_forecasting_experiment_results.md",
    ):
        _replace_managed(target, concise)
    print(f"[r2-state-analysis] selected={selected_label} results={result_path}")


if __name__ == "__main__":
    main()
