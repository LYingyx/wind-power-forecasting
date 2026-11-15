from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_state_spatial_interaction_v1"
DOC = PROJECT / "exp_docs" / "local_state_spatial_interaction_v1_results.md"
SEEDS = tuple(range(42, 52))
SPATIALS = ("bilinear", "fixed4", "learned4")
OPERATORS = ("none", "add", "int")
GROUPS = tuple(f"{spatial}_{operator}" for spatial in SPATIALS for operator in OPERATORS)
INTERVENTIONS = ("strong_sample", "station_permutation", "reference")
METRICS = (
    "station_mae_cf",
    "cluster_mae_mw",
    "cluster_rmse_mw",
    "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw",
    "horizon_1_3h_mae_mw",
    "horizon_3_6h_mae_mw",
    "parameter_count",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(group: str, seed: int, split: str) -> dict[str, Any]:
    return load(ROOT / "formal" / group / f"seed{seed}" / f"metrics_{split}.json")


def values(group: str, key: str, split: str) -> np.ndarray:
    return np.asarray(
        [float(metric(group, seed, split)[key]) for seed in SEEDS], dtype=np.float64
    )


def summarize(group: str, split: str) -> dict[str, Any]:
    result: dict[str, Any] = {"group": group, "split": split}
    for key in METRICS:
        array = values(group, key, split)
        result[key] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
        }
    return result


def contrast_from_delta(
    name: str, delta: np.ndarray, *, candidate: str, control: str, split: str
) -> dict[str, Any]:
    critical = 2.262
    half = critical * float(delta.std(ddof=1)) / math.sqrt(len(delta))
    return {
        "name": name,
        "candidate": candidate,
        "control": control,
        "split": split,
        "mean_delta": float(delta.mean()),
        "ci95": [float(delta.mean() - half), float(delta.mean() + half)],
        "wins": int((delta < 0.0).sum()),
        "seed_delta": delta.tolist(),
    }


def paired(
    candidate: str, control: str, split: str, key: str = "station_mae_cf"
) -> dict[str, Any]:
    return contrast_from_delta(
        f"{candidate}_minus_{control}",
        values(candidate, key, split) - values(control, key, split),
        candidate=candidate,
        control=control,
        split=split,
    )


def interaction_did(spatial: str, split: str) -> dict[str, Any]:
    spatial_delta = (
        values(f"{spatial}_int", "station_mae_cf", split)
        - values(f"{spatial}_add", "station_mae_cf", split)
    )
    bilinear_delta = (
        values("bilinear_int", "station_mae_cf", split)
        - values("bilinear_add", "station_mae_cf", split)
    )
    return contrast_from_delta(
        f"{spatial}_interaction_advantage_minus_bilinear",
        spatial_delta - bilinear_delta,
        candidate=f"({spatial}_int-{spatial}_add)",
        control="(bilinear_int-bilinear_add)",
        split=split,
    )


def per_station(candidate: str, control: str, split: str) -> dict[str, Any]:
    candidate_values = np.asarray(
        [metric(candidate, seed, split)["per_station_mae_cf"] for seed in SEEDS]
    )
    control_values = np.asarray(
        [metric(control, seed, split)["per_station_mae_cf"] for seed in SEEDS]
    )
    delta = (candidate_values - control_values).mean(axis=0)
    return {
        "candidate": candidate,
        "control": control,
        "split": split,
        "farms_improved": int((delta < 0.0).sum()),
        "farms_degraded": int((delta > 0.0).sum()),
        "mean_delta_cf": float(delta.mean()),
        "median_delta_cf": float(np.median(delta)),
        "per_farm_delta_cf": delta.tolist(),
    }


def intervention(group: str, name: str, split: str) -> dict[str, Any]:
    normal = values(group, "station_mae_cf", split)
    changed = np.asarray(
        [
            float(
                load(
                    ROOT
                    / "interventions"
                    / group
                    / name
                    / f"seed{seed}"
                    / f"metrics_{split}.json"
                )["station_mae_cf"]
            )
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    degradation = changed - normal
    return {
        "group": group,
        "intervention": name,
        "split": split,
        "mean_degradation": float(degradation.mean()),
        "positive_seeds": int((degradation > 0.0).sum()),
        "seed_degradation": degradation.tolist(),
    }


def format_metric(item: dict[str, Any], key: str) -> str:
    return f"{item[key]['mean']:.6f} ± {item[key]['sd']:.6f}"


def format_contrast(item: dict[str, Any]) -> str:
    return (
        f"{item['mean_delta']:+.6f}; "
        f"[{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]; "
        f"{item['wins']}/10"
    )


def main() -> None:
    freeze = load(ROOT / "selection" / "validation_freeze.json")
    summaries = {
        split: {group: summarize(group, split) for group in GROUPS}
        for split in ("val", "test")
    }
    contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    farm_distributions: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("val", "test"):
        contrasts[split] = {}
        farm_distributions[split] = {}
        for spatial in SPATIALS:
            for candidate_operator, control_operator in (
                ("add", "none"),
                ("int", "none"),
                ("int", "add"),
            ):
                candidate = f"{spatial}_{candidate_operator}"
                control = f"{spatial}_{control_operator}"
                key = f"{candidate}_minus_{control}"
                contrasts[split][key] = paired(candidate, control, split)
                farm_distributions[split][key] = per_station(candidate, control, split)
        for operator in OPERATORS:
            for spatial in ("fixed4", "learned4"):
                candidate = f"{spatial}_{operator}"
                control = f"bilinear_{operator}"
                key = f"{candidate}_minus_{control}"
                contrasts[split][key] = paired(candidate, control, split)
                farm_distributions[split][key] = per_station(candidate, control, split)
        for spatial in ("fixed4", "learned4"):
            contrasts[split][f"{spatial}_interaction_did"] = interaction_did(
                spatial, split
            )

    interventions = {
        split: {
            group: {
                name: intervention(group, name, split) for name in INTERVENTIONS
            }
            for group in GROUPS
            if not group.endswith("_none")
        }
        for split in ("val", "test")
    }
    intervention_did: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("val", "test"):
        intervention_did[split] = {}
        for spatial in SPATIALS:
            for name in INTERVENTIONS:
                interaction_values = np.asarray(
                    interventions[split][f"{spatial}_int"][name]["seed_degradation"]
                )
                additive_values = np.asarray(
                    interventions[split][f"{spatial}_add"][name]["seed_degradation"]
                )
                intervention_did[split][f"{spatial}_{name}"] = contrast_from_delta(
                    f"{spatial}_{name}_interaction_minus_additive_degradation",
                    interaction_values - additive_values,
                    candidate=f"{spatial}_int degradation",
                    control=f"{spatial}_add degradation",
                    split=split,
                )

    interaction_support = {}
    for spatial in SPATIALS:
        val = contrasts["val"][f"{spatial}_int_minus_{spatial}_add"]
        test = contrasts["test"][f"{spatial}_int_minus_{spatial}_add"]
        interaction_support[spatial] = bool(
            val["ci95"][1] < 0.0
            and val["wins"] >= 7
            and test["mean_delta"] < 0.0
            and test["wins"] >= 7
        )
    surrounding_specific = {}
    for spatial in ("fixed4", "learned4"):
        val = contrasts["val"][f"{spatial}_interaction_did"]
        test = contrasts["test"][f"{spatial}_interaction_did"]
        surrounding_specific[spatial] = bool(
            interaction_support[spatial]
            and val["ci95"][1] < 0.0
            and test["mean_delta"] < 0.0
        )

    decision = {
        "interaction_supported_by_spatial_mode": interaction_support,
        "surrounding_weather_specific_interaction_supported": surrounding_specific,
        "switch_primary_task_to_single_farm_state_modulation": bool(
            any(surrounding_specific.values())
        ),
        "rule": (
            "Switch only if a 4x4 Int-Add contrast is significant on frozen validation, "
            "confirms direction on 2024, and its interaction advantage is stronger than "
            "bilinear under the paired support DID."
        ),
    }
    output = {
        "validation_freeze": freeze,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "farm_distributions": farm_distributions,
        "interventions": interventions,
        "intervention_interaction_minus_additive": intervention_did,
        "decision": decision,
    }
    summary_path = ROOT / "analysis" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    main_rows = []
    for spatial in SPATIALS:
        for operator in OPERATORS:
            group = f"{spatial}_{operator}"
            main_rows.append(
                f"| `{group}` | {format_metric(summaries['val'][group], 'station_mae_cf')} | "
                f"{format_metric(summaries['test'][group], 'station_mae_cf')} | "
                f"{summaries['val'][group]['cluster_mae_mw']['mean']:.3f} | "
                f"{summaries['test'][group]['cluster_mae_mw']['mean']:.3f} |"
            )
    contrast_rows = []
    for spatial in SPATIALS:
        for candidate_operator in ("add", "int"):
            key = f"{spatial}_{candidate_operator}_minus_{spatial}_none"
            contrast_rows.append(
                f"| `{key}` | {format_contrast(contrasts['val'][key])} | "
                f"{format_contrast(contrasts['test'][key])} |"
            )
        key = f"{spatial}_int_minus_{spatial}_add"
        contrast_rows.append(
            f"| **`{key}`** | **{format_contrast(contrasts['val'][key])}** | "
            f"**{format_contrast(contrasts['test'][key])}** |"
        )
    did_rows = []
    for spatial in ("fixed4", "learned4"):
        key = f"{spatial}_interaction_did"
        did_rows.append(
            f"| `{key}` | {format_contrast(contrasts['val'][key])} | "
            f"{format_contrast(contrasts['test'][key])} |"
        )
    intervention_rows = []
    for spatial in SPATIALS:
        for operator in ("add", "int"):
            group = f"{spatial}_{operator}"
            values_text = []
            for name in INTERVENTIONS:
                item = interventions["val"][group][name]
                values_text.append(
                    f"{item['mean_degradation']:+.6f} ({item['positive_seeds']}/10)"
                )
            intervention_rows.append(
                f"| `{group}` | " + " | ".join(values_text) + " |"
            )

    decision_text = (
        "至少一种周边 4×4 表示满足预注册的 interaction 与 support-DID 判据，"
        "可以把单场 state modulation 作为主任务候选。"
        if decision["switch_primary_task_to_single_farm_state_modulation"]
        else "没有周边 4×4 表示同时通过 interaction 与 support-DID 判据；停止把 state modulation 作为主线，保留可分离局地动力学的集群路线。"
    )
    document = f"""# 单场空间支持 × 历史利用方式决定性实验

## 实验目的

本实验补上此前缺失的严格交叉矩阵。所有模型使用同一批 56 场、同一 3 h history baseline、同一共享 backbone、相同 station-only loss 和相同训练预算，只改变：

1. 气象空间支持：raw bilinear、fixed 4×4 Gaussian、learned 4×4 + wind geometry；
2. response branch：no-extra-history、additive history、concat interaction。

主指标为 56 场等权 station macro CF MAE。2023 完成全矩阵和冻结后，才读取 2024。主机制比较是每种空间支持内部的 `Int−Add`，而不是 `Int−None`。周边天气必要性由以下差分中的差分判断：

\[
\bigl(\mathrm{{Int}}-\mathrm{{Add}}\bigr)_{{4\times4}}
-
\bigl(\mathrm{{Int}}-\mathrm{{Add}}\bigr)_{{bilinear}}.
\]

负值且配对区间不跨 0，才说明 4×4 特异地增强 interaction。

## 10-seed 主结果

| 模型 | 2023 station CF MAE | 2024 station CF MAE | 2023 summed cluster MAE | 2024 summed cluster MAE |
|---|---:|---:|---:|---:|
{chr(10).join(main_rows)}

## 历史机制配对比较

表中每个单元依次为 `mean Δ | paired 95% CI | wins`；负值代表前者更好。

| 比较 | 2023 validation | 2024 frozen test |
|---|---:|---:|
{chr(10).join(contrast_rows)}

## 周边天气是否特异地增强 interaction

| support DID | 2023 validation | 2024 frozen test |
|---|---:|---:|
{chr(10).join(did_rows)}

## 状态干预

下表为 2023 station CF MAE degradation，括号内为正退化 seeds。Add 与 Int 都执行相同干预，避免把“使用历史”误写成“识别 interaction”。

| 模型 | strong donor | far-station permutation | reference state |
|---|---:|---:|---:|
{chr(10).join(intervention_rows)}

## 决策

{decision_text}

机器判定：

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

场站不能按结果事后筛选。逐场结果、完整配对 seed 差值、干预及参数量保存在机器汇总中。

## 证据

- `artifacts/local_state_spatial_interaction_v1/selection/validation_freeze.json`
- `artifacts/local_state_spatial_interaction_v1/analysis/summary.json`
- `artifacts/local_state_spatial_interaction_v1/COMPLETE`
"""
    DOC.write_text(document, encoding="utf-8")
    print(f"[analysis] wrote {summary_path}")
    print(f"[analysis] wrote {DOC}")


if __name__ == "__main__":
    main()
