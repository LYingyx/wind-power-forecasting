from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_state_transfer_multistage_v1"
DOC = PROJECT / "exp_docs" / "local_state_transfer_multistage_v1_results.md"
SEEDS = tuple(range(42, 52))
HORIZON_KEYS = (
    "horizon_0_1h_mae_mw",
    "horizon_1_3h_mae_mw",
    "horizon_3_6h_mae_mw",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    return load(ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json")


def values(stage: str, group: str, key: str, split: str = "val") -> np.ndarray:
    return np.asarray([float(metric(stage, group, seed, split)[key]) for seed in SEEDS])


def paired(stage: str, candidate: str, control: str, key: str, split: str = "val") -> dict[str, Any]:
    delta = values(stage, candidate, key, split) - values(stage, control, key, split)
    critical = 2.262
    half = critical * float(delta.std(ddof=1)) / math.sqrt(len(delta))
    return {
        "candidate": candidate,
        "control": control,
        "key": key,
        "split": split,
        "candidate_mean": float(values(stage, candidate, key, split).mean()),
        "control_mean": float(values(stage, control, key, split).mean()),
        "mean_delta": float(delta.mean()),
        "ci95": [float(delta.mean() - half), float(delta.mean() + half)],
        "wins": int((delta < 0.0).sum()),
        "seed_delta": delta.tolist(),
    }


def intervention(
    view: str,
    stage: str,
    group: str,
    name: str,
    key: str,
    split: str = "val",
) -> dict[str, Any]:
    normal = values(stage, group, key, split)
    changed = np.asarray(
        [
            float(
                load(
                    ROOT / "interventions" / view / group / name / f"seed{seed}"
                    / f"metrics_{split}.json"
                )[key]
            )
            for seed in SEEDS
        ]
    )
    delta = changed - normal
    return {
        "intervention": name,
        "key": key,
        "mean_degradation": float(delta.mean()),
        "positive": int((delta > 0.0).sum()),
        "seed_degradation": delta.tolist(),
    }


def summary(stage: str, group: str, split: str = "val") -> dict[str, Any]:
    keys = (
        "cluster_mae_mw", "cluster_rmse_mw", "cluster_ramp_mae_mw",
        "station_mae_cf", *HORIZON_KEYS,
    )
    result: dict[str, Any] = {"stage": stage, "group": group, "split": split}
    for key in keys:
        array = values(stage, group, key, split)
        result[key] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
        }
    return result


def per_station_comparison(stage: str, candidate: str, control: str) -> dict[str, Any]:
    candidate_values = np.asarray(
        [metric(stage, candidate, seed)["per_station_mae_cf"] for seed in SEEDS]
    )
    control_values = np.asarray(
        [metric(stage, control, seed)["per_station_mae_cf"] for seed in SEEDS]
    )
    delta = (candidate_values - control_values).mean(axis=0)
    return {
        "farms_improved": int((delta < 0.0).sum()),
        "farms_degraded": int((delta > 0.0).sum()),
        "median_delta_cf": float(np.median(delta)),
        "best_farm_indices": np.argsort(delta)[:10].tolist(),
        "worst_farm_indices": np.argsort(delta)[-10:][::-1].tolist(),
        "per_farm_delta_cf": delta.tolist(),
    }


def format_cell(result: dict[str, Any], key: str = "cluster_mae_mw") -> str:
    value = result[key]
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


def main() -> None:
    freeze = load(ROOT / "selection" / "validation_freeze.json")
    final_cells = [tuple(value.split("/", 1)) for value in freeze["formal_cluster_mae_means"]]
    model_summaries = {
        f"{stage}/{group}": {
            "val": summary(stage, group, "val"),
            "test": summary(stage, group, "test"),
        }
        for stage, group in final_cells
    }
    comparisons = {
        "stage1_state": paired("formal_station", "s_int", "s0", "station_mae_cf"),
        "e2e_state": paired("formal_joint", "e_int", "e0", "cluster_mae_mw"),
        "pretrained_free_state": paired(
            "formal_transfer", "p_int_free", "p0_free", "cluster_mae_mw"
        ),
        "pretrained_preserve_state": paired(
            "formal_transfer", "p_int_preserve", "p0_free", "cluster_mae_mw"
        ),
        "preserve_vs_free": paired(
            "formal_transfer", "p_int_preserve", "p_int_free", "cluster_mae_mw"
        ),
    }
    if (ROOT / "formal_joint" / "e_add").exists():
        comparisons["e2e_interaction_vs_additive"] = paired(
            "formal_joint", "e_int", "e_add", "cluster_mae_mw"
        )
        comparisons["pretrained_interaction_vs_additive"] = paired(
            "formal_transfer", "p_int_free", "p_add_free", "cluster_mae_mw"
        )
        comparisons["stage1_interaction_vs_additive"] = paired(
            "formal_station", "s_int", "s_add", "station_mae_cf"
        )

    interventions = {
        "stage1": {
            name: intervention(
                "formal_station", "formal_station", "s_int", name, "station_mae_cf"
            )
            for name in ("strong_sample", "station_permutation", "reference")
        },
        "e2e": {
            name: intervention(
                "formal_joint", "formal_joint", "e_int", name, "cluster_mae_mw"
            )
            for name in ("strong_sample", "station_permutation", "reference")
        },
        "pretrained_free": {
            name: intervention(
                "formal_transfer", "formal_transfer", "p_int_free", name,
                "cluster_mae_mw",
            )
            for name in ("strong_sample", "station_permutation", "reference")
        },
        "pretrained_preserve": {
            name: intervention(
                "formal_transfer", "formal_transfer", "p_int_preserve", name,
                "cluster_mae_mw",
            )
            for name in ("strong_sample", "station_permutation", "reference")
        },
    }
    farm_distribution = {
        "stage1": per_station_comparison("formal_station", "s_int", "s0"),
        "e2e": per_station_comparison("formal_joint", "e_int", "e0"),
        "pretrained_free": per_station_comparison(
            "formal_transfer", "p_int_free", "p0_free"
        ),
        "pretrained_preserve": per_station_comparison(
            "formal_transfer", "p_int_preserve", "p0_free"
        ),
    }

    stage1 = comparisons["stage1_state"]
    stage1_donor = interventions["stage1"]["strong_sample"]
    local_state_identified = bool(
        stage1["mean_delta"] < 0.0
        and stage1["wins"] >= 7
        and stage1_donor["mean_degradation"] > 0.0
        and stage1_donor["positive"] >= 7
    )
    e2e = comparisons["e2e_state"]
    free = comparisons["pretrained_free_state"]
    preserve = comparisons["pretrained_preserve_state"]
    cluster_state_survives = bool(
        (e2e["mean_delta"] < 0.0 and e2e["wins"] >= 7)
        or (free["mean_delta"] < 0.0 and free["wins"] >= 7)
        or (preserve["mean_delta"] < 0.0 and preserve["wins"] >= 7)
    )
    schedule_did = float(
        free["mean_delta"] - e2e["mean_delta"]
    )
    if local_state_identified and cluster_state_survives:
        route = (
            "继续 local-physical-unit → shared learning → physical sum 的集群路线；"
            "是否采用多阶段训练由 P1-vs-E1 和差分中的差分决定。"
        )
    elif local_state_identified:
        route = (
            "局地状态对单场目标可辨识、但没有稳定转化为集群收益。主论文若强调状态机理，"
            "应以场站预测为主任务；集群求和作为尺度效应分析，而非强行声称集群增益。"
        )
    else:
        route = (
            "在当前双线性局地 forcing 下，历史状态未被严格识别。停止继续增加 state operator/ID；"
            "在彻底放弃前只保留一个必要复核：用旧单场实验完全相同的 4×4 气象代理复现输入。"
        )

    result = {
        "validation_freeze": freeze,
        "model_summaries": model_summaries,
        "paired_comparisons": comparisons,
        "interventions": interventions,
        "farm_distribution": farm_distribution,
        "decision": {
            "local_state_identified": local_state_identified,
            "cluster_state_survives": cluster_state_survives,
            "multistage_state_effect_minus_e2e_state_effect": schedule_did,
            "route": route,
        },
    }
    atomic = ROOT / "analysis" / "summary.json"
    atomic.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for stage, group in final_cells:
        item = model_summaries[f"{stage}/{group}"]
        rows.append(
            f"| {group} | {format_cell(item['val'])} | {format_cell(item['test'])} | "
            f"{item['val']['station_mae_cf']['mean']:.5f} |"
        )
    comp_rows = []
    for name, item in comparisons.items():
        comp_rows.append(
            f"| {name} | {item['mean_delta']:.6f} | "
            f"[{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] | {item['wins']}/10 |"
        )
    intervention_rows = []
    for phase, phase_values in interventions.items():
        for name, item in phase_values.items():
            intervention_rows.append(
                f"| {phase} | {name} | {item['mean_degradation']:.6f} | "
                f"{item['positive']}/10 |"
            )

    text = f"""# Local state 多阶段迁移实验结果

## 实验回答的问题

本实验不预设多阶段训练优于端到端训练。它以场站为预测单元、跨场共享编码器与响应网络、容量加权物理求和为集群输出，严格比较：

- `E0/E1`：从头端到端联合训练，无/有 local state interaction；
- `P0/P1-Free`：station-only 预训练后联合微调，无/有 local state interaction；
- `P1-Preserve`：联合微调时冻结已学得的 state encoder 与 interaction；
- `Additive`：同参数化的历史主效应控制，用来区分 state 本身预测与 weather×state interaction。

Stage 1 只按等权场站 capacity-factor MAE 选 checkpoint；Stage 2 才按 cluster MAE 选。2024 test 在 `selection/validation_freeze.json` 写入后才读取。

> 适用边界：本轮使用固定双线性局地 Pangu forcing。它检验训练日程与局地状态是否被 aggregate optimization 舍弃，但不等价于旧单场实验的 4×4 周边气象代理。

## 10-seed 主结果

| 模型 | 2023 cluster MAE (MW) | 2024 cluster MAE (MW) | 2023 station macro CF MAE |
|---|---:|---:|---:|
{chr(10).join(rows)}

配对差值均为 candidate − control，负值代表候选更好：

| 比较 | mean Δ | paired 95% CI | wins |
|---|---:|---:|---:|
{chr(10).join(comp_rows)}

## 状态干预

正 degradation 表示替换状态后性能恶化，即模型确实依赖正确状态：

| 阶段 | 干预 | mean degradation | positive seeds |
|---|---|---:|---:|
{chr(10).join(intervention_rows)}

## 严格判定

- 局地状态在 station-only 阶段被识别：`{local_state_identified}`
- 状态收益在集群目标下保留：`{cluster_state_survives}`
- 多阶段相对端到端对“状态增量”的差分中的差分：`{schedule_did:.6f}`（负值表示多阶段更有利于保留状态）
- 冻结式与自由微调差异：`{comparisons['preserve_vs_free']['mean_delta']:.6f} MW`

结论：{route}

不能按“挑选支持结论的场站”形成主结果。逐场结果只用于异质性分析或预注册分层；完整 56 场 macro 与 cluster sum 始终是主证据。逐场差值见机器可读文件。

## 证据文件

- `artifacts/local_state_transfer_multistage_v1/analysis/summary.json`
- `artifacts/local_state_transfer_multistage_v1/selection/screening.json`
- `artifacts/local_state_transfer_multistage_v1/selection/validation_freeze.json`
"""
    DOC.write_text(text, encoding="utf-8")
    print(f"[analysis] wrote {atomic}")
    print(f"[analysis] wrote {DOC}")


if __name__ == "__main__":
    main()
