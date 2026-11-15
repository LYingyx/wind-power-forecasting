from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_state_cluster_transfer_v2"
DOC = PROJECT / "exp_docs" / "local_state_cluster_transfer_v2_results.md"
SOURCE_DOC = PROJECT / "exp_docs" / "local_state_spatial_interaction_v1_2023_report.md"
SEEDS = tuple(range(42, 52))
GROUPS = (
    "b_shared_add",
    "b_independent_add",
    "b_int",
    "f_shared_add",
    "f_independent_add",
    "f_int",
)
ADDITIVES = ("shared_add", "independent_add")
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
HORIZON_SLICES = {
    "all": slice(0, 24),
    "0_1h": slice(0, 4),
    "1_3h": slice(4, 12),
    "3_6h": slice(12, 24),
}
T_CRITICAL_10_SEEDS = 2.262


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_dir(stage: str, group: str, seed: int) -> Path:
    return ROOT / stage / group / f"seed{seed}"


def metric(stage: str, group: str, seed: int, split: str) -> dict[str, Any]:
    return load_json(run_dir(stage, group, seed) / f"metrics_{split}.json")


def metric_values(stage: str, group: str, key: str, split: str) -> np.ndarray:
    return np.asarray(
        [float(metric(stage, group, seed, split)[key]) for seed in SEEDS],
        dtype=np.float64,
    )


def paired_summary(
    delta: Sequence[float] | np.ndarray,
    *,
    candidate: str,
    control: str,
    metric_name: str,
    split: str,
) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a finite one-dimensional array")
    mean = float(values.mean())
    half = T_CRITICAL_10_SEEDS * float(values.std(ddof=1)) / math.sqrt(len(values))
    return {
        "candidate": candidate,
        "control": control,
        "metric": metric_name,
        "split": split,
        "candidate_minus_control_mean": mean,
        "ci95": [mean - half, mean + half],
        "wins": int((values < 0.0).sum()),
        "seed_delta": values.tolist(),
    }


def paired_metric(
    stage: str, candidate: str, control: str, key: str, split: str
) -> dict[str, Any]:
    return paired_summary(
        metric_values(stage, candidate, key, split)
        - metric_values(stage, control, key, split),
        candidate=f"{stage}/{candidate}",
        control=f"{stage}/{control}",
        metric_name=key,
        split=split,
    )


def passes_paired(result: Mapping[str, Any], minimum_wins: int = 7) -> bool:
    """The frozen directional rule used for both validation and test confirmation."""
    return bool(
        float(result["candidate_minus_control_mean"]) < 0.0
        and float(result["ci95"][1]) < 0.0
        and int(result["wins"]) >= minimum_wins
    )


def summarize_model(stage: str, group: str, split: str) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage, "group": group, "split": split}
    for key in METRICS:
        values = metric_values(stage, group, key, split)
        result[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return result


def per_farm(stage: str, candidate: str, control: str, split: str) -> dict[str, Any]:
    candidate_values = np.asarray(
        [metric(stage, candidate, seed, split)["per_station_mae_cf"] for seed in SEEDS],
        dtype=np.float64,
    )
    control_values = np.asarray(
        [metric(stage, control, seed, split)["per_station_mae_cf"] for seed in SEEDS],
        dtype=np.float64,
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


def support_did(
    stage: str, additive: str, key: str, split: str
) -> dict[str, Any]:
    fixed_advantage = metric_values(stage, "f_int", key, split) - metric_values(
        stage, f"f_{additive}", key, split
    )
    bilinear_advantage = metric_values(stage, "b_int", key, split) - metric_values(
        stage, f"b_{additive}", key, split
    )
    return paired_summary(
        fixed_advantage - bilinear_advantage,
        candidate=f"({stage}/f_int-f_{additive})",
        control=f"({stage}/b_int-b_{additive})",
        metric_name=f"support_DID/{key}",
        split=split,
    )


def intervention_result(
    view: str,
    stage: str,
    group: str,
    intervention: str,
    key: str,
    split: str,
) -> dict[str, Any]:
    normal = metric_values(stage, group, key, split)
    changed = np.asarray(
        [
            float(
                load_json(
                    ROOT
                    / "interventions"
                    / view
                    / group
                    / intervention
                    / f"seed{seed}"
                    / f"metrics_{split}.json"
                )[key]
            )
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    degradation = changed - normal
    return {
        "view": view,
        "group": group,
        "intervention": intervention,
        "metric": key,
        "split": split,
        "mean_degradation": float(degradation.mean()),
        "positive_seeds": int((degradation > 0.0).sum()),
        "seed_degradation": degradation.tolist(),
    }


def residual_decomposition(
    station_prediction_mw: np.ndarray,
    station_target_mw: np.ndarray,
) -> dict[str, float]:
    """Exact population-moment decomposition of physical-sum squared error."""
    prediction = np.asarray(station_prediction_mw, dtype=np.float64)
    target = np.asarray(station_target_mw, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Station prediction and target must have matching [sample,farm,lead]")
    # Treat every sample-lead pair in the requested horizon as one observation.
    errors = np.moveaxis(prediction - target, 1, -1).reshape(-1, prediction.shape[1])
    farm_bias = errors.mean(axis=0)
    centered = errors - farm_bias
    diagonal_variance = float(np.square(centered).mean(axis=0).sum())
    aggregate_error = errors.sum(axis=1)
    aggregate_bias = float(aggregate_error.mean())
    aggregate_variance = float(np.square(aggregate_error - aggregate_bias).mean())
    cross_covariance = aggregate_variance - diagonal_variance
    aggregate_mse = float(np.square(aggregate_error).mean())
    reconstructed = aggregate_bias**2 + diagonal_variance + cross_covariance
    cancellation_ratio = (
        float(-cross_covariance / diagonal_variance)
        if diagonal_variance > 0.0
        else float("nan")
    )
    return {
        "aggregate_mse_mw2": aggregate_mse,
        "aggregate_bias_mw": aggregate_bias,
        "aggregate_bias_squared_mw2": aggregate_bias**2,
        "diagonal_variance_mw2": diagonal_variance,
        "cross_covariance_mw2": cross_covariance,
        "aggregate_variance_mw2": aggregate_variance,
        "cancellation_ratio": cancellation_ratio,
        "closure_error_mw2": aggregate_mse - reconstructed,
    }


def decomposition_for_archive(path: Path) -> dict[str, dict[str, float]]:
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["station_prediction_mw"], dtype=np.float64)
        target = np.asarray(archive["station_target_mw"], dtype=np.float64)
        if "cluster_prediction_mw" in archive.files:
            summed = prediction.sum(axis=1)
            cluster = np.asarray(archive["cluster_prediction_mw"], dtype=np.float64)
            if not np.allclose(summed, cluster, atol=2e-3, rtol=2e-6):
                raise ValueError(f"Station predictions do not physically sum in {path}")
    return {
        label: residual_decomposition(prediction[:, :, leads], target[:, :, leads])
        for label, leads in HORIZON_SLICES.items()
    }


def covariance_analysis() -> dict[str, Any]:
    decompositions: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    terms = (
        "aggregate_mse_mw2",
        "aggregate_bias_squared_mw2",
        "diagonal_variance_mw2",
        "cross_covariance_mw2",
        "cancellation_ratio",
        "closure_error_mw2",
    )
    for split in ("val", "test"):
        decompositions[split] = {}
        paired[split] = {}
        for stage in ("station", "transfer"):
            decompositions[split][stage] = {}
            for group in GROUPS:
                decompositions[split][stage][group] = [
                    decomposition_for_archive(
                        run_dir(stage, group, seed) / f"predictions_{split}.npz"
                    )
                    for seed in SEEDS
                ]
            paired[split][stage] = {}
            for response in ("shared_add", "independent_add", "int"):
                for horizon in HORIZON_SLICES:
                    for term in terms:
                        fixed = np.asarray(
                            [
                                item[horizon][term]
                                for item in decompositions[split][stage][f"f_{response}"]
                            ]
                        )
                        bilinear = np.asarray(
                            [
                                item[horizon][term]
                                for item in decompositions[split][stage][f"b_{response}"]
                            ]
                        )
                        key = f"{response}/{horizon}/{term}"
                        paired[split][stage][key] = paired_summary(
                            fixed - bilinear,
                            candidate=f"{stage}/f_{response}",
                            control=f"{stage}/b_{response}",
                            metric_name=f"{horizon}/{term}",
                            split=split,
                        )
    max_closure = max(
        abs(item[horizon]["closure_error_mw2"])
        for split_values in decompositions.values()
        for stage_values in split_values.values()
        for group_values in stage_values.values()
        for item in group_values
        for horizon in HORIZON_SLICES
    )
    return {
        "definition": (
            "MSE(sum e_i)=bias(sum e_i)^2+sum_i Var(e_i)+"
            "2*sum_{i<j}Cov(e_i,e_j); cancellation_ratio=-cross/diagonal, "
            "so positive values mean cross-farm covariance reduces aggregate variance."
        ),
        "decompositions": decompositions,
        "fixed_minus_bilinear_paired": paired,
        "maximum_absolute_closure_error_mw2": float(max_closure),
    }


def format_summary(item: Mapping[str, Any], key: str) -> str:
    return f"{item[key]['mean']:.3f} ± {item[key]['sd']:.3f}"


def format_contrast(item: Mapping[str, Any], digits: int = 3) -> str:
    delta = float(item["candidate_minus_control_mean"])
    low, high = item["ci95"]
    return f"{delta:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}], {item['wins']}/10"


def replace_marked_section(path: Path, section: str) -> None:
    start = "<!-- CLUSTER_TRANSFER_V2_START -->"
    end = "<!-- CLUSTER_TRANSFER_V2_END -->"
    marked = f"{start}\n{section.rstrip()}\n{end}"
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    if start in original and end in original:
        before = original.split(start, 1)[0].rstrip()
        after = original.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{marked}\n"
        if after:
            updated += f"\n{after}"
    else:
        updated = f"{original.rstrip()}\n\n{marked}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    freeze = load_json(ROOT / "selection" / "validation_freeze.json")
    summaries = {
        split: {
            stage: {group: summarize_model(stage, group, split) for group in GROUPS}
            for stage in ("station", "transfer")
        }
        for split in ("val", "test")
    }
    contrasts: dict[str, Any] = {}
    farm_results: dict[str, Any] = {}
    did: dict[str, Any] = {}
    for split in ("val", "test"):
        contrasts[split], farm_results[split], did[split] = {}, {}, {}
        for stage in ("station", "transfer"):
            contrasts[split][stage], farm_results[split][stage] = {}, {}
            primary = "station_mae_cf" if stage == "station" else "cluster_mae_mw"
            for prefix in ("b", "f"):
                for additive in ADDITIVES:
                    key = f"{prefix}_int-minus-{prefix}_{additive}"
                    contrasts[split][stage][key] = paired_metric(
                        stage, f"{prefix}_int", f"{prefix}_{additive}", primary, split
                    )
                    farm_results[split][stage][key] = per_farm(
                        stage, f"{prefix}_int", f"{prefix}_{additive}", split
                    )
                for response in ("shared_add", "independent_add", "int"):
                    key = f"f_{response}-minus-b_{response}"
                    contrasts[split][stage][key] = paired_metric(
                        stage, f"f_{response}", f"b_{response}", primary, split
                    )
                    farm_results[split][stage][key] = per_farm(
                        stage, f"f_{response}", f"b_{response}", split
                    )
            did[split][stage] = {
                additive: support_did(stage, additive, primary, split)
                for additive in ADDITIVES
            }

    interventions: dict[str, Any] = {}
    sensitivity_change: dict[str, Any] = {}
    for split in ("val", "test"):
        interventions[split], sensitivity_change[split] = {}, {}
        for stage in ("station", "transfer"):
            interventions[split][stage] = {}
            for group in GROUPS:
                interventions[split][stage][group] = {}
                for name in INTERVENTIONS:
                    interventions[split][stage][group][name] = {
                        key: intervention_result(stage, stage, group, name, key, split)
                        for key in ("station_mae_cf", "cluster_mae_mw")
                    }
        for group in GROUPS:
            sensitivity_change[split][group] = {}
            for name in INTERVENTIONS:
                sensitivity_change[split][group][name] = {}
                for key in ("station_mae_cf", "cluster_mae_mw"):
                    station_delta = np.asarray(
                        interventions[split]["station"][group][name][key]["seed_degradation"]
                    )
                    transfer_delta = np.asarray(
                        interventions[split]["transfer"][group][name][key]["seed_degradation"]
                    )
                    sensitivity_change[split][group][name][key] = paired_summary(
                        transfer_delta - station_delta,
                        candidate=f"transfer sensitivity/{group}",
                        control=f"station sensitivity/{group}",
                        metric_name=f"intervention_degradation/{key}",
                        split=split,
                    )

    covariance = covariance_analysis()
    station_support: dict[str, bool] = {}
    cluster_support: dict[str, bool] = {}
    for prefix in ("b", "f"):
        station_support[prefix] = all(
            passes_paired(contrasts[split]["station"][f"{prefix}_int-minus-{prefix}_{additive}"])
            for split in ("val", "test")
            for additive in ADDITIVES
        )
        cluster_support[prefix] = all(
            passes_paired(contrasts[split]["transfer"][f"{prefix}_int-minus-{prefix}_{additive}"])
            for split in ("val", "test")
            for additive in ADDITIVES
        )
    fixed_specific = all(
        passes_paired(did[split]["transfer"][additive])
        for split in ("val", "test")
        for additive in ADDITIVES
    )
    covariance_keys = [
        covariance["fixed_minus_bilinear_paired"][split]["transfer"][
            "int/all/cross_covariance_mw2"
        ]
        for split in ("val", "test")
    ]
    stable_cross_covariance_increase = all(
        float(item["candidate_minus_control_mean"]) > 0.0
        and float(item["ci95"][0]) > 0.0
        and int(item["wins"]) <= 3
        for item in covariance_keys
    )
    if any(cluster_support.values()):
        route = (
            "interaction survives aggregation-aware optimization; retain regional cluster "
            "forecasting with the supported local response as its physical computation unit."
        )
    elif any(station_support.values()):
        route = (
            "interaction is identifiable for farms but does not survive the cluster target; "
            "use farm forecasting as the mechanism-focused primary task and cluster aggregation "
            "as the scale-effect extension."
        )
    else:
        route = (
            "interaction does not beat both additive controls under the frozen rule; stop the "
            "state-modulation claim and retain the best additive cluster model."
        )
    decision = {
        "station_interaction_supported": station_support,
        "cluster_interaction_supported": cluster_support,
        "fixed4_specific_interaction_supported": fixed_specific,
        "fixed4_cross_covariance_increase_supported": stable_cross_covariance_increase,
        "route": route,
        "rule": (
            "Int must beat both shared-output Add and equal-parameter independent dual-branch "
            "Add with upper paired 95% CI<0 and >=7/10 wins on 2023 and frozen 2024. "
            "Fixed4-specific interaction additionally requires both support-DIDs to pass."
        ),
    }
    result = {
        "validation_freeze": freeze,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "per_farm_contrasts": farm_results,
        "support_did": did,
        "state_interventions": interventions,
        "pretraining_to_transfer_sensitivity_change": sensitivity_change,
        "residual_covariance_decomposition": covariance,
        "decision": decision,
    }
    write_json(ROOT / "analysis" / "summary.json", result)

    rows = []
    for group in GROUPS:
        rows.append(
            f"| `{group}` | {format_summary(summaries['val']['station'][group], 'station_mae_cf')} | "
            f"{format_summary(summaries['test']['station'][group], 'station_mae_cf')} | "
            f"{format_summary(summaries['val']['transfer'][group], 'cluster_mae_mw')} | "
            f"{format_summary(summaries['test']['transfer'][group], 'cluster_mae_mw')} |"
        )
    contrast_rows = []
    for prefix in ("b", "f"):
        for additive in ADDITIVES:
            key = f"{prefix}_int-minus-{prefix}_{additive}"
            contrast_rows.append(
                f"| `{key}` | {format_contrast(contrasts['val']['station'][key], 6)} | "
                f"{format_contrast(contrasts['test']['station'][key], 6)} | "
                f"{format_contrast(contrasts['val']['transfer'][key])} | "
                f"{format_contrast(contrasts['test']['transfer'][key])} |"
            )
    spatial_rows = []
    for response in ("shared_add", "independent_add", "int"):
        key = f"f_{response}-minus-b_{response}"
        spatial_rows.append(
            f"| `{key}` | {format_contrast(contrasts['val']['station'][key], 6)} | "
            f"{format_contrast(contrasts['test']['station'][key], 6)} | "
            f"{format_contrast(contrasts['val']['transfer'][key])} | "
            f"{format_contrast(contrasts['test']['transfer'][key])} |"
        )
    did_rows = [
        f"| `{additive}` | {format_contrast(did['val']['transfer'][additive])} | "
        f"{format_contrast(did['test']['transfer'][additive])} |"
        for additive in ADDITIVES
    ]
    covariance_rows = []
    for term in (
        "aggregate_bias_squared_mw2",
        "diagonal_variance_mw2",
        "cross_covariance_mw2",
        "cancellation_ratio",
    ):
        val_item = covariance["fixed_minus_bilinear_paired"]["val"]["transfer"][
            f"int/all/{term}"
        ]
        test_item = covariance["fixed_minus_bilinear_paired"]["test"]["transfer"][
            f"int/all/{term}"
        ]
        digits = 6 if term == "cancellation_ratio" else 1
        covariance_rows.append(
            f"| `{term}` | {format_contrast(val_item, digits)} | "
            f"{format_contrast(test_item, digits)} |"
        )
    document = f"""# Bilinear / Fixed 4×4 × Add / Interaction 集群迁移实验

## 目的与冻结协议

本实验不是继续搜索结构，而是补足最后一个归因控制：已在 station-only 阶段观察到的 history–weather 非可分离性，经过 cluster-aware fine-tuning 后是否仍然成立。完整矩阵为两种气象支持乘三种响应：共享输出 Add、等参数独立双分支 Add、Interaction。独立 Add 使用互不共享的 history/weather 分支，且响应模块与 Interaction 参数量严格相同。

56 个场站全部保留；2022 train、2023 validation、冻结后 2024 test；seeds 42–51。先用 station CF loss 预训练 40 epochs，再从对应 checkpoint 自由微调 20 epochs，目标为 `station loss + 2 × cluster loss`。主结论要求 Interaction 同时超过两个 Add 对照。

## 主结果

| 模型 | 2023 station CF MAE | 2024 station CF MAE | 2023 transfer cluster MAE (MW) | 2024 transfer cluster MAE (MW) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

配对项为 candidate−control；负值更好，括号为配对 95% CI 与获胜 seeds。

| Interaction 比较 | 2023 station | 2024 station | 2023 cluster transfer | 2024 cluster transfer |
|---|---:|---:|---:|---:|
{chr(10).join(contrast_rows)}

## Bilinear 与 Fixed 4×4

| Fixed−Bilinear | 2023 station | 2024 station | 2023 cluster transfer | 2024 cluster transfer |
|---|---:|---:|---:|---:|
{chr(10).join(spatial_rows)}

Fixed 4×4 是否特异增强 interaction 由 support-DID 判断：

| Add 对照 | 2023 transfer DID | 2024 transfer DID |
|---|---:|---:|
{chr(10).join(did_rows)}

## 求和误差协方差分解

严格使用导出的 56 场 MW 残差验证：

\[
\operatorname{{MSE}}(\sum_i e_i)=\operatorname{{Bias}}(\sum_i e_i)^2
+\sum_i\operatorname{{Var}}(e_i)+2\sum_{{i<j}}\operatorname{{Cov}}(e_i,e_j).
\]

下表为 transfer-stage Interaction 的 Fixed4−Bilinear 配对差。`cancellation_ratio=-cross/diagonal`，正值表示跨场协方差帮助抵消误差。只有 cross-covariance 的配对区间跨年稳定不跨 0，才把“4×4 改变空间误差相关性”写成结论。

| 分解项 | 2023 | 2024 |
|---|---:|---:|
{chr(10).join(covariance_rows)}

所有 horizon（all、0–1 h、1–3 h、3–6 h）、station/transfer 两阶段和逐 seed 的精确分解均在 `analysis/summary.json`。最大恒等式闭合误差为 `{covariance['maximum_absolute_closure_error_mw2']:.3e}` MW²。

## 严格结论与路线判定

- Station interaction：`{json.dumps(station_support, ensure_ascii=False)}`
- Cluster-transfer interaction：`{json.dumps(cluster_support, ensure_ascii=False)}`
- Fixed 4×4 特异增强 interaction：`{fixed_specific}`
- Fixed 4×4 稳定增加跨场协方差：`{stable_cross_covariance_increase}`

结论：{route}

这里的“interaction”只表示历史发电与未来 Pangu 在预测上不可由匹配的独立主效应替代；没有额外观测时，不把它强行解释为已辨识的风机内部因果状态。不能按结果筛选场站形成主结论。

## 证据文件

- `artifacts/local_state_cluster_transfer_v2/selection/validation_freeze.json`
- `artifacts/local_state_cluster_transfer_v2/source_checkpoint_audit.json`
- `artifacts/local_state_cluster_transfer_v2/analysis/summary.json`
- `artifacts/local_state_cluster_transfer_v2/COMPLETE`
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(document, encoding="utf-8")
    replace_marked_section(SOURCE_DOC, "## Cluster-aware transfer 补充结论\n\n" + document)
    print(f"[analysis] wrote {ROOT / 'analysis' / 'summary.json'}")
    print(f"[analysis] wrote {DOC}")
    print(f"[analysis] updated {SOURCE_DOC}")


if __name__ == "__main__":
    main()
