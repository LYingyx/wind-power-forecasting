from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_state_identifiability_v2"
CACHE = PROJECT / "data" / "local_bilinear_56farm_v1"
SEEDS = tuple(range(42, 52))
METRICS = (
    "cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "cluster_ramp_mae_mw",
    "horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw",
    "station_mae_mw", "samples_per_second", "parameter_count",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(group: str, seed: int, split: str) -> dict[str, Any]:
    return load_json(ROOT / "formal" / group / f"seed{seed}" / f"metrics_{split}.json")


def run_archive(group: str, seed: int, split: str) -> Path:
    return ROOT / "formal" / group / f"seed{seed}" / f"predictions_{split}.npz"


def intervention_archive(group: str, seed: int, name: str, split: str) -> Path:
    return ROOT / "interventions" / group / name / f"seed{seed}" / f"predictions_{split}.npz"


def reference(group: str) -> str:
    if group == "p_noid":
        return "b0"
    if group == "p_id":
        return "id_only"
    if group == "p_affine":
        return "affine_only"
    return "b_static"


def summarize(group: str, split: str) -> dict[str, Any]:
    rows = [metric(group, seed, split) for seed in SEEDS]
    result: dict[str, Any] = {"group": group, "split": split}
    for key in METRICS:
        value = [float(row[key]) for row in rows if row.get(key) is not None]
        if value:
            result[key + "_mean"] = statistics.mean(value)
            result[key + "_sd"] = statistics.stdev(value) if len(value) > 1 else 0.0
    return result


def paired(candidate: str, control: str, split: str) -> dict[str, Any]:
    difference = np.asarray(
        [
            float(metric(candidate, seed, split)["cluster_mae_mw"])
            - float(metric(control, seed, split)["cluster_mae_mw"])
            for seed in SEEDS
        ]
    )
    mean = float(difference.mean())
    half = 2.262 * float(difference.std(ddof=1)) / math.sqrt(len(difference))
    try:
        from scipy.stats import ttest_rel

        pvalue = float(
            ttest_rel(
                [metric(candidate, seed, split)["cluster_mae_mw"] for seed in SEEDS],
                [metric(control, seed, split)["cluster_mae_mw"] for seed in SEEDS],
            ).pvalue
        )
    except ImportError:
        pvalue = float("nan")
    return {
        "candidate": candidate, "reference": control, "split": split,
        "mean_delta_mw": mean, "ci_low_mw": mean - half, "ci_high_mw": mean + half,
        "wins": int(np.sum(difference < 0.0)), "differences_mw": difference.tolist(),
        "pvalue_two_sided": pvalue,
    }


def holm(rows: list[dict[str, Any]]) -> None:
    finite = sorted(
        ((index, row["pvalue_two_sided"]) for index, row in enumerate(rows) if np.isfinite(row["pvalue_two_sided"])),
        key=lambda item: item[1],
    )
    running = 0.0
    count = len(finite)
    for rank, (index, pvalue) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * pvalue))
        rows[index]["holm_adjusted_pvalue"] = running


def block_bootstrap_intervention(
    group: str, name: str, split: str, repeats: int = 2000
) -> dict[str, Any]:
    daily: list[float] = []
    for seed in SEEDS:
        with np.load(run_archive(group, seed, split), allow_pickle=False) as normal, np.load(
            intervention_archive(group, seed, name, split), allow_pickle=False
        ) as changed:
            np.testing.assert_array_equal(normal["origin_unix"], changed["origin_unix"])
            base_error = np.abs(normal["cluster_prediction_mw"] - normal["cluster_target_mw"])
            changed_error = np.abs(changed["cluster_prediction_mw"] - changed["cluster_target_mw"])
            day = normal["origin_unix"] // 86400
            for value in np.unique(day):
                selected = day == value
                daily.append(float((changed_error[selected] - base_error[selected]).mean()))
    values = np.asarray(daily)
    rng = np.random.default_rng(9401 + sum(map(ord, group + name + split)))
    draws = np.empty(repeats)
    for repeat in range(repeats):
        draws[repeat] = rng.choice(values, len(values), replace=True).mean()
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "group": group, "intervention": name, "split": split,
        "mean_degradation_mw": float(values.mean()),
        "block_bootstrap_ci_low_mw": float(low),
        "block_bootstrap_ci_high_mw": float(high),
        "day_seed_blocks": len(values),
    }


def kmeans(coordinates: np.ndarray, groups: int) -> np.ndarray:
    if groups == 1:
        return np.zeros(len(coordinates), dtype=np.int64)
    if groups == len(coordinates):
        return np.arange(len(coordinates), dtype=np.int64)
    rng = np.random.default_rng(0)
    centers = coordinates[rng.choice(len(coordinates), groups, replace=False)].copy()
    labels = np.zeros(len(coordinates), dtype=np.int64)
    for _ in range(100):
        distance = ((coordinates[:, None] - centers[None]) ** 2).sum(-1)
        new = distance.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for group in range(groups):
            members = coordinates[labels == group]
            if len(members):
                centers[group] = members.mean(0)
    return labels


def scale_metrics(path: Path, labels: np.ndarray, capacity: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["station_prediction_mw"]
        target = archive["station_target_mw"]
    groups = int(labels.max()) + 1
    group_prediction = np.stack([prediction[:, labels == k].sum(1) for k in range(groups)], 1)
    group_target = np.stack([target[:, labels == k].sum(1) for k in range(groups)], 1)
    group_capacity = np.asarray([capacity[labels == k].sum() for k in range(groups)])
    error = group_prediction - group_target
    flat = error.transpose(0, 2, 1).reshape(-1, groups)
    if groups > 1:
        correlation = np.corrcoef(flat, rowvar=False)
        mean_correlation = float(correlation[np.triu_indices(groups, 1)].mean())
    else:
        mean_correlation = float("nan")
    station_error = prediction - target
    cancellation = float(
        np.abs(station_error.sum(1)).mean() / np.abs(station_error).sum(1).mean()
    )
    return {
        "group_count": groups,
        "mae_mw": float(np.abs(error).mean()),
        "capacity_normalized_mae": float(
            (np.abs(error) / group_capacity[None, :, None]).mean()
        ),
        "ramp_mae_mw": float(
            np.abs(np.diff(group_prediction, axis=2) - np.diff(group_target, axis=2)).mean()
        ),
        "mean_error_correlation": mean_correlation,
        "full_cluster_cancellation_ratio": cancellation,
        "group_station_counts": [int(np.sum(labels == k)) for k in range(groups)],
        "group_capacity_mw": group_capacity.tolist(),
    }


def multiscale(candidate: str, split: str) -> list[dict[str, Any]]:
    latitude = np.load(CACHE / "static" / "station_latitude.npy")
    longitude = np.load(CACHE / "static" / "station_longitude.npy")
    capacity = np.load(CACHE / "static" / "station_capacity_mw.npy")
    coordinates = np.stack((latitude, longitude), 1)
    coordinates = (coordinates - coordinates.mean(0)) / coordinates.std(0)
    control = reference(candidate)
    rows: list[dict[str, Any]] = []
    for count in (56, 8, 4, 2, 1):
        labels = kmeans(coordinates, count)
        candidate_rows = [scale_metrics(run_archive(candidate, seed, split), labels, capacity) for seed in SEEDS]
        control_rows = [scale_metrics(run_archive(control, seed, split), labels, capacity) for seed in SEEDS]
        shuffle_rows = [
            scale_metrics(intervention_archive(candidate, seed, "strong_sample", split), labels, capacity)
            for seed in SEEDS
        ]
        row = dict(candidate_rows[0])
        for key in (
            "mae_mw", "capacity_normalized_mae", "ramp_mae_mw",
            "mean_error_correlation", "full_cluster_cancellation_ratio",
        ):
            row[key] = float(np.nanmean([value[key] for value in candidate_rows]))
        row["state_gain_mw"] = float(
            np.mean([base["mae_mw"] - value["mae_mw"] for value, base in zip(candidate_rows, control_rows)])
        )
        row["strong_shuffle_degradation_mw"] = float(
            np.mean([changed["mae_mw"] - value["mae_mw"] for value, changed in zip(candidate_rows, shuffle_rows)])
        )
        rows.append(row)
    return rows


def station_correspondence(candidate: str, split: str) -> dict[str, float]:
    latitude = np.load(CACHE / "static" / "station_latitude.npy")
    longitude = np.load(CACHE / "static" / "station_longitude.npy")
    capacity = np.load(CACHE / "static" / "station_capacity_mw.npy")
    coordinates = np.stack((latitude, longitude), 1)
    coordinates = (coordinates - coordinates.mean(0)) / coordinates.std(0)
    result: dict[str, float] = {}
    for count, label in ((56, "farm"), (8, "subcluster")):
        labels = kmeans(coordinates, count)
        degradation = []
        for seed in SEEDS:
            normal = scale_metrics(run_archive(candidate, seed, split), labels, capacity)
            changed = scale_metrics(
                intervention_archive(candidate, seed, "station_permutation", split),
                labels,
                capacity,
            )
            degradation.append(changed["capacity_normalized_mae"] - normal["capacity_normalized_mae"])
        result[f"{label}_normalized_mae_degradation"] = float(np.mean(degradation))
        result[f"{label}_positive_seeds"] = int(np.sum(np.asarray(degradation) > 0.0))
    return result


def zero_invariance_error(group: str) -> float | None:
    if group == "p_direct":
        return None
    import torch

    from wind_forcing.data.dataset import PreparedCache
    from wind_forcing.models import build_forecast_model

    checkpoint = torch.load(
        ROOT / "formal" / group / "seed42" / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    static = PreparedCache(CACHE).static
    model = build_forecast_model(
        checkpoint["model_config"], static, history_steps=24, forecast_steps=24
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    weather = torch.zeros(3, len(static["station_capacity_mw"]), 24, model.d_model)
    condition = torch.randn(3, len(static["station_capacity_mw"]), model.d_model)
    with torch.no_grad():
        changed, _ = model._conditioned_weather(weather, condition)
    return float(changed.abs().max())


def table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Model | MAE / MW | RMSE / MW | R² | Ramp / MW | 0–1 h | 1–3 h | 3–6 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def f(key: str) -> str:
            return f"{row[key + '_mean']:.3f} ± {row[key + '_sd']:.3f}"
        lines.append(
            f"| {row['group']} | {f('cluster_mae_mw')} | {f('cluster_rmse_mw')} | "
            f"{f('cluster_r2')} | {f('cluster_ramp_mae_mw')} | "
            f"{f('horizon_0_1h_mae_mw')} | {f('horizon_1_3h_mae_mw')} | "
            f"{f('horizon_3_6h_mae_mw')} |"
        )
    return "\n".join(lines)


def main() -> None:
    screening = load_json(ROOT / "selection" / "screening.json")
    freeze = load_json(ROOT / "selection" / "validation_freeze.json")
    groups = freeze["formal_groups"]
    promoted = freeze["promoted"]
    summaries = {
        split: [summarize(group, split) for group in groups] for split in ("val", "test")
    }
    comparisons = []
    for group in promoted:
        comparisons.append(paired(group, reference(group), "val"))
        comparisons.append(paired(group, f"{group}_shuf", "val"))
        comparisons.append(paired(group, reference(group), "test"))
    holm(
        [
            row for row in comparisons
            if row["split"] == "val" and row["reference"] == reference(row["candidate"])
        ]
    )
    interventions = [
        block_bootstrap_intervention(group, name, split)
        for group in promoted
        for split in ("val", "test")
        for name in (("strong_sample", "station_permutation") if group in {
            "p_noid", "p_static", "p_direct", "gl_static", "p12_static", "physical_static",
            "hierarchical_static", "io_pair_static", "io_pair_id_static", "p_id", "p_affine",
        } else ("strong_sample",))
    ]
    scale = {
        f"{group}_{split}": multiscale(group, split)
        for group in promoted for split in ("val", "test")
    }
    formal_decisions: dict[str, Any] = {}
    for group in promoted:
        vs_ref = next(row for row in comparisons if row["candidate"] == group and row["reference"] == reference(group) and row["split"] == "val")
        vs_shuffle = next(row for row in comparisons if row["candidate"] == group and row["reference"] == f"{group}_shuf" and row["split"] == "val")
        intervention = next(row for row in interventions if row["group"] == group and row["intervention"] == "strong_sample" and row["split"] == "val")
        test = next(row for row in comparisons if row["candidate"] == group and row["reference"] == reference(group) and row["split"] == "test")
        correspondence = station_correspondence(group, "val") if group in {
            "p_noid", "p_static", "p_direct", "gl_static", "p12_static", "physical_static",
            "hierarchical_static", "io_pair_static", "io_pair_id_static", "p_id", "p_affine",
        } else None
        local_pass = bool(
            correspondence is None
            or (
                correspondence["farm_normalized_mae_degradation"] > 0.0
                and correspondence["farm_positive_seeds"] >= 8
                and correspondence["subcluster_normalized_mae_degradation"] > 0.0
                and correspondence["subcluster_positive_seeds"] >= 8
            )
        )
        invariance = zero_invariance_error(group)
        passed = bool(
            vs_ref["ci_high_mw"] < 0.0 and vs_ref["wins"] >= 8
            and vs_ref.get("holm_adjusted_pvalue", 1.0) < 0.05
            and vs_shuffle["ci_high_mw"] < 0.0
            and intervention["block_bootstrap_ci_low_mw"] > 0.0
            and test["mean_delta_mw"] < 0.0
            and local_pass
            and (invariance is None or invariance < 1e-6)
        )
        formal_decisions[group] = {
            "passed": passed,
            "paired_reference": vs_ref,
            "paired_train_shuffle": vs_shuffle,
            "strong_intervention": intervention,
            "2024_same_direction": test["mean_delta_mw"] < 0.0,
            "local_correspondence": correspondence,
            "local_correspondence_passed": local_pass,
            "zero_weather_invariant_max_cf": invariance,
        }
    passed = [group for group, row in formal_decisions.items() if row["passed"]]
    if any(group in passed for group in ("io_pair_static", "io_pair_id_static")):
        route = "pivot_to_response_state_identification"
    elif any(group in passed for group in ("p_static", "p12_static", "physical_static", "hierarchical_static", "gl_static", "p_id", "p_affine")):
        route = "continue_strict_local_or_hierarchical_state"
    elif "g_static" in passed:
        route = "retain_global_common_mode_state_only"
    elif "p_direct" in passed:
        route = "history_is_direct_predictive_input_not_weather_modulation"
    else:
        route = "stop_response_state_route"
    scale_effect = False
    for group in promoted:
        rows = scale[f"{group}_val"]
        if rows[-1]["state_gain_mw"] <= 0.0 and any(row["state_gain_mw"] > 0.0 for row in rows[:-1]):
            scale_effect = True
    recommendation = (
        "Keep all 56 farms as the primary cluster task. Report farm/subcluster effects as a predeclared "
        "scale analysis; never select farms because their validation/test gain supports the claim."
    )
    if scale_effect and not passed:
        recommendation += (
            " The evidence is a scale-effect finding, not support for a final full-cluster state model."
        )
    result = {
        "screening": screening,
        "validation_freeze": freeze,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "intervention_block_bootstrap": interventions,
        "multiscale": scale,
        "formal_decisions": formal_decisions,
        "route_decision": route,
        "scale_effect_detected": scale_effect,
        "cluster_vs_selected_farms_recommendation": recommendation,
        "scientific_limits": {
            "primary_population": "all 56 farms",
            "posthoc_farm_selection": False,
            "2024_pristine_blind": False,
            "causal_historical_pangu": True,
        },
    }
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    temporary = analysis / f".summary.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, analysis / "summary.json")
    scale_lines = []
    for group in promoted:
        scale_lines.append(f"### {group} / 2023")
        scale_lines.append("| groups | MAE MW | nMAE | Ramp MW | state gain MW | shuffle degradation MW |")
        scale_lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in scale[f"{group}_val"]:
            scale_lines.append(
                f"| {row['group_count']} | {row['mae_mw']:.3f} | {row['capacity_normalized_mae']:.5f} | "
                f"{row['ramp_mae_mw']:.3f} | {row['state_gain_mw']:+.3f} | "
                f"{row['strong_shuffle_degradation_mw']:+.3f} |"
            )
    decision_lines = [
        f"- `{group}`: formal pass = **{row['passed']}**, 2024同方向 = {row['2024_same_direction']}"
        for group, row in formal_decisions.items()
    ] or ["- 没有候选通过预注册 screening，因此没有状态候选进入正式阶段。"]
    document = f"""# 历史状态可辨识性、聚合尺度与路线停止实验结果

状态：全部完成。结构选择只使用2023 validation；2024在冻结后运行，但因既往研究已查看该年份，不能称为纯盲测试。

## 2023 validation（10 seeds）

{table(summaries['val'])}

## 冻结后的2024（10 seeds）

{table(summaries['test'])}

## 严格判定

{chr(10).join(decision_lines)}

路线结论：**`{route}`**。

## 多尺度证据

{chr(10).join(scale_lines) if scale_lines else '无候选通过screening，未进行事后挑选。'}

## 是否继续集群预测、是否筛选场站

{recommendation}

56/56场站在仅使用2022训练集定义的数据质量门槛下均合格；因此不存在按结果删除场站的客观依据。若局地或子集群有效而完整集群无效，应报告“误差抵消导致的尺度效应”，不能从验证/测试集挑几个场站重构主结论。若研究目标转为单场机制，必须预先按地理/容量/数据质量分层选取或对56场全部报告，而不能按效果选取。

机器可读证据：`artifacts/local_state_identifiability_v2/analysis/summary.json`。
"""
    (PROJECT / "exp_docs" / "local_state_identifiability_v2_results.md").write_text(
        document, encoding="utf-8"
    )
    print(f"[analysis] route={route} promoted={promoted}", flush=True)


if __name__ == "__main__":
    main()
