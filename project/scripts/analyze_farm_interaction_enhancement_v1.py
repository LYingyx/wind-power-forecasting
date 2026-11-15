from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from wind_forcing.experiment_guard import read_freeze

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts/farm_interaction_enhancement_v1"
CACHE = PROJECT / "data/model_cache_pad5_f8"
DOC = PROJECT / "exp_docs/farm_interaction_enhancement_v1_results.md"
FORMAL = tuple(range(42, 52))
SCREEN = (42, 44, 46, 48, 50)
LAGS = {1: "15m", 4: "1h", 12: "3h"}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_path(stage: str, group: str, role: str, seed: int) -> Path:
    return ROOT / ("baseline" if group == "m0" else stage) / f"{group}_{role}" / f"seed{seed}"


def train_thresholds() -> dict[int, tuple[float, float]]:
    history = np.load(CACHE / "train/station_history.npy", mmap_mode="r")
    future = np.load(CACHE / "train/station_future.npy", mmap_mode="r")
    capacity = np.load(CACHE / "static/station_capacity_mw.npy")
    result = {}
    for lag in LAGS:
        previous = np.concatenate((history[:, :, -lag:], future[:, :, :-lag]), axis=2)
        delta = np.asarray(future) - previous
        result[lag] = (float(np.quantile(np.abs(delta), 0.9)),
                       float(np.quantile(np.abs((delta * capacity[None, :, None]).sum(1)), 0.9)))
    return result


def safe_f1(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(predicted & target))
    fp = float(np.sum(predicted & ~target))
    fn = float(np.sum(~predicted & target))
    precision, recall = tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


def aligned_positions(split_sample: np.ndarray, sample: np.ndarray) -> np.ndarray:
    if len(np.unique(sample)) != len(sample) or np.any(np.diff(split_sample) <= 0):
        raise ValueError("Duplicate prediction IDs or unsorted cache IDs")
    positions = np.searchsorted(split_sample, sample)
    if np.any(positions >= len(split_sample)) or not np.array_equal(split_sample[positions], sample):
        raise ValueError("Prediction sample indices do not match the split cache")
    return positions


def array_metrics(prediction, target, history, capacity, thresholds):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 24:
        raise ValueError("Expected matching [sample, farm, 24] predictions/targets")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite predictions/targets")
    prediction, target = np.asarray(prediction,dtype=np.float64), np.asarray(target,dtype=np.float64)
    error = prediction - target
    farm = np.abs(error).mean((0, 2))
    lead = np.abs(error).mean((0, 1))
    result = {
        "station_mae_cf": float(farm.mean()),
        "station_rmse_cf": float(np.sqrt(np.square(error).mean((0, 2))).mean()),
        "horizon_0_1h_station_mae_cf": float(lead[:4].mean()),
        "horizon_1_3h_station_mae_cf": float(lead[4:12].mean()),
        "horizon_3_6h_station_mae_cf": float(lead[12:].mean()),
        "physical_sum_cluster_mae_mw": float(np.abs((error * capacity[None, :, None]).sum(1)).mean()),
        "out_of_range_cf_fraction": float(((prediction < 0) | (prediction > 1)).mean()),
    }
    level = error.mean(-1, keepdims=True)
    result.update(station_mse_cf2=float(np.square(error).mean()),
                  trajectory_mean_error_mse_cf2=float(np.square(level).mean()),
                  trajectory_shape_error_mse_cf2=float(np.square(error-level).mean()))
    for lag, label in LAGS.items():
        pd = prediction - np.concatenate((history[:, :, -lag:], prediction[:, :, :-lag]), axis=2)
        td = target - np.concatenate((history[:, :, -lag:], target[:, :, :-lag]), axis=2)
        result[f"station_ramp_{label}_forecast_internal_mae_cf"] = float(
            np.abs(error[:, :, lag:] - error[:, :, :-lag]).mean(dtype=np.float64))
        result[f"station_ramp_{label}_history_boundary_mae_cf"] = float(
            np.abs(error[:, :, :lag]).mean(dtype=np.float64))
        for level, pred, truth, threshold, unit in (
            ("station", pd, td, thresholds[lag][0], "cf"),
            ("cluster", (pd * capacity[None, :, None]).sum(1),
             (td * capacity[None, :, None]).sum(1), thresholds[lag][1], "mw"),
        ):
            result[f"{level}_ramp_{label}_mae_{unit}"] = float(np.abs(pred - truth).mean(dtype=np.float64))
            active = np.abs(truth) > 1e-8
            result[f"{level}_ramp_{label}_direction_accuracy"] = (
                float(np.mean(np.sign(pred[active]) == np.sign(truth[active]))) if active.any() else None)
            for name, value in zip(("precision", "recall", "f1"),
                                   safe_f1(np.abs(pred) >= threshold, np.abs(truth) >= threshold)):
                result[f"{level}_extreme_ramp_{label}_{name}"] = value
            extreme = np.abs(truth) >= threshold
            result[f"{level}_extreme_ramp_{label}_conditional_mae_{unit}"] = (
                float(np.abs(pred[extreme] - truth[extreme]).mean(dtype=np.float64)) if extreme.any() else None)
    return result, farm


def archive_metrics(path: Path, split: str, thresholds):
    if split == "test":
        read_freeze(ROOT)
    with np.load(path, allow_pickle=False) as archive:
        capacity = np.load(CACHE / "static/station_capacity_mw.npy")
        sample = archive["sample_index"].astype(np.int64)
        split_sample = np.load(CACHE / split / "sample_index.npy", mmap_mode="r")
        position = aligned_positions(split_sample, sample)
        if len(position) != len(split_sample):
            raise ValueError("Incomplete prediction archive")
        history = np.load(CACHE / split / "station_history.npy", mmap_mode="r")[position]
        prediction = archive["station_prediction_mw"].astype(np.float64) / capacity[None, :, None]
        target = archive["station_target_mw"].astype(np.float64) / capacity[None, :, None]
        expected_target = np.load(CACHE / split / "station_future.npy", mmap_mode="r")[position]
        if not np.allclose(target, expected_target, atol=2e-6, rtol=2e-6):
            raise ValueError("Prediction targets do not match frozen cache")
        metrics, farm = array_metrics(prediction, target, history, capacity, thresholds)
        attention = archive["history_response_attention_mean"].mean(0) if "history_response_attention_mean" in archive.files else None
        origins = np.asarray(archive["origin_unix"])
        days, inverse = np.unique(origins // 86400, return_inverse=True)
        sample_mae = np.abs(prediction - target).mean((1, 2))
        day_sums = np.bincount(inverse, weights=sample_mae)
        day_counts = np.bincount(inverse)
    return metrics, farm, attention, (days, day_sums, day_counts)


def derived_metrics(stage, group, role, seed, split, thresholds):
    return archive_metrics(run_path(stage, group, role, seed) / f"predictions_{split}.npz", split, thresholds)[:3]


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a - b).all():
        raise ValueError("Paired CI requires matching finite seed vectors, n >= 2")
    delta = a - b
    rng = np.random.default_rng(20260908)
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
            "wins": int((delta < 0).sum()), "n": len(delta), "ci_method": "10000 paired seed bootstrap"}


def day_paired_summary(a, b) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError("Unmatched seed/day arrays")
    days = a[0][0]
    for row in a + b:
        if not np.array_equal(row[0], days) or not np.array_equal(row[2], a[0][2]):
            raise ValueError("Paired day bootstrap requires identical dates and sample counts")
    sums = np.stack([x[1] - y[1] for x, y in zip(a, b)]).mean(0)
    counts = a[0][2]
    rng = np.random.default_rng(20260908)
    boot = []
    for _ in range(100):
        sampled = rng.integers(0, len(days), (100, len(days)))
        boot.extend((sums[sampled].sum(1) / counts[sampled].sum(1)).tolist())
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"day_ci_low": float(low), "day_ci_high": float(high),
            "day_ci_method": "10000 shared UTC-day blocks; mean of paired seed errors"}


def load_pairs() -> tuple[list[dict], list[dict]]:
    freeze = read_freeze(ROOT)
    formal = freeze["formal_pairs"]
    seen = {p["group"] for p in formal}
    if len(seen) != len(formal):
        raise ValueError("Duplicate formal groups")
    screen = []
    for p in freeze["screen_only_pairs"]:
        if p["group"] not in seen:
            screen.append(p)
            seen.add(p["group"])
    return formal, screen


def baseline_table(thresholds) -> list[dict]:
    read_freeze(ROOT)
    root = PROJECT / "artifacts/farm_level_history_weather_v1"
    selection = json.loads((root / "selection/history_validation_freeze.json").read_text())
    strong = {4: "h_strong_4", 12: "h_matched", 24: "h_strong_24"}[int(selection["selected_history_visible_steps"])]
    groups = {"H-Matched": root / "baseline/h_matched", "H-Strong": root / f"baseline/{strong}",
              "W": root / "baseline/w_anchor", "HW-None": root / "reused/hw_none",
              "HW-SAdd": root / "reused/hw_sadd", "HW-IAdd": root / "reused/hw_iadd",
              "HW-Int": root / "reused/hw_int"}
    rows = []
    for split in ("val", "test"):
        for name, directory in groups.items():
            for seed in FORMAL:
                metrics, _, _, _ = archive_metrics(directory / f"seed{seed}/predictions_{split}.npz", split, thresholds)
                rows.append({"split": split, "model": name, "seed": seed, "protocol": "legacy_logit_reference", **metrics})
        history = np.load(CACHE / split / "station_history.npy", mmap_mode="r")
        target = np.load(CACHE / split / "station_future.npy", mmap_mode="r")
        capacity = np.load(CACHE / "static/station_capacity_mw.npy")
        prediction = np.broadcast_to(history[:, :, -1:], target.shape)
        metrics, _ = array_metrics(prediction, target, history, capacity, thresholds)
        rows.append({"split": split, "model": "P", "seed": "deterministic", "protocol": "persistence", **metrics})
    return rows


def strengthening_results(freeze, pairs, rows, thresholds):
    # Older synthetic fixtures have no strengthening registration.
    if freeze.get("protocol_version", 2) < 3:
        return {}
    by_group = {p["group"]:p for p in pairs}
    lookup = {(r["group"],r["role"],r["seed"],r["split"]):r for r in rows}
    comparisons = []
    intervention_rows = []
    for group in dict.fromkeys(("m0", freeze["selected_group"])):
        pair = by_group[group]
        for role in ("iadd","int"):
            for split in ("val","test"):
                for mode in ("reference","weather"):
                    changed, clean = [], []
                    for seed in FORMAL:
                        path = run_path(pair["stage"],group,role,seed) / f"predictions_{split}_{mode}.npz"
                        metrics, _, _, _ = archive_metrics(path,split,thresholds)
                        base = lookup[(group,role,seed,split)]["station_mae_cf"]
                        changed.append(metrics["station_mae_cf"])
                        clean.append(base)
                        intervention_rows.append({"group":group,"role":role,"split":split,"mode":mode,"seed":seed,
                                                  "clean_mae_cf":base,**metrics})
                    comparisons.append({"group":group,"role":role,"split":split,"mode":mode,
                                        "contrast":"intervention-minus-clean",**paired_summary(changed,clean)})
    write_csv(ROOT/"analysis/strengthening_intervention_seed_metrics.csv",intervention_rows)
    write_csv(ROOT/"analysis/strengthening_interventions.csv",comparisons)
    ablations = []
    for item in freeze.get("component_ablations",[]):
        for role in ("iadd","int"):
            for split in ("val","test"):
                removed=[lookup[(item["group"],role,s,split)]["station_mae_cf"] for s in SCREEN]
                full=[lookup[(item["parent_group"],role,s,split)]["station_mae_cf"] for s in SCREEN]
                ablations.append({"removed":item["removed"],"group":item["group"],"role":role,"split":split,
                                  "contrast":"removed-minus-full",**paired_summary(removed,full)})
    write_csv(ROOT/"analysis/component_ablations.csv",ablations)
    output_rows = []
    for protocol,base,final in (("cf_residual","m0",freeze["selected_group"]),
                               ("bounded_cf_mix","m0_bounded",freeze["bounded_finalist_group"])):
        for group in dict.fromkeys((base,final)):
            for split in ("val","test"):
                a=[lookup[(group,"int",s,split)]["station_mae_cf"] for s in FORMAL]
                b=[lookup[(group,"iadd",s,split)]["station_mae_cf"] for s in FORMAL]
                output_rows.append({"output_protocol":protocol,"group":group,"split":split,
                                    "contrast":"Int-IAdd",**paired_summary(a,b)})
    write_csv(ROOT/"analysis/output_link_robustness.csv",output_rows)
    return {"interventions":comparisons,"component_ablations":ablations,"output_link_robustness":output_rows}


def main() -> None:
    freeze = read_freeze(ROOT)
    thresholds = {int(lag): (v["station_cf"], v["cluster_mw"])
                  for lag, v in freeze["train_q90_ramp_thresholds"].items()}
    formal, screen = load_pairs()
    rows, attention_rows, farms, days = [], [], {}, {}
    for pair in formal + screen:
        stage, group = pair["stage"], pair["group"]
        for split in ("val", "test"):
            for role in ("iadd", "int"):
                for seed in pair["seeds"]:
                    metrics, farm, attention, day = archive_metrics(
                        run_path(stage, group, role, seed) / f"predictions_{split}.npz", split, thresholds)
                    key = (group, role, seed, split)
                    farms[key], days[key] = farm, day
                    rows.append({"stage": stage, "group": group, "role": role, "seed": seed, "split": split,
                                 "diagnostic_only": pair["spec"]["context"] == "id_film",
                                 "output_protocol":pair["spec"].get("output", "cf_residual"),
                                 "readout":pair["spec"].get("readout", "legacy"), **metrics})
                    if attention is not None:
                        for lead, weights in enumerate(attention):
                            entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum())
                            for lag, weight in enumerate(weights):
                                attention_rows.append({"stage": stage, "group": group, "role": role, "seed": seed,
                                    "split": split, "lead": lead + 1, "history_lag": lag-len(weights)+1,
                                    "weight": float(weight), "entropy": entropy})
        print(f"[analysis] {stage}/{group} complete", flush=True)
    analysis = ROOT / "analysis"
    write_csv(analysis / "seed_metrics.csv", rows)
    write_csv(analysis / "per_farm_seed_metrics.csv", [
        {"group":g,"role":r,"seed":s,"split":split,"farm_index":i,"station_mae_cf":float(v)}
        for (g,r,s,split), values in farms.items() for i,v in enumerate(values)])
    write_csv(analysis / "history_attention.csv", attention_rows)
    baseline = baseline_table(thresholds)
    write_csv(analysis / "complete_baselines.csv", baseline)
    lookup = {(r["group"], r["role"], r["seed"], r["split"]): r for r in rows}
    comparisons = []
    metric_keys = [k for k in rows[0] if k.endswith(("_cf", "_cf2", "_mw", "_f1", "_accuracy", "_precision", "_recall"))]
    for p in formal + screen:
        group, seeds = p["group"], p["seeds"]
        protocol_m0 = "m0_bounded" if p["spec"].get("output") == "bounded_cf_mix" else "m0"
        for split in ("val", "test"):
            contrasts = [("Int-IAdd", group, "iadd")]
            if group != protocol_m0:
                contrasts.append(("Int-ParentInt", p["parent_group"], "int"))
                if p["parent_group"] != protocol_m0:
                    contrasts.append(("Int-ProtocolM0Int", protocol_m0, "int"))
            for contrast, ref, role in contrasts:
                for metric in metric_keys:
                    left = [lookup[(group, "int", s, split)][metric] for s in seeds]
                    right = [lookup[(ref, role, s, split)][metric] for s in seeds]
                    if any(v is None for v in left + right):
                        continue
                    values = paired_summary(left, right)
                    higher_is_better = metric.endswith(("_accuracy", "_precision", "_recall", "_f1"))
                    values["wins"] = sum(a > b for a, b in zip(left, right)) if higher_is_better else values["wins"]
                    row = {"stage": p["stage"], "group": group, "split": split, "formal": p in formal,
                           "contrast": contrast, "metric": metric, **values, "farm_wins": "", "farm_count": "", "farm_coverage": ""}
                    if metric == "station_mae_cf":
                        fw = int((np.mean([farms[(group,"int",s,split)] for s in seeds], 0)
                                  < np.mean([farms[(ref,role,s,split)] for s in seeds], 0)).sum())
                        row.update(farm_wins=fw, farm_count=len(farms[(group,"int",seeds[0],split)]),
                                   farm_coverage=fw/len(farms[(group,"int",seeds[0],split)]))
                        row.update(day_paired_summary([days[(group,"int",s,split)] for s in seeds],
                                                      [days[(ref,role,s,split)] for s in seeds]))
                    comparisons.append(row)
            if group != protocol_m0:
                parent = p["parent_group"]
                delta = [lookup[(group,"int",s,split)]["station_mae_cf"] - lookup[(group,"iadd",s,split)]["station_mae_cf"] for s in seeds]
                base = [lookup[(parent,"int",s,split)]["station_mae_cf"] - lookup[(parent,"iadd",s,split)]["station_mae_cf"] for s in seeds]
                comparisons.append({"stage": p["stage"], "group": group, "split": split, "formal": p in formal,
                                    "contrast": "Interaction-DID", "metric": "station_mae_cf", **paired_summary(delta, base)})
    write_csv(analysis / "formal_comparisons.csv", comparisons)
    strengthening = strengthening_results(freeze, formal + screen, rows, thresholds)
    summary = {"validation_freeze": freeze, "train_q90_ramp_thresholds": freeze["train_q90_ramp_thresholds"],
               "formal_comparisons": comparisons, "unique_training_runs": len(rows)//2,
               "legacy_baseline_seed_rows": len(baseline), "strengthening": strengthening}
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    lines = ["# Farm-level Interaction Enhancement v1 完整结果", "",
             f"2023 冻结主模型：{freeze['selected_stage']}/{freeze['selected_group']}/{freeze.get('selected_role','int')}。固定 56 场，2022 训练，2024 冻结测试。",
             "本轮采用 CF residual 输出，M0 同协议重训；最终 CF 不裁剪，以保留严格可分离性。旧 logit baseline 单列为历史参考。",
             "每组 IAdd/Int 等参数、同 seed/schedule。ID-FiLM 仅作诊断，metadata 仅经纬度及容量。", ""]
    for split in ("val", "test"):
        lines += [f"## {split}: 全部目标模型", "",
                  "| Group | Seeds | Int MAE CF | IAdd MAE CF | Int-IAdd 95% seed CI | Farms improved |",
                  "|---|---:|---:|---:|---|---:|"]
        for p in formal + screen:
            values = [r for r in rows if r["group"] == p["group"] and r["split"] == split]
            a = np.mean([r["station_mae_cf"] for r in values if r["role"] == "int"])
            b = np.mean([r["station_mae_cf"] for r in values if r["role"] == "iadd"])
            comp = next(c for c in comparisons if c["group"] == p["group"] and c["split"] == split and c["contrast"] == "Int-IAdd" and c["metric"] == "station_mae_cf")
            lines.append(f"| {p['group']} | {len(p['seeds'])} | {a:.6f} | {b:.6f} | {comp['mean']:+.6f} [{comp['ci_low']:+.6f}, {comp['ci_high']:+.6f}] | {comp['farm_wins']}/{comp['farm_count']} |")
        lines += ["", "### 全部历史 baseline", "", "| Model | MAE CF | RMSE CF | Physical-sum MAE MW |", "|---|---:|---:|---:|"]
        for name in dict.fromkeys(r["model"] for r in baseline):
            values = [r for r in baseline if r["model"] == name and r["split"] == split]
            means = [np.mean([r[k] for r in values]) for k in ("station_mae_cf", "station_rmse_cf", "physical_sum_cluster_mae_mw")]
            lines.append(f"| {name} | {means[0]:.6f} | {means[1]:.6f} | {means[2]:.3f} |")
        lines.append("")
    lines += ["## 指标口径与证据", "",
              "seed_metrics.csv 与 complete_baselines.csv 完整报告 station CF 与 physical-sum cluster MW 的 MAE、15 min/1 h/3 h RampMAE、方向准确率和极端 ramp precision/recall/F1。极端阈值为 freeze 中的 2022 q90。",
              "formal_comparisons.csv 同时含 screening 与 formal 标记、各指标 paired seed bootstrap CI、station MAE 的共享 UTC-day block CI、per-farm coverage、parent 对照与 interaction DID。",
              "站场 ramp 绝对值的平均不等于容量加权后集群 ramp 的绝对值。不能合并声称 ramp 改善。Attention 是诊断而非因果证据；见 history_attention.csv。",
              "cf_residual 的越界比例另报 out_of_range_cf_fraction；不事后裁剪或用 2024 改选模型。",
              "strengthening_interventions.csv 报告冻结后的 history-reference / stratified weather permutation 干预；它们只说明输入依赖，不识别因果物理状态。",
              "component_ablations.csv 报告删去各组件的配对差；output_link_robustness.csv 在 CF residual 和 bounded CF mixture 两套严格可分离协议内分别复核 Int-IAdd。",
              "D1 level-shape 的归因对照是等参数 D0 capacity-aux，详见 formal_comparisons.csv 的 Int-ParentInt 和 Interaction-DID。",
              "所有 mean-error/shape MSE、预测内部及历史边界 ramp 分别报告；不能用平均水平校正替代局地动态恢复证据。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
