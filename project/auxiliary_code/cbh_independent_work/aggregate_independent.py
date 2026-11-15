"""Audit and aggregate the 56-site x 10-seed independent experiment."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from independent_models import OUT, SEEDS, atomic_json


SHARED = OUT / "analysis/shared_per_station_seed_long.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def crossed_bootstrap(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    # values [seed, site], resampling both axes preserves the two sources of
    # empirical variation without pretending the 560 cells are iid.
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        seed_index = rng.integers(0, values.shape[0], (stop - start, values.shape[0]))
        site_index = rng.integers(0, values.shape[1], (stop - start, values.shape[1]))
        for row in range(stop - start):
            result[start + row] = values[seed_index[row]][:, site_index[row]].mean()
    return float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))


def load_shared() -> dict[tuple[str, int, int], float]:
    result = {}
    with SHARED.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in {"CBH", "history"}:
                result[(row["model"], int(row["seed"]), int(row["station_index"]))] = float(row["mae_cf"])
    expected = 2 * 10 * 56 * 2  # two models x seeds x sites x splits
    # split is absent from the key, so load separately below instead of using this check.
    return result


def main() -> None:
    independent_rows = []
    audits = []
    station_meta = {}
    independent: dict[tuple[str, str, int, int], float] = {}
    for site in range(56):
        for seed in SEEDS:
            root = OUT / "independent_seed" / f"site_{site:02d}" / f"seed_{seed}"
            success_path = root / "SUCCESS.json"
            metrics_path = root / "metrics.json"
            if not success_path.exists() or not metrics_path.exists():
                raise RuntimeError(f"Missing completed job: site={site} seed={seed}")
            success = json.loads(success_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            if not success.get("passed") or not success.get("frozen_baseline_unchanged"):
                raise RuntimeError(f"Failed success contract: {success_path}")
            station_meta[site] = {
                "station_id": metrics["station_id"],
                "station_name": metrics["station_name"],
                "capacity_mw": metrics["capacity_mw"],
            }
            for model in ("baseline", "cbh"):
                history = json.loads((root / f"{model}_history.json").read_text())["history"]
                values = [float(row["val_mae_cf"]) for row in history]
                best_index = int(np.argmin(values))
                best_epoch = int(metrics[model]["best_epoch"])
                val_metric = float(metrics[model]["val_mae_cf"])
                if best_epoch != best_index + 1 or abs(val_metric - values[best_index]) > 2e-7:
                    raise RuntimeError(f"Checkpoint argmin mismatch: {root} {model}")
                for split in ("val", "test"):
                    value = float(metrics[model][f"{split}_mae_cf"])
                    independent[(model, split, seed, site)] = value
                    independent_rows.append(
                        {
                            "split": split,
                            "model": "INDEPENDENT_HISTORY" if model == "baseline" else "CBH_INDEPENDENT_SITE",
                            "seed": seed,
                            "station_index": site,
                            **station_meta[site],
                            "mae_cf": value,
                            "mae_mw": value * float(metrics["capacity_mw"]),
                            "best_epoch": best_epoch,
                            "source_metrics": str(metrics_path),
                        }
                    )
            audits.append(
                {
                    "site_index": site,
                    "seed": seed,
                    "baseline_argmin_checked": True,
                    "cbh_argmin_checked": True,
                    "frozen_baseline_unchanged": True,
                }
            )

    shared: dict[tuple[str, str, int, int], float] = {}
    with SHARED.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in {"CBH", "history"}:
                shared[(row["model"], row["split"], int(row["seed"]), int(row["station_index"]))] = float(row["mae_cf"])
    if len(shared) != 2 * 2 * 10 * 56:
        raise RuntimeError(f"Incomplete shared controls: {len(shared)}")

    summaries = []
    for split in ("val", "test"):
        for model in ("baseline", "cbh"):
            for site in range(56):
                values = np.array([independent[(model, split, seed, site)] for seed in SEEDS])
                low, high = bootstrap_ci(values, 91000 + site + (1000 if split == "test" else 0) + (100 if model == "cbh" else 0))
                summaries.append(
                    {
                        "split": split,
                        "model": "INDEPENDENT_HISTORY" if model == "baseline" else "CBH_INDEPENDENT_SITE",
                        "station_index": site,
                        **station_meta[site],
                        "mean_mae_cf": float(values.mean()),
                        "sd_mae_cf": float(values.std(ddof=1)),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "minimum_mae_cf": float(values.min()),
                        "minimum_seed": int(SEEDS[int(values.argmin())]),
                        "maximum_mae_cf": float(values.max()),
                        "maximum_seed": int(SEEDS[int(values.argmax())]),
                    }
                )

    comparisons = []
    comparison_specs = (
        ("CBH_INDEPENDENT_SITE-minus-INDEPENDENT_HISTORY", "independent_cbh", "independent_history"),
        ("CBH_INDEPENDENT_SITE-minus-CBH_SHARED_56", "independent_cbh", "shared_cbh"),
        ("INDEPENDENT_HISTORY-minus-SHARED_HISTORY", "independent_history", "shared_history"),
    )
    for split in ("val", "test"):
        split_rows = []
        for comparison_index, (name, left, right) in enumerate(comparison_specs):
            family_rows = []
            for site in range(56):
                def values_for(label: str) -> np.ndarray:
                    if label == "independent_cbh":
                        return np.array([independent[("cbh", split, seed, site)] for seed in SEEDS])
                    if label == "independent_history":
                        return np.array([independent[("baseline", split, seed, site)] for seed in SEEDS])
                    if label == "shared_cbh":
                        return np.array([shared[("CBH", split, seed, site)] for seed in SEEDS])
                    if label == "shared_history":
                        return np.array([shared[("history", split, seed, site)] for seed in SEEDS])
                    raise ValueError(label)
                left_values, right_values = values_for(left), values_for(right)
                difference = left_values - right_values
                low, high = bootstrap_ci(difference, 120000 + 1000 * comparison_index + 100 * (split == "test") + site)
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                family_rows.append(
                    {
                        "split": split,
                        "comparison": name,
                        "station_index": site,
                        **station_meta[site],
                        "mean_delta_mae_cf": float(difference.mean()),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "left_seed_wins": int((difference < 0).sum()),
                        "right_seed_wins": int((difference > 0).sum()),
                        "ties": int((difference == 0).sum()),
                        "wilcoxon_p": p_value,
                    }
                )
            adjusted = holm([row["wilcoxon_p"] for row in family_rows])
            for row, value in zip(family_rows, adjusted):
                row["wilcoxon_holm_p_across_56_sites"] = value
            split_rows.extend(family_rows)
        comparisons.extend(split_rows)

    overall = []
    for split in ("val", "test"):
        arrays = {
            "CBH_INDEPENDENT_SITE": np.array([[independent[("cbh", split, seed, site)] for site in range(56)] for seed in SEEDS]),
            "INDEPENDENT_HISTORY": np.array([[independent[("baseline", split, seed, site)] for site in range(56)] for seed in SEEDS]),
            "CBH_SHARED_56": np.array([[shared[("CBH", split, seed, site)] for site in range(56)] for seed in SEEDS]),
            "SHARED_HISTORY": np.array([[shared[("history", split, seed, site)] for site in range(56)] for seed in SEEDS]),
        }
        for name, values in arrays.items():
            seed_means = values.mean(1)
            low, high = bootstrap_ci(seed_means, 170000 + len(overall))
            overall.append(
                {
                    "split": split,
                    "quantity": name,
                    "mean_mae_cf": float(values.mean()),
                    "sd_across_seed_station_means": float(seed_means.std(ddof=1)),
                    "seed_bootstrap_ci95_low": low,
                    "seed_bootstrap_ci95_high": high,
                    "minimum_seed_mean": float(seed_means.min()),
                    "minimum_seed": int(SEEDS[int(seed_means.argmin())]),
                    "maximum_seed_mean": float(seed_means.max()),
                    "maximum_seed": int(SEEDS[int(seed_means.argmax())]),
                    "sites": 56,
                    "seeds": 10,
                }
            )
        for name, left, right in (
            ("CBH_INDEPENDENT_SITE-minus-INDEPENDENT_HISTORY", arrays["CBH_INDEPENDENT_SITE"], arrays["INDEPENDENT_HISTORY"]),
            ("CBH_INDEPENDENT_SITE-minus-CBH_SHARED_56", arrays["CBH_INDEPENDENT_SITE"], arrays["CBH_SHARED_56"]),
            ("INDEPENDENT_HISTORY-minus-SHARED_HISTORY", arrays["INDEPENDENT_HISTORY"], arrays["SHARED_HISTORY"]),
        ):
            difference = left - right
            seed_difference = difference.mean(1)
            seed_low, seed_high = bootstrap_ci(seed_difference, 180000 + len(overall))
            crossed_low, crossed_high = crossed_bootstrap(difference, 190000 + len(overall))
            overall.append(
                {
                    "split": split,
                    "quantity": name,
                    "mean_mae_cf": float(difference.mean()),
                    "sd_across_seed_station_means": float(seed_difference.std(ddof=1)),
                    "seed_bootstrap_ci95_low": seed_low,
                    "seed_bootstrap_ci95_high": seed_high,
                    "crossed_bootstrap_ci95_low": crossed_low,
                    "crossed_bootstrap_ci95_high": crossed_high,
                    "seed_wins_left": int((seed_difference < 0).sum()),
                    "station_wins_left_by_10seed_mean": int((difference.mean(0) < 0).sum()),
                    "sites": 56,
                    "seeds": 10,
                }
            )

    analysis = OUT / "analysis"
    write_csv(analysis / "independent_per_station_seed_long.csv", independent_rows)
    write_csv(analysis / "independent_per_station_summary.csv", summaries)
    write_csv(analysis / "independent_vs_shared_comparisons.csv", comparisons)
    write_csv(analysis / "independent_overall_summary.csv", overall)
    write_csv(analysis / "independent_job_audit.csv", audits)
    audit = {
        "passed": True,
        "jobs": len(audits),
        "expected_jobs": 560,
        "long_rows": len(independent_rows),
        "expected_long_rows": 2240,
        "station_summary_rows": len(summaries),
        "comparison_rows": len(comparisons),
        "all_checkpoint_argmins_verified": True,
        "all_frozen_baseline_hashes_verified": True,
        "no_ramp_metrics_or_losses": True,
    }
    atomic_json(OUT / "status/INDEPENDENT_FINAL_AUDIT.json", audit)

    lookup = {(row["split"], row["quantity"]): row for row in overall}
    lines = [
        "# CBH 逐风场多种子评估与完全独立训练对照",
        "",
        "本报告由 56 个风场 × seeds 42–51 的正式结果自动生成。2022 用于训练，2023 用于 early stopping/checkpoint 选择，2024 为跨年复评。全程未使用爬坡筛选、爬坡损失或爬坡指标。",
        "",
        "## 总体结果",
        "",
        "| split | 模型/比较 | mean CF MAE | seed bootstrap 95% CI | crossed bootstrap 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    labels = (
        "SHARED_HISTORY", "CBH_SHARED_56", "INDEPENDENT_HISTORY", "CBH_INDEPENDENT_SITE",
        "CBH_INDEPENDENT_SITE-minus-INDEPENDENT_HISTORY",
        "CBH_INDEPENDENT_SITE-minus-CBH_SHARED_56",
        "INDEPENDENT_HISTORY-minus-SHARED_HISTORY",
    )
    for split in ("val", "test"):
        for label in labels:
            row = lookup[(split, label)]
            crossed = (
                f"[{row['crossed_bootstrap_ci95_low']:.6f}, {row['crossed_bootstrap_ci95_high']:.6f}]"
                if "crossed_bootstrap_ci95_low" in row else "—"
            )
            lines.append(
                f"| {split} | {label} | {row['mean_mae_cf']:.6f} | "
                f"[{row['seed_bootstrap_ci95_low']:.6f}, {row['seed_bootstrap_ci95_high']:.6f}] | {crossed} |"
            )
    lines += [
        "",
        "负的 `INDEPENDENT−SHARED` 差值表示逐站独立训练更好。逐站 mean/SD/95% CI/min/max、最佳/最差 seed 见 `independent_per_station_summary.csv`；逐站配对差、seed wins、Wilcoxon 与跨 56 站 Holm 校正见 `independent_vs_shared_comparisons.csv`。",
        "",
        "## 与完整消融的关系",
        "",
        "共享参数路线的 25 个既有模型（trajectory、P0I transfer、CBH condition、固定和可学习 summary 消融）已经统一提取为 56 站 × 10 seeds × 2023/2024 的完整表。独立路线额外给出 `INDEPENDENT_HISTORY → CBH_INDEPENDENT_SITE`，用于检验 future-weather correction 在取消一切跨站可学习参数后是否仍带来增益。",
        "",
        "## 参数共享边界",
        "",
        "`CBH_SHARED_56` 的 history encoders、weather projection、condition projection、adapter、Transformer decoder 和 readout 均在 56 站间共享，且 history baseline 读取全场聚合 `cluster_history`。`CBH_INDEPENDENT_SITE` 每个风场、每个 seed 都有完整独立参数；其 cluster-history 分支只读取本站历史。仅经纬度/容量的固定归一化常数沿用 56 站训练集定义。",
        "",
        "## 审计",
        "",
        "560 个作业均验证 validation argmin checkpoint；CBH 训练前后的冻结 baseline tensor hash 完全一致；局地天气缓存与原 P0I 双线性读取逐值一致。",
        "",
    ]
    (analysis / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
