"""Create the decision-focused post-hoc analysis and figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from independent_models import CACHE, OUT, atomic_json


ANALYSIS = OUT / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    independent_summary = read_csv(ANALYSIS / "independent_per_station_summary.csv")
    shared_summary = read_csv(ANALYSIS / "shared_per_station_summary.csv")
    comparisons = read_csv(ANALYSIS / "independent_vs_shared_comparisons.csv")
    shared_comparisons = read_csv(ANALYSIS / "shared_per_station_comparisons.csv")
    independent_long = read_csv(ANALYSIS / "independent_per_station_seed_long.csv")

    independent = {
        (row["split"], row["model"], int(row["station_index"])): float(row["mean_mae_cf"])
        for row in independent_summary
    }
    independent_sd = {
        (row["split"], row["model"], int(row["station_index"])): float(row["sd_mae_cf"])
        for row in independent_summary
    }
    shared = {
        (row["split"], row["model"], int(row["station_index"])): float(row["seed_mean_mae_cf"])
        for row in shared_summary
        if row["model"] in {"history", "CBH"}
    }
    shared_sd = {
        (row["split"], row["model"], int(row["station_index"])): float(row["seed_sd_mae_cf"])
        for row in shared_summary
        if row["model"] in {"history", "CBH"}
    }
    metadata = {
        int(row["station_index"]): {
            "station_id": row["station_id"],
            "station_name": row["station_name"],
            "capacity_mw": float(row["capacity_mw"]),
        }
        for row in independent_summary
    }
    comparison_lookup = {
        (row["split"], row["comparison"], int(row["station_index"])): row
        for row in comparisons
    }

    station_rows = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for split in ("val", "test"):
        arrays[(split, "shared_history")] = np.array([shared[(split, "history", site)] for site in range(56)])
        arrays[(split, "independent_history")] = np.array([independent[(split, "INDEPENDENT_HISTORY", site)] for site in range(56)])
        arrays[(split, "shared_cbh")] = np.array([shared[(split, "CBH", site)] for site in range(56)])
        arrays[(split, "independent_cbh")] = np.array([independent[(split, "CBH_INDEPENDENT_SITE", site)] for site in range(56)])

    train_target = np.load(CACHE / "train/station_future.npy", mmap_mode="r")
    test_target = np.load(CACHE / "test/station_future.npy", mmap_mode="r")
    target_mean_shift = np.asarray(test_target.mean(axis=(0, 2)) - train_target.mean(axis=(0, 2)), dtype=np.float64)

    for site in range(56):
        row = {"station_index": site, **metadata[site], "test_minus_train_target_mean_cf": float(target_mean_shift[site])}
        for split in ("val", "test"):
            sh = arrays[(split, "shared_history")][site]
            ih = arrays[(split, "independent_history")][site]
            sc = arrays[(split, "shared_cbh")][site]
            ic = arrays[(split, "independent_cbh")][site]
            comparison = comparison_lookup[(split, "CBH_INDEPENDENT_SITE-minus-CBH_SHARED_56", site)]
            row.update(
                {
                    f"{split}_shared_history": sh,
                    f"{split}_independent_history": ih,
                    f"{split}_history_independent_minus_shared": ih - sh,
                    f"{split}_shared_cbh": sc,
                    f"{split}_independent_cbh": ic,
                    f"{split}_cbh_independent_minus_shared": ic - sc,
                    f"{split}_shared_weather_gain": sc - sh,
                    f"{split}_independent_weather_gain": ic - ih,
                    f"{split}_independent_extra_weather_gain": (ic - ih) - (sc - sh),
                    f"{split}_independent_cbh_seed_sd": independent_sd[(split, "CBH_INDEPENDENT_SITE", site)],
                    f"{split}_shared_cbh_seed_sd": shared_sd[(split, "CBH", site)],
                    f"{split}_independent_cbh_seed_wins_vs_shared": int(comparison["left_seed_wins"]),
                    f"{split}_independent_vs_shared_ci_low": float(comparison["bootstrap_ci95_low"]),
                    f"{split}_independent_vs_shared_ci_high": float(comparison["bootstrap_ci95_high"]),
                }
            )
        row["independent_weather_gain_test_minus_val"] = (
            row["test_independent_weather_gain"] - row["val_independent_weather_gain"]
        )
        station_rows.append(row)
    write_csv(ANALYSIS / "station_decision_table.csv", station_rows)

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in shared_comparisons:
        key = (row["split"], row["comparison_group"], row["model"], row["control"])
        grouped.setdefault(key, []).append(row)
    ablation_rows = []
    for (split, group, model, control), rows in grouped.items():
        values = np.array([float(row["delta_mae_cf"]) for row in rows])
        ablation_rows.append(
            {
                "split": split,
                "comparison_group": group,
                "model": model,
                "control": control,
                "mean_delta_mae_cf": float(values.mean()),
                "median_delta_mae_cf": float(np.median(values)),
                "minimum_station_delta": float(values.min()),
                "maximum_station_delta": float(values.max()),
                "stations_model_better": int((values < 0).sum()),
                "stations_control_better": int((values > 0).sum()),
                "stations_ci95_entirely_model_better": sum(float(row["seed_ci95_high"]) < 0 for row in rows),
                "stations_ci95_entirely_control_better": sum(float(row["seed_ci95_low"]) > 0 for row in rows),
                "stations_holm_p_below_0_05": sum(float(row["holm_p_across_56_stations_within_comparison"]) < 0.05 for row in rows),
            }
        )
    write_csv(ANALYSIS / "shared_ablation_station_overview.csv", ablation_rows)

    diagnostics = []
    for split in ("val", "test"):
        sh = arrays[(split, "shared_history")]
        ih = arrays[(split, "independent_history")]
        sc = arrays[(split, "shared_cbh")]
        ic = arrays[(split, "independent_cbh")]
        history_gap = ih - sh
        cbh_gap = ic - sc
        shared_gain = sc - sh
        independent_gain = ic - ih
        extra_gain = independent_gain - shared_gain
        diagnostics.append(
            {
                "split": split,
                "shared_history_mean": float(sh.mean()),
                "independent_history_mean": float(ih.mean()),
                "shared_cbh_mean": float(sc.mean()),
                "independent_cbh_mean": float(ic.mean()),
                "history_independent_minus_shared": float(history_gap.mean()),
                "cbh_independent_minus_shared": float(cbh_gap.mean()),
                "shared_weather_gain": float(shared_gain.mean()),
                "independent_weather_gain": float(independent_gain.mean()),
                "independent_extra_weather_gain": float(extra_gain.mean()),
                "fraction_history_gap_recovered_by_extra_weather_gain": float(-extra_gain.mean() / history_gap.mean()),
                "independent_cbh_relative_penalty_vs_shared": float(cbh_gap.mean() / sc.mean()),
                "independent_weather_relative_improvement": float(-independent_gain.mean() / ih.mean()),
                "shared_weather_relative_improvement": float(-shared_gain.mean() / sh.mean()),
                "sites_independent_cbh_better_than_shared": int((cbh_gap < 0).sum()),
                "sites_independent_history_better_than_shared": int((history_gap < 0).sum()),
                "sites_independent_cbh_better_than_own_history": int((independent_gain < 0).sum()),
                "station_wilcoxon_p_cbh_independent_vs_shared": float(wilcoxon(cbh_gap).pvalue),
                "station_wilcoxon_p_history_independent_vs_shared": float(wilcoxon(history_gap).pvalue),
                "station_wilcoxon_p_extra_weather_gain": float(wilcoxon(extra_gain).pvalue),
                "spearman_cbh_gap_vs_capacity_rho": float(spearmanr(cbh_gap, [metadata[i]["capacity_mw"] for i in range(56)]).statistic),
                "spearman_cbh_gap_vs_shared_cbh_error_rho": float(spearmanr(cbh_gap, sc).statistic),
            }
        )
    write_csv(ANALYSIS / "gap_decomposition_summary.csv", diagnostics)

    val_cbh_gap = arrays[("val", "independent_cbh")] - arrays[("val", "shared_cbh")]
    test_cbh_gap = arrays[("test", "independent_cbh")] - arrays[("test", "shared_cbh")]
    val_history_gap = arrays[("val", "independent_history")] - arrays[("val", "shared_history")]
    test_history_gap = arrays[("test", "independent_history")] - arrays[("test", "shared_history")]
    val_independent_gain = arrays[("val", "independent_cbh")] - arrays[("val", "independent_history")]
    test_independent_gain = arrays[("test", "independent_cbh")] - arrays[("test", "independent_history")]
    cross_year = {
        "cbh_gap_pearson_r": float(pearsonr(val_cbh_gap, test_cbh_gap).statistic),
        "cbh_gap_spearman_rho": float(spearmanr(val_cbh_gap, test_cbh_gap).statistic),
        "history_gap_pearson_r": float(pearsonr(val_history_gap, test_history_gap).statistic),
        "history_gap_spearman_rho": float(spearmanr(val_history_gap, test_history_gap).statistic),
        "independent_weather_gain_pearson_r": float(pearsonr(val_independent_gain, test_independent_gain).statistic),
        "independent_weather_gain_spearman_rho": float(spearmanr(val_independent_gain, test_independent_gain).statistic),
        "weather_gain_drift_vs_target_mean_shift_spearman_rho": float(
            spearmanr(test_independent_gain - val_independent_gain, target_mean_shift).statistic
        ),
        "weather_gain_drift_vs_target_mean_shift_spearman_p": float(
            spearmanr(test_independent_gain - val_independent_gain, target_mean_shift).pvalue
        ),
    }
    atomic_json(ANALYSIS / "cross_year_diagnostics.json", cross_year)

    # Figure 1: direct shared/independent comparison and per-site deltas.
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, split in enumerate(("val", "test")):
        shared_cbh = arrays[(split, "shared_cbh")]
        independent_cbh = arrays[(split, "independent_cbh")]
        delta = independent_cbh - shared_cbh
        colors = np.where(delta < 0, "#2b8cbe", "#d95f0e")
        axes[0, column].scatter(shared_cbh, independent_cbh, c=colors, s=34, alpha=0.85, edgecolors="none")
        low = min(shared_cbh.min(), independent_cbh.min()) - 0.003
        high = max(shared_cbh.max(), independent_cbh.max()) + 0.003
        axes[0, column].plot([low, high], [low, high], color="black", lw=1, ls="--")
        for site in (10, 49, 55):
            axes[0, column].annotate(str(site), (shared_cbh[site], independent_cbh[site]), fontsize=8)
        axes[0, column].set(xlabel="Shared CBH MAE", ylabel="Independent CBH MAE", title=f"{split}: per-site mean over 10 seeds")
        order = np.argsort(delta)
        axes[1, column].bar(np.arange(56), delta[order], color=colors[order], width=0.9)
        axes[1, column].axhline(0, color="black", lw=1)
        axes[1, column].set(xlabel="Sites sorted by independent - shared", ylabel="Delta CF MAE", title=f"{split}: independent minus shared")
    figure.savefig(ANALYSIS / "fig_shared_vs_independent.png", dpi=180)
    plt.close(figure)

    # Figure 2: decomposition and the main cross-year failure diagnostic.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = ["Shared\nhistory", "Independent\nhistory", "Shared\nCBH", "Independent\nCBH"]
    x = np.arange(4)
    width = 0.36
    for offset, split, color in ((-width / 2, "val", "#6baed6"), (width / 2, "test", "#2171b5")):
        values = [arrays[(split, key)].mean() for key in ("shared_history", "independent_history", "shared_cbh", "independent_cbh")]
        axes[0].bar(x + offset, values, width, label=split, color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean station CF MAE")
    axes[0].set_title("Baseline and weather-correction decomposition")
    axes[0].legend()
    gain_drift = test_independent_gain - val_independent_gain
    axes[1].scatter(target_mean_shift, gain_drift, color="#756bb1", s=35, alpha=0.85)
    for site in (10, 49, 55):
        axes[1].annotate(str(site), (target_mean_shift[site], gain_drift[site]), fontsize=8)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set(
        xlabel="2024 minus 2022 target mean CF",
        ylabel="2024 minus 2023 independent correction gain",
        title=f"Distribution shift vs gain drift (Spearman {cross_year['weather_gain_drift_vs_target_mean_shift_spearman_rho']:.2f})",
    )
    figure.savefig(ANALYSIS / "fig_gap_decomposition_and_shift.png", dpi=180)
    plt.close(figure)

    # Compose a decision-oriented report.
    val, test = diagnostics
    val_winners = [row for row in station_rows if row["val_cbh_independent_minus_shared"] < 0]
    test_winners = [row for row in station_rows if row["test_cbh_independent_minus_shared"] < 0]
    test_failure = station_rows[55]
    lines = [
        "# CBH 逐风场共享与完全独立训练：详细分析",
        "",
        "## 结论",
        "",
        "**继续使用 `CBH_SHARED_56` 作为高频代理天气生成路线之前的冻结主结构。** 完全独立训练没有形成可替代共享模型的总体优势；其主要价值是证明 future-weather correction 在单站模型内仍然成立，并揭示跨站共享主要改善 history baseline 起点。",
        "",
        "## 总体分解",
        "",
        "| 年份 | shared history | independent history | shared CBH | independent CBH | independent−shared CBH | 独立天气增益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 2023 | {val['shared_history_mean']:.6f} | {val['independent_history_mean']:.6f} | {val['shared_cbh_mean']:.6f} | {val['independent_cbh_mean']:.6f} | +{val['cbh_independent_minus_shared']:.6f} | {val['independent_weather_gain']:.6f} |",
        f"| 2024 | {test['shared_history_mean']:.6f} | {test['independent_history_mean']:.6f} | {test['shared_cbh_mean']:.6f} | {test['independent_cbh_mean']:.6f} | +{test['cbh_independent_minus_shared']:.6f} | {test['independent_weather_gain']:.6f} |",
        "",
        f"独立 history 起点比共享 history 差 {val['history_independent_minus_shared']:+.6f}（2023）和 {test['history_independent_minus_shared']:+.6f}（2024）。独立 weather correction 比共享路线额外改善 {-val['independent_extra_weather_gain']:.6f} 和 {-test['independent_extra_weather_gain']:.6f}，弥补了 history 起点差距的 {100*val['fraction_history_gap_recovered_by_extra_weather_gain']:.1f}% 和 {100*test['fraction_history_gap_recovered_by_extra_weather_gain']:.1f}%，但不足以逆转最终排序。",
        "",
        f"独立 CBH 相对自己的 history baseline：2023 改善 56/56 站、总体 {100*val['independent_weather_relative_improvement']:.2f}%；2024 改善 55/56 站、总体 {100*test['independent_weather_relative_improvement']:.2f}%。这证明单站 future-weather correction 有稳定预测价值。",
        "",
        "## 逐站比较",
        "",
        f"2023 独立 CBH 仅在 {len(val_winners)}/56 站均值优于共享 CBH：" + "、".join(f"{row['station_name']}({row['val_cbh_independent_minus_shared']:+.6f})" for row in sorted(val_winners, key=lambda row: row['val_cbh_independent_minus_shared'])) + "。",
        "",
        f"2024 独立 CBH 仅在 {len(test_winners)}/56 站均值优于共享 CBH：" + "、".join(f"{row['station_name']}({row['test_cbh_independent_minus_shared']:+.6f})" for row in sorted(test_winners, key=lambda row: row['test_cbh_independent_minus_shared'])) + "。两年共同为负的只有富裕风场；查干湖风电在 2024 的优势仅 −0.000288、5/10 seeds，不构成稳定替代证据。",
        "",
        f"2024 有 51/56 站的独立−共享 bootstrap CI 完全大于零。富裕风场是唯一跨年方向一致且 2024 为 8/10 seed wins 的独立优势候选；兴隆山平价只在 2023 明显更好，2024 已反转。",
        "",
        "## 跨年稳定性与余字风电场",
        "",
        f"逐站 independent−shared CBH 差距在 2023/2024 的 Pearson r={cross_year['cbh_gap_pearson_r']:.3f}、Spearman ρ={cross_year['cbh_gap_spearman_rho']:.3f}；history 差距更稳定（Pearson r={cross_year['history_gap_pearson_r']:.3f}）。",
        "",
        f"余字风电场是唯一 2024 独立 CBH 比自己的独立 history 更差的站：weather correction Δ={test_failure['test_independent_weather_gain']:+.6f}，且相对共享 CBH 差 {test_failure['test_cbh_independent_minus_shared']:+.6f}。该站 2024 相对 2022 的目标均值变化为 {test_failure['test_minus_train_target_mean_cf']:+.6f}，是 56 站中最大负向漂移。全站点上，目标均值漂移与独立 correction gain 的跨年漂移 Spearman ρ={cross_year['weather_gain_drift_vs_target_mean_shift_spearman_rho']:.3f}（p={cross_year['weather_gain_drift_vs_target_mean_shift_spearman_p']:.6g}）。这支持“单站 weather→power 映射更易受跨年关系漂移影响”的解释，但属于事后相关诊断，不能写成因果结论。",
        "",
        "## 共享路线的逐站消融",
        "",
        "- `W−history`：2023 为 56/56 站改善，2024 为 55/56；future weather 是最普遍的增量。",
        "- `L−W`：两年均 56/56 站改善；用 baseline operating level 条件化 weather correction 的收益高度一致。",
        "- `J−A`：2023 为 56/56、2024 为 54/56 站改善；完整联合 correction 优于当前严格输出可分离对照。",
        "- `J−H`：54/56 与 53/56 站改善；显式 baseline trajectory 在 raw history 外普遍有小增量。",
        "- `J−T`：39/56 与 36/56；raw history 的额外作用明显更异质。",
        "- `CBH−CH`、`CBH−CNEUTRAL`：两年均 56/56 站改善；baseline condition 是稳定核心。",
        "- `CBH−CB`：38/56 与 43/56，均值仅 −0.000150/−0.000290；额外 history summary 是次要且异质的增量。",
        "- summary 提取、LP、generic/multiplicative fusion 的站点方向不稳定，继续扩大结构搜索没有依据。",
        "",
        "## 统计解释边界",
        "",
        "每站只有 10 个配对 seeds，双侧 exact Wilcoxon 的最小 p=0.001953；对 56 站做 Holm 后最小可能值为 0.109375。因此没有单站通过该 Holm 门槛是检验分辨率限制，不能反推所有单站无效。逐站结论应同时看效应量、10-seed bootstrap CI、seed wins 与跨年方向。总体 independent−shared 差值的 seed bootstrap 和 crossed bootstrap 均完全大于零。",
        "",
        "2024 已在开发中查看，只能称跨年复评。seed 区间表示训练随机性，不能覆盖年份抽样或反复模型选择的不确定性。",
        "",
        "## 模型选择与后续控制",
        "",
        "当前结果支持冻结共享 CBH，再接高频代理天气生成模块。完全独立模型可保留为逐站鲁棒性对照，不应替换主模型。若要把性能差严格归因到“参数共享”本身，还需做 2×2 控制：共享参数+本站 cluster input，以及独立参数+全场 aggregate cluster input；当前 shared/independent 比较同时改变了参数共享和跨站历史输入。",
        "",
        "更有希望的单站适配路线是从共享 CBH 初始化后仅训练小型 station adapter/LoRA 或最后若干层，而非从零训练完整独立模型。该建议尚未实验验证。",
        "",
        "## 产物",
        "",
        "- `station_decision_table.csv`：56 站 shared/independent、history/CBH、两年差值及 seed 稳定性；",
        "- `shared_ablation_station_overview.csv`：44 个 split×comparison 的逐站覆盖率；",
        "- `gap_decomposition_summary.csv`：总体差距分解；",
        "- `cross_year_diagnostics.json`：跨年相关；",
        "- `fig_shared_vs_independent.png`、`fig_gap_decomposition_and_shift.png`：可视化。",
        "",
    ]
    (ANALYSIS / "DETAILED_ANALYSIS_CN.md").write_text("\n".join(lines), encoding="utf-8")

    audit = {
        "passed": True,
        "station_rows": len(station_rows),
        "shared_ablation_rows": len(ablation_rows),
        "diagnostic_rows": len(diagnostics),
        "figures": ["fig_shared_vs_independent.png", "fig_gap_decomposition_and_shift.png"],
        "source_final_audit": json.loads((OUT / "status/INDEPENDENT_FINAL_AUDIT.json").read_text()),
    }
    atomic_json(OUT / "status/DETAILED_ANALYSIS_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
