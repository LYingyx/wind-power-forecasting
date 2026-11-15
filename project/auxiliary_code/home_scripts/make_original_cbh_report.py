"""Render the Chinese final report from audited aggregate artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_original_cbh_20260917_v1")
ORIGINAL = "CBH_ORIGINAL_56_SUBSET"
FAMILIES = [
    "CBH_ADAPT_LINEAR", "CBH_ADAPT_DET", "CBH_ADAPT_ICFM", "CBH_ADAPT_VPCFM",
    "CBH_ADAPT_SINE_SI", "CBH_ADAPT_VP_EPS", "CBH_ADAPT_VP_V",
    "CBH_ADAPT_EDM", "CBH_ADAPT_MEANFLOW",
]
LABEL = {
    ORIGINAL: "原 CBH（56站模型在10站直接评测）",
    "CBH_ADAPT_LINEAR": "CBH + LINEAR",
    "CBH_ADAPT_DET": "CBH + DET",
    "CBH_ADAPT_ICFM": "CBH + I-CFM",
    "CBH_ADAPT_VPCFM": "CBH + VP-CFM",
    "CBH_ADAPT_SINE_SI": "CBH + Sine-SI",
    "CBH_ADAPT_VP_EPS": "CBH + VP-epsilon",
    "CBH_ADAPT_VP_V": "CBH + VP-v",
    "CBH_ADAPT_EDM": "CBH + EDM",
    "CBH_ADAPT_MEANFLOW": "CBH + MeanFlow",
}


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def signed(value: float) -> str:
    return f"{float(value):+.6f}"


def main() -> None:
    overall = json.loads((ROOT / "analysis/overall_seed_summary.json").read_text())
    paired = json.loads((ROOT / "analysis/paired_comparisons.json").read_text())
    proxy = json.loads((ROOT / "analysis/proxy_summary.json").read_text())
    audit = json.loads((ROOT / "status/FINAL_AUDIT.json").read_text())
    by_model = {row["model"]: row for row in overall}
    pair_key = {(row["model"], row["control"]): row for row in paired}
    proxy_key = {row["model"]: row for row in proxy}

    result_lines = [
        "| 模型 | 10-seed station-equal MAE 均值±SD | 最低（seed） | 最高（seed） |",
        "|---|---:|---:|---:|",
    ]
    for model in [ORIGINAL, *FAMILIES]:
        row = by_model[model]
        result_lines.append(
            f"| {LABEL[model]} | {fmt(row['mean'])} ± {fmt(row['std'])} | "
            f"{fmt(row['min'])} ({row['min_seed']}) | {fmt(row['max'])} ({row['max_seed']}) |"
        )

    linear_lines = [
        "| 代理方法 − LINEAR | ΔMAE | crossed-bootstrap 95% CI | seed胜数 | station胜数 | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in FAMILIES[1:]:
        row = pair_key[(model, "CBH_ADAPT_LINEAR")]
        linear_lines.append(
            f"| {LABEL[model]} − {LABEL['CBH_ADAPT_LINEAR']} | {signed(row['delta'])} | "
            f"[{signed(row['ci95_low'])}, {signed(row['ci95_high'])}] | "
            f"{row['seed_wins']}/10 | {row['station_wins']}/10 | {row['holm_p']:.6f} |"
        )

    original_lines = [
        "| 适配路线 − 原 CBH | ΔMAE | crossed-bootstrap 95% CI | seed胜数 | station胜数 | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in FAMILIES:
        row = pair_key[(model, ORIGINAL)]
        original_lines.append(
            f"| {LABEL[model]} − {LABEL[ORIGINAL]} | {signed(row['delta'])} | "
            f"[{signed(row['ci95_low'])}, {signed(row['ci95_high'])}] | "
            f"{row['seed_wins']}/10 | {row['station_wins']}/10 | {row['holm_p']:.6f} |"
        )

    proxy_lines = [
        "| 方法 | 4h→1h linear / joint MAE | 2h→1h linear / joint MAE | joint sample SD（4h / 2h） |",
        "|---|---:|---:|---:|",
    ]
    for model in FAMILIES[1:]:
        row = proxy_key[model]
        proxy_lines.append(
            f"| {LABEL[model]} | {row['4h_linear_mae']:.4f} / {row['4h_after_joint_generated_mae']:.4f} | "
            f"{row['2h_linear_mae']:.4f} / {row['2h_after_joint_generated_mae']:.4f} | "
            f"{row['4h_after_joint_sample_std']:.4f} / {row['2h_after_joint_sample_std']:.4f} |"
        )

    with (ROOT / "analysis/v2_named_baseline_context.csv").open() as handle:
        v2_rows = list(csv.DictReader(handle))
    baseline_names = ["DLinear", "PatchTST", "HistGradientBoostingRegressor", "TiDE", "TFT", "NBEATSx"]
    v2_key = {row["model"]: row for row in v2_rows}
    baseline_lines = [
        "| 实际模型 | 输入 | 10-seed MAE 均值±SD | 最低（seed） | 最高（seed） |",
        "|---|---|---:|---:|---:|",
    ]
    inputs = {
        "DLinear": "历史功率",
        "PatchTST": "历史功率",
        "HistGradientBoostingRegressor": "未来小时天气 only",
        "TiDE": "历史功率 + 未来已知天气",
        "TFT": "历史功率 + 未来已知天气",
        "NBEATSx": "历史功率 + 未来已知天气",
    }
    for model in baseline_names:
        row = v2_key[model]
        baseline_lines.append(
            f"| {model} | {inputs[model]} | {fmt(row['mean'])} ± {fmt(row['std'])} | "
            f"{fmt(row['min'])} ({row['min_seed']}) | {fmt(row['max'])} ({row['max_seed']}) |"
        )

    generated = FAMILIES[1:]
    best_generated = min(generated, key=lambda name: by_model[name]["mean"])
    best_pair = pair_key[(best_generated, "CBH_ADAPT_LINEAR")]
    beating_linear = [name for name in generated if by_model[name]["mean"] < by_model["CBH_ADAPT_LINEAR"]["mean"]]
    confirmed_linear = [
        name for name in generated
        if pair_key[(name, "CBH_ADAPT_LINEAR")]["ci95_high"] < 0
        and pair_key[(name, "CBH_ADAPT_LINEAR")]["holm_p"] < 0.05
    ]
    linear_vs_original = pair_key[("CBH_ADAPT_LINEAR", ORIGINAL)]
    best_vs_original = pair_key[(best_generated, ORIGINAL)]
    best_proxy = proxy_key[best_generated]
    best_overall = min([ORIGINAL, *FAMILIES], key=lambda name: by_model[name]["mean"])

    baseline_text = "\n".join(baseline_lines)
    result_text = "\n".join(result_lines)
    linear_text = "\n".join(linear_lines)
    original_text = "\n".join(original_lines)
    proxy_text = "\n".join(proxy_lines)
    confirmed_text = "、".join(LABEL[name] for name in confirmed_linear) if confirmed_linear else "无"
    beating_text = "、".join(LABEL[name] for name in beating_linear) if beating_linear else "无"

    report = f"""# 潜在高频代理天气接入原 CBH：10 站 × 10 seeds 最终探索报告

## 结论

在当前固定协议下，保留交接文档确定的原 CBH 是明确选择。原 CBH 在所选 10 站的 10-seed station-equal MAE 为 **{fmt(by_model[ORIGINAL]['mean'])} ± {fmt(by_model[ORIGINAL]['std'])}**；本批最好的非 LINEAR 代理是 **{LABEL[best_generated]}**，MAE 为 **{fmt(by_model[best_generated]['mean'])} ± {fmt(by_model[best_generated]['std'])}**。它相对同样经过单站 pilot 微调的 LINEAR 控制，Δ={signed(best_pair['delta'])}，95% CI [{signed(best_pair['ci95_low'])}, {signed(best_pair['ci95_high'])}]，Holm p={best_pair['holm_p']:.6f}；相对未微调原 CBH，Δ={signed(best_vs_original['delta'])}。

生成方法中，均值低于 LINEAR 的有：{beating_text}；同时满足 crossed-bootstrap CI 全负且 Holm p<0.05 的有：{confirmed_text}。当前所有模型里最低均值是 **{LABEL[best_overall]}**。结果不支持用本批潜在高频生成路线替换原 CBH。

这里没有真实 15 分钟天气是任务设定，不是实验遗漏。模型学习的是受小时锚点和小时遮盖正则约束的潜在高频 future covariates，并以 15 分钟功率误差作为主评价。天气代理指标只诊断正则行为，不承担气象真实性证明。

## 引入生成路线前的正式对照

交接文档的主模型是 **CBH：History-supported baseline + conditional future-weather correction**。它包含冻结 history-only ForcedEvolution 轨迹 `b`、显式 `Δb`、完整历史 Transformer memory、P0I 天气投影、condition projection、transfer adapter、Transformer decoder 和 `b + correction` 输出。旧批次的 `P0I_CBH_SINGLE_*` 用 PatchTST 替换原 baseline 且令 `Δb=0`，本报告只把它视为简化控制，不再称为原结构。

原 CBH seeds 42–51 checkpoint 已在同一 10 站、同一 2023 report 窗口直接运行。站点历史、目标、origin、phase、样本索引和 source rows 均逐值对齐。抽取到单站计算图后，100 个 station×seed 初始化前向的最大误差是 `{audit['max_initial_original_equivalence_abs_error']:.8g}`，低于全量初始化扫描后固定的 FP32 guard `{audit['initial_equivalence_tolerance']:.1g}`。

## 生成完以后接什么

完整路径是：

`小时 NWP(7×14) → 6个区间内各3个潜在天气 residual → 15min future covariates(24×14) → 原 CBH 天气投影/condition adapter/Transformer decoder → b + correction → 功率(24)`

已知小时节点旁路生成器并在输出端写回。联合目标是 `power MAE + 0.1 × weather proxy loss`；功率 MAE 可独立回传至八种生成器，新的 CBH gradient audit 已逐方法验证非零梯度。随机方法固定保存 5 个天气/功率样本，点预测预先固定为 5 个功率预测的中位数。

没有先把每帧压成唯一 CLS token。当前单站数据已经是固定 2×2 双线性提取后的 14 个物理变量，直接生成变量级 residual 更容易保留整点约束。若以后接完整天气 patch，适合保留多个 patch token，并让 CLS 只承担全局条件。

## 已有模型 baseline

以下是上一隔离批次使用的实际模型名，不是把“纯历史/纯天气”等类别当模型名：

{baseline_text}

DLinear、PatchTST 是历史功率模型；HistGradientBoostingRegressor 是 weather-only；TiDE、TFT、NBEATSx 都把未来天气作为已知未来协变量。它们使用同一站点、seed 和 report 窗口，但从 10% pilot 训练；原 CBH 使用完整 2022 数据且保留 56 站 history-only baseline，所以绝对 MAE 只能作背景，不能归因于某个单一架构差异。完整 2000 个结果和输入扰动审计保存在 `temporal_sr_named_models_20260917_v2`。

## 原 CBH 与九种适配路线

{result_text}

`LINEAR` 也从原 CBH checkpoint 初始化并在同一 2022 pilot 上微调 correction mapping。因此它相对原 CBH 的 Δ={signed(linear_vs_original['delta'])}，CI [{signed(linear_vs_original['ci95_low'])}, {signed(linear_vs_original['ci95_high'])}]，反映的是小样本单站再适配的整体影响。生成器与 LINEAR 的差才主要隔离潜在天气生成方法。

## 不同 flow/diffusion 对同一 LINEAR 控制

{linear_text}

Δ<0 表示前者更好。I-CFM 是 paired linear conditional flow；VP-CFM 更换为三角 variance-preserving path；Sine-SI 加入中间随机项；VP-epsilon 与 VP-v 共用 VP path、只改变 target parameterization；EDM 使用 Karras 预条件、rho=7 schedule 和 17-NFE Heun；MeanFlow 学区间平均速度并以 1 NFE 采样。DET 是确定性 residual 容量控制。

DDIM 属于采样器，没有被重复列作训练模型。OT-CFM 会破坏当前样本已绑定的站点/发行周期/物理时间配对，故没有做无条件 minibatch 重配。SB-CFM、I²SB、DDBM、TimeBridge 需要另外定义数据依赖 prior 和 paired bridge；当前结果没有冒称这些路线。

## 整条适配路线对未微调原 CBH

{original_text}

这一层比较回答实际是否值得替换原结构；它同时包含单站 pilot 微调和天气代理变化。结果应与上一节结合解释，不能把适配路线整体退化全部归到生成器。

## 小时遮盖代理诊断

{proxy_text}

最优生成路线 {LABEL[best_generated]} 的 4h→1h joint proxy MAE 为 {best_proxy['4h_after_joint_generated_mae']:.4f}，线性插值为 {best_proxy['4h_linear_mae']:.4f}；2h→1h 分别为 {best_proxy['2h_after_joint_generated_mae']:.4f} 和 {best_proxy['2h_linear_mae']:.4f}。这反映跨尺度正则的行为，不是 1h→15min 的气象真值误差。即使功率改善，也只能说明代理 covariates 对下游映射有用；即使代理遮盖 MAE变差，也不单独否定潜变量对功率的用途。

## 站点、协议和统计边界

固定 10 站为 `1043019, 1043010, 1043036, 1043007, 1043028, 1043033, 1043011, 1043058, 1043015, 1043061`。筛选只使用 2022 覆盖率、容量检查、非恒定信号、同期天气功率相关与地理/容量多样性；没有使用爬坡、功率差分、验证性能或本批结果。

每个模型运行 seeds 42–51；主汇总先按 seed 对 10 站等权，再报告 10 seeds 的均值、样本 SD、最小/最大及 seed。配对 CI 独立重采样 station 和 seed 20,000 次；Wilcoxon 在每个 seed 的 10 站等权差上计算，17 个预列比较统一做 Holm 校正。逐站 10-seed 明细在 `analysis/per_station_seed_summary.csv`。

原 CBH checkpoint 已使用完整 2023 validation 选模，随后 adapter 又使用独立的 2023 `val_stop` 选 checkpoint；因此这些是固定 report split 的探索结果，不是盲测。所选站点也不是 56 站的随机样本，bootstrap CI 不能解释为全省总体区间。

## 下一步决策

1. 主模型继续使用原 CBH；不依据本批结果替换生产/论文主结构。
2. 若继续潜在高频路线，先冻结原 CBH mapping，仅训练小型 linear-prior bridge/generator，再与“联合微调 mapping”做正交消融，以分离当前最明显的小样本再适配损失。
3. 生成状态仍使用 14 变量 missing residual；完整空间天气到位后再比较 patch tokens + 全局 CLS 条件。
4. 优先尝试 TimeBridge/I²SB 风格的 linear-spline data-dependent prior，并把观测点写回作为硬约束；仍以功率指标选路。只有要主张真实气象恢复时，才需要另取高频天气做外部真实性验证。
5. 新一轮必须使用未参与原 checkpoint 选择的新年份或新站点作确认；否则只保留探索性结论。

## 审计与文件

- `status/FINAL_AUDIT.json`：900 个适配结果、原 CBH 推理、checkpoint/protocol/hash、预测重算和 sample median 审计；
- `status/CBH_GENERATOR_GRADIENT_AUDIT.json`：只用功率 MAE、排除代理损失后的下游梯度；
- `analysis/overall_seed_summary.csv`：全部模型的 seed 等权汇总；
- `analysis/per_station_seed_summary.csv`：逐站 10 seeds 的均值、SD、最低/最高；
- `analysis/paired_comparisons.csv`：17 个预列配对比较；
- `analysis/proxy_summary.csv`：小时遮盖代理指标；
- `analysis/RESEARCH_AND_PROTOCOL_CN.md`：计算图、方法定义、训练协议与口径；
- `analysis/LITERATURE_SYNTHESIS_CN.md`、`analysis/METHOD_MATRIX_CN.md`：扩散、flow matching、bridge 与天气降尺度调研。
"""
    (ROOT / "analysis/FINAL_REPORT_CN.md").write_text(report)


if __name__ == "__main__":
    main()
