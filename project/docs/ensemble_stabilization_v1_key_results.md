# Ensemble 稳定化与蒸馏：关键结果

日期：2026-08-29

## 结论

原计划的主方向合理：冻结科学结构为

\[
\hat Y=B(H)+C(S_{\rm recent},W,\tau),
\]

把机制发现用于决定模型输入与职责，把 ensemble 用于识别和压缩训练方差。完整验证后，最终推荐为：

\[
\boxed{B(H)+C(S_{\rm recent},W,\tau)+\text{plain ensemble-consensus KD},\quad \lambda_{\rm KD}=0.5}
\]

它是当前最强单模型：validation cluster MAE 为 **251.116 ± 1.617 MW**，较未稳定化 Normal 改善 **7.447 MW**，95% CI `[6.655, 8.238]`，10/10 seeds 胜；回收了 **53.8%** 的 Normal single–ensemble gap。其 10-model ensemble 为 **242.899 MW**。

不推荐 checkpoint averaging 或 disagreement-aware KD。EMA 是有效且便宜的次选，但显著弱于最终 KD。

## 对原计划的修正

- `Corr(disagreement, ensemble gain)>0` 不能单独证明“可蒸馏方差”。MAE 的凸性使逐点 ensemble gain 几乎必然非负；本实验补充 signed-error correlation、disagreement decile 和 MSE 精确 ambiguity decomposition。
- KD 使用 `L_GT + λ_KD L_KD`，不是缩小 ground-truth 项的凸组合。`λ∈{0.25,0.5}`，未做大范围搜索。
- teacher 对训练集产生预测是标准 train-only KD，不构成 train–validation 泄漏；本轮未做昂贵 cross-fitting。
- EMA 和 checkpoint average 都使用 Normal 选出的最佳 epoch，不单独借 validation 重新选点。
- 旧 strongest fusion 只有 seeds 42–44，只保留为描述性结果，正式推断均使用相同 seeds 42–51。
- 仓库旧实验已存在 test 指标与预测，所以现有 test 不是 pristine。本阶段没有读取 test；最终无偏确认必须新建时间外或区域外 holdout。

## 实验口径

- 随机种子：42–51；每个正式组 10 次独立训练。
- 选择与分析：仅 2023 validation；test 未读取。
- 主指标：10-seed 单模型 cluster MAE 均值 ± 样本标准差。
- 单模型差异：同 seed 配对 t 区间；稳定化比较做 Holm 校正。
- ensemble 差异：10,000 次 sample-weighted paired UTC-day block bootstrap，共 359 个 UTC 日。
- EMA：decay 0.999；checkpoint average：截至 Normal 最佳 epoch 的最近 5 个 epoch。
- teacher：10 个 Normal checkpoint 的训练集算术均值与标准差。

## 公平排行榜

| 模型 | seeds | 单模型 MAE（MW） | 10-model ensemble（MW） | single–ensemble gap（MW） |
|---|---:|---:|---:|---:|
| Power-only | 10 | 410.471 ± 0.754 | 407.598 | 2.873 |
| Weather-only | 10 | 274.231 ± 1.595 | 260.461 | 13.770 |
| Direct joint d64 | 10 | 262.089 ± 1.787 | 250.203 | 11.885 |
| 旧 prediction-weight fusion，仅描述性 | 3 | 265.367 ± 3.118 | 255.600 | 9.766 |
| Full-history anchor 0.20 | 10 | 258.311 ± 1.583 | 242.915 | 15.396 |
| Recent-state Normal | 10 | 258.563 ± 1.816 | 244.732 | 13.831 |
| Recent-state EMA | 10 | 253.494 ± 2.163 | 245.251 | 8.243 |
| Recent-state checkpoint average | 10 | 256.731 ± 2.535 | 247.410 | 9.321 |
| Plain KD, λ=0.25 | 10 | 254.438 ± 1.687 | 243.437 | 11.001 |
| **Plain KD, λ=0.5** | **10** | **251.116 ± 1.617** | **242.899** | **8.217** |
| Disagreement-aware KD, λ=0.5 | 10 | 252.396 ± 1.871 | 243.086 | 9.310 |

Recent-state Normal 相对 Direct 单模型改善 3.526 MW，95% CI `[2.236, 4.816]`，10/10 胜；ensemble 改善 5.471 MW，按日 CI `[1.740, 9.108]`。它与 full-history anchor 的单模型差异不可靠，但 recent-state ensemble 差 1.816 MW，CI `[0.476, 3.248]`。因此最近 1 h 状态足以维持单模型能力，却损失了一部分跨 seed 的 ensemble center/diversity；KD 基本补回了这部分损失。

## Ensemble 机制证据

Recent-state Normal 的 single–ensemble gap 为 13.831 MW，平均 seed disagreement 为 103.214 MW，平均 signed-error correlation 为 0.8977；Direct 对应为 96.578 MW 和 0.9084。Recent 模型的误差相关更低、ensemble center 更强，满足预注册 plain-KD 门控。

分歧和 ensemble gain 都随时效增加：

| 时效 | disagreement（MW） | ensemble gain（MW） | MSE ambiguity（MW²） |
|---|---:|---:|---:|
| 0.25–1 h | 34.247 | 4.081 | 1,383 |
| 1.25–3 h | 90.075 | 11.313 | 9,515 |
| 3.25–6 h | 134.962 | 18.759 | 20,252 |

在 3.25–6 h，最低 disagreement decile 的平均 gain 为 3.867 MW，最高 decile 为 49.313 MW。不过这不能单独当因果证据：逐点 MAE gain 的最小值仅为数值误差 `-4.5e-13 MW`，体现了 Jensen/三角不等式的机械约束。更可靠的辅助证据是 signed-error correlation 降低，以及 MSE 恒等式

\[
\operatorname{MSE}_{\rm single}
=\operatorname{MSE}_{\rm ensemble\ center}
+\operatorname{ambiguity}
\]

的最大数值误差仅约 `2.5e-9 MW²`。

## 稳定化结果与判断

| 候选 vs Normal | 单模型配对改善（MW） | 95% CI | 胜场 | recovery |
|---|---:|---:|---:|---:|
| EMA | 5.069 | `[3.695, 6.443]` | 10/10 | 36.7% |
| Checkpoint average | 1.832 | `[0.565, 3.098]` | 9/10 | 13.2% |
| Plain KD 0.25 | 4.125 | `[3.199, 5.050]` | 10/10 | 29.8% |
| **Plain KD 0.5** | **7.447** | **`[6.655, 8.238]`** | **10/10** | **53.8%** |
| Aware KD 0.5 | 6.166 | `[4.851, 7.481]` | 10/10 | 44.6% |

关键直接比较：

- Plain KD 0.5 相对 EMA 改善 2.378 MW，95% CI `[0.697, 4.058]`，8/10 胜；ensemble 也改善 2.351 MW，按日 CI `[1.002, 3.640]`。
- Plain KD 0.5 相对 KD 0.25 改善 3.322 MW，95% CI `[2.775, 3.869]`，10/10 胜。
- Aware KD 相对 plain KD 0.5 **变差 1.280 MW**，95% CI `[0.635, 1.926]`，0/10 胜，Holm 校正 `p=0.0076`；ensemble 差异不可靠。
- EMA 稳定改善单模型，但没有改善 ensemble center；checkpoint average 虽改善单模型，却使 ensemble 显著变差 2.679 MW，按日 CI `[1.162, 4.249]`。

KD 0.5 使平均 seed disagreement 从 103.214 降到 78.355 MW，signed-error correlation 从 0.8977 升到 0.9349，single–ensemble gap 从 13.831 降到 8.217 MW，同时 ensemble center 仍改善 1.832 MW。因此它不只是让成员变得相似，而是把 ensemble 共识有效转入单模型。

Aware-KD 的失败说明 disagreement 不能直接等同于 teacher 不可信度。高分歧区域同时也是 ensemble gain 最大的区域；简单降低这些位置的 teacher 权重会丢失有价值的平均化信号。若未来继续研究 uncertainty weighting，应先区分“可消除的成员方差”和“ensemble center 本身的误差”，不能只用成员标准差。

## Stop / Go

- **Go**：冻结 plain ensemble-consensus KD、`λ_KD=0.5` 作为当前最终单模型；若部署允许多模型，KD ensemble 可继续从 251.116 降到 242.899 MW。
- **Stop**：不再扩展 fusion operator、DCT、checkpoint averaging、aware-KD 或更多 KD 权重。
- **边界**：当前是 validation 上的模型选择结果，不是最终泛化确认。下一次只应在新建的时间外或区域外 pristine holdout 上一次性评估 Direct、full-history anchor、最终 KD single 和最终 KD ensemble。

## 结果位置

- 统一机器可读分析：`artifacts/ensemble_stabilization_v1/analysis/analysis.json`
- 排行榜与配对统计：`artifacts/ensemble_stabilization_v1/analysis/summaries.csv`、`paired_comparisons.csv`
- 机制诊断：`ensemble_diagnostic_bands.csv`、`disagreement_deciles.csv` 及三张 PNG 图
- 正式模型与预测：`artifacts/ensemble_stabilization_v1/groups/`
- 训练集 teacher：`artifacts/ensemble_stabilization_v1/teacher/train_teacher_normal.npz`
- 因执行期代码变更而中止的非正式目录 `artifacts/ensemble_stabilization_v1_aborted_live_code_change_20260828/` 未进入任何分析。

最终代码检查：38/38 tests 通过，Python compileall 和 shell 语法检查通过。
