# Regional generation composition：关键实验记录

日期：2026-08-31

## 结论

区域内 56 个场站的近期状态确实存在明显异质性，但在当前信息、模型和训练协议下，无论手工分布统计还是 tiny DeepSets，都没有在 `aggregate + self` 状态之外提供稳定且具有部署价值的集群预测增益。因此不把 regional composition 纳入现有主模型，不重建 teacher/KD，并停止 graph、learned adjacency、邻居/K 搜索和 lag-aligned propagation 等空间状态架构搜索。

这是“未发现可稳定利用的额外增量”，不是“证明跨场站信息在任何条件下都无效”。

## 对原建议的修正

原建议把问题从场站间消息传播改为“区域发电状态构成”，方向合理：同步邻居均值会丢失场站间 dispersion、quantile 和 distribution shape，因而值得做最后一次简单验证。但原始 R0/R1/R2 会同时改变输入信息和网络容量，不能把差异归因于 composition。

正式实验改为严格嵌套、完全参数匹配的三组（均为 `1,318,895` 参数）：

- R0 matched control：self 通道不变；statistics 和 set 分支只接收 aggregate recent 的冗余重表达。
- R1 handcrafted statistics：仅将 statistics 分支换成真实 56 站统计。对最近四个 15 min 时刻分别计算 mean/std/q25/q50/q75，并对最后一个 15 min ramp 计算同五项，共 25 维。
- R2 tiny DeepSets：保留 R1 的真实统计，并将 set 分支换成 56 站的四点 recent trajectory 与最后 ramp；共享小 MLP 后做 mean/max pooling。无 station ID、attention 或 graph，保持纯 permutation-invariant composition test。

因此 R1−R0 检验手工构成信息，R2−R1 检验 learned set 的额外价值，R2−R0 检验 composition 总价值。旧 G1 与 R0 的容量和优化轨迹不同，只用于审计，不作为 composition 的主比较。

正式协议为 seeds 42--51、Normal training、无 KD、相同初始化来源、weak anchor `0.2`、4×4 NWP 前端、预算和 early stopping；只使用 2023 validation。单模型采用同 seed 配对 t 区间，10-model ensemble 采用 10,000 次 UTC-day block bootstrap（359 天），R1/R2 vs R0 做 Holm 校正。候选进入主模型须同时满足：seed 和 ensemble 区间支持改善、Holm 后成立、无显著超过 2 MW 的 horizon 退化，且 single/ensemble 中心均至少改善 2 MW。

## 输入诊断

- aggregate 与无权 station mean 的相关系数为 `0.9957`：均值信息几乎已被 aggregate 覆盖。
- 跨站 std 的 mean/median/p95 为 `0.1600/0.1619/0.2923`。
- 跨站 IQR 的 mean/median/p95 为 `0.2147/0.1938/0.4863`。
- last-ramp 跨站 std 的 mean/median/p95 为 `0.0459/0.0415/0.0975`。

所以 composition 输入不是退化常数，但可能的新增信息只能主要来自离散度和分布形状，而非重新编码均值。

## 10-seed 结果

| 组别 | 单模型 MAE mean ± std（MW） | 10-model ensemble（MW） |
|---|---:|---:|
| R0 matched control | 258.258 ± 2.134 | **242.496** |
| R1 handcrafted statistics | 258.775 ± 2.091 | 243.022 |
| R2 tiny DeepSets | **258.093 ± 2.101** | 242.668 |

下表均为“候选减参考”，负值表示候选更好：

| 比较 | 单模型配对差 [95% CI]（MW） | 胜场 | Holm p | ensemble 差 [按日 95% CI]（MW） |
|---|---:|---:|---:|---:|
| R1 vs R0 | +0.518 [-0.221, 1.257] | 4/10 | 0.295 | +0.526 [-0.367, 1.421] |
| R2 vs R0 | -0.165 [-0.785, 0.456] | 8/10 | 0.563 | +0.172 [-0.623, 0.975] |
| R2 vs R1 | -0.682 [-1.515, 0.150] | 8/10 | 0.097（未作主族 Holm） | -0.354 [-1.446, 0.729] |

R1 的 single 与 ensemble 中心都更差。R2 虽有 8/10 seed 获胜，但平均 single 只改善 `0.165 MW`，区间跨 0；其 ensemble 反而恶化 `0.172 MW`。8/10 胜场主要表示多数变化很小，不能覆盖少数 seed 的较大反向变化。R2 相对 R1 有弱方向性优势，但 seed 与 ensemble 区间仍跨 0，而且 R2 并未超越 R0。因此不能把 DeepSets 写成有效机制。

四个主比较 CI（R1/R2 的 single/ensemble vs R0）均完整落在 ±2 MW 内：本实验不仅未证明改善，也排除了达到预设 2 MW 部署门槛的稳定中心效应。

## Horizon 与对照审计

R1 vs R0 在 `0.25--1 h` 恶化 `+0.928 [0.671, 1.182] MW`（未作跨 band 多重校正的区间不跨 0），中、长时效区间跨 0。R2 vs R0 的三个时效分别为：

- `0.25--1 h`：`+0.030 [-0.122, 0.185] MW`；
- `1.25--3 h`：`-0.267 [-0.928, 0.388] MW`；
- `3.25--6 h`：`+0.513 [-0.669, 1.678] MW`。

R2 没有严重 horizon 风险，但也没有任何时效段的稳定改善。

旧 G1 的单模型/ensemble 为 `257.480/243.916 MW`；新 R0 为 `258.258/242.496 MW`。R0 单模型更差、ensemble 更好，说明冗余匹配分支会改变容量、优化轨迹和 ensemble 多样性，不能把 R0 vs 旧 G1 解释为信息机制。这也证明使用新 R0 作为 R1/R2 的同容量对照是必要的。

## 最终 insight 与决策

1. 观测到 regional heterogeneity 不等于它在 aggregate/self/NWP 之外仍有稳定的条件预测价值；本轮结果支持后者很小或被现有状态表示吸收。
2. Handcrafted composition 对主目标集群 MAE 没有价值，并且近时效有小幅退化信号；不再扩展更多分位数、ramp 或窗口搜索。
3. Tiny DeepSets 比手工统计略有方向性优势，但没有超过 matched control，也没有 ensemble 支持；不升级为论文模块。
4. 不继续 lag-aligned neighbors。该方向需要更强的平流、时延和邻居先验，且可能与未来 NWP double counting；现有同步邻居与 composition 证据都不足以支撑开启新的搜索族。
5. 冻结当前 architecture，保留已有 consensus-KD 主模型；不为 R1/R2 重训 teacher 或 KD。论文可将这一轮作为 `empirically constrained architecture` 的关键设计依据，而非标题或贡献中心。

## 产物与审计

- 正式训练与 30 份 validation 预测：`artifacts/regional_composition_v1/groups/`
- 完整统计：`artifacts/regional_composition_v1/analysis/analysis.json`
- 汇总与比较：`artifacts/regional_composition_v1/analysis/summaries.csv`、`comparisons.csv`
- Horizon 与决策：`artifacts/regional_composition_v1/analysis/horizon_bands.csv`、`decisions.csv`
- 实现：`src/wind_forcing/models/incremental_correction.py`
- 运行/分析入口：`shell/launch_regional_composition_matrix.sh`、`scripts/analyze_regional_composition.py`

审计结果：30/30 正式 runs、metrics、predictions 和 configs 齐全；实验目录内无 test 产物；三组参数完全相同，配置仅在信息模式和 experiment ID 上不同；信息嵌套、参数匹配和 permutation invariance 的定向回归检查通过。
