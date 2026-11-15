# 论文科学问题与贡献点草案

日期：2026-10-16。依据已完成的 canonical、P0I 迁移、条件入口、summary 和路径诊断实验。模块取舍采用用户当前接受小幅收益的标准，详见[模块证据与取舍](MODULE_EVIDENCE_AND_DECISIONS_20261016_CN.md)。

## 核心定位

研究未来已知 NWP 与已实现发电历史如何协同完成未来 6 小时、15 分钟分辨率的风场功率预测。当前证据最强的设计是：历史模型先形成未来功率参考轨迹，修正网络显式读取该参考，并结合未来天气与历史 memory 预测修正。

\[
\mathbf b=B_0(H_i,H_{agg},context),\qquad M_i=E_H(H_i),
\]
\[
\hat{\mathbf y}_i=\mathbf b_i+C_\theta(W_i^+,M_i,\mathbf b_i,\Delta\mathbf b_i,context).
\]

冻结 B₀ 使参考轨迹在 correction 训练中保持固定。该分解是预测建模设计；b 不是无天气世界的物理反事实，correction 也不是被单独识别的纯天气因果贡献。

## 三个关键科学问题

### Q1：天气修正网络是否需要显式知道它正在修正的历史预测轨迹？

即使网络可以读取历史隐表示，是否仍有必要把历史模型在功率空间的未来预测 b 显式提供给修正分支？这里讨论的是表示和计算路径的建模效用；b 来自历史，不是与历史独立的新观测源。

证据：canonical L−W 为 −0.007802/−0.007107；最终骨干 CB−CNEUTRAL 为 −0.004680/−0.004442。两组中两个年份均为 10/10 seeds 改善。后者在公共 decoder memory 和输出参照都保留的条件下改变 condition 的样本特异入口，直接支持显式轨迹条件的价值。

边界：最终 CB 含 b 和 Δb；收益也可能包含参考校准，不能直接解释为物理 operating-point sensitivity。

### Q2：历史信息应如何进入修正过程，显式轨迹与历史 memory 是否互补？

比较严格输出加性修正、联合修正，以及显式轨迹与 raw-history 条件的相互消融。进一步区分完整 decoder memory 与额外 summary 两条历史路径。

证据：canonical J−A 为 −0.003004/−0.002122；J−H 为 −0.000953/−0.000838；J−T 为 −0.000918/−0.001857，均 10/10 seeds。最终骨干 SLAST−CB 为 −0.000338/−0.000447，9/10、8/10。

边界：canonical 互补性限定该架构和协议；A 与 L 不是嵌套控制，不能用 A−L 宣称独立 history 分支有害。最后 token summary 有小幅增益，不等于完整 decoder memory 已被单独验证为必需；特定乘法算子没有获得优越性支持。

### Q3：模型是否实际利用样本匹配的预测参考，其收益体现在哪些时效？

区分 reference 的输出相加路径与 condition 路径，以 condition-only 置换、常数替换和分路径 Jacobian 检查模型如何使用 b；配合逐 lead 分析，描述增益在预测时效上的分布。

证据：CBH condition-only donor shuffle 在两年造成约 +0.01369/+0.01320 MAE 退化；CNEUTRAL 对照保持不变。CBH combined Jacobian 对角均值约 0.9085/0.9060，不支持普遍完全抵消 baseline。轨迹相对 neutral 的改善主要出现在中远期。

边界：内部替换可能制造输入不一致，不能把退化直接视为因果效应。结果支持模型依赖参考路径，但一般非线性校准仍是兼容解释。canonical raw-history 条件的增益也随 horizon 增大，不能讲“raw history 只管近端、trajectory 只管远端”。

## 推荐的三条贡献

1. **以历史预测轨迹为显式参照的天气条件修正框架。** 通过两阶段训练固定历史支持的预测参考，使修正网络同时访问未来 NWP、完整历史 memory 与功率空间的参考轨迹。方法贡献集中在信息组织和参考条件入口，不把普通 Transformer、残差相加或池化单独包装成创新。

2. **历史预测参考与历史上下文作用的分层实证分析。** 通过严格输出加性对照、canonical 双向条件消融、最终骨干 neutral 对照及 summary 比较，确认显式轨迹条件的稳定效用，并量化历史摘要入口更小的增益。保留小效应及其不确定性，不将所有模块表述为不可替代。

3. **跨实现的性能验证与分路径依赖诊断。** 在统一 P0I 协议中验证条件修正组织对已有骨干的增量，并用只改变 condition 的干预、Jacobian 与逐时效分析约束机制解释。区分预测性能、计算依赖与物理因果。

这三条是当前实验支持的论文贡献候选；学术新颖性仍需与 residual forecasting、forecast calibration、future-covariate conditioning 等相关文献逐项比较，不能未做文献核对就使用“首次”或普适 SOTA。

## 论文关键数值应如何使用

| 证据层次 | 代表结果 | 可以支持什么 |
| --- | --- | --- |
| canonical 参考条件 | L−W：−0.007802/−0.007107 | 在该修正模型中显式 baseline level 有预测价值 |
| canonical 联合依赖 | J−A：−0.003004/−0.002122 | 当前联合修正优于当前严格输出可分离对照 |
| 最终骨干条件入口 | CB−CNEUTRAL：−0.004680/−0.004442 | 显式样本轨迹条件有稳定价值 |
| 共同协议性能迁移 | PJ−P0：−0.000367/−0.000328；PA 也改善 | 整套条件修正组织可对已有骨干产生小幅增量 |
| summary 去留 | SLAST−CB：−0.000338/−0.000447 | 附加历史摘要入口值得保留 |
| summary 替换 | SLAST−CBH：−0.000188/−0.000157 | 最后 token 作为零新增参数候选有小幅优势 |

各列差分均按原报告统计，顺序为 2023/2024。不要把约 5.5% 的 CBH−neutral 改善写成相对最强 P0 的改善；不要将不同训练协议的绝对 MAE 拼接成排行榜。PJ−PA 没有确认优势，因此不能主张联合 adapter 优于所有更简单的条件接口。

原正式 CBH 的 MAE 为 0.083814/0.081507；候选 SLAST 为 0.083626/0.081351。若采用 SLAST，须明确记录实现更新和选择依据；原 CBH 的路径诊断、共享/独立实验保持原模型标签，不能只改名称后当作 SLAST 的实验。

## 主文与附录分工

主文优先呈现总体模型、共同协议性能表、参考条件消融、canonical 互补性、summary 选择，以及一组 condition-only 诊断。summary 不必单列创新，但需要结果支撑其存在。

乘法、不同 summary、斜率、native/local/global 天气表示、clipping、场站异质性和 cluster 次级指标可进入附录或讨论。其小效应和负结果有助于说明复杂度取舍，不应被改写为所有复杂方法无效。

共享/独立的后续 2×2 统计已提供参数共享收益的补充证据，但属于不同问题；少样本迁移、低秩场站适配和代理天气生成不并入本篇主贡献。全站命名 baseline 使用独立单站 pilot 协议，不能用它与共享主模型的数值差直接宣称公平架构排名。

## 证据索引

- [canonical 归因](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/trajectory_attribution_20260914_v1/analysis/REPORT_CN.md)
- [最终骨干条件入口与 horizon 复核](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_attribution_closure_review_20260916/REPORT_CN.md)
- [P0I 共同协议迁移](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_condition_transfer_20260915_v1/analysis/REPORT_CN.md)
- [summary 固定读出](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_summary_calibration_20260916_v1/analysis/REPORT_CN.md)
- [MLP summary](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md)
- [CBH baseline 路径诊断](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_summary_diagnostic_seed49_20260916_v1/analysis/DIAGNOSTICS_CN.md)
- [共享/独立 2×2 汇总](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/cross_farm_shared_specific_20260920_v1/analysis/two_by_two_summary.json)

2023 已反复用于开发，2024 也已查看；本文应称跨年复评。seed CI 不能代替未见年份、未见场站或整个研究选择过程的不确定性估计。
