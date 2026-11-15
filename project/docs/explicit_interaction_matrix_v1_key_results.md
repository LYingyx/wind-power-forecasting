# 显式修正分解与统一融合：关键实验结论

日期：2026-08-27

## 核心结论
 
当前实验不应被表述为识别出了两个真实且唯一的修正项。更准确的总模型是：

$$
\boxed{
\hat Y_k
=B(H)_k+C\!\left(
\mathcal M\left(F_W(W,\tau_k);F_S(S_{\rm recent}),\tau_k\right)
\right)
}
$$

其中只有一个总 correction。`weather main effect + state-conditioning benefit` 可以作为四格信息消融的**效应分析语言**，不能直接等同于可唯一辨识的 $C_W(W)+C_{SW}(S,W)$。

本轮结果没有证据要求在输出层永久保留两个显式修正分支：低秩 multiplicative 数值最好，但相对一般 black-box 的两类 95% 区间都跨 0。它适合作为下一步 horizon-aware interaction 的候选归纳偏置，而不是已经识别的物理机制。

## 附件分析与必要修正

- 附件关于“一般联合函数 $C(S,W)$ 已包含 weather-only 函数”和“无约束 $C_W+C_{SW}$ 不可唯一分解”的判断正确。
- 当前矩阵并非每组都显式区分两类作用：`multiplicative` 才是 $C_W(W)+C_{SW}^{\rm low-rank}(S,W)$；`additive` 是 $C_W(W)+C_S(S)$；`asymmetric_modulation` 在表征层统一为 $z_W\odot(1+\tanh z_S)$，不输出伪可辨识的子修正；旧 `blackbox` 是一般联合函数。
- 四格实验中的 $C(S)\approx C(\varnothing)$ 只能解释为：**在冻结基线已经使用完整历史后，recent state 对剩余误差几乎没有额外主效应，但能提高未来天气修正的有效性。**它不能推出状态在整个预测问题中不是独立信息源。
- “状态调制天气响应”仍是预测交互假设，不是天气干预、物理因果或唯一函数分解。
- scale-only asymmetric 只是统一融合的一个最小原型；它未胜出不能否定统一 correction，只能否定当前具体参数化是最佳选择。

## 实验协议与审计

- 三个新结构各使用 seeds `42–51`，30/30 次训练完成；旧 black-box 复用相同 10 seeds。
- 所有模型共享同一个完全冻结的 $B_0(H)$、aggregate recent-1h 四个状态点、未来天气、full-frequency correction、训练预算和 early stopping。
- 仅使用 2023 validation；每组 32,550 个对齐 origins、359 个 UTC 日；未生成或使用 test 产物。
- 新三组均为 1,287,795 参数；black-box 为 1,287,343，差 452（约 0.04%）。新三组总参数匹配，但未参与某一前向模式的模块不同，因此不能声称有效函数容量完全相同。
- 30/30 个新 checkpoint 的冻结基线逐张量完全一致。基线预测重放平均绝对差 0.011 MW，单次最大 0.040 MW。
- 单模型差异按相同 seed 配对；ensemble 差异使用 10,000 次 sample-weighted paired UTC-day block bootstrap。

## 主结果

| correction 内部融合 | 10-seed 单模型 MAE（MW） | 10-model ensemble MAE（MW） |
|---|---:|---:|
| black-box joint | 267.214 ± 3.985 | 252.577 |
| additive：weather main + state main | 267.053 ± 3.628 | 253.458 |
| multiplicative：weather main + low-rank interaction | **266.261 ± 4.631** | **251.946** |
| asymmetric：统一 scale-only modulation | 266.879 ± 5.254 | 253.185 |

下表把相对 black-box 的改善写成正值：

| 候选结构 | 单模型配对改善 [95% CI] | 胜场 | ensemble 改善 [按日 95% CI] | 判断 |
|---|---:|---:|---:|---|
| additive | 0.161 [-0.926, 1.247] MW | 4/10 | -0.881 [-2.297, 0.840] MW | 无优势 |
| multiplicative | 0.953 [-0.225, 2.130] MW | 6/10 | 0.631 [-0.433, 1.760] MW | 数值最佳，但证据不稳定 |
| asymmetric | 0.334 [-1.235, 1.904] MW | 5/10 | -0.608 [-1.606, 0.429] MW | 无优势且波动更大 |

multiplicative 相对 additive 的 ensemble 改善为 1.512 MW `[0.251, 2.753]`，但单模型配对区间跨 0；相对 asymmetric 的 ensemble 改善为 1.239 MW `[0.473, 2.038]`，单模型配对区间仍跨 0。因此可以把 multiplicative 选为下一轮结构候选，但不能宣称它已稳定优于 black-box。

## 为什么不能解释显式分支

- additive 的 weather-main 平均绝对幅度为 576.8 MW，state-main 为 390.3 MW，最终 raw correction 却只有 386.1 MW；80.4% 的样本中两分支异号。
- multiplicative 的 weather-main 和 interaction 平均绝对幅度分别为 492.1、492.3 MW，最终 raw correction 为 390.5 MW；77.0% 的样本中两分支异号。
- 两组显式分支在数值上都精确加回 raw correction，最大误差小于 0.001 MW；问题不是保存错误，而是优化后存在大量职责抵消。

因此，显式分支只能说明该参数化如何得到最终预测，不能被命名为“天气真实贡献”和“状态真实贡献”。如未来确实需要效应识别，应另做条件中心化，例如约束 $E_S[C_{SW}(S,W)\mid W]=0$，或使用 post-hoc functional ANOVA、置换与反事实分析，而不是直接解释原始分支输出。

## 时效结构与下一步

multiplicative 相对 black-box 在 0.25h 恶化 0.692 MW `[-0.890, -0.502]`，到 6h 改善 2.285 MW `[0.343, 4.284]`。按时段汇总，0.25–1h、1.25–3h、3.25–6h 的改善依次为 -0.262、0.218、1.203 MW；区间显示总体形态明确，但后两个时段仍不能单独判定稳定胜出。asymmetric 在 0.25–1h 稳定恶化 1.125 MW `[-1.364, -0.863]`。

下一步只做一个受控因素：

1. 以 ungated multiplicative 为候选，运行 `multiplicative` vs `horizon_gated_multiplicative` 的 10-seed 配对实验；gate 只控制 state–weather interaction，不控制完整 weather correction。
2. black-box 继续作为性能参考。只有 gated multiplicative 同时改善 seed 稳定性、ensemble 和短/长时效形态，才把显式交互定为主结构。
3. horizon gate 是延后验证，不是被否定；当前不同时加入 scale+shift、差分阶数或空间图，避免无法归因。
4. 融合基型确定后，再在弱 anchor、可训练 $B(H)$ 下复验；方案冻结后才一次性使用未触碰 test。

当前统一判断是：**分析上保留天气主效应与状态条件收益的区分；总公式和机制叙事采用单一 correction；网络内部可以使用 multiplicative 交互作为归纳偏置，但不赋予两个输出分支唯一作用边界。**

## 结果位置

- 正式训练：`artifacts/explicit_interaction_matrix_v1/groups/`
- 汇总、配对区间、时效和分支诊断：`artifacts/explicit_interaction_matrix_v1/analysis/`
- 关键文件：`summaries.csv`、`comparisons.csv`、`ensemble_contrasts.csv`、`horizon_bands.csv`、`component_seed.csv`、`analysis.json`

