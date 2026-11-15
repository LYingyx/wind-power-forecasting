# Horizon-adaptive interaction 控制矩阵：关键实验结论

日期：2026-08-28

## 统一结论

这轮实验完成了预定的停止性验证：五类 horizon-adaptive operator 都没有同时通过 10-seed 主比较、短中长时效一致性和结构诊断。因此，不再继续搜索新的 fusion operator；默认主线保留一般的 **black-box recent-state-conditioned correction**。旧 ungated multiplicative 只保留为数值候选，不能上升为已识别机制。

实验否定了最有吸引力的具体故事：**recent state 的主要作用不是随 horizon 增长而改变模型读取未来天气轨迹的位置。** Conditioned query 和 attention bias 的天气注意力变化没有随 horizon 稳定增强，且两者预测性能均不优；attention bias 还是五个主比较中唯一经 Holm 校正后明确劣于 black-box 的结构。

FiLM 出现了长时 ensemble 改善信号，但 seedwise 证据不稳且短时显著退化，不能据此选为统一主结构。如果业务单独重视 3.25–6h，只值得做一次独立确认；否则本轮即为 fusion architecture search 的终点。

## 对原计划的分析与修正

原计划中正确且已落实的部分：

- 不再围绕乘法做近邻变体，而是比较 state 作用于 feature、retrieval、compatibility 和 mapping 等不同位置。
- 所有新候选共享冻结的 $B_0(H)$、recent-1h state、完整未来天气、full-frequency correction、rank 16、训练预算及 $g_k=2\sigma(a_k)$；24 个 gate 初值均为 1，第一轮不加平滑正则。
- gate 只控制 state-conditioned route，不切断 weather 主路径；MoE 因容量和优化差异更大而不进入首轮。
- 预先同时评价总体 accuracy、短中长 horizon consistency 和结构稳定性，而不是只选择最低单次 MAE。

需要收窄的部分：

- 五类 operator 是不同的**控制位置和归纳偏置**，不是五种可唯一辨识的物理机制。在线性 attention 中，conditioned query 本身会产生一类 state-key score bias；conditional LoRA 与低秩 feature transform 也可被前后网络近似重参数化。
- FiLM 的 $\beta(S)$ 是不依赖天气的 state-only offset，能够绕过 weather 路径。因此 FiLM 的长时收益不能全部命名为“状态调制天气”。
- learned gate 与 operator 权重尺度耦合，gate 大小不等于真实作用强度；跨 seed 形态稳定也不等于预测有效。
- 对已训练 checkpoint 设 $g=0$ 虽在函数上关闭 state route，但这是分布外干预，不等价于重新训练的 weather-only 对照，也不能作为因果或必要性证明。
- 旧 ungated M2 与新 gated M2 不只相差 gate：旧模型的 control state 含 history response/target-time token，新模型使用纯 recent-state encoder。因此二者差异不能全部归因于 horizon gate。

## 实现中的关键纠错

预检时发现第一版 operator state 输入混入 target-time token，可能让 operator 学到 horizon-only 捷径。该版本在 0/50 个正式任务完成时即中止，结果不进入比较，并归档至：

`artifacts/horizon_control_matrix_v1_aborted_target_token_input_20260828/`

正式版本的 operator control 只来自 recent-state encoder；target time 只通过各模型共同的 horizon gate/weather query 进入。这个修正确保比较的是“recent state 如何影响 weather correction”，而不是某种结构更容易重复读取 horizon embedding。

## 实验协议与完整性审计

- 新训练 5 类结构，每类 seeds `42–51`，共 50/50 个 checkpoint；另复用相同 seeds 的 black-box 和旧 ungated M2。
- gate 干预对每个新 checkpoint 分别生成 $g=0$、$g=1$ 预测，共 100/100 套消融预测。
- 本轮模型选择只使用 2023 validation：32,550 个对齐 origins、每个 24 leads、359 个 UTC 日；test 未使用。
- 单模型比较按相同 seed 配对；ensemble 使用 10,000 次 sample-weighted paired UTC-day block bootstrap；相对 black-box 的五个主 seedwise 比较做 Holm 校正。
- 50 个新 checkpoint 的冻结基线参数逐张量完全一致；baseline replay 平均绝对差 0.011 MW，单次最大 0.040 MW。
- 所有正式及消融 prediction 均为 $32{,}550\times24$，origin/target 完全对齐。
- 新结构参数量为 1,290,919–1,292,055，最大差 1,136，仅占约 0.088%。

## 主结果

| 结构 | 10-seed 单模型 MAE（MW） | 10-model ensemble MAE（MW） |
|---|---:|---:|
| black-box reference | 267.214 ± 3.985 | 252.577 |
| 旧 ungated M2 | 266.261 ± 4.631 | 251.946 |
| Horizon-FiLM | **266.106 ± 5.043** | **251.467** |
| Gated M2 | 267.276 ± 3.798 | 252.908 |
| Conditioned query | 268.787 ± 3.710 | 254.121 |
| Attention bias | 269.962 ± 4.834 | 254.836 |
| Conditional LoRA | 266.802 ± 4.233 | 251.966 |

相对 black-box 的改善记为正值：

| 新候选 | seedwise 配对改善 [95% CI] | 胜场 | Holm $p$ | ensemble 改善 [按日 95% CI] | 判断 |
|---|---:|---:|---:|---:|---|
| FiLM | 1.108 [-0.375, 2.590] | 7/10 | .424 | 1.110 [0.222, 1.983] | 有 ensemble 信号，seedwise 不稳 |
| Gated M2 | -0.062 [-0.892, 0.768] | 4/10 | 1.000 | -0.332 [-1.338, 0.659] | 与 black-box 持平 |
| Conditioned query | -1.574 [-3.555, 0.408] | 4/10 | .424 | -1.545 [-3.259, 0.282] | 无优势 |
| Attention bias | -2.748 [-4.208, -1.288] | 2/10 | .0106 | -2.259 [-3.998, -0.452] | 明确淘汰 |
| Conditional LoRA | 0.412 [-0.917, 1.741] | 7/10 | 1.000 | 0.611 [-0.321, 1.633] | 数值略好但不稳定 |

最低均值不等于主线胜出：FiLM 的 seed 标准差最大，且五个主比较中只有 attention bias 经多重比较校正后达到明确差异；这个差异方向还是更差。

## Horizon consistency

下表仍以“相对 black-box 改善”为正值，区间来自 paired UTC-day block bootstrap：

| 新候选 | 0.25–1h | 1.25–3h | 3.25–6h | 关键判断 |
|---|---:|---:|---:|---|
| FiLM | -0.392 [-0.625, -0.162] | 0.526 [-0.162, 1.252] | **1.999 [0.744, 3.333]** | 以显著短时损失换长时收益 |
| Gated M2 | 0.137 [-0.092, 0.370] | -0.695 [-1.439, 0.075] | -0.246 [-1.796, 1.326] | 无一致改善 |
| Conditioned query | -1.519 [-1.891, -1.130] | -1.617 [-3.100, -0.096] | -1.505 [-4.143, 1.157] | 短、中时显著更差 |
| Attention bias | -0.405 [-0.760, -0.020] | -1.479 [-2.889, -0.005] | -3.397 [-6.154, -0.673] | 三段均退化，长时最差 |
| Conditional LoRA | -0.242 [-0.412, -0.065] | 0.168 [-0.626, 1.042] | 1.191 [-0.236, 2.683] | 短时显著退化，长时不确定 |

新 gated M2 相对旧 ungated M2：短时改善 0.399 MW `[0.207, 0.586]`，但中、长时分别恶化 0.913 `[0.315, 1.531]` 和 1.449 `[0.348, 2.556]` MW；总体 seedwise 恶化 1.015 MW `[0.062, 1.967]`，ensemble 恶化 0.963 MW `[0.229, 1.730]`。由于 control-state 定义同时变化，这只能说明正式的新整体设计没有胜出，不能单独判定 gate 有害。

## Gate 与内部结构诊断

| 结构 | gate 长时−短时 | 跨 seed 曲线相关 | 归一化内部变化长时−短时 | 解释 |
|---|---:|---:|---:|---|
| FiLM | +0.030（10/10） | .809 | update +0.054（9/10） | 长时 modulation 增强，但未带来统一收益 |
| Gated M2 | +0.106（10/10） | .938 | update +0.102（10/10） | 形态稳定，accuracy 仍未胜出 |
| Conditioned query | -0.038 | .780 | attention shift -0.021（CI 跨 0） | 不支持长时更强地改变天气读取位置 |
| Attention bias | -0.047 | .803 | attention shift -0.011（CI 跨 0） | 同样不支持 selection 故事 |
| Conditional LoRA | +0.070（10/10） | .938 | update +0.298（10/10） | 长时更新增强，但没有稳定转化为收益 |

这些内部量只适合比较同一 operator 内的 horizon 形态，不能跨 operator 比较绝对强弱。最重要的反证是：query/bias 的 attention shift 没有随 horizon 增强，同时两类模型也不优，因此“时效越长，state 越改变读取哪些天气阶段”的预期没有得到支持。

## Gate 干预结论

- 将训练后 checkpoint 直接设为 $g=0$，五类模型 seedwise MAE 分别恶化约 72.1–186.3 MW，全部为 10/10。这只表明已训练网络强依赖 state-conditioned route；因为网络从未在 $g=0$ 分布下重新适配，所以不能据此声称 state route 带来同等大小的真实增益。
- learned gate 相对固定 $g=1$：只有 gated M2 有较可靠的 checkpoint 内总体价值，seedwise 改善 0.823 MW `[0.444, 1.203]`，ensemble 改善 0.313 `[0.179, 0.444]` MW。FiLM、query 和 LoRA 总体近零；attention bias 只有约 0.1 MW 的小改善，而其整体结构仍明确劣于 black-box。
- FiLM 和 LoRA 的 learned gate 在短时分别比 $g=1$ 恶化 0.242 和 0.392 MW。由此可见 horizon gate 不是跨 operator 通用的增益机制。

## Stop/Go 决策

1. **Stop：**结束 fusion architecture search；淘汰 conditioned query 和 attention bias，不进入 MoE、动态卷积、SSM 或 frequency expert 扩展。
2. **Go：**默认采用一般 black-box recent-state-conditioned correction。它是当前最稳妥的结构结论；科学表述保留为“recent state 有助于条件化未来天气修正”，不宣称具体控制对象已被识别。
3. **保留但不升级：**旧 ungated M2 仍是数值候选，但其相对 black-box 的既有区间跨 0，新 gated 版本又未复现优势，不能作为论文中的已验证主机制。
4. **仅在长时效具有独立价值时确认 FiLM：**重新训练 gamma-only、beta-only 和完整 FiLM，对照 black-box，使用一组独立 seeds；必须同时消除短时退化并获得 seedwise 稳定改善才继续。beta-only 用于检验收益是否实际来自绕开 weather 的 state offset。若独立确认失败，停止该分支。
5. test 继续保持未触碰；只有模型和选择规则完全冻结后，才进行一次性 test 评估。

## 结果位置

- 正式训练与预测：`artifacts/horizon_control_matrix_v1/groups/`
- 统一统计分析：`artifacts/horizon_control_matrix_v1/analysis/`
- 关键汇总：`summaries.csv`、`paired_comparisons.csv`、`ensemble_contrasts.csv`、`horizon_bands.csv`、`gate_ablation.csv`、`gate_ablation_horizon_bands.csv`、`analysis.json`
- 中止的 shortcut 版本：`artifacts/horizon_control_matrix_v1_aborted_target_token_input_20260828/`
