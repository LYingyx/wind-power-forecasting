# 集群风功率预测关键结论总览

更新日期：2026-09-05。

本文档汇总截至当前所有与 56 场集群未来 6 h 功率预测直接相关的关键实验。目标是统一数值口径、区分已经成立的预测结论与尚未成立的物理解释，并给出是否继续集群路线或切换单风场研究的决策依据。

## 1. 一页结论

1. **集群预测本身仍值得继续。** 当前最有效的信息组织不是先把区域天气和场站状态全部聚合，而是先在 56 个场站位置构造局地天气表示、共享地预测场站功率，再按装机容量物理求和。
2. **当前证据选择的正式模型是 G-Static。** 2023 validation MAE 为 `267.985 ± 3.067 MW`，冻结后的 2024 跨年检查为 `264.817 ± 1.919 MW`。
3. **G-Static 相对匹配无动态状态控制 B-Static 的纯 aggregate-state 增量为 2.177/1.986 MW。** 2023 配对区间不跨 0，10/10 seeds 获胜；强 donor 错配会造成约 31–34 MW 退化，证明网络依赖正确的逐样本 aggregate state。
4. **该结果具有预测价值，但尚无确定物理机理。** aggregate P4 可能代理共同 NWP 偏差、区域天气相位、共同有效容量，也可能只是一个低噪声、容易学习的历史摘要。现有数据不能区分这些原因。
5. **pure local power state 没有获得增量支持。** 更长历史、乘法、zero-invariant MLP、concat、station ID 或 affine calibration 均未稳定救活 local state。hierarchical local deviation 会被读取，但没有超过 aggregate-only state。
6. **旧 4×4 模型不能证明 local weather-response state 已成立。** 旧模型允许 local/global history 和 additive bias 直接形成预测；旧 state ladder 中 `station recent` 也同时包含 aggregate history。旧结论主要证明历史与天气互补，不能全部解释为纯 local modulation。
7. **不能把 local state 失败直接归因于 bilinear。** same-support 实验没有证明 learned 4×4 稳定优于 fixed bilinear/fixed 4×4。
8. **不应根据 validation/test 挑选“支持故事”的场站。** 若研究切换到单风场，必须先在全部 56 场上严格复现 state effect，再根据预注册判据决定，而不能事后选择表现好的风场。

## 2. 任务与统一实验口径

主任务为：

\[
P_{1:56,t-3h:t},\quad W^P_{future}
\longrightarrow
P_{\Sigma,t+15min:t+6h}.
\]

- 输出 24 个 15 min 集群总功率点；
- 2022 用于训练；
- 2023 validation 用于结构选择；
- 2024 只在结构冻结后检查，但由于它在更早研究中已经被查看过，只称为跨年检查，不称为 pristine blind test；
- 正式结论以 10 个 seeds 42–51 的独立 checkpoint 均值和配对比较为主；
- 不使用 ensemble 或挑选最佳 seed 作为主表结果；
- 所有场站内部使用 capacity factor，最终按装机容量恢复 MW 并求和。

## 3. 模型演化与关键数值

### 3.1 直接 aggregate 强控制 A0-FT

A0-FT 的结构为：

\[
\hat P=B(H_{\Sigma,3h})+C(W^P_{future}).
\]

未来天气先做区域 CNN 编码和静态 capacity pooling，再在 latent space 从小时节点适配到 24 个 15 min 时刻。correction 不重复读取功率状态。A0-FT 是旧 R2-W 的同 seed、同拓扑低学习率继续训练控制。

| 模型 | 2023 MAE / MW | 2024 MAE / MW | 结论 |
|---|---:|---:|---|
| R2-W | 275.363 ± 2.599 | 272.717 ± 4.578 | 旧天气路线 |
| **A0-FT** | **273.195 ± 2.435** | **270.325 ± 3.949** | 公平训练预算下的 aggregate 强控制 |

A0-FT 相对旧 R2-W 改善 `2.169/2.392 MW`。因此任何从旧 R2-W 继续训练的新模块，都必须与 A0-FT 比较，不能把继续训练收益误记为结构收益。

### 3.2 Raw-bilinear local-to-global

该路线先把 14 个 Pangu 变量固定双线性插值到 56 个场站，所有场站共享 weather/history encoder 和 decoder，输出逐场 capacity factor，最后严格求和。

| 模型 | 状态 | 2023 MAE / MW | 2024 MAE / MW |
|---|---|---:|---:|
| A0-FT | aggregate architecture | 273.195 ± 2.435 | 270.325 ± 3.949 |
| L1 | local-to-global，无 correction state | 269.437 ± 2.790 | 266.837 ± 1.996 |
| L2 / G-RMul | aggregate P4 residual gain | 268.792 ± 2.791 | 265.672 ± 2.200 |
| L3 / L-RMul | pure local P4 residual gain | 269.781 ± 2.849 | 266.625 ± 1.396 |

L1 相对 A0-FT 改善 `3.758/3.488 MW`，说明 local-to-global 信息组织具有工程价值。该差值同时改变了局地天气、逐场历史、共享逐场预测和 station auxiliary loss，不能被严格归因于单一的“先转换还是先聚合”算子。

L2 相对 L1 进一步改善 `0.645/1.165 MW`；L3 相对 L1 为 `+0.344/-0.212 MW`，没有稳定增量。

### 3.3 Interaction operator 与严格状态可辨识性

更强 aggregate interaction 的结果为：

| 模型 | 约束 | 2023 MAE / MW | 2024 MAE / MW | 定位 |
|---|---|---:|---:|---|
| G-Int | zero-weather invariant | 267.818 ± 2.960 | 265.252 ± 2.448 | 上一轮严格候选 |
| G-Cat | 允许 state-only path | **267.655 ± 2.738** | 264.998 ± 2.436 | 数值诊断，不用于调制机理主张 |
| **G-Static** | strict interaction + static context + strong donor audit | 267.985 ± 3.067 | **264.817 ± 1.919** | 当前正式推荐 |
| P-Direct | local state direct path | 269.416 ± 3.175 | **264.525 ± 3.220** | 2024事后数值，不可反向选型 |

G-Cat 的 2023 点估计最低，但允许 state-only shortcut；P-Direct 的 2024 点估计最低，但2023不是选择结果且不满足 local correspondence gate。当前推荐 G-Static 不是因为它在每个表格单元都取得绝对最低点估计，而是因为它同时具备：

- 2023 预定选择口径；
- strict zero-weather invariance；
- matched no-state control；
- matched train-time shuffled-state control；
- 低相关 inference donor intervention；
- 10-seed 配对和冻结跨年检查。

## 4. 当前正式模型 G-Static

### 4.1 局地 baseline

每场最近 3 h、12 个 15 min 历史点经共享 history encoder：

\[
\Delta b_{i,1:24}=B(H_{i,t-3h:t}).
\]

当前场站 capacity factor 是 logit anchor：

\[
b_{i,q}=\operatorname{logit}(p_{i,t})+\Delta b_{i,q}.
\]

### 4.2 当前 aggregate state

定义集群 capacity factor：

\[
p_{\Sigma,k}
=\frac{P_{\Sigma,k}}{K_\Sigma}
=\frac{\sum_iC_ip_{i,k}}{\sum_iC_i}.
\]

状态只看最近四点：

\[
\mathbf p_t^g=
[p_{\Sigma,t-45m},p_{\Sigma,t-30m},p_{\Sigma,t-15m},p_{\Sigma,t}].
\]

三层 causal TCN 取最后 token 得到 64 维 \(s_g\)。标准化 latitude、longitude、log-capacity 经静态编码器得到 \(e_i\)。同一个 \(s_g\) 广播给所有场站，但可结合 \(e_i\) 形成不同局地作用。

### 4.3 状态如何作用

Pangu 在场站坐标双线性插值，经共享 weather encoder、difference path 和目标时刻查询得到 \(h^w_{i,q}\)。状态只通过：

\[
h'_{i,q}
=h^w_{i,q}
+0.25\left[M([h^w_{i,q},s_g+e_i])-M([0,s_g+e_i])\right]
\]

改变天气表示。因此 \(h^w=0\) 时状态相关 update 精确为零。最终：

\[
\hat p_{i,q}
=\sigma\left[b_{i,q}+\Delta c_{i,q}(h'_{i,q})\right],
\qquad
\hat P_q=\sum_iC_i\hat p_{i,q}.
\]

训练损失等价于：

\[
\mathcal L=\mathcal L_{cluster}+0.5\mathcal L_{station,CF}.
\]

## 5. Aggregate state 的严格证据

### 5.1 匹配控制

| Split | 比较 | ΔMAE / MW | 95% paired CI | wins |
|---|---|---:|---:|---:|
| 2023 | G-Static − B-Static | **−2.177** | **[−3.150,−1.203]** | **10/10** |
| 2024 | G-Static − B-Static | **−1.986** | **[−3.181,−0.791]** | 8/10 |
| 2023 | G-Static − A0-FT | −5.210 | [−7.844,−2.575] | 8/10 |
| 2024 | G-Static − A0-FT | −5.508 | [−8.896,−2.120] | 8/10 |

相对 A0-FT 的约 5 MW 改善包含 local-to-global 天气、逐场历史与监督结构变化，不能全部归因于 state。纯 state 增量只能使用 G-Static 对 B-Static，即 `2.177/1.986 MW`。

### 5.2 强状态错配

- matched train-time shuffled G-Static 相对正确状态模型在2023差 `2.228 MW`，10/10 seeds；
- inference donor 在 month、origin phase、NWP age 内无放回错排，与原状态相关性仅约 0.10；
- donor 错配使 2023/2024 MAE 恶化 `30.652/34.311 MW`。

这证明模型依赖正确的逐样本 aggregate state，而不仅是对一个参考向量或状态分布敏感。但巨大 intervention degradation 不等于正常预测有 30 MW state gain；正常净增量仍是约 2 MW。

### 5.3 时效和尺度

G-Static 相对 B-Static 的收益主要来自 3–6 h：2023/2024 改善 `4.112/3.903 MW`；0–1 h 反而恶化 `0.675/0.622 MW`。Ramp MAE 仅改善 `0.299/0.263 MW`。

state gain 随聚合尺度的结果为：

| 评价尺度 | 2023 gain / MW | 2024 gain / MW |
|---|---:|---:|
| 56 单场 macro | +0.135 | +0.106 |
| K=8 子集群 | +0.190 | +0.075 |
| K=4 子集群 | +0.381 | +0.091 |
| K=2 子集群 | +1.094 | +0.084 |
| 完整56场集群 | **+2.177** | **+1.986** |

最清楚的状态收益出现在完整集群尺度，而不是单场尺度。它支持“aggregate summary 对集群任务有效”，但不能单独证明某个具体物理 common-mode mechanism。

## 6. Local state 路线的结论

### 6.1 已否定的解释

- pure local P4 相对 no-state 恶化 `0.325 MW`（3-seed screening）；
- local P12 虽有 `−0.340 MW` 点估计，但 farm wins 为 0/3 且 shuffle 无信号；
- L-Int 和 L-Cat 相对 L1 分别恶化 `0.157/0.169 MW`，更强 operator 和 state-only path 都没有救活 local state；
- local + learned ID 相对 ID-only 恶化 `0.507 MW`；
- local + affine 相对 affine-only 仅 `−0.010 MW`。

因此，“local state 只是被乘法算子压死”已经被否定；也没有证据表明加 station embedding 就能使 local state 产生增量。

### 6.2 “被使用”与“有增量”不同

GL-Static 使用 aggregate state 加 local-deviation state。far-station permutation 使 2023/2024 恶化 `5.774/2.661 MW`，说明网络确实读取了正确局地对应关系。然而 G-Static 比 GL-Static：

- 2023 好 `0.563 MW`，95% CI `[0.259,0.868]`，10/10；
- 2024 好 `0.381 MW`，但区间跨 0。

因此准确结论是：

> local-deviation state 被模型使用，但没有在 aggregate/common-mode state 之上提高正常预测。

不能把 intervention 敏感性单独写成 local-state route 成功。

## 7. 旧 4×4 模型为何曾表现为“local state 有用”

旧 `state_conditioned` 使用：

\[
\operatorname{forced}
=(1+\tanh\gamma)\odot h_w+\beta,
\]

之后 decoder 仍直接读取 local history、global history、forced、history×forced 和 target token。状态能够通过 \(\beta\) 和直接 history path 绕开天气。它证明的是 history+weather joint forecasting 有效，不是纯粹的 local weather-response modulation。

旧融合结果中 state-conditioned 为 `265.738 MW`，但 prediction weighting 为 `265.367 MW`，bidirectional cross-attention 为 `265.669 MW`，因此旧 state-conditioned 并未明确领先非调制融合。

旧 frozen-B0 state ladder 的 `station recent 1h` 也同时向 correction 暴露 station recent 和 aggregate recent。最公平的 local 增量是：

\[
267.214-265.759=1.455\ \mathrm{MW},
\]

95% CI `[0.434,2.476]`，9/10 seeds；不是 weather-only 到 station-recent 的全部三十多 MW。

所以旧结果与当前结果不构成直接矛盾：旧实验估计的是历史直接预测、aggregate dynamics 和天气融合的总价值；当前实验估计的是强 local-history baseline 与 aggregate state 已存在后，pure local state 对严格 weather interaction 的边际价值。

## 8. 4×4 与 bilinear 的空间结论

same-support 3-seed结果：

| 空间表示 | 2023 MAE / MW |
|---|---:|
| raw 2×2 bilinear | 257.826 ± 1.034 |
| latent 2×2 bilinear | 257.716 ± 0.606 |
| learned 2×2，无 wind geometry | 257.922 ± 1.663 |
| fixed 4×4 Gaussian | 257.763 ± 0.660 |
| learned 4×4，无 wind geometry | 258.728 ± 1.550 |
| learned 4×4 + wind geometry | **256.589 ± 1.967** |

可以成立的结论：

- raw interpolation 与 latent interpolation 基本相同，差 `0.110 MW`；
- learned 2×2 没有优于 fixed bilinear；
- learned 4×4 without geometry 比 fixed 4×4 差 `0.965 MW`；
- fixed 4×4 与 latent bilinear 只差 `0.047 MW`；
- learned 4×4 + geometry 只有3-seed工程均值迹象，没有10-seed机制确认。

因此不能声称 learned 4×4 比 bilinear 更好，也不能把 local state 失败归因于 bilinear。若要验证“local state 需要周边天气上下文”，必须完成：

\[
\{\text{bilinear},\text{fixed 4×4},\text{learned 4×4+geometry}\}
\times
\{\text{aggregate-only},\text{aggregate+local}\}
\]

的同主干交叉矩阵，并做 far-station permutation。

## 9. 已经停止的结构路线

### 9.1 Aggregate 强物理结构

相对 A0-FT，3-seed筛选中：

- baseline headroom：`+1.720 MW`；
- anomaly headroom：`+49.593 MW`；
- recursive headroom：`+27.448 MW`；
- fixed/adaptive equilibrium：`+30.751/+25.328 MW`；
- envelope × realization：`+55.591 MW`；
- fixed/adaptive error decay：`+45.328/+45.081 MW`。

唯一晋级的 M1 在10-seed validation/2024仍比 A0-FT 差 `1.906/2.547 MW`。当前6 h集群任务不支持这些低维动力学约束，停止继续搜索 headroom、equilibrium、envelope 和 decay 变体。

### 9.2 Aggregate 历史天气—功率 response state

10-seed结果：

| response-state source | 2023 MAE / MW | 2024 MAE / MW |
|---|---:|---:|
| A0-FT，无response state | 273.195 | 270.325 |
| power-history latent | 274.693 | 272.176 |
| weather-power transition pair | 273.173 | 269.714 |
| historical weather only | **273.168** | **269.706** |

transition pair 与 historical-weather-only 只差 `0.005/0.008 MW`；power shuffle 和 pair shift 的干预影响精确为零，weather shuffle 则恶化 `4.321/5.540 MW`。因此模型实际主要利用历史天气，不依赖历史功率响应配对。当前实现没有证明 input-output system identification 成立。

### 9.3 复杂 aggregate 天气编码与调制器搜索

- 动态空间 pooling、区域语义 token、空间 patch Transformer 和联合时空 patch query 均未超过简单 R1/R2 路线；
- 严格专家/低秩/channel modulation 会强烈使用状态，但没有超过同预算 A0-FT；
- 最佳严格 E1 相对 A0-FT 在2023/2024分别差 `1.439/2.140 MW`，尽管 state shuffle 会使其恶化 `9.005/10.468 MW`。

这证明“模型使用了状态”不能替代 matched no-state 性能比较。继续增加专家、gate或低秩调制没有依据。

## 10. 当前物理解释边界

当前最严谨的事实是：最近 aggregate power trajectory 是一个对集群预测有用的低噪声共同信号。它可能代理：

1. Pangu 未解析或有偏的区域共同边界层天气；
2. 区域天气系统的相位/传播误差；
3. 区域共同限电、覆冰、故障或有效可用容量；
4. 由聚合降噪和显式摘要带来的 representation/optimization benefit。

现有数据没有场站实测风、限电、开机容量、故障或覆冰标签，也没有在正式 G-Static 中显式构造历史 Pangu–realization residual。因此不能声称已经辨识了真实运行状态、设备可用率或因果 weather-response law。

zero-weather invariance 只保证状态不能在零天气 latent 下独立产生 correction；它不保证网络学习到的状态依赖变化就是物理功率曲线变化。强 donor 只证明网络依赖正确状态，不辨认其物理来源。

## 11. 当前科学问题与路线决策

### 11.1 能严谨提出的物理科学问题

> 在未来 0–6 h 内，区域内多个风场的天气—功率关系是否受到一个跨场站共享、短时持续但未被 Pangu 与静态场站参数完整观测的区域条件影响；最近区域总功率能否作为该共同条件的可靠代理？

当前结果只回答了后半句的“预测上有用”，没有辨识这个共同条件的物理身份。

### 11.2 是否继续集群预测

**建议继续集群预测任务。** 理由是：

- local-to-global 结构跨年份优于强 aggregate A0-FT；
- aggregate state 的最稳定收益出现在完整56场尺度；
- 当前模型已经内部逐场预测后求和，不存在必须“退回逐场输出”的结构问题；
- 56/56场均通过train-only数据质量门槛，没有预先定义的剔除理由。

但应区分两个目标：

- **工程预测目标：**可以保留 G-Static；
- **物理机理论文目标：**不能只凭当前 aggregate P4 继续包装状态机理，也不应继续搜索调制 operator。

### 11.3 是否切换单风场研究

当前不应直接切换，也不应选择少数表现好的场站。旧单风场结论存在 direct-history 和 aggregate-history 混杂；当前 farm macro 的 state gain 又很小。

若要决定是否切换，必须先在全部56场上严格复现：

\[
\{\text{bilinear},\text{4×4 weather}\}
\times
\{\text{no state},\text{strict local state}\}.
\]

要求相同 local baseline、禁止 state-only path、强 sample/station mismatch、2023选择后冻结到2024，并报告所有场站或预先定义的train-only分层。只有整体 station-CF MAE 的配对区间稳定低于0、跨年同方向且多数场站受益，才有依据把论文主任务切换为单风场 state-dependent response。

### 11.4 状态路线最后值得做的判别实验

如果继续状态研究，只建议最后完成两个可证伪对照：

1. **真实 response-state 对照：**历史同源 Pangu–实际功率 residual/transition pair，比较 baseline-only 使用与 strict weather-interaction 使用；
2. **空间支持对照：**bilinear/fixed-4×4/learned-4×4 与 aggregate-only/aggregate+local 的交叉矩阵。

若 IO/residual state 不能超过 historical-weather-only，或者 baseline-only 与 strict interaction 获得相同收益，应停止 weather-response-state 叙事。此后把 G-Static 仅作为工程预测模型，论文重点转向 local-to-global 信息组织、简单可靠的空间局地化与跨时间分辨率天气表示。

## 12. 当前可写与不可写的结论

### 可以写

- 固定双线性局地化、跨场共享响应和物理求和构成有效的集群预测结构；
- 显式 recent aggregate summary 在 matched local-to-global control 上提供约2 MW跨年增量；
- 状态增益主要出现在中长时效和完整集群尺度；
- pure local power state 没有在 aggregate state 上提供增量；
- 更复杂的空间采样、状态operator和强动力学约束没有稳定优势。

### 不可以写

- 已经发现或辨识了真实区域运行状态；
- aggregate power 导致局地功率曲线发生变化；
- learned 4×4 interpolation 优于 bilinear；
- local state 在单风场有效、只是在集群求和时被抵消；
- station embedding 已经改善了 state interpretation；
- 根据表现挑出的少数风场能够代表整体物理规律。

## 13. 关键证据索引

- aggregate A0-FT：`exp_docs/aggregate_cluster_interim_key_results.md`
- strict aggregate modulation：`exp_docs/aggregate_strict_state_weather_modulation_results.md`
- structured dynamics：`exp_docs/aggregate_weather_driven_dynamics_experiment_results.md`
- historical weather–power response state：`exp_docs/weather_power_response_state_experiment_results.md`
- raw-bilinear local-to-global：`exp_docs/local_to_global_raw_bilinear_experiment_results.md`
- local state operator：`exp_docs/local_state_operator_experiment_results.md`
- strong state identifiability：`exp_docs/local_state_identifiability_v2_results.md`
- spatial same-support：`exp_docs/cross_resolution_next_experiment_results.md`
- old direct fusion：`EXPERIMENT_SUMMARY_AND_NEXT_STEPS.md`
- old frozen-B0 state ladder：`docs/state_forcing_ladder_v1_key_results.md`

当前主机器汇总：`artifacts/local_state_identifiability_v2/analysis/summary.json`。

当前冻结记录：`artifacts/local_state_identifiability_v2/selection/validation_freeze.json`。
