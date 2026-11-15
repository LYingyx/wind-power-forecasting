# 时间跨分辨率 T0--T4 筛选实验结果

日期：2026-09-02  
状态：2023 validation 三随机种子筛选已完成；尚未进入 10-seed 正式验证，也未查看 2024 test

## 1. 本轮只回答什么

在其余结构全部固定时，只改变“小时级 NWP 如何变成目标时刻的天气表示”：

- T0 Raw linear：14 个原始天气变量先线性插值，再编码。
- T1 Latent linear：小时节点先编码，再对编码表示线性插值。
- T2 Bracket nonlinear：保留目标两侧的小时节点及相位，用共享 MLP 非线性映射。
- T3 Nearest：只使用最近的一个小时节点。
- T4 Full native path：对完整小时天气路径做相对时间 attention。

三次训练使用 seeds 42、46、50。每个数都是一个独立 checkpoint 的结果；没有对三个模型的预测取平均。所有模型参数量均为 1,287,343，2023 validation 预测均为 32,550 个 origin × 24 个 lead。

## 2. 主结果

主指标为 cluster MAE，单位 MW，越低越好。均值和标准差只描述单次训练的稳定性，不是 ensemble 成绩。

| 方法 | seed42 | seed46 | seed50 | 单 checkpoint 均值 ± 标准差 |
|---|---:|---:|---:|---:|
| T0 Raw linear | 260.54 | 263.85 | 258.00 | 260.80 ± 2.93 |
| T1 Latent linear | 254.76 | 258.67 | 256.34 | 256.59 ± 1.97 |
| **T2 Bracket nonlinear** | **253.17** | **256.82** | **255.26** | **255.09 ± 1.83** |
| T3 Nearest | 258.04 | 258.10 | 257.73 | 257.95 ± 0.20 |
| T4 Full native path | 255.52 | 257.75 | 256.86 | 256.71 ± 1.12 |

关键配对差异约定为“候选减参考”，负值表示候选更好。

| 配对比较 | 平均差异 | 95% CI | 胜场 |
|---|---:|---:|---:|
| T1 − T0 | -4.21 | [-9.73, 1.31] | 3/3 |
| T2 − T0 | -5.71 | [-12.12, 0.70] | 3/3 |
| **T2 − T1** | **-1.50** | **[-2.48, -0.53]** | **3/3** |
| T2 − T3 | -2.87 | [-7.41, 1.68] | 3/3 |
| T2 − T4 | -1.62 | [-3.40, 0.16] | 3/3 |
| **T4 − T1** | **+0.12** | **[-2.15, 2.39]** | **1/3** |

三 seed 下最可靠的新结果是 T2 稳定优于 T1。T2 相对 T4 也是 3/3 获胜，但 CI 上界略高于 0，现阶段只能称为边界证据。

汇总图：`artifacts/cross_resolution_temporal_v1/figures/cross_resolution_screening.png`

## 3. 机制结果

### 3.1 原始值空间插值较弱，但原因不能简单归为“非整点天气缺失”

T1、T2、T3、T4 相对 T0 均为 3/3 seed 获胜，说明“先在原始 14 变量上构造 15 分钟天气，再统一编码”不是当前任务中最有效的计算顺序。

但收益没有只出现在小时节点之间：

- T4 相对 T0：anchor 改善 4.64 MW，三个 non-anchor phase 平均改善 3.91 MW。
- T2 相对 T0：anchor 改善 6.66 MW，三个 non-anchor phase 平均改善约 5.40 MW。

因此，当前实验支持的是“先获得任务相关的小时表示，再做目标时刻响应映射”，而不是“模型主要补出了不存在的 15 分钟天气”。

### 3.2 局部非线性映射优于 latent 线性插值

T2 相对 T1 的逐 phase 差异为：

| Phase | T2 − T1 | 95% CI | 胜场 |
|---|---:|---:|---:|
| anchor | -2.59 | [-4.25, -0.93] | 3/3 |
| +15 min | -1.40 | [-1.74, -1.06] | 3/3 |
| +30 min | -0.94 | [-1.98, 0.09] | 3/3 |
| +45 min | -1.08 | [-2.26, 0.10] | 3/3 |

这说明收益不是只靠“有两个节点”，而更可能来自同时保留两侧节点、相对时间和非线性响应映射。anchor 和 +15 min 的证据最稳定。

按预测时距看，T2 相对 T1 的差异从短期 -0.04 MW、中期 -0.87 MW 增长到长期 -2.41 MW，但各时段只有 3 seeds，分段 CI 仍跨 0，暂不能声称长期优势已经成立。

### 3.3 完整路径 attention 没有表现出必要性

T4 与 T1 几乎相同：总体差 0.12 MW，只有 1/3 seed 获胜；三个 non-anchor phase 上 T4 平均还差 0.35 MW。

在低 forcing-variation 样本中，T4 相对 T1 变差 5.27 MW，95% CI 为 [0.57, 9.98] MW。中、高变化条件也没有稳定优势。因此不能声称“完整未来路径在天气变化快时更有效”。

T2 相对 T4 在短期差 1.26 MW，但在中期和长期分别改善 2.22、2.19 MW。这个现象说明完整路径可能在很短时距提供一些优化便利，但没有形成总体优势；仍需 10 seeds 才能判断是否是真实的 horizon trade-off。

## 4. 对原假设的判定

| 原假设 | 当前判定 | 原因 |
|---|---|---|
| Raw weather interpolation 足够 | 倾向否定 | 所有 encoded 方法均 3/3 优于 T0，T2 平均改善 5.71 MW |
| Full native-path query 优于合理插值 | 否定 | T4 与 T1 等价，且落后 T2 |
| 收益主要来自 non-anchor phase | 否定 | anchor 收益不小于 non-anchor |
| 收益在 high forcing variation 最大 | 否定 | 没有该排序，T4 在 low variation 反而明显变差 |
| 局部 nonlinear bracket 优于 latent linear | 初步支持 | T2 − T1 = -1.50 MW，3/3，CI 不跨 0 |
| 现在应继续 continuous-time 新架构 | 不支持 | 完整路径 query 尚无必要性证据，先做更小的机制识别 |

预注册的“T4 相对 T0 改善至少 1 MW、2/3 seed 获胜、non-anchor 改善”规则形式上通过。但该规则只能说明 raw interpolation 较弱，不能证明 full-path attention 必需，也不能单独证明收益源于跨分辨率的非整点位置。论文表述必须采用后一个、更严格的解释。

## 5. 建议修改后的科学问题

不建议继续写成：

> 如何用完整原生天气路径连续查询缺失的 15 分钟天气响应？

当前证据更适合写成：

> **在没有 15 分钟未来气象真值时，应当在哪个表示空间、利用多大的时间邻域和何种目标时刻条件，将原生小时级未来强迫转化为 15 分钟功率响应？**

当前候选机制是：

> **先在原生小时节点学习任务相关的天气表示，再保留目标两侧节点及真实相位，直接学习状态条件化的局部非线性功率响应；不重建细粒度天气，也不默认完整未来路径 attention 必要。**

本轮没有重新识别 state-conditioning；recent-state 和 black-box 状态—天气融合在五组中完全固定。状态条件化的论述应引用此前专门消融，本轮只负责识别时间适配模块。

## 6. 下一步最小实验，而不是立即扩展 continuous-time

1. 先对 T2 做正式 10-seed 验证，并将 T1 和 T4 补到同一 10-seed 口径；当前三 seed 只用于结构筛选。
2. 增加两个最小消融：`two brackets without phase` 和 `two brackets + phase but no nonlinear MLP`，分离“两节点信息、相位、非线性”三个因素。
3. 只有 T2 的优势在 10 seeds 中成立后，再做 missing-node、2 h forcing 和 held-out target phase 的 resolution stress test。
4. 暂停 continuous relative-time bias、连续 kernel 和任意 target-grid decoder；当前结果尚未证明这些复杂度必要。
5. 继续只用 2023 validation 冻结选择，不查看 2024 test 反向修改科学问题。

## 7. 产物位置

- 完整模型与 validation 预测：`artifacts/cross_resolution_temporal_v1/groups/`
- 主表：`artifacts/cross_resolution_temporal_v1/analysis_val/summaries.csv`
- 相对 T0 的配对结果：`artifacts/cross_resolution_temporal_v1/analysis_val/comparisons_vs_t0.csv`
- 核心方法直接配对：`artifacts/cross_resolution_temporal_v1/analysis_val/focused_pairwise.csv`
- phase/horizon/variation/NWP-age 分层：`artifacts/cross_resolution_temporal_v1/analysis_val/focused_pairwise_stratified_summary.csv`
- 决策记录：`artifacts/cross_resolution_temporal_v1/analysis_val/decision.json`
- 汇总图：`artifacts/cross_resolution_temporal_v1/figures/cross_resolution_screening.png`

统计说明：CI 为三个配对 seed 差异的 t 区间，仅用于筛选；没有把多个 checkpoint 的预测平均，也没有做 2024 test 或多重比较后的论文级确认。
