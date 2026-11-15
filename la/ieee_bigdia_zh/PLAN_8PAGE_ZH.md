# BigDIA 2026 IEEE 8页中文底稿规划

## 目标约束

- 投稿格式：IEEE conference template。
- 页数上限：8 pages including references。
- 当前中文迁移稿：12 页。
- 新稿目标：正文、图表、参考文献全部控制在 8 页内。
- 写作语言阶段：先写中文底稿，后续再翻译英文。

## 核心主线

本文不写成“一个复杂网络结构介绍”，而写成一篇围绕气象-功率融合预测的短会论文。

核心问题：

> 在业务化区域风电集群总功率 24 h 预测中，未来气象是主驱动，但单靠未来气象会忽略预测起点前的真实运行状态。如何在严格时间可用性约束下，将目标窗口前可用的场站级功率状态与未来网格化气象预报进行有效融合，使气象驱动预测受到当前运行状态条件化约束，从而修正近端预测误差？

方法落脚：

> 功率状态条件化的气象-功率融合预测网络。

一句话方法：

> 将未来气象编码为可查询的气象记忆，将近期场站级功率编码为动态功率上下文 token，并在 memory 层面进行非对称融合，使 future queries 在状态条件化后的融合记忆中读取气象主驱动和功率状态信息。

## 建议标题

中文：

> 功率状态条件化的气象-功率融合网络用于区域风电集群总功率预测

英文后续可译为：

> Power-State-Conditioned Meteorology-Power Fusion Network for Regional Aggregate Wind Power Forecasting

## 8页页数预算

| 部分 | 目标页数 | 主要内容 |
|---|---:|---|
| Title/Abstract/Keywords | 0.35 | IEEE 标题区、摘要、关键词 |
| I. Introduction | 0.90 | 问题、痛点、方法概述、贡献 |
| II. Related Work and Gap | 0.55 | 三段式相关工作，压缩到半页 |
| III. Problem and Operational Time Alignment | 0.70 | 业务时间可用性、输入输出定义 |
| IV. Method | 2.10 | 气象记忆、功率状态、memory-level fusion、query decoding；保留4张紧凑方法图 |
| V. Experiments | 1.95 | 数据文字说明、主结果、逐 horizon、消融 |
| VI. Discussion and Conclusion | 0.55 | 作用边界与结论 |
| References | 0.90 | 约 20--24 条 |
| 合计 | 8.00 | 含图表和参考文献 |

## 章节结构

### I. Introduction

目标：1 页内把问题立住。

必须包含：

1. 区域风电集群总功率预测面向调度，是区域总出力预测，不是单场站预测。
2. 业务化 24 h 预测中，未来网格化气象预报是预测窗口内主驱动。
3. Weather-only 的不足：忽略预测起点前真实功率水平、短期惯性和多场站同步状态。
4. 现有融合方法多为拼接、并行分支或对称注意力，没有充分表达“气象主驱动 + 功率状态条件”的非对称角色。
5. 本文提出功率状态条件化的气象-功率融合预测网络。
6. 实验发现：整体优于 Weather-only，H1/H6 改善明显，远端收益减弱，符合功率状态作用边界。

贡献点压成三条：

- 提出业务时间可用性约束下的非对称气象-功率融合预测问题表述。
- 提出 memory-level fusion 网络：气象构成主记忆，近期场站功率生成动态上下文 token 条件化 memory 读取。
- 在吉林 58 场站数据上验证融合有效性，并通过逐 horizon 与消融分析说明收益来自真实功率状态，主要改善近端预测。

删除/避免：

- 不要把 Pangu-Weather 写成主要贡献。
- 不要在引言列举所有模型细节。
- 不要展开太多文献列表。

### II. Related Work and Gap

目标：半页左右。

三段即可：

1. 区域/场群风电预测与图建模：说明 GNN 能建模多场站关系。
2. 气象预报驱动的风电预测：说明 NWP/Pangu 类气象是多步预测主信号。
3. 气象-功率融合预测：指出已有方法多为 feature-level 或对称融合；本文是 memory-level、非对称、状态条件化融合。

删除/避免：

- 不单列 Transformer/Perceiver/Memory Networks 大段综述。
- query-based memory decoding 只用一句话作为技术支撑。

### III. Problem and Operational Time Alignment

目标：约 0.7 页。

保留内容：

- 预测日 \(D\)，目标窗口 \(D\) 00:00--23:00。
- 最新功率上下文为 \(D-1\) 08:00--23:00，长度 16。
- Pangu 起报为 \(D-1\) 08:00；lead 0--15 为 hindcast/context，lead 16--39 为目标窗口 future forecast。
- 输入：
  - \(\mathbf{X}^{h}\in\mathbb{R}^{16\times C\times H\times W}\)
  - \(\mathbf{X}^{f}\in\mathbb{R}^{24\times C\times H\times W}\)
  - \(\mathbf{P}^{c}\in\mathbb{R}^{N\times16}\)
  - \(\mathbf{A}\in\mathbb{R}^{N\times N}\)
- 输出：
  - \(\hat{\mathbf{y}}\in\mathbb{R}^{24}\)，区域集群总功率。

图表：

- 保留 Fig. 1：业务时间协议图。
- Fig. 1 做紧凑双栏，caption 聚焦“operational time availability”。图内文字需要缩短，避免占据半页以上。

删除/压缩：

- 不展开所有符号说明。
- 邻接矩阵构造只写一句，不在这里展开。

### IV. Method

目标：约 2.1 页。

小节建议：

#### A. Overview of Asymmetric Meteorology-Power Fusion

核心句：

> 未来气象是主驱动，近期功率不是另一个对称输入分支，而是预测起点状态条件。

保留 Fig. 2 总框架图，使用双栏 `figure*`。  
Fig. 3 和 Fig. 4 不删除，但必须改成紧凑双栏图：减少图内说明文字，caption 控制在 1--2 句，正文只解释与融合主线直接相关的部分。

#### B. Weather Memory Construction

压缩写法：

- 将 \(\mathbf{X}=[\mathbf{X}^{h};\mathbf{X}^{f}]\) 输入 Weather Encoder。
- 气象通道分为风场、背景场和一阶差分三类。
- 经 frame-wise CNN、时空卷积和自适应池化得到：

\[
\mathbf{M}^{w}\in\mathbb{R}^{L_w\times d}.
\]

不要展开每个卷积 block 细节。Fig. 3 只作为视觉辅助，正文不重复图中每个模块名称。

#### C. Power-State Encoder

压缩写法：

- 最近 16 步场站功率和邻接矩阵输入多尺度 GCN。
- point/patch/global 三种尺度分别提取瞬时状态、局部变化和整体趋势。
- 跨尺度注意力融合后得到功率状态向量：

\[
\mathbf{z}^{p}\in\mathbb{R}^{d}.
\]

只保留一个 GCN 归一化公式即可。Fig. 4 保留 point/patch/global 和跨尺度交互的视觉结构，但正文删除 point/patch/global 的逐项公式。

#### D. Memory-Level Fusion and Query Decoding

这是方法核心，必须写重。

\[
\mathbf{C}^{p}=\operatorname{reshape}(a_{\theta}(\mathbf{z}^{p}))\in\mathbb{R}^{K\times d},
\]

\[
\mathbf{M}^{pc}=[\mathbf{C}^{p};\mathbf{M}^{w}].
\]

强调：

- 不是输出层拼接。
- 功率状态作为 memory token 参与 cross-attention。
- 每个 future query 可同时读取气象主驱动和功率状态条件。

Query decoder 简写：

\[
\mathbf{Q}^{(\ell)}=\operatorname{DecoderBlock}(\mathbf{Q}^{(\ell-1)},\mathbf{M}^{pc}),
\quad
\hat{\mathbf{y}}=g_{\phi}(\mathbf{Q}^{(L_d)}).
\]

训练目标保留一行：

\[
\mathcal{L}=0.6\operatorname{MAE}+0.4\operatorname{RMSE}.
\]

### V. Experiments

目标：约 2.0 页。

#### A. Dataset and Setup

压缩到 2 段文字，不再设置实验设置表。

保留：

- 吉林 58 风电场站。
- 2022 train sliding，2023 validation/test daily。
- 17 通道，19x33 网格。
- 12 次重复。
- 指标 MAE、RMSE、MAE(MW)、RMSE(MW)、\(R^2\)。

删除原 Table I。训练/验证/测试划分、场站数、网格大小、通道数、重复次数和指标全部写进正文段落。

#### B. Overall Forecasting Performance

保留主结果表，后续短版中编号为 Table I：

- Ours
- Weather-only
- CNN
- 3D CNN
- ViT
- Swin
- Video Transformer

重点解释：

- Weather-only 是最强基线，说明气象是主驱动。
- Ours 比 Weather-only 更好，说明功率状态条件化融合有效。

删除：

- Fig. 6 benchmark bar plot，因为 Table II 已经足够。

#### C. Horizon-Wise Effect of Fusion

保留 Table IV 或 Fig. 9 二选一。

建议保留 Fig. 9，删除 Table IV，或者把 Table IV 压缩成只含 H1/H6/H12/H24 的小表。

核心解释：

- H1 改善最大：MAE 0.0747 -> 0.0430。
- H6/H12 仍有小幅优势。
- H24 Weather-only 略优，说明功率状态条件的影响随 horizon 衰减。

这一节是文章最关键证据，应该放在消融之前。

#### D. Ablation Study

保留消融表，后续短版中编号为 Table II，并精简行：

- Ours
- Weather-only
- Random power
- Zero power
- Basic GCN
- No memory
- No wind

删除或移到补充：

- No delta
- Single branch
- No synoptic
- Fig. 7
- Fig. 8

消融解释分两段：

1. Power-side ablation：随机/零功率接近 Weather-only，说明收益来自真实功率状态。
2. Weather-side ablation：No memory 和 No wind 退化，说明气象记忆和风场主驱动不可替代。

## 图表取舍

当前：9 张图/表 + 43 条参考文献，12 页。  
目标：4 张紧凑方法图 + 2 张实验表 + 1 个逐 horizon 小图或小表 + 约 20--22 条参考文献，8 页。

保留：

- Fig. 1：业务时间协议。
- Fig. 2：整体融合框架。
- Fig. 3：Weather Encoder，紧凑双栏。
- Fig. 4：Graph Encoder，紧凑双栏。
- Table I：主结果，单栏，原 Table II 精简而来。
- Table II：精简消融，单栏，原 Table III 精简而来。
- Fig. 5 或 Table III：逐 horizon 结果，二选一，单栏。

删除：

- Benchmark bar plot。
- Ablation bar plot。
- Error distribution plot。
- 过长的完整 horizon 表和冗余讨论。
- 原实验设置表。

如果页数仍超：

1. 将逐 horizon 结果用 3 行文字 + 单栏简图替代。
2. 进一步减少参考文献到 18--20 条。
3. 压缩 Fig. 3/Fig. 4 的显示高度，而不是删除 Fig. 1/Fig. 2。

## 参考文献压缩策略

目标：20--24 条。

保留类别：

- 区域/场群预测与 GNN：约 5 条。
- NWP/气象驱动风电预测：约 5 条。
- 网格气象/CNN/区域预测：约 3 条。
- Pangu-Weather：1 条。
- Transformer/query/memory 基础：约 4 条。
- 气象-功率融合或注意力融合：约 4 条。

优先保留：

- ref2, ref5, ref6, ref7, ref11
- ref13, ref14, ref15, ref18
- ref19, ref20, ref21
- ref22
- ref24, ref27, ref35, ref36 or ref65
- ref41, ref72, ref73

可删除：

- 过多 2024--2026 同类 GNN 论文。
- Flamingo/FiLM 等多模态泛化文献，如果篇幅紧张。
- Memory Networks 早期两篇可二选一或删除。

## 需要重写的文件

建议不要直接在当前长章节上小修，而是新建短版章节，避免反复删改变乱。

建议文件：

- `manuscript_zh_short.tex`
- `sections_short/introduction_zh.tex`
- `sections_short/related_work_zh.tex`
- `sections_short/problem_zh.tex`
- `sections_short/method_zh.tex`
- `sections_short/experiments_zh.tex`
- `sections_short/conclusion_zh.tex`
- `sections_short/references_zh.tex`

当前长版 `manuscript_zh.tex` 保留作为材料来源。

## 改写顺序

1. 新建短版主文件和章节目录。
2. 先写 Introduction、Problem、Method 三节，确保主线集中。
3. 删除实验设置表，精简实验表格，生成 short 版主结果表和消融表。
4. 写 Experiments，先放逐 horizon，再放消融。
5. 压缩 references。
6. 编译并检查页数。
7. 若超过 8 页，优先删图和参考文献，而不是削弱核心问题叙述。
