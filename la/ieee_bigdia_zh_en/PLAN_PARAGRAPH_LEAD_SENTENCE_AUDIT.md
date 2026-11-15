# 段首短句问题扫描与改写建议

## 处理范围

本报告只扫描当前 short 主稿，不修改 `.tex`，不编译。

重点扫描中文正文：

- `manuscript_zh_short.tex`
- `sections_short/introduction_zh.tex`
- `sections_short/related_work_zh.tex`
- `sections_short/problem_zh.tex`
- `sections_short/method_zh.tex`
- `sections_short/experiments_zh.tex`
- `sections_short/conclusion_zh.tex`

同步检查英文对应文件，用于后续同步修改：

- `manuscript_en_short.tex`
- `sections_short/*_en.tex`

参考文献列表不纳入段首短句判断，因为参考文献条目不是论文叙述段落。

## 判断标准

本次问题不是单纯“第一句短”，而是以下组合会显得像 AIGC 模板：

1. 段首第一句是泛化主题句，例如“本文关注……”“未来气象预报是……”“Transformer 及其变体为……”“上述两类研究分别……”

2. 第一句很短，第二句才展开实质信息，形成“先喊口号、再解释”的节奏。

3. 多个相邻段落都采用同一种开头模式，造成机械化结构。

4. 第一句没有提供新的具体信息，只起到“分类标签”作用。

不一定需要改的情况：

1. 段首是定义、公式、图表引导或实验结果引导，例如“设预测日为 \(D\)……”“表~\ref{...} 给出了……”。这类句子承担结构功能，不属于模板化空句。

2. 段首虽然短，但后面紧接公式或符号定义，短句用于降低技术段落阅读负担。

3. 贡献列表、图题、表题、参考文献不按普通段落处理。

## 总体结论

需要优先改的段落集中在：

- 摘要：1 处
- 引言：2 处
- 相关工作：5 处
- 问题定义：1 处
- 方法：2 处
- 实验：4 处
- 结论：1 处

共建议修改 16 处。

不建议修改的段落主要包括：

- 问题定义第一段
- 方法中公式前的技术说明段
- 图表结果段
- 消融变体说明段
- 贡献列表

这些段落的段首句承担定义、图表引导、公式引导或实验说明功能，贸然合并会降低可读性。

## 逐段扫描与建议

### 0. 摘要

文件：`manuscript_zh_short.tex`

现段首：

在区域风电集群 24 h 预测中，未来气象预报是主要驱动信息，但预测起点前的真实功率状态会影响气象到功率映射的初始条件。

判定：需要小改。

理由：摘要第一句不是很短，但后续“仅依赖未来气象容易……”构成典型的“背景-问题”模板。摘要可以保留这个逻辑，但应把问题对象和方法定位压得更具体，减少泛化开头。

建议改法：

区域风电集群 24 h 预测依赖未来气象预报刻画目标窗口内的风场演化，但气象到功率的映射还受到预测起点前真实出力水平、短期惯性和多场站同步变化的影响。

英文同步：

In 24 h regional wind farm cluster forecasting, future weather describes wind-field evolution over the target window, but the weather-to-power mapping is also affected by the observed generation level, short-term inertia, and synchronized multi-site variations before the forecast origin.

### 1. 引言第 1 段

文件：`sections_short/introduction_zh.tex`

现段首：

区域风电集群预测直接服务于调度、备用容量配置和新能源消纳。

判定：建议修改。

理由：这是典型的短泛化开头，句子本身正确，但信息量偏空，后面才进入“单场站 vs 区域集群”的具体问题。作为全文第一段，最好直接进入区域集群预测的对象和困难。

建议改法：

区域风电集群预测面向调度、备用容量配置和新能源消纳等业务场景，其预测对象不是单个场站的独立出力，而是区域内多场站聚合后的功率变化过程。

后文可接：

与单场站预测相比，区域集群预测既受区域天气过程驱动，也受预测起点前场群真实运行状态、场站间同步变化和短期功率惯性影响。

英文同步：

Regional wind farm cluster forecasting supports dispatch, reserve allocation, and renewable-energy accommodation by predicting the aggregated power evolution of multiple wind farms rather than the output of a single site.

### 2. 引言第 2 段

文件：`sections_short/introduction_zh.tex`

现段首：

本文关注业务化区域风电集群 24 h 预测。

判定：需要修改。

理由：这是用户已明确指出的典型问题。第一句只是声明研究对象，第二句才进入“未来气象”和“场站功率”的核心矛盾。应合并为一个带业务时间约束的完整论断。

建议改法：

在业务化区域风电集群 24 h 预测中，未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。

后文可接：

本文将预测起点前的真实功率状态定位为未来气象读取的运行状态约束……

英文同步：

In operational 24 h regional wind farm cluster forecasting, future gridded weather forecasts cover the full target window and provide the main driver of future power variation, whereas station-level measured power is available only before the target window and characterizes the current generation level, short-term inertia, and multi-site coupling state.

### 3. 引言第 3 段

文件：`sections_short/introduction_zh.tex`

现段首：

已有风电预测研究从风速时序深度模型……等角度提升预测性能。

判定：不需要改。

理由：虽然句子较长，但它不是短泛化句，而是带有具体研究脉络和引用分组的综述句。该句承担“文献空间定位”功能，信息密度高，不属于 AIGC 式短开头。后面“然而”自然引出不足，结构合理。

可选轻微优化：

如果后续压缩，可以把“已有风电预测研究从……”改为“现有风电预测研究主要从……展开”，但不属于必须修改项。

### 4. 引言第 4 段

文件：`sections_short/introduction_zh.tex`

现段首：

基于这一思想，本文提出运行状态约束的气象-功率融合预测网络。

判定：需要修改。

理由：该句属于典型“基于这一思想，本文提出……”模板。后面两句才说明模型实际怎么做。建议直接以模型机制开头。

建议改法：

为使预测起点前的功率状态参与未来气象读取，本文将后报气象和未来预报气象编码为可查询气象记忆，并从近期场站级功率中生成动态功率上下文 token。

后文可接：

这些 token 被写入气象记忆，使 future queries 在读取未来气象时同时感知当前出力水平……

英文同步：

To make the pre-window power state participate in future-weather reading, the proposed network encodes hindcast and future forecast meteorology into queryable weather memory and generates dynamic power context tokens from recent station-level power.

### 5. 引言贡献列表

文件：`sections_short/introduction_zh.tex`

现段首：

本文主要贡献如下：

判定：不需要改。

理由：这是贡献列表的标准引导句，不是普通正文段落。IEEE 会议论文中这种写法可接受，且它不会被看作段落式 AIGC 短句。

### 6. 相关工作第 1 段

文件：`sections_short/related_work_zh.tex`

现段首：

区域或场群风电预测需要同时刻画时间变化和场站间空间相关性。

判定：需要修改。

理由：这是用户明确指出的短泛化句。后面才给出 CNN-LSTM、规则网格、图建模等具体研究。应让段首直接进入“现有研究如何处理时空相关性”。

建议改法：

围绕区域或场群风电预测中的时间变化和场站间空间相关性，已有研究从风速序列建模、规则网格嵌入和图结构建模等方向展开。

后文可接：

早期深度学习方法利用 CNN-LSTM 建模风速时序特征……

英文同步：

Existing studies on regional or cluster-level wind power forecasting model temporal evolution and inter-site spatial correlations through wind-speed sequence models, grid-based spatial representations, and graph-structured dependencies.

### 7. 相关工作第 2 段

文件：`sections_short/related_work_zh.tex`

现段首：

未来气象预报是多步风电预测中的主要外生信息。

判定：需要修改。

理由：这是用户明确指出的短泛化句。它像教材式判断，后面才有 NWP 融合、误差、滞后和网格气象编码。应把“未来气象主驱动”和“已有工作怎么利用 NWP”合在一起。

建议改法：

围绕多步风电预测中的未来气象驱动，已有方法将 NWP 与历史功率进行多模态融合，或针对 NWP 预报误差、时间滞后、多源时空特征和空间相关性进行建模 \cite{ref13,ref14,ref15,ref16,ref17,ref18}。

后文可接：

对于网格化气象场，也有研究使用 3D CNN……

英文同步：

For future-weather-driven multi-step wind power forecasting, existing methods fuse NWP with historical power or model NWP forecast errors, temporal lag, multi-source spatiotemporal features, and spatial correlations.

### 8. 相关工作第 3 段

文件：`sections_short/related_work_zh.tex`

现段首：

Transformer 及其变体为多步时间序列预测提供了并行建模基础。

判定：需要修改。

理由：这是用户明确指出的短泛化句。它先给一个概念标签，再列方法。建议改成“以 Informer、PatchTST 等为代表的研究……”。

建议改法：

以 Informer、PatchTST、Crossformer 和 Non-stationary Transformers 为代表的序列预测研究，分别从长序列注意力、时间片段表示、跨维依赖和非平稳性角度改进多步时间序列预测 \cite{ref23,ref24,ref25,ref26}。

后文可接：

这些方法说明，注意力机制适合处理多步预测中的长时依赖……

英文同步：

Informer, PatchTST, Crossformer, and Non-stationary Transformers improve multi-step time-series forecasting through long-sequence attention, temporal patch representation, cross-dimension dependency, and non-stationarity modeling, respectively.

### 9. 相关工作第 4 段

文件：`sections_short/related_work_zh.tex`

现段首：

上述两类研究分别强调场站功率时空关系和未来气象驱动。

判定：需要修改。

理由：这是用户明确指出的短泛化句。“上述两类研究”也是明显的机械承接。建议直接写差异：业务预测协议下两类信息时间可用性不同。

建议改法：

在业务预测协议下，场站功率状态和未来气象驱动的关键差异不在于模态类型，而在于时间可用性：未来气象贯穿整个预测窗口，真实功率仅出现在预测窗口前。

后文可接：

基于这一差异，本文让功率状态在气象读取阶段发挥运行状态约束作用……

英文同步：

Under an operational forecasting protocol, the key difference between station-level power state and future weather driver is their time availability rather than modality type: future weather covers the full target window, whereas observed power appears only before the target window.

### 10. 相关工作第 5 段

文件：`sections_short/related_work_zh.tex`

现段首：

针对功率状态与未来气象在业务时间轴上的可用性差异，本文将融合过程放在 future queries 读取气象记忆之前。

判定：不需要强制修改，但建议小改。

理由：这句虽然是段首论断，但不短，也有具体机制位置；不是空泛模板。问题在于“本文将融合过程放在……”仍略像自我声明。可以改得更论文化，但不是最高风险。

建议改法：

功率状态与未来气象在业务时间轴上的可用性差异，要求融合机制在 future queries 读取气象记忆之前完成，而不是只在预测头前拼接两类编码特征。

英文同步：

The time-availability difference between power state and future weather requires fusion before future queries read weather memory, rather than only concatenating encoded features before the prediction head.

### 11. 问题定义第 1 段

文件：`sections_short/problem_zh.tex`

现段首：

设预测日为 \(D\)，目标窗口为北京时间 \(D\) 日 00:00--23:00。

判定：不需要改。

理由：这是数学问题定义的标准开头，承担符号定义功能。虽然短，但它不是泛化主题句，而是给出具体变量和时间窗口。合并或改写反而会降低定义清晰度。

### 12. 问题定义第 2 段

文件：`sections_short/problem_zh.tex`

现段首：

本文遵循业务化时间可用性约束，业务时间协议如图~\ref{fig:time-protocol} 所示。

判定：建议修改。

理由：这句不是特别短，但“本文遵循……”仍有模板感。由于用户刚要求 Fig. 1 引用放在这一节，可以改成更具体的协议描述。

建议改法：

业务时间协议如图~\ref{fig:time-protocol} 所示：日样本评价中，最新可用功率上下文覆盖 \(D-1\) 日 08:00--23:00，Pangu-Weather 预报在 \(D-1\) 日 08:00 起报。

后文可接：

lead 0--15 对应目标窗口之前的后报气象段……

英文同步：

As shown in Fig.~\ref{fig:time-protocol}, the latest available power context in daily-sample evaluation covers 08:00--23:00 on day \(D-1\), and the Pangu-Weather forecast is initialized at 08:00 on day \(D-1\).

### 13. 问题定义第 3 段

文件：`sections_short/problem_zh.tex`

现段首：

训练阶段采用滑动窗口样本，以提高样本数量并覆盖更多预测起点；

判定：不需要改。

理由：这是实验协议/数据切分说明，第一句直接给出训练与测试采样方式，没有空泛主题句。句子虽然以“训练阶段”开头，但内容具体，属于技术设置说明。

### 14. 方法 A 第 1 段

文件：`sections_short/method_zh.tex`

现段首：

本文方法采用运行状态约束的非对称气象-功率融合设计，如图~\ref{fig:framework} 所示。

判定：建议修改。

理由：这是方法总览段，短句后面才列 Weather Encoder、Power-State Encoder、Query Decoder。建议让段首直接包含三模块结构，减少“本文方法采用……”模板。

建议改法：

如图~\ref{fig:framework} 所示，运行状态约束的非对称气象-功率融合网络由 Weather Encoder、Power-State Encoder 和 Query Decoder 三部分组成。

后文可接：

Weather Encoder 将后报气象和未来预报气象编码为 weather memory tokens……

英文同步：

As shown in Fig.~\ref{fig:framework}, the operating-state-constrained asymmetric weather-power fusion network consists of a Weather Encoder, a Power-State Encoder, and a Query Decoder.

### 15. 方法 A 第 2 段

文件：`sections_short/method_zh.tex`

现段首：

在该框架中，气象特征先被组织为可查询记忆……

判定：不需要改。

理由：这句不是短泛化句，而是直接解释框架内信息流。后文说明 memory-level 写入的区别和运行状态约束定义，逻辑具体。

### 16. 方法 B 气象记忆构建第 1 段

文件：`sections_short/method_zh.tex`

现段首：

气象记忆构建过程如图~\ref{fig:weather-memory} 所示。

判定：可不改，但建议小改。

理由：这是图引导句，技术论文中常见，可以保留。若希望完全避免短句段首，也可把图引用和输入定义合并。

建议改法：

如图~\ref{fig:weather-memory} 所示，Weather Encoder 以 \(\mathbf{X}=[\mathbf{X}^{h};\mathbf{X}^{f}]\) 为输入，其中 \(\mathbf{X}^{h}\) 为目标窗口前的后报气象，\(\mathbf{X}^{f}\) 为目标窗口内的未来预报气象。

英文同步：

As illustrated in Fig.~\ref{fig:weather-memory}, the Weather Encoder takes \(\mathbf{X}=[\mathbf{X}^{h};\mathbf{X}^{f}]\) as input, where ...

### 17. 方法 B 后续公式段

文件：`sections_short/method_zh.tex`

段首包括：

三路特征经 \(1\times1\) 融合后加入时间位置嵌入和阶段嵌入：

随后，将 \(\mathbf{H}_{1:T}\) 输入时空卷积主干 \(\mathcal{E}_{w}\)。

最终气象记忆定义为

判定：不需要改。

理由：这些段落直接引出公式或模块计算，短句承担“公式导入”功能。若强行合并，会让公式前说明变得拥挤。技术方法部分允许这种写法。

### 18. 方法 C 功率状态编码第 1 段

文件：`sections_short/method_zh.tex`

现段首：

功率状态编码过程如图~\ref{fig:power-state} 所示。

判定：可不改，但建议小改。

理由：同 Fig. 3 段，是图引导句，技术上可接受；但为了减少短句段首，可合并输入定义。

建议改法：

如图~\ref{fig:power-state} 所示，功率状态编码器输入目标窗口前最近 \(T_c\) 步场站级功率 \(\mathbf{P}^{c}\in\mathbb{R}^{N\times T_c}\) 和场站邻接矩阵 \(\mathbf{A}\in\mathbb{R}^{N\times N}\)。

英文同步：

As shown in Fig.~\ref{fig:power-state}, the power-state encoder takes as input ...

### 19. 方法 C 后续公式段

文件：`sections_short/method_zh.tex`

段首包括：

模型在 point、patch 和 global 三种时间尺度上编码场站功率……

为使三种尺度的信息相互校准……

其中 \(J\) 为 patch 数量……

判定：不需要改。

理由：这些句子直接解释模型结构或公式符号，信息具体，不是空泛短句。尤其“为使三种尺度的信息相互校准”虽然是目的句，但后面立即展开 cross-scale attention，属于技术说明。

### 20. 方法 D Memory-Level 融合第 1 段

文件：`sections_short/method_zh.tex`

现段首：

为在 memory 层写入功率状态，Power-to-context adapter 将功率状态向量映射为 \(K\) 个动态功率上下文 token：

判定：不需要改。

理由：该句直接引出公式，包含具体模块、输入输出和 token 数，不属于泛化短句。

### 21. 方法 D Query Decoder 段

文件：`sections_short/method_zh.tex`

现段首：

Query Decoder 维护 \(24\) 个 learnable future queries。

判定：不需要改。

理由：这是具体结构定义，短但有明确技术信息。后面紧接 decoder block 公式，作为公式导入合理。

### 22. 方法 D 末段

文件：`sections_short/method_zh.tex`

现段首：

所有 future queries 并行解码，避免自回归误差累积；

判定：不需要改。

理由：这是方法属性总结，信息具体，且本段很短是为了收束方法部分。不是模板式空句。

### 23. 实验数据与设置第 1 段

文件：`sections_short/experiments_zh.tex`

现段首：

本文基于吉林省区域风电场群构建实验数据集……

判定：不需要强制修改。

理由：这句直接给出数据来源、预测对象和气象数据源，信息密度高。虽然以“本文基于”开头，但不是空泛短句。

可选改法：

实验数据集来自吉林省区域风电场群，预测对象为 58 个风电场站形成的区域聚合出力，并配套使用覆盖研究区域的 Pangu-Weather 网格化气象预报。

### 24. 实验数据与设置第 2 段

文件：`sections_short/experiments_zh.tex`

现段首：

训练集使用 2022 年滑动窗口样本……

判定：不需要改。

理由：直接描述数据划分和采样设置，具体且必要。

### 25. 实验数据与设置第 3 段

文件：`sections_short/experiments_zh.tex`

现段首：

对比模型包括 Weather-only 以及多种网格气象编码基线。

判定：建议修改。

理由：这是实验部分较明显的“分类标签”短句。可以让段首直接说明对比的目的：检验运行状态约束增益和气象编码器差异。

建议改法：

为分别检验运行状态约束的增益和不同网格气象编码方式的影响，本文设置 Weather-only 以及 CNN、3D CNN、ViT、Swin 和 Video Transformer 等对比模型。

后文可接：

Weather-only 与本文方法共享气象输入和 query decoder……

英文同步：

To separately evaluate the gain from operating-state constraints and the effect of different gridded-weather encoders, the experiments include Weather-only and CNN, 3D CNN, ViT, Swin, and Video Transformer baselines.

### 26. 实验数据与设置第 4 段

文件：`sections_short/experiments_zh.tex`

现段首：

为保证时间信息使用一致，所有 weather-only 基线均输入完整的 16 步后报气象和 24 步未来预报气象。

判定：不需要改。

理由：目的明确，直接说明基线输入公平性，信息具体。

### 27. 整体预测性能第 1 段

文件：`sections_short/experiments_zh.tex`

现段首：

表~\ref{tab:benchmark-short} 给出了不同模型的整体测试结果。

判定：不需要改。

理由：这是结果表引导句，符合论文写法。后文立即给出数值和解释，不属于 AIGC 空泛开头。

### 28. 整体预测性能第 2 段

文件：`sections_short/experiments_zh.tex`

现段首：

除 Weather-only 外，3D CNN 是误差最低的网格气象编码基线……

判定：不需要改。

理由：句子直接给出对比结果和分析对象，信息具体。不是泛化短句。

### 29. 逐预测步融合效果第 1 段

文件：`sections_short/experiments_zh.tex`

现段首：

表~\ref{tab:horizon-short} 和图~\ref{fig:horizon} 展示了本文方法与 Weather-only 在不同预测步长上的误差差异……

判定：不需要改。

理由：图表引导句，后面立即报告 H1 数值和解释。结构自然。

### 30. 逐预测步融合效果第 2 段

文件：`sections_short/experiments_zh.tex`

现段首：

逐步长曲线显示，运行状态约束带来的收益在目标窗口起始阶段最为突出……

判定：不需要改。

理由：这是图后分析句，直接总结曲线趋势和数值变化，信息具体。

### 31. 消融实验第 1 段

文件：`sections_short/experiments_zh.tex`

现段首：

各消融变体用于分别检验功率状态、功率编码结构和气象记忆的作用。

判定：建议修改。

理由：这句是实验目的概括，略像模板。可直接写具体变体和对应检验目标。

建议改法：

消融实验从功率状态、功率编码结构和气象记忆三个方面拆解模型贡献：Weather-only 移除功率状态上下文，仅使用气象输入；Random power 和 Zero power 保留功率侧结构，但分别用随机功率和零功率替代真实场站功率；Basic GCN 将多尺度功率编码器替换为基础图卷积结构。

后文接：

No memory 关闭时空气象 memory，No wind 移除风场相关气象变量。

英文同步：

The ablation study decomposes model contributions from power state, power encoding structure, and weather memory: Weather-only removes the power-state context and uses only meteorological input; Random power and Zero power keep the power-side structure but replace true station-level power with random or zero values; Basic GCN replaces the multi-scale power encoder with a basic graph convolution structure.

### 32. 消融实验第 2 段

文件：`sections_short/experiments_zh.tex`

现段首：

表~\ref{tab:ablation-short} 给出了关键消融结果。

判定：不需要改。

理由：表格引导句，后文立即给出 MAE 增加和变体解释。可接受。

### 33. 消融实验第 3 段

文件：`sections_short/experiments_zh.tex`

现段首：

气象侧消融进一步说明，本文方法仍以未来气象为主驱动。

判定：需要修改。

理由：这是短结论句，后面才列 No memory、No wind。可以改成直接以两个消融结果开头。

建议改法：

No memory 和 No wind 的退化结果表明，本文方法在引入运行状态约束的同时仍以未来气象为主驱动。

后文可接：

No memory 关闭气象空间记忆后误差明显上升……

英文同步：

The degradation of No memory and No wind shows that the proposed method remains mainly driven by future weather while introducing operating-state constraints.

### 34. 消融实验第 4 段

文件：`sections_short/experiments_zh.tex`

现段首：

结合逐预测步结果可以看到，功率侧和气象侧消融回答的是不同问题。

判定：建议修改。

理由：这句有一定模板感，但内容有分析价值。建议改成更直接的归纳句。

建议改法：

逐预测步结果与消融结果共同表明，功率侧退化和气象侧退化对应两类不同的信息来源。

后文可接：

随机或零功率退化说明模型确实依赖真实运行状态……

英文同步：

The horizon-wise and ablation results jointly indicate that power-side and weather-side degradations correspond to two different information sources.

### 35. 讨论与结论第 1 段

文件：`sections_short/conclusion_zh.tex`

现段首：

实验结果表明，在区域风电集群 24 h 预测中……

判定：不需要改。

理由：结论段开头直接总结实验发现，属于标准论文结论句。它不是“一句短标签”，而是带有场景、指标关系和基线定位的完整论断。

### 36. 讨论与结论第 2 段

文件：`sections_short/conclusion_zh.tex`

现段首：

本文提出了运行状态约束的气象-功率融合网络……

判定：建议修改。

理由：结论末段使用“本文提出了……”可以接受，但与前文多处“本文……”叠加后显得模板化。建议以方法机制收束，而不是自我声明。

建议改法：

运行状态约束的气象-功率融合网络将未来气象编码为主记忆，将近期场站级功率编码为动态上下文 token，并通过 memory-level fusion 将运行状态约束写入 future queries 的读取过程。

英文同步：

The operating-state-constrained weather-power fusion network encodes future weather as the main memory, maps recent station-level power to dynamic context tokens, and imposes operating-state constraints on future-query reading through memory-level fusion.

## 不需要修改项的详细说明

### 技术定义类段落不建议改

包括：

- 问题定义第 1 段
- 方法中公式前后的符号定义段
- Query Decoder 维护 24 个 future queries 段

理由：这些段落的段首虽然可能较短，但它们承担定义、公式导入或变量说明功能。技术论文中这类短句是正常写法，不会被看作 AIGC 模板。

### 图表引导类段落不建议改

包括：

- 表~\ref{tab:benchmark-short} 给出了……
- 表~\ref{tab:horizon-short} 和图~\ref{fig:horizon} 展示了……
- 表~\ref{tab:ablation-short} 给出了……

理由：图表引导句是实验部分的常规结构，目的是让读者快速定位证据来源。只要后文紧接数值和解释，就不是空泛短句。

### 结果分析类段落多数不建议改

包括：

- 除 Weather-only 外，3D CNN 是误差最低的网格气象编码基线……
- 逐步长曲线显示……
- 实验结果表明……

理由：这些句子都直接给出实验事实或归纳结论，信息密度高，不属于“先给空标签、再展开”的写法。

### 贡献列表不按普通段落处理

“本文主要贡献如下”可以保留，因为它只是列表引导，不是正文论证段。

## 建议修改清单

优先改：

1. 摘要第一句
2. 引言第 1 段
3. 引言第 2 段
4. 引言第 4 段
5. 相关工作第 1 段
6. 相关工作第 2 段
7. 相关工作第 3 段
8. 相关工作第 4 段
9. 问题定义第 2 段
10. 方法整体框架第 1 段
11. 实验对比模型段
12. 消融实验变体说明段
13. 气象侧消融段
14. 消融总结段
15. 结论第 2 段

可选改：

1. 相关工作第 5 段
2. 方法气象记忆构建第 1 段
3. 方法功率状态编码第 1 段
4. 实验数据与设置第 1 段

不建议改：

1. 问题定义第 1 段和第 3 段
2. 方法中公式导入和符号解释段
3. 实验图表引导段
4. 实验结果分析段
5. 贡献列表

## 后续执行建议

如果确认修改，建议分两步做：

1. 先改“优先改”15 处，并同步英文。

2. 再看篇幅和语气，如果仍显模板化，再处理“可选改”4 处。

不建议一次性把所有短句都消灭，因为方法和实验部分需要保留一定的结构性短句，否则可读性会变差。
