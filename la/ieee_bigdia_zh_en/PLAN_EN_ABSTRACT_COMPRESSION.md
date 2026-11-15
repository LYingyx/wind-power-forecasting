# 摘要压缩大方案：英文与中文对照

## 1. 当前问题

当前英文摘要约 236 words，对 IEEE 双栏首页偏长。它同时承担了背景、问题动机、方法细节、完整实验指标和结论解释，导致首屏信息密度过高。

主要冗余点包括：

- 应用背景写得过满：`power dispatch, reserve allocation, and renewable energy integration` 可以压缩为 `grid operation` 或直接删除。
- 问题动机重复：未来气象、真实运行状态、状态不一致等表达在摘要中连续展开，和引言功能有重叠。
- 方法细节过长：`pre-forecast-day meteorological context and intraday future weather`、`weather-driven decoding process`、`remaining aware of the current cluster state` 可以合并。
- 实验指标过密：摘要同时写 MAE、RMSE、R2 的具体值和百分比提升，占用空间较大；前面讨论过正文中更适合突出百分比，具体数值可留在实验表格。
- 结论句偏解释型：`true power states before the forecast origin can effectively constrain...` 可以改成更短的结果归纳。

## 2. 压缩目标

建议把英文摘要压到 130--150 words 左右。

目标不是简单删短，而是保留四个核心信息：

1. 任务：regional wind farm cluster 24 h forecasting。
2. 核心问题：future weather is dominant, but pre-origin operating state is missing in weather-only prediction。
3. 方法落点：operating-state-constrained weather-power fusion network。
4. 结果亮点：MAE improvement about 4.1%; first-hour MAE improvement about 42.4%; RMSE and R2 consistent。

## 3. 建议删减原则

### 3.1 背景句压缩

原思路：

Intraday regional wind farm cluster power forecasting is an important basis for power dispatch, reserve allocation, and renewable energy integration.

建议处理：

删除或压缩。IEEE 摘要更需要直接进入问题，不必展开业务价值。

建议表达：

Regional wind farm cluster forecasting is mainly driven by future weather forecasts.

### 3.2 问题句合并

原思路：

Weather-driven forecasting models usually rely on future weather forecasts to describe wind-field evolution during the forecast day. However, under the same weather situation, actual power output is also affected by the station-level generation level, short-term inertia, and synchronized multi-site variations before the forecast origin...

建议处理：

合并为一句，突出“未来气象是主驱动，但预测起点前运行状态缺失会造成偏差”。

建议表达：

Future weather forecasts describe wind-field evolution, but weather-only models can miss the measured operating state before the forecast origin, including generation level, short-term inertia, and coordinated multi-site variations.

### 3.3 方法句压缩

原思路：

The method extracts weather-driven representations from pre-forecast-day meteorological context and intraday future weather, encodes station-level measured power available before the forecast origin into an operating-state representation, and then incorporates this state into the weather-driven decoding process so that the model can use future wind-field evolution while remaining aware of the current cluster state.

建议处理：

方法只保留“weather memory + operating-state representation + decoding constraint”。不要在摘要里解释太细。

建议表达：

The model builds queryable weather memory from meteorological context and future forecasts, encodes pre-window station-level power as an operating-state representation, and injects it into the decoding process so that future queries read weather information under current-state constraints.

### 3.4 实验结果压缩

原思路：

Experiments on 58 wind farms in Jilin Province show that the proposed method achieves an MAE of 0.0767, an RMSE of 0.1041, and an R2 of 0.8330..., reducing MAE by about 4.1% relative to the best baseline.

建议处理：

摘要中不写具体 MAE/RMSE/R2 数值，只写百分比和一致改进。具体数值留在 Table I 和实验分析。

建议表达：

Experiments on 58 wind farms in Jilin Province show that the proposed method reduces MAE by about 4.1% over the best baseline, with consistent RMSE and R2 improvements.

### 3.5 逐步长结果包装

原思路：

Horizon-wise analysis shows that the gain is most pronounced at the beginning of the forecast day, with first-hour MAE reduced by about 42.4%.

建议保留，但避免暗示只改善前段。写成“最明显”，不是“只在前段有效”。

建议表达：

Horizon-wise results show the most pronounced gains at the initial forecast horizons, reducing first-hour MAE by about 42.4%.

## 4. 推荐摘要版本

推荐版本约 140 words，适合作为下一版直接替换：

### 4.1 English

Regional wind farm cluster forecasting is mainly driven by future weather forecasts, but weather-only models can miss the measured operating state before the forecast origin, including generation level, short-term inertia, and coordinated multi-site variations. This paper proposes an operating-state-constrained weather-power fusion network for 24 h cluster forecasting. The model builds queryable weather memory from meteorological context and future forecasts, encodes pre-window station-level power as an operating-state representation, and injects it into the decoding process so that future queries read weather information under current-state constraints. Experiments on 58 wind farms in Jilin Province show that the proposed method reduces MAE by about 4.1% over the best baseline, with consistent RMSE and R2 improvements. Horizon-wise results show the most pronounced gains at the initial forecast horizons, reducing first-hour MAE by about 42.4%. These results demonstrate that pre-window power observations provide effective operating-state constraints for weather-power fusion.

### 4.2 中文

区域风电集群预测主要受未来气象预报驱动，但仅依赖气象的模型容易忽略预测起点前真实可观测的运行状态，包括出力水平、短时惯性和多场站协同变化。为此，本文提出一种运行状态约束的气象-功率融合网络，用于 24 h 集群预测。该模型由气象上下文和未来气象预报构建可查询气象记忆，将目标窗口前场站级功率编码为运行状态表示，并将其注入解码过程，使 future queries 在当前状态约束下读取气象信息。基于吉林区域 58 个风电场站的实验表明，本文方法相较最优基线 MAE 降低约 4.1%，RMSE 和 R2 也取得一致改进。逐预测步结果显示，收益在预测初始步长最为明显，首小时 MAE 降低约 42.4%。结果说明，目标窗口前功率观测能够为气象-功率融合提供有效的运行状态约束。

## 5. 更激进压缩版本

如果需要进一步给首页腾空间，可以压到约 115--125 words：

### 5.1 English

Regional wind farm cluster forecasting is mainly driven by future weather forecasts, but weather-only models can miss the measured operating state before the forecast origin. This paper proposes an operating-state-constrained weather-power fusion network for 24 h cluster forecasting. It builds queryable weather memory from meteorological context and future forecasts, encodes pre-window station-level power as an operating-state representation, and injects this representation into decoding so that future queries read weather information under current-state constraints. Experiments on 58 wind farms in Jilin Province show that the proposed method reduces MAE by about 4.1% over the best baseline, with consistent RMSE and R2 improvements. The gain is most pronounced at initial horizons, where first-hour MAE is reduced by about 42.4%.

### 5.2 中文

区域风电集群预测主要受未来气象预报驱动，但仅依赖气象的模型容易忽略预测起点前真实可观测的运行状态。本文提出一种运行状态约束的气象-功率融合网络，用于 24 h 集群预测。该模型由气象上下文和未来气象预报构建可查询气象记忆，将目标窗口前场站级功率编码为运行状态表示，并将其注入解码过程，使 future queries 在当前状态约束下读取气象信息。基于吉林区域 58 个风电场站的实验表明，本文方法相较最优基线 MAE 降低约 4.1%，RMSE 和 R2 也取得一致改进。逐预测步结果显示，收益在预测初始步长最为明显，首小时 MAE 降低约 42.4%。

## 6. 建议采用方案

建议先采用第 4 节的推荐版本，并中英文同步。

理由：

- 长度从约 236 words 降到约 140 words，压缩幅度足够明显。
- 保留核心问题、方法机制和实验亮点。
- 不暴露“只改善前段”的弱点，而是表述为“initial horizons 上收益最明显”。
- 不再堆叠具体 MAE/RMSE/R2 数值，摘要更像论文摘要而不是实验段落。

## 7. 预计源文件改动

只需要修改：

- `manuscript_en_short.tex` 的 `abstract` 环境。

暂不建议同步改关键词和正文。等摘要确定后，再检查关键词是否需要与摘要里的 `operating-state-constrained`、`weather-power fusion` 保持一致。

## 8. 编译预期

摘要缩短后，英文稿首页压力会下降。当前英文 PDF 已为 8 页，摘要压缩主要用于增强版面余量和改善首页观感，不一定改变总页数，但能降低后续图表、作者块和 references 轻微调整带来的超页风险。
