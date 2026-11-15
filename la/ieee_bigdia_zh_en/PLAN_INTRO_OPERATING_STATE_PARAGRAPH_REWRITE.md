# 引言中“运行状态约束”段落改写方案

## 修改目标

用户指出引言中以下段落读起来不自然：

在业务化区域风电集群 24 h 预测中，未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。本文将预测起点前的真实功率状态定位为未来气象读取的运行状态约束，用于引导模型解释目标窗口内的气象驱动信息。仅使用未来气象会忽略预测起点处的真实运行状态，简单对称拼接历史功率和未来气象也会弱化二者在业务时间轴上的不同角色。因此，本文的核心问题是让预测起点前可用的运行状态有效参与未来气象读取过程。

问题主要在于“定位为未来气象读取的运行状态约束”“解释目标窗口内的气象驱动信息”“有效参与未来气象读取过程”等表达过于概念化。

## 中文建议改写

在业务化区域风电集群 24 h 预测中，未来网格化气象预报给出目标窗口内的风场演化，是区域功率变化的主要驱动；预测起点前的场站级实测功率则记录了当前出力水平、短期惯性和多场站同步变化，是模型读取未来气象时需要参照的运行状态约束。若只依赖未来气象，模型容易将相似气象形势映射到与实际起点不一致的出力状态；若将历史功率与未来气象简单拼接，又难以体现二者在业务时间轴上的不同角色。因此，本文关注的问题是：如何在未来气象主驱动预测中引入预测起点前可用的真实运行状态，使模型在读取未来气象时同时保持对当前场群状态的感知。

## 英文同步改写

In operational 24 h regional wind farm cluster forecasting, future gridded weather forecasts describe wind-field evolution within the target window and provide the main driver of regional power variation. Station-level measured power before the forecast origin records the current generation level, short-term inertia, and synchronized multi-site variations, and therefore serves as an operating-state constraint that the model should refer to when reading future weather. If only future weather is used, similar meteorological situations may be mapped to power states inconsistent with the actual forecast origin; if historical power and future weather are simply concatenated, their different roles along the operational time axis are weakened. This paper therefore studies how to introduce the truly available operating state before the forecast origin into future-weather-driven prediction, so that the model can read future weather while remaining aware of the current cluster state.

## 修改文件

- `sections_short/introduction_zh.tex`
- `sections_short/introduction_en.tex`

## 编译

用户已明确允许本次修改并编译。修改后编译：

- `manuscript_zh_short.tex`
- `manuscript_en_short.tex`
