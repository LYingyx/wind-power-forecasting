# Fig. 1 正文引用修订方案

## 问题

当前 Fig. 1 在 `sections_short/introduction_zh.tex` 和 `sections_short/introduction_en.tex` 中被插入，但使用的是 `\figwide` 宏：

中文：

\figwide{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{业务时间协议。}{近期场站功率仅在目标窗口前可用，未来气象预报覆盖完整预测窗口；本文在该时间可用性约束下进行气象-功率融合预测。}

英文：

\figwide{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{Operational time protocol.}{Recent station-level power is available only before the target window, whereas future meteorological forecasts cover the full target window. The proposed method performs weather-power fusion under this information-availability constraint.}

`figwide` 不生成 `\label`，因此正文无法使用 `图~\ref{...}` / `Fig.~\ref{...}` 引用。正文目前也没有显式提到 Fig. 1。

## 修改目标

只解决 Fig. 1 没有正文引用的问题。

本次不处理“段首短句”问题，不重写引言和相关工作。

本次不编译，除非后续明确要求。

## 建议修改

### 1. 增加带 label 的 Fig. 1

将 Fig. 1 的插图命令从 `\figwide` 改为 `\figwidelabel`。

中文改为：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{业务时间协议。}{近期场站功率仅在目标窗口前可用，未来气象预报覆盖完整预测窗口；本文在该时间可用性约束下进行气象-功率融合预测。}{fig:time-protocol}

英文改为：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{Operational time protocol.}{Recent station-level power is available only before the target window, whereas future meteorological forecasts cover the full target window. The proposed method performs weather-power fusion under this information-availability constraint.}{fig:time-protocol}

### 2. 在正文第二段加入引用

中文原句：

本文关注业务化区域风电集群 24 h 预测。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。

中文建议改为：

本文关注业务化区域风电集群 24 h 预测，其时间可用性关系如图~\ref{fig:time-protocol} 所示。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。

英文原句：

This paper studies operational 24 h regional wind farm cluster forecasting. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation, whereas station-level measured power is available only before the target window and characterizes the current generation level, short-term inertia, and multi-site coupling state.

英文建议改为：

This paper studies operational 24 h regional wind farm cluster forecasting, whose time-availability relation is shown in Fig.~\ref{fig:time-protocol}. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation, whereas station-level measured power is available only before the target window and characterizes the current generation level, short-term inertia, and multi-site coupling state.

## 修改理由

1. Fig. 1 是业务时间协议图，最自然的引用位置就是引言第二段中解释“未来气象覆盖目标窗口、功率只在目标窗口前可用”的句子。

2. 使用 `fig:time-protocol` 作为 label，语义清楚，不影响 Fig. 2 的 `fig:framework`。

3. 只增加引用和 label，不调整段落结构，避免提前处理“短句开头”问题。
