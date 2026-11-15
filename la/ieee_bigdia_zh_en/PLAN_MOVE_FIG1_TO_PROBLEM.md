# Fig. 1 移到“问题定义与业务时间对齐”的修改方案

## 修改目标

Fig. 1 是业务时间协议图，内容对应“目标窗口、功率上下文、后报气象、未来气象”的时间可用性关系。相比放在引言中，它更适合放在 `III. 问题定义与业务时间对齐` 部分。

本次只处理 Fig. 1 的引用位置和插图位置，不处理段首短句问题，不改其它图。

## 当前问题

当前 Fig. 1 位于：

- `sections_short/introduction_zh.tex`
- `sections_short/introduction_en.tex`

当前正文引用也在引言第二段：

中文现句：

本文关注业务化区域风电集群 24 h 预测，其时间可用性关系如图~\ref{fig:time-protocol} 所示。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。

英文现句：

This paper studies operational 24 h regional wind farm cluster forecasting, whose time-availability relation is shown in Fig.~\ref{fig:time-protocol}. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation, whereas station-level measured power is available only before the target window and characterizes the current generation level, short-term inertia, and multi-site coupling state.

当前 Fig. 1 插图命令：

中文：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{业务时间协议。}{近期场站功率仅在目标窗口前可用，未来气象预报覆盖完整预测窗口；本文在该时间可用性约束下进行气象-功率融合预测。}{fig:time-protocol}

英文：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{Operational time protocol.}{Recent station-level power is available only before the target window, whereas future meteorological forecasts cover the full target window. The proposed method performs weather-power fusion under this information-availability constraint.}{fig:time-protocol}

## 建议修改

### 1. 从引言中移除 Fig. 1 引用

文件：`sections_short/introduction_zh.tex`

将中文现句改为：

本文关注业务化区域风电集群 24 h 预测。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；场站级实测功率只在目标窗口前可用，主要表征当前出力水平、短期惯性和多场站耦合状态。

文件：`sections_short/introduction_en.tex`

将英文现句改为：

This paper studies operational 24 h regional wind farm cluster forecasting. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation, whereas station-level measured power is available only before the target window and characterizes the current generation level, short-term inertia, and multi-site coupling state.

说明：这一步只移除 Fig. 1 引用，不处理“短句开头”问题。短句问题后续集中改。

### 2. 从引言末尾移除 Fig. 1 插图命令

从 `sections_short/introduction_zh.tex` 删除 Fig. 1 的 `\figwidelabel{...Fig01_svg.pdf...}{fig:time-protocol}`。

从 `sections_short/introduction_en.tex` 删除 Fig. 1 的 `\figwidelabel{...Fig01_svg.pdf...}{fig:time-protocol}`。

Fig. 2 仍保留在引言中，不移动：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig02_svg.pdf}{...}{...}{fig:framework}

### 3. 在问题定义部分加入 Fig. 1 正文引用

文件：`sections_short/problem_zh.tex`

原句：

本文遵循业务化时间可用性约束。对于日样本评价，最新可用功率上下文覆盖 \(D-1\) 日 08:00--23:00。Pangu-Weather 预报在 \(D-1\) 日 08:00 起报，lead 0--15 对应目标窗口之前的后报气象段，lead 16--39 对应 \(D\) 日 00:00--23:00 的未来预报气象段。

建议改为：

本文遵循业务化时间可用性约束，其时间协议如图~\ref{fig:time-protocol} 所示。对于日样本评价，最新可用功率上下文覆盖 \(D-1\) 日 08:00--23:00。Pangu-Weather 预报在 \(D-1\) 日 08:00 起报，lead 0--15 对应目标窗口之前的后报气象段，lead 16--39 对应 \(D\) 日 00:00--23:00 的未来预报气象段。

文件：`sections_short/problem_en.tex`

原句：

The task follows an operational information-availability constraint. For daily-sample evaluation, the latest available power context covers 08:00--23:00 on day \(D-1\). The Pangu-Weather forecast is initialized at 08:00 on day \(D-1\).

建议改为：

The task follows an operational information-availability constraint, whose time protocol is shown in Fig.~\ref{fig:time-protocol}. For daily-sample evaluation, the latest available power context covers 08:00--23:00 on day \(D-1\). The Pangu-Weather forecast is initialized at 08:00 on day \(D-1\).

### 4. 在问题定义部分插入 Fig. 1

建议把 Fig. 1 插图命令放在 `problem_zh.tex` 和 `problem_en.tex` 的第二段之后，也就是完整说明功率上下文、起报时间、lead 0--15、lead 16--39 和气象输入之后。

中文插入：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{业务时间协议。}{近期场站功率仅在目标窗口前可用，未来气象预报覆盖完整预测窗口；本文在该时间可用性约束下进行气象-功率融合预测。}{fig:time-protocol}

英文插入：

\figwidelabel{0.78\textwidth}{../../paper_image_use/method/Fig01_svg.pdf}{Operational time protocol.}{Recent station-level power is available only before the target window, whereas future meteorological forecasts cover the full target window. The proposed method performs weather-power fusion under this information-availability constraint.}{fig:time-protocol}

## 修改后结构

修改后：

- Fig. 1 位于 `III. 问题定义与业务时间对齐`。
- Fig. 1 的正文引用也位于该节。
- Fig. 2 仍位于引言，负责承接整体方法框架。
- `fig:time-protocol` 标签不变，避免影响引用编号。

## 风险与注意

1. 使用 `figure*` 的 Fig. 1 仍可能浮动到相邻页面顶部，这是 IEEE 双栏模板的正常行为；但源码位置和正文引用会属于问题定义部分。

2. 本次移动可能轻微影响图表浮动和页数，需要你同意后再编译检查。

3. 本次不处理段首短句问题，避免把两个修改主题混在一起。
