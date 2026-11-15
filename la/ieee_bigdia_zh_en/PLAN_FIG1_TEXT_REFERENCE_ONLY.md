# Fig. 1 仅调整正文引用的修改方案

## 目标

图片位置不变，只修改正文文字。

也就是说：

- Fig. 1 的插图命令仍留在 `sections_short/introduction_zh.tex` 和 `sections_short/introduction_en.tex` 原位置。
- 不移动图片。
- 只把正文中对 Fig. 1 的说明放到 `III. 问题定义与业务时间对齐` 这一节。
- 本方案不处理段首短句问题。

## 当前问题

上一次为了让 Fig. 1 有正文引用，在引言第二段加入了：

中文：

本文关注业务化区域风电集群 24 h 预测，其时间可用性关系如图~\ref{fig:time-protocol} 所示。

英文：

This paper studies operational 24 h regional wind farm cluster forecasting, whose time-availability relation is shown in Fig.~\ref{fig:time-protocol}.

现在需要把这类正文引用放到 `III. 问题定义与业务时间对齐`，而不是放在引言。

## 建议修改

### 1. 引言中删除 Fig. 1 引用

文件：`sections_short/introduction_zh.tex`

将：

本文关注业务化区域风电集群 24 h 预测，其时间可用性关系如图~\ref{fig:time-protocol} 所示。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；

改为：

本文关注业务化区域风电集群 24 h 预测。未来网格化气象预报覆盖完整目标窗口，是预测未来功率变化的主要驱动；

文件：`sections_short/introduction_en.tex`

将：

This paper studies operational 24 h regional wind farm cluster forecasting, whose time-availability relation is shown in Fig.~\ref{fig:time-protocol}. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation,

改为：

This paper studies operational 24 h regional wind farm cluster forecasting. Future gridded weather forecasts cover the full target window and provide the main driver of future power variation,

### 2. 在 III 问题定义中加入 Fig. 1 引用

文件：`sections_short/problem_zh.tex`

将第二段开头：

本文遵循业务化时间可用性约束。对于日样本评价，最新可用功率上下文覆盖 \(D-1\) 日 08:00--23:00。

改为：

本文遵循业务化时间可用性约束，业务时间协议如图~\ref{fig:time-protocol} 所示。对于日样本评价，最新可用功率上下文覆盖 \(D-1\) 日 08:00--23:00。

文件：`sections_short/problem_en.tex`

将第二段开头：

The task follows an operational information-availability constraint. For daily-sample evaluation, the latest available power context covers 08:00--23:00 on day \(D-1\).

改为：

The task follows an operational information-availability constraint, with the time protocol shown in Fig.~\ref{fig:time-protocol}. For daily-sample evaluation, the latest available power context covers 08:00--23:00 on day \(D-1\).

## 不修改的内容

Fig. 1 的插图命令不移动：

- `sections_short/introduction_zh.tex`
- `sections_short/introduction_en.tex`

Fig. 1 的 label 继续使用：

`fig:time-protocol`

## 说明

这次只是文字层面的引用位置调整。源码中的图片位置保持不变。
