# 功率状态编码中 CSA 与 \(\mathbf{S}^{p}\) 的修正方案

## 当前问题

涉及文件：

- `sections_short/method_zh.tex`
- `sections_short/method_en.tex`

当前短版在“功率状态编码”里存在两个不一致：

1. 文字写了自底向上和自顶向下的跨尺度注意力，但没有给出 CSA 的定义或公式。

2. 当前公式把跨尺度融合后的 token 序列写成

\[
\mathbf{S}^{p}=[\mathbf{o}^{gl};\mathbf{o}^{pa}_{1:J};\mathbf{o}^{pt}_{1:T_c}],
\]

这实际上只是原始 global、patch、point tokens 的直接拼接，没有体现 CSA 之后的结果。按长版写法，\(\mathbf{S}^{p}\) 应该来自跨尺度交互后的 \(\mathbf{H}^{g}_{bu}\)、\(\mathbf{H}^{pat}_{td}\)、\(\mathbf{H}^{pt}_{td}\)。

另外，当前 patch 写成 \(\mathbf{P}^{c}_{:,jr:jr+s}\)，从论文符号角度不够清楚。长版用局部片段索引集合 \(\mathcal I_j\) 表达更稳。

## 拟修改点 1：修正 patch token 的索引

当前写法：

\[
\mathbf{o}^{pa}_{j}
=g_{pa}(\operatorname{GCN}_{pa}(\mathbf{P}^{c}_{:,jr:jr+s},\mathbf{A})).
\]

拟改为先定义 patch 索引：

\[
\mathcal{I}_{j}=\{1+(j-1)r,\ldots,1+(j-1)r+s-1\},
\quad
J=\lfloor (T_c-s)/r\rfloor+1.
\]

然后写：

\[
\mathbf{o}^{pa}_{j}
=g_{pa}(\operatorname{GCN}_{pa}(\mathbf{P}^{c}_{:,\mathcal{I}_{j}},\mathbf{A})).
\]

理由：

这样避免 0-based / 1-based 索引混乱，也更贴近长版。

## 拟修改点 2：补充 CSA 过程

在 point、patch、global 三类 token 生成后，增加一小段：

加入位置编码后，三类 token 分别记为 \(\mathbf{H}^{pt}\)、\(\mathbf{H}^{pa}\) 和 \(\mathbf{H}^{gl}\)。跨尺度交互先自底向上，再自顶向下：

\[
\begin{aligned}
\mathbf{H}^{pa}_{bu} &= \operatorname{CSA}_{pt\rightarrow pa}(\mathbf{H}^{pa},\mathbf{H}^{pt}),\\
\mathbf{H}^{gl}_{bu} &= \operatorname{CSA}_{pa\rightarrow gl}(\mathbf{H}^{gl},\mathbf{H}^{pa}_{bu}),\\
\mathbf{H}^{pa}_{td} &= \operatorname{CSA}_{gl\rightarrow pa}(\mathbf{H}^{pa}_{bu},\mathbf{H}^{gl}_{bu}),\\
\mathbf{H}^{pt}_{td} &= \operatorname{CSA}_{pa\rightarrow pt}(\mathbf{H}^{pt},\mathbf{H}^{pa}_{td}).
\end{aligned}
\]

其中 \(\operatorname{CSA}_{source\rightarrow target}\) 表示以 target tokens 为 query、以 source tokens 为 key 和 value 的跨尺度注意力。

理由：

这能明确“patch 读取 point、global 读取 patch、global 反向调制 patch、patch 反向调制 point”的实际计算路径。

## 拟修改点 3：修正 \(\mathbf{S}^{p}\) 的来源

当前写法：

\[
\mathbf{S}^{p}=[\mathbf{o}^{gl};\mathbf{o}^{pa}_{1:J};\mathbf{o}^{pt}_{1:T_c}]
\in\mathbb{R}^{(1+J+T_c)\times d}.
\]

拟改为：

\[
\mathbf{S}^{p}=[\mathbf{H}^{gl}_{bu};\mathbf{H}^{pa}_{td};\mathbf{H}^{pt}_{td}]
\in\mathbb{R}^{(1+J+T_c)\times d}.
\]

随后保持：

\[
\mathbf{z}^{p}=f_{p}(\operatorname{Transformer}(\mathbf{S}^{p}))
\in\mathbb{R}^{d}.
\]

理由：

这样 \(\mathbf{S}^{p}\) 才是真正的跨尺度融合后 token 序列，而不是 CSA 前的原始 token 拼接。

## 英文同步原则

英文 `method_en.tex` 同步做三处对应修改：

1. 用 \(\mathcal I_j\) 定义 patch segment。
2. 补充 CSA 四步公式和 `source -> target` 的 query/key/value 解释。
3. 将 \(\mathbf{S}^{p}\) 改为由 \(\mathbf{H}^{gl}_{bu}\)、\(\mathbf{H}^{pa}_{td}\)、\(\mathbf{H}^{pt}_{td}\) 拼接得到。

## 篇幅控制

不引入长版里的以下实现细节：

- point/patch/global GCN encoder 的具体层数。
- CSA block 的数量。
- 每个 CSA block 的完整 MHA/FFN/残差公式。

这样能修正逻辑错误，同时尽量控制英文 8 页限制。

## 拟执行动作

如确认执行：

1. 修改 `sections_short/method_zh.tex` 的功率状态编码段。
2. 同步修改 `sections_short/method_en.tex`。
3. 编译中英文短版。
4. 检查英文是否仍为 8 页。
