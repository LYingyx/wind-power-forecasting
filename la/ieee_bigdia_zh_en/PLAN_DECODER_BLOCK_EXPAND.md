# Query Decoder Block 公式展开方案

## 问题

当前短版在 `Memory-Level 融合与查询解码` 小节中写：

\[
\mathbf{Q}^{(\ell)}=\operatorname{DecoderBlock}(\mathbf{Q}^{(\ell-1)},\mathbf{M}^{pc}).
\]

虽然文字说明了 decoder block 包含 query self-attention、query-to-memory cross-attention 和 FFN，但公式本身仍然是黑箱。用户希望“打开一下，写清楚点”。

## 参考长版

长版 `sections/methodology_zh.tex` 中展开为：

\[
\begin{aligned}
\tilde{\mathbf{Q}}^{(\ell)} &= \operatorname{LN}\left(\mathbf{Q}^{(\ell-1)}+\operatorname{MHA}(\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)})\right), \\
\bar{\mathbf{Q}}^{(\ell)} &= \operatorname{LN}\left(\tilde{\mathbf{Q}}^{(\ell)}+\operatorname{MHA}(\tilde{\mathbf{Q}}^{(\ell)},\mathbf{M}^{pc},\mathbf{M}^{pc})\right), \\
\mathbf{Q}^{(\ell)} &= \operatorname{LN}\left(\bar{\mathbf{Q}}^{(\ell)}+\operatorname{FFN}(\bar{\mathbf{Q}}^{(\ell)})\right).
\end{aligned}
\]

## 拟修改中文

保留现有文字：

Query Decoder 维护 \(24\) 个 learnable future queries。每个 decoder block 由 query self-attention、query-to-memory cross-attention 和 FFN 组成，并配合残差连接和 LayerNorm；其中 self-attention 建模不同预测步之间的依赖，cross-attention 从融合记忆 \(\mathbf{M}^{pc}\) 读取气象记忆和功率上下文：

将原来的占位公式替换为：

\[
\begin{aligned}
\tilde{\mathbf{Q}}^{(\ell)}
&=\operatorname{LN}\!\left(\mathbf{Q}^{(\ell-1)}
+\operatorname{MHA}(\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)})\right),\\
\bar{\mathbf{Q}}^{(\ell)}
&=\operatorname{LN}\!\left(\tilde{\mathbf{Q}}^{(\ell)}
+\operatorname{MHA}(\tilde{\mathbf{Q}}^{(\ell)},\mathbf{M}^{pc},\mathbf{M}^{pc})\right),\\
\mathbf{Q}^{(\ell)}
&=\operatorname{LN}\!\left(\bar{\mathbf{Q}}^{(\ell)}
+\operatorname{FFN}(\bar{\mathbf{Q}}^{(\ell)})\right).
\end{aligned}
\]

其中 \(\operatorname{MHA}(Q,K,V)\) 表示多头注意力；第一行对应 query self-attention，第二行对应 query-to-memory cross-attention，第三行对应 FFN 更新。

## 拟修改英文

Keep the current explanatory sentence and replace the placeholder formula with:

\[
\begin{aligned}
\tilde{\mathbf{Q}}^{(\ell)}
&=\operatorname{LN}\!\left(\mathbf{Q}^{(\ell-1)}
+\operatorname{MHA}(\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)},\mathbf{Q}^{(\ell-1)})\right),\\
\bar{\mathbf{Q}}^{(\ell)}
&=\operatorname{LN}\!\left(\tilde{\mathbf{Q}}^{(\ell)}
+\operatorname{MHA}(\tilde{\mathbf{Q}}^{(\ell)},\mathbf{M}^{pc},\mathbf{M}^{pc})\right),\\
\mathbf{Q}^{(\ell)}
&=\operatorname{LN}\!\left(\bar{\mathbf{Q}}^{(\ell)}
+\operatorname{FFN}(\bar{\mathbf{Q}}^{(\ell)})\right).
\end{aligned}
\]

Then add:

Here, \(\operatorname{MHA}(Q,K,V)\) denotes multi-head attention. The three lines correspond to query self-attention, query-to-memory cross-attention, and FFN updating, respectively.

## 是否写 decoder block 数量

不建议写具体数量。

理由：

长版中写了当前实现使用 2 个 QueryDecoderBlock，但短版当前没有集中列所有层数超参数。只补公式结构即可，不额外引入实现细节。

## 拟执行动作

如确认：

1. 修改 `sections_short/method_zh.tex`。
2. 同步修改 `sections_short/method_en.tex`。
3. 编译中英文短版。
