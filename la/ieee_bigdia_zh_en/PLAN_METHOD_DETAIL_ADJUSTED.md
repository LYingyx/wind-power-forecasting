# 方法细节修改方案：按最新意见收敛

## 用户确认后的处理原则

1. 气象记忆位置编码不改。
2. GCN 作为通识图卷积操作，不再保留完整公式解释。
3. 补充 \(g_\phi\) 的定义。
4. 补充 `DecoderBlock` 的组成，但不展开成多行公式。

## 1. 气象记忆位置编码

不修改。

理由：

当前短版已经写出时间、行、列三类可学习位置嵌入相加：

\[
\mathbf{E}^{pos}_{\tau,i,j}=
\mathbf{p}^{time}_{\tau}+\mathbf{p}^{row}_{i}+\mathbf{p}^{col}_{j}.
\]

这对会议短文已经足够，不再增加实现说明。

## 2. GCN 公式处理

涉及文件：

- `sections_short/method_zh.tex`
- `sections_short/method_en.tex`

### 当前中文

图卷积使用自环增强和对称归一化：

\[
\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I},\quad
\hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2},
\]

\[
\operatorname{GCN}(\mathbf{H})=\hat{\mathbf{A}}\mathbf{H}\mathbf{W}+\mathbf{b}.
\]

### 拟改中文

功率状态编码器采用带自环增强和对称归一化邻接矩阵的 GCN 提取场站间耦合关系。

### 当前英文

Graph convolution uses self-loop augmentation and symmetric normalization:

\[
\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I},\quad
\hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2},
\]

\[
\operatorname{GCN}(\mathbf{H})=\hat{\mathbf{A}}\mathbf{H}\mathbf{W}+\mathbf{b}.
\]

### 拟改英文

The power-state encoder uses GCNs with self-loop augmentation and symmetrically normalized adjacency to extract inter-station coupling.

### 理由

GCN 与 CNN 类似，属于常见基础操作。本文重点不是重新解释图卷积，而是说明近期场站功率如何被编码成运行状态约束。因此删除完整 GCN 公式更合适，也能节省篇幅。

注意：删除两个公式会导致后续公式编号自动前移，这是 LaTeX 正常行为。正文目前没有用“公式 6/7”硬引用，因此不会造成引用错误。

## 3. 补充 \(g_\phi\)

涉及位置：

公式

\[
\hat{\mathbf{y}}=g_{\phi}(\operatorname{LN}(\mathbf{Q}^{(L_d)})).
\]

### 拟补中文

其中 \(g_{\phi}\) 为预测头，将每个 horizon 的 query 表示映射为对应预测步的区域聚合功率；实现上由线性层、GELU、dropout 和输出线性层组成。

### 拟补英文

Here, \(g_{\phi}\) is the prediction head that maps each horizon query representation to the corresponding regional aggregate power output, and is implemented by a linear layer, GELU, dropout, and an output linear layer.

### 理由

这是符号定义缺失，必须补。

## 4. 补充 DecoderBlock 组成

涉及位置：

当前句子：

Query Decoder 维护 \(24\) 个 learnable future queries。第 \(\ell\) 层 decoder block 通过 query self-attention 建模不同预测步之间的依赖，并通过 cross-attention 读取融合记忆：

### 拟改中文

Query Decoder 维护 \(24\) 个 learnable future queries。每个 decoder block 由 query self-attention、query-to-memory cross-attention 和 FFN 组成，并配合残差连接和 LayerNorm；其中 self-attention 建模不同预测步之间的依赖，cross-attention 从融合记忆 \(\mathbf{M}^{pc}\) 读取气象记忆和功率上下文：

### 拟改英文

The Query Decoder maintains 24 learnable future queries. Each decoder block consists of query self-attention, query-to-memory cross-attention, and an FFN with residual connections and LayerNorm. The self-attention models dependencies among forecast horizons, while the cross-attention reads weather memory and power context from the fused memory \(\mathbf{M}^{pc}\):

### 理由

这样能解释 `DecoderBlock` 的内部含义，但不把原版三行公式搬回来，避免英文短版超过 8 页。

## 拟执行动作

如确认执行：

1. 删除短版中 GCN 的两个公式，并替换为一句通识实现说明。
2. 补充 `DecoderBlock` 的组成说明。
3. 补充 \(g_\phi\) 的定义。
4. 中英文同步修改。
5. 编译中英文短版，并检查英文是否仍为 8 页。
