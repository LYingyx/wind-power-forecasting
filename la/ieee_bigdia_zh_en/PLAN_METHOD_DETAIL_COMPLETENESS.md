# 方法部分细节完整性检查与修改建议

## 背景

用户指出 `V. 运行状态约束的气象-功率融合网络` 中有四个细节问题：

1. 气象记忆构建中的几个位置编码是否需要解释具体做法。
2. GCN 公式 6、7 是否需要保留。
3. 公式 15 中 \(g_\phi\) 没有解释。
4. `DecoderBlock` 是否需要展开。

参考原版：

- `build/manuscript_zh.pdf`
- 对应源码：`sections/methodology_zh.tex`

当前短版：

- 中文：`sections_short/method_zh.tex`
- 英文：`sections_short/method_en.tex`

## 1. 气象记忆位置编码

### 当前情况

短版已经写了两类位置/阶段信息：

- 编码前：\(\mathbf{e}^{time}_{t}\) 和 \(\mathbf{e}^{stage}_{t}\)，用于逐帧气象特征。
- memory 网格后：\(\mathbf{p}^{time}_{\tau}\)、\(\mathbf{p}^{row}_{i}\)、\(\mathbf{p}^{col}_{j}\)，相加得到 \(\mathbf{E}^{pos}_{\tau,i,j}\)。

### 判断

需要解释，但不需要展开成原版那么长。

当前文字已经解释了 `time`、`stage` 的含义，但对 memory 网格里的 `time/row/col` 位置嵌入解释偏短。读者能看出是相加，但不一定知道它们分别对应池化后网格的哪个轴。

### 建议改法

在公式 \(\mathbf{E}^{pos}_{\tau,i,j}\) 后补一句：

其中 \(\mathbf{p}^{time}_{\tau}\)、\(\mathbf{p}^{row}_{i}\) 和 \(\mathbf{p}^{col}_{j}\) 分别对应 memory 网格的时间索引、纬向/行索引和经向/列索引，并通过广播加到每个时空 memory token 上。

英文同步：

Here, \(\mathbf{p}^{time}_{\tau}\), \(\mathbf{p}^{row}_{i}\), and \(\mathbf{p}^{col}_{j}\) correspond to the temporal, row, and column indices of the pooled memory grid, respectively, and are broadcast to each spatiotemporal memory token.

### 理由

这样可以补足“具体怎么做”，但只增加一小句，不会把方法部分写成实现说明。

## 2. GCN 公式 6、7 是否保留

### 当前情况

短版保留了：

- 公式 6：自环增强和对称归一化。
- 公式 7：图卷积计算。

后文 point、patch、global token 的公式直接调用 \(\operatorname{GCN}_{pt}\)、\(\operatorname{GCN}_{pa}\)、\(\operatorname{GCN}_{gl}\)。

### 判断

建议保留，但可以补一句解释 \(\mathbf{H}\)、\(\mathbf{W}\)、\(\mathbf{b}\)。

### 理由

如果删掉公式 6、7，后面的 \(\operatorname{GCN}\) 操作会变成黑箱，功率状态编码的图结构来源也会弱化。由于本文方法名和贡献强调运行状态约束，而这个约束来自场站级功率及场站间耦合关系，GCN 的基本定义应保留。

更稳的短版写法是在公式 7 后补一句：

其中 \(\mathbf{H}\) 为节点特征，\(\mathbf{W}\) 和 \(\mathbf{b}\) 为图卷积层的可学习权重和偏置。

英文同步：

Here, \(\mathbf{H}\) denotes node features, and \(\mathbf{W}\) and \(\mathbf{b}\) are the learnable weight and bias of the graph convolution layer.

## 3. 公式 15 中 \(g_\phi\) 需要解释

### 当前情况

短版公式 15：

\(\hat{\mathbf{y}}=g_{\phi}(\operatorname{LN}(\mathbf{Q}^{(L_d)}))\).

但没有说明 \(g_\phi\) 是什么。

### 判断

必须补。这个属于符号定义缺失。

### 建议改法

在公式 15 后加一句：

其中 \(g_{\phi}\) 为预测头，将每个 horizon 的 query 表示映射为对应预测步的区域聚合功率；实现上由线性层、GELU、dropout 和输出线性层组成。

英文同步：

Here, \(g_{\phi}\) is the prediction head that maps each horizon query representation to the corresponding regional aggregate power output, and is implemented by a linear layer, GELU, dropout, and an output linear layer.

### 理由

这句直接来自原版，但比原版更紧凑。它补足符号定义，也解释了输出维度逻辑。

## 4. DecoderBlock 是否需要展开

### 当前情况

短版只写：

\(\mathbf{Q}^{(\ell)}=\operatorname{DecoderBlock}(\mathbf{Q}^{(\ell-1)},\mathbf{M}^{pc})\).

原版展开为三步：

- query self-attention
- query-to-memory cross-attention
- FFN

并说明当前实现使用 2 个 QueryDecoderBlock。

### 判断

需要解释 DecoderBlock 的内部组成，但不建议把原版三行公式完整搬回短版。

### 理由

英文短版当前正好 8 页。完整展开三行公式会占比较多空间，容易超过 8 页。更稳妥的写法是保留紧凑公式，并在公式前后加一句说明：

每个 decoder block 依次包含 query self-attention、query-to-memory cross-attention 和 FFN，并配合残差连接和 LayerNorm；其中 self-attention 建模不同预测步之间的依赖，cross-attention 从 \(\mathbf{M}^{pc}\) 读取气象记忆和功率上下文。

英文同步：

Each decoder block consists of query self-attention, query-to-memory cross-attention, and an FFN with residual connections and LayerNorm. The self-attention models dependencies among forecast horizons, while the cross-attention reads weather memory and power context from \(\mathbf{M}^{pc}\).

### 是否写“2 个 decoder block”

不建议在短版中写具体层数。

理由：

具体层数是实现超参数，和核心方法逻辑关系不大。实验设置里已经放了一些必要超参数，方法部分如果继续加入层数，会增加实现细节负担。除非审稿人需要复现级细节，否则短版写 block 组成即可。

## 总体建议

建议做以下最小修改：

1. 气象 memory 位置嵌入后补一句，说明 time/row/col 三类 embedding 的轴向含义和广播相加方式。
2. 保留 GCN 公式 6、7，并在公式 7 后补一句解释 \(\mathbf{H}\)、\(\mathbf{W}\)、\(\mathbf{b}\)。
3. 在公式 15 后补一句解释 \(g_\phi\)。
4. 不展开三行 DecoderBlock 公式，只用一句话说明 block 包含 self-attention、cross-attention、FFN、残差和 LayerNorm。

## 拟执行动作

如确认修改：

1. 修改 `sections_short/method_zh.tex`。
2. 同步修改 `sections_short/method_en.tex`。
3. 编译中文和英文短版。
4. 检查页数，重点确认英文仍为 8 页。
