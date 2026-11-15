# 短句开头与单句段落问题检查

目标：处理“段落开头先抛一个很短概念句，再用后文解释”的写法。重点不是机械删除所有短句，而是避免正文呈现提纲式、补丁式表达。

## 需要修改的 3 处

### 1. 方法：整体框架末尾单句段落

文件：sections_short/method_zh.tex

位置：整体框架小节，第 9 行附近

原文上下文：

本文方法采用运行状态约束的非对称气象-功率融合设计，如图~\ref{fig:framework} 所示。Weather Encoder 将后报气象和未来预报气象编码为 weather memory tokens，作为预测窗口内的主驱动信息；Power-State Encoder 从目标窗口前可用的场站级功率中提取当前场群状态，并生成 dynamic power context tokens；Query Decoder 使用 24 个可学习 future queries 从融入运行状态约束的融合记忆中并行读取信息，得到 24 步区域集群预测序列。

在该框架中，气象特征先被组织为可查询记忆，预测起点前的功率状态再被转换为同一隐空间中的上下文 token，并写入 future queries 的读取路径。传统拼接式融合通常在编码后或预测头前合并两类特征，模型需要自行学习二者的角色差异；本文通过 memory-level 写入，使每个未来预测步在 cross-attention 读取气象信息时都能显式访问预测起点状态。

因此，本文将功率状态定义为面向未来气象读取过程的运行状态约束。

下一段：

气象记忆构建过程如图~\ref{fig:weather-memory} 所示。Weather Encoder 的输入为 \(\mathbf{X}=[\mathbf{X}^{h};\mathbf{X}^{f}]\)，其中 \(\mathbf{X}^{h}\) 为目标窗口前的后报气象，\(\mathbf{X}^{f}\) 为目标窗口内的未来预报气象。

问题：

这句话单独成段，像给概念下标签，和前一段的机制说明割裂。它本身内容重要，但不适合独立作为短段落。

准备改法：

并入上一段末尾，让“气象记忆、功率上下文 token、cross-attention 读取路径”和“运行状态约束”的定义自然连在一起。

建议改为：

本文方法采用运行状态约束的非对称气象-功率融合设计，如图~\ref{fig:framework} 所示。Weather Encoder 将后报气象和未来预报气象编码为 weather memory tokens，作为预测窗口内的主驱动信息；Power-State Encoder 从目标窗口前可用的场站级功率中提取当前场群状态，并生成 dynamic power context tokens；Query Decoder 使用 24 个可学习 future queries 从融入运行状态约束的融合记忆中并行读取信息，得到 24 步区域集群预测序列。

在该框架中，气象特征先被组织为可查询记忆，预测起点前的功率状态再被转换为同一隐空间中的上下文 token，并写入 future queries 的读取路径。传统拼接式融合通常在编码后或预测头前合并两类特征，模型需要自行学习二者的角色差异；本文通过 memory-level 写入，使每个未来预测步在 cross-attention 读取气象信息时都能显式访问预测起点状态，并由此将功率状态定义为面向未来气象读取过程的运行状态约束。

下一段保持不变：

气象记忆构建过程如图~\ref{fig:weather-memory} 所示。Weather Encoder 的输入为 \(\mathbf{X}=[\mathbf{X}^{h};\mathbf{X}^{f}]\)，其中 \(\mathbf{X}^{h}\) 为目标窗口前的后报气象，\(\mathbf{X}^{f}\) 为目标窗口内的未来预报气象。

同步英文方向：

删除英文中单独的 “Thus, this work defines ...” 段，把定义并入上一段末尾。

### 2. 方法：Memory-Level 融合开头

文件：sections_short/method_zh.tex

位置：Memory-Level 融合与查询解码小节，第 88 行附近

原文上下文：

其中 \(J\) 为 patch 数量，\(f_p\) 为投影头。预测目标面向区域聚合出力，功率编码器的输出为样本特定的运行状态表示，压缩当前出力水平、局部变化和场站间耦合关系，供后续气象读取过程使用。

\figonelabel{\columnwidth}{../../paper_image_use/method/Fig04_svg.pdf}{功率状态编码。}{多尺度 GCN 从近期场站功率中提取当前运行状态、局部变化和多场站耦合关系。}{fig:power-state}

\subsection{Memory-Level 融合与查询解码}

本文的关键融合机制是在 memory 层写入功率状态。Power-to-context adapter 将功率状态向量映射为 \(K\) 个动态功率上下文 token：

问题：

第一句是典型“先抛概念短句，再解释”的写法，而且 “关键融合机制” 这种概括放在段首显得硬。这里可以直接从 adapter 如何把功率状态写入 memory 讲起。

准备改法：

把机制名词融入动作描述，直接写 adapter 的功能。

建议改为：

其中 \(J\) 为 patch 数量，\(f_p\) 为投影头。预测目标面向区域聚合出力，功率编码器的输出为样本特定的运行状态表示，压缩当前出力水平、局部变化和场站间耦合关系，供后续气象读取过程使用。

\figonelabel{\columnwidth}{../../paper_image_use/method/Fig04_svg.pdf}{功率状态编码。}{多尺度 GCN 从近期场站功率中提取当前运行状态、局部变化和多场站耦合关系。}{fig:power-state}

\subsection{Memory-Level 融合与查询解码}

为在 memory 层写入功率状态，Power-to-context adapter 将功率状态向量映射为 \(K\) 个动态功率上下文 token：

后文保持不变：

随后将其前置到气象记忆：

同步英文方向：

将 “The key fusion mechanism writes the power state at the memory level.” 改为 “To write the power state at the memory level, a power-to-context adapter maps ...”

### 3. 实验：数据与设置中的单句段落

文件：sections_short/experiments_zh.tex

位置：数据与设置小节，第 9 行附近

原文上下文：

本文基于吉林省区域风电场群构建实验数据集，预测对象为 58 个风电场站形成的区域聚合出力，并配套使用覆盖研究区域的 Pangu-Weather 网格化气象预报。气象网格大小为 \(19\times33\)，输入通道数为 17，包括 10 m 风场、不同气压层风场、温度、湿度、位势高度和海平面气压等变量。所有功率和气象数据均按北京时间对齐。

训练集使用 2022 年滑动窗口样本，以增加训练样本数量并覆盖更多天气和运行状态；验证集和测试集使用 2023 年严格日样本，其中每月前 10 天作为验证集，其余日期作为测试集。所有模型采用相同的数据划分、归一化方式和 24 h 预测目标。评价指标包括归一化 MAE、RMSE 和 \(R^2\)。

本文方法使用目标窗口前 16 步场站级功率作为运行状态上下文。

下一段：

对比模型包括 Weather-only 以及多种网格气象编码基线。Weather-only 与本文方法共享气象输入和 query decoder，但不使用功率状态条件，用于直接检验功率状态融合的增益。

问题：

这是一个单句段落，像后补的实验细节。它应该并入前面的数据划分和评价设置段，形成完整的实验设置信息。

准备改法：

并入前一段末尾，同时保持简洁。

建议改为：

本文基于吉林省区域风电场群构建实验数据集，预测对象为 58 个风电场站形成的区域聚合出力，并配套使用覆盖研究区域的 Pangu-Weather 网格化气象预报。气象网格大小为 \(19\times33\)，输入通道数为 17，包括 10 m 风场、不同气压层风场、温度、湿度、位势高度和海平面气压等变量。所有功率和气象数据均按北京时间对齐。

训练集使用 2022 年滑动窗口样本，以增加训练样本数量并覆盖更多天气和运行状态；验证集和测试集使用 2023 年严格日样本，其中每月前 10 天作为验证集，其余日期作为测试集。所有模型采用相同的数据划分、归一化方式和 24 h 预测目标。评价指标包括归一化 MAE、RMSE 和 \(R^2\)。本文方法使用目标窗口前 16 步场站级功率作为运行状态上下文。

下一段保持不变：

对比模型包括 Weather-only 以及多种网格气象编码基线。Weather-only 与本文方法共享气象输入和 query decoder，但不使用功率状态条件，用于直接检验功率状态融合的增益。

同步英文方向：

删除英文中单独的 “The proposed method uses ...” 段，把该句并入上一段。

## 暂不修改的短句

以下句子虽然位于段首或较短，但属于正常 topic sentence，不建议机械修改：

- related_work_zh.tex：未来气象预报是多步风电预测中的主要外生信息。
- related_work_zh.tex：Transformer 及其变体为多步时间序列预测提供了并行建模基础。
- experiments_zh.tex：各消融变体用于分别检验功率状态、功率编码结构和气象记忆的作用。
- experiments_zh.tex：表~\ref{tab:ablation-short} 给出了关键消融结果。

这些句子承担段落主题引入功能，后文自然展开，和“突兀概念短句”不同。

## 执行顺序

1. 先改中文正文三处。
2. 同步英文对应三处。
3. 重新检索单句段落和短句开头。
4. 编译中英文短稿，确认 PDF 正常生成并检查英文页数。
