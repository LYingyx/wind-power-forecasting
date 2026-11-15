# 删除冗余表述与 Query Decoder 表述调整

## 1. 删除 related work 中的 prompt learning / 外部检索说明

涉及文件：

- `sections_short/related_work_zh.tex`
- `sections_short/related_work_en.tex`

原中文句子：

整个上下文生成过程仅依赖样本内功率状态，不涉及语言 prompt learning 或外部检索增强。

拟处理：

直接删除该句。

对应英文句子：

The context-generation process relies only on within-sample power state and involves no language prompt learning or external retrieval augmentation.

拟处理：

直接删除该句。

理由：

这句话是在回应可能的 prompt / retrieval 概念混淆，但正文并没有真正展开相关方法，放在 related work 末尾会显得防御性较强，也会把论文焦点带偏。删除后段落更集中于本文的 memory-level 气象读取融合。

## 2. 删除 problem definition 中“并覆盖更多预测起点”

涉及文件：

- `sections_short/problem_zh.tex`

原中文句子：

训练阶段采用滑动窗口样本，以提高样本数量并覆盖更多预测起点；验证和测试阶段采用严格日样本，以模拟固定起报的业务预测流程。所有阶段均只使用目标窗口开始前真实可用的信息，避免未来功率泄漏。

拟改为：

训练阶段采用滑动窗口样本，以提高样本数量；验证和测试阶段采用严格日样本，以模拟固定起报的业务预测流程。所有阶段均只使用目标窗口开始前真实可用的信息，避免未来功率泄漏。

英文同步检查：

当前英文 `problem_en.tex` 中没有完全对应的“cover more forecast origins”表述，如无新增冗余句，不需要改英文。

理由：

“覆盖更多预测起点”和“提高样本数量”在这里含义接近，前者增加解释负担但不增加核心信息。删除后更简洁。

## 3. Query Decoder 是否强调“24 个可学习 future queries”

涉及文件：

- `sections_short/method_zh.tex`
- `sections_short/method_en.tex`

当前中文总览句：

Query Decoder 使用 24 个可学习 future queries 从融入运行状态约束的融合记忆中并行读取信息，得到 24 步区域集群预测序列。

当前英文总览句：

The Query Decoder uses 24 learnable future queries to read information in parallel from the fusion memory with operating-state constraints and produces a 24-step regional cluster power forecast.

建议：

保留“24 个可学习 future queries”，但只在方法总览和 Query Decoder 小节中各出现一次，不再扩展解释。

理由：

这里的“24 个”有必要，因为论文任务是 24 h 多步预测，每个 future query 对应一个预测 horizon，可以说明模型是直接并行多步解码，而不是自回归滚动预测。“可学习”也有必要，因为它说明这些 queries 不是由输入序列直接拷贝得到，而是作为解码端的 horizon-specific 查询向量学习得到。

但不建议继续强调或反复解释，因为方法小节后文已经有：

Query Decoder 维护 \(24\) 个 learnable future queries。

因此总览句保留即可，后文公式处承担技术定义，不需要再增加说明。

## 拟执行动作

1. 删除 related work 中 prompt learning / 外部检索增强句，中英文同步。
2. 删除 problem definition 中“并覆盖更多预测起点”。
3. Query Decoder 总览句暂不改，保留“24 个可学习 future queries”。
4. 不编译，除非后续明确要求。
