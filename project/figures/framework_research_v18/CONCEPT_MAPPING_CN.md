# 概念图与方法定义的对应

本版以矩形功能模块组织信息流，模块内部保留概念示意，不逐层展开计算图。以下对应关系用于核对图像，不能将示意形状读作新增模块。

| 图中概念 | 方法中的对应 |
| --- | --- |
| Historical evidence | 本站与区域历史观测；冻结基准使用两者，修正侧历史编码器使用本站历史 |
| History-based reference | 冻结历史预测器 B0 产生的参考轨迹 b |
| Historical context | 独立历史编码器的完整记忆 M；摘要取最后一个上下文化 token |
| Reference condition | 历史摘要、基准值、基准邻步变化和提前量构成的条件表示 |
| Future weather | 起点可用的 NWP，经空间采样、时间对齐与特征投影得到天气特征 |
| Feature-space adjustment | 条件与天气共同形成的隐特征调整；不表示修改物理天气 |
| Direct weather features | 天气特征直接参与未来特征构造的保留路径 |
| Lead / site / phase | 目标上下文；不是天气观测 |
| History-aware readout | 概念上合并未来特征融合、读取完整历史的解码器及功率读出；不是新增的单一网络层 |
| Power correction | 可正可负的容量因子修正 c |
| Wind-power forecast | b+c；基准输出支路直接接入最终求和 |

必须保留：参考的条件/输出双路径，历史的摘要/完整记忆双路径，天气直接路径，以及最终求和恰好两项输入。

表示方式：功能模块有明确边界，模块间连线使用水平、垂直与直角折线；不显示 feature plus、decoder 层框、token 方格或单独的局部计算面板。图中天气切片、特征等高线、历史带与曲线均为定性示意，不表示经验数据、注意力权重、置信区间或物理机制。
