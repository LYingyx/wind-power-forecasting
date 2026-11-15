# 方法整体框架第二段改写方案

## 问题位置

涉及文件：

- `sections_short/method_zh.tex`
- `sections_short/method_en.tex`

位置：

`III/IV` 方法部分的 `整体框架 / Overall Framework` 小节第二段。

## 原中文段落

在该框架中，气象特征先被组织为可查询记忆，预测起点前的功率状态再被转换为同一隐空间中的上下文 token，并写入 future queries 的读取路径。传统拼接式融合通常在编码后或预测头前合并两类特征，模型需要自行学习二者的角色差异；本文通过 memory-level 写入，使每个未来预测步在 cross-attention 读取气象信息时都能显式访问预测起点状态，并由此将功率状态定义为面向未来气象读取过程的运行状态约束。

## 主要问题

1. 逻辑顺序不顺。段落先描述“写入 future queries 的读取路径”，但这个动作为什么重要还没有铺垫。

2. “传统拼接式融合”突然插入，造成段落从方法说明跳到对比批评，再跳回本文方法，阅读节奏断裂。

3. “并由此将功率状态定义为……”像是在通过模型操作反向定义概念。更自然的写法应该是先说明功率状态在业务时间轴上的角色，再说明模型如何实现这个角色。

4. 长句过多，多个抽象词连续出现，包括“可查询记忆”“同一隐空间”“读取路径”“memory-level 写入”“运行状态约束”，压缩在一段里会显得概念堆叠。

## 拟改中文段落

该框架的融合逻辑是在气象记忆的读取阶段引入预测起点前的功率状态，从结构上区分两类信息在业务时间轴上的角色：未来气象提供目标窗口内的主驱动，近期功率提供预测起点处的运行状态约束。Weather Encoder 保留目标窗口前后气象场的时空结构；Power-State Encoder 将近期场站功率压缩为样本特定的动态上下文 token，并映射到与气象记忆一致的隐空间。Query Decoder 面向不同预测步读取融合记忆时，可以同时利用未来气象演化和当前场群状态，使运行状态约束贯穿并行 24 步解码过程。

## 对应英文同步改写

The fusion logic of this framework introduces the pre-origin power state at the weather-memory reading stage, thereby separating the roles of the two information sources along the operational time axis: future weather provides the main driver within the target window, while recent power provides the operating-state constraint at the forecast origin. The Weather Encoder preserves the spatiotemporal structure of meteorological fields before and within the target window. The Power-State Encoder compresses recent station-level power into sample-specific dynamic context tokens and maps them into the same latent space as the weather memory. When the Query Decoder reads the fused memory for different forecast horizons, it can use both future weather evolution and the current cluster state, so that the operating-state constraint is carried through the parallel 24-step decoding process.

## 修改理由

这版按“业务角色 → 结构实现 → 解码效果”的顺序展开：

- 先说明为什么要在气象记忆读取阶段引入功率状态。
- 再说明 Weather Encoder 和 Power-State Encoder 各自做什么。
- 最后说明 Query Decoder 如何从融合记忆中并行读取。

同时删除“传统拼接式融合”这一句，避免段落中途切换到对比批评。这个对比已经在引言和 related work 中出现过，方法总览里不需要重复。

## 拟执行动作

如确认，下一步只替换 `method_zh.tex` 和 `method_en.tex` 中该段，不移动图、不调整公式、不编译，除非另行要求编译。
