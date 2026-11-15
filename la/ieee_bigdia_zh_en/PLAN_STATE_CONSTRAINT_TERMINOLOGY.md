# “状态条件/条件化”残留统计与改写方案

本文当前标题和主线已经改为“运行状态约束”，因此正文中继续使用“状态条件”“条件化”等旧表述会造成概念不一致。下面统计只做文本检查和改写建议，尚未改动正文 `.tex` 文件。

## 统计范围

优先统计当前正在使用的 short 版本：

- `manuscript_zh_short.tex`
- `manuscript_en_short.tex`
- `sections_short/*.tex`

同时检查了旧版长文稿：

- `manuscript_zh.tex`
- `sections/*.tex`

旧版长文稿不是当前编译主版本，但其中仍保留大量“功率状态条件化”旧术语，后续如果还需要维护长版，也应同步清理。

## 当前 Short 版本统计

宽松关键词包括：

- 中文：`状态条件`、`条件化`、`功率状态条件`、`起点状态条件`
- 英文：`state condition`、`power-state condition(s)`、`forecast-origin state condition`、`conditional constraint`、`condition how`、`conditions future-query reading`

当前 short 版本共检出 15 处宽松匹配。

其中需要改写的旧术语共 13 处：

- `manuscript_zh_short.tex`：1 处
- `manuscript_en_short.tex`：1 处
- `sections_short/introduction_zh.tex`：2 处
- `sections_short/introduction_en.tex`：2 处
- `sections_short/related_work_en.tex`：1 处
- `sections_short/experiments_zh.tex`：3 处
- `sections_short/experiments_en.tex`：3 处
- `sections_short/conclusion_zh.tex`：1 处
- `sections_short/conclusion_en.tex`：1 处

其中可以保留的普通语义共 2 处：

- `manuscript_en_short.tex` 摘要里的 `initial condition of the weather-to-power mapping`
- `sections_short/conclusion_en.tex` 里的 `complex operating conditions`

这两处不是“状态条件化”术语，不会削弱“运行状态约束”的主线。

## 需要改写的具体位置

### 1. 中文摘要

文件：`manuscript_zh_short.tex`

原句：

在 memory 层面对 future queries 的气象读取过程进行条件化。

建议改为：

在 memory 层面向 future queries 的气象读取过程写入运行状态约束。

理由：

“条件化”是旧概念，建议改成“写入运行状态约束”，与方法机制和标题一致。

### 2. 英文摘要

文件：`manuscript_en_short.tex`

原句：

and uses these tokens to condition how future queries read weather memory.

建议改为：

and uses these tokens to impose operating-state constraints on how future queries read weather memory.

理由：

英文中保留 `condition how` 会继续对应中文“条件化”。用 `impose operating-state constraints` 更贴合标题。

保留句：

the observed power state before the forecast origin affects the initial condition of the weather-to-power mapping.

理由：

这里的 `initial condition` 是数学/物理意义上的初始条件，不是方法命名中的“状态条件化”，可以保留。

### 3. 中文引言

文件：`sections_short/introduction_zh.tex`

原句：

未来气象作为主驱动记忆，近期功率作为预测起点状态条件，对未来气象信息的读取过程进行约束。

建议改为：

未来气象作为主驱动记忆，近期功率作为预测起点处的运行状态约束，作用于未来气象信息的读取过程。

理由：

用户指出的问题就在这里。“预测起点状态条件”应直接改为“预测起点处的运行状态约束”。

原句：

提出运行状态约束的气象-功率融合网络，在 memory 层面注入动态功率上下文 token，以条件化 future queries 对气象记忆的读取。

建议改为：

提出运行状态约束的气象-功率融合网络，在 memory 层面注入动态功率上下文 token，使 future queries 在读取气象记忆时受到预测起点运行状态的约束。

理由：

保留 dynamic power context token 的机制，但把“条件化”改为“受到运行状态约束”。

### 4. 英文引言

文件：`sections_short/introduction_en.tex`

原句：

future weather acts as the driving memory, while recent power serves as a forecast-origin state condition that constrains how future weather information is read.

建议改为：

future weather acts as the driving memory, while recent power imposes an operating-state constraint at the forecast origin on how future weather information is read.

原句：

We propose an operating-state-constrained weather-power fusion network that injects dynamic power context tokens at the memory level to condition how future queries read weather memory.

建议改为：

We propose an operating-state-constrained weather-power fusion network that injects dynamic power context tokens at the memory level, so that future queries read weather memory under forecast-origin operating-state constraints.

理由：

英文主线统一为 `operating-state constraint(s)`，避免 `state condition` 和 `condition how`。

### 5. 英文 Related Work

文件：`sections_short/related_work_en.tex`

原句：

this paper uses power state as a conditional constraint during weather reading

建议改为：

this paper uses the observed power state as an operating-state constraint during weather reading

理由：

`conditional constraint` 不够明确，也会让读者联想到“条件化”。改为标题术语。

### 6. 中文实验设置

文件：`sections_short/experiments_zh.tex`

原句：

Weather-only 与本文方法共享气象输入和 query decoder，但不使用功率状态条件，用于直接检验功率状态融合的增益。

建议改为：

Weather-only 与本文方法共享气象输入和 query decoder，但不引入运行状态约束，用于直接检验功率状态融合的增益。

原句：

Weather-only 移除所有功率状态条件，MAE 增加 4.27\%。

建议改为：

Weather-only 移除功率侧运行状态约束，MAE 增加 4.27\%。

原句：

上述结果共同支持本文的非对称融合设定：气象提供未来主驱动，功率提供起点状态条件。

建议改为：

上述结果共同支持本文的非对称融合设定：气象提供未来主驱动，功率提供预测起点处的运行状态约束。

理由：

实验部分是支撑标题的关键位置，应明确说消融去掉的是“运行状态约束”，而不是“状态条件”。

### 7. 英文实验设置

文件：`sections_short/experiments_en.tex`

原句：

but does not use the power-state condition

建议改为：

but does not introduce the operating-state constraint from power observations

原句：

Removing all power-state conditions in Weather-only increases MAE by 4.27\%.

建议改为：

Removing the power-side operating-state constraint in Weather-only increases MAE by 4.27\%.

原句：

weather provides the future driver, while power provides the forecast-origin state condition.

建议改为：

weather provides the future driver, while power provides the operating-state constraint at the forecast origin.

理由：

英文实验部分同步中文，统一为 `operating-state constraint`。

### 8. 中文结论

文件：`sections_short/conclusion_zh.tex`

原句：

并通过 memory-level fusion 条件化 future queries 的读取过程。

建议改为：

并通过 memory-level fusion 将运行状态约束写入 future queries 的读取过程。

理由：

结论中应回扣标题，而不是回到“条件化”。

### 9. 英文结论

文件：`sections_short/conclusion_en.tex`

原句：

and conditions future-query reading through memory-level fusion.

建议改为：

and imposes operating-state constraints on future-query reading through memory-level fusion.

保留句：

complex operating conditions

理由：

这里的 `conditions` 表示“运行场景/工况”，不是旧术语。

## 旧版长文稿统计

旧版长文稿共检出 21 处宽松匹配，主要集中在：

- `manuscript_zh.tex`：5 处
- `sections/introduction_zh.tex`：7 处
- `sections/related_work_zh.tex`：7 处
- `sections/methodology_zh.tex`：6 处
- `sections/case_study_zh.tex`：4 处
- `sections/conclusion_zh.tex`：2 处
- `sections/references_zh.tex`：1 处

这里的 `sections/references_zh.tex` 是文献题名中的 `conditioning layer`，不建议改。

旧版长文稿如果不再用于提交，可以先不处理；如果后续还要保留长版 PDF，则应按 short 版同样原则统一为“运行状态约束”。

## 总体改写原则

1. 中文主术语统一为“运行状态约束”。

2. 不再使用“状态条件”“功率状态条件”“起点状态条件”“状态条件化”“条件化读取”等表述。

3. dynamic power context tokens 的角色不改，但解释方式改为：

动态功率上下文 token 是由预测起点前真实功率状态生成的模型载体，用于在 memory 层面向气象读取过程写入运行状态约束。

4. 英文主术语统一为 `operating-state constraint(s)` 和 `operating-state-constrained`。

5. 英文中避免 `state condition`、`power-state condition`、`condition how future queries read...`。

6. 普通语义的 `initial condition` 和 `complex operating conditions` 可以保留，不属于术语冲突。

## 建议修改顺序

1. 先改 short 版本中 13 处术语冲突。

2. 重新 `rg` 检查，目标是 short 版本中只剩下 2 处可保留的普通 `condition` 语义。

3. 编译中文和英文 short PDF。

4. 如后续还维护旧版长文稿，再清理 `sections/` 和 `manuscript_zh.tex`。
