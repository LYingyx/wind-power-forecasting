# 中英文同步润色计划

## 修改原则

- 只做局部措辞润色，不调整作者、参考文献、字号、图表和实验数值。
- 中文仍作为主稿；英文只做对应同步，避免出现中文有、英文无或英文含义偏移。
- 修改后重新编译中英文 PDF，并检查页数和关键文本。

## 计划修改项

### 1. 摘要中 H1 误差表述

位置：
- `manuscript_en_short.tex` 摘要
- 可选同步：`manuscript_zh_short.tex` 摘要

英文原句：

> with first-hour MAE reduced by about 42.4\%.

英文拟改：

> with first-horizon MAE reduced by about 42.4\%.

说明：`first-horizon` 与正文中的 H1/forecast step 表述更一致，避免 `first-hour` 被理解为一个连续小时区间。

中文建议同步为：

> 首个预测步 MAE 降低约 42.4\%。

如果希望保持中文原有表达，也可以仅改英文；但从中英文一致性看，建议同步改为“首个预测步”。

### 2. 引言中 regional weather propagation

位置：
- `sections_short/introduction_en.tex`

英文原句：

> Weather propagation, synchronized ramps, and curtailment recovery can all cause substantial forecast errors

英文拟改：

> Regional weather propagation, synchronized ramps, and curtailment recovery can all cause substantial forecast errors

说明：中文为“区域天气传播”，英文加回 `Regional`，语义更完整。中文原句不需要改。

### 3. 第 II 节“信息分工”表述

位置：
- `sections_short/related_work_zh.tex`
- `sections_short/related_work_en.tex`

中文原句：

> 业务化风电预测中，未来气象驱动与预测起点运行状态之间的信息分工仍值得进一步刻画。

中文拟改：

> 业务化风电预测中，未来气象驱动与预测起点运行状态之间的信息分工仍需进一步显式刻画。

英文原句：

> the information division between future weather drivers and the forecast-origin operating state still needs to be characterized more explicitly.

英文拟改：

> the distinct roles of future weather drivers and the forecast-origin operating state still need to be formalized more explicitly.

说明：`distinct roles` 比 `information division` 更自然；`formalized` 更贴近“显式建模/刻画”的论文语境。

### 4. 中文摘要和引言的轻量润色

位置：
- `manuscript_zh_short.tex`
- `sections_short/introduction_zh.tex`
- 对应英文视情况微调，不强行重写。

摘要中文原句：

> 如果模型只关注未来气象，容易产生与真实运行状态不一致的偏差。

摘要中文拟改：

> 如果模型只关注未来气象，易产生与真实运行状态不一致的偏差。

说明：更书面，含义不变。

引言中文原句：

> 在业务化区域风电集群日内预测中，未来网格化气象预报给出预测日内的风场演化，是区域功率变化的主要驱动；预测起点前的场站级实测功率则记录了当前出力水平、短期惯性和多场站同步变化，是模型读取未来气象时需要参照的运行状态约束。

引言中文拟改：

> 在业务化区域风电集群日内预测中，未来网格化气象预报给出预测日内的风场演化，是区域功率变化的主要驱动。预测起点前的场站级实测功率则记录当前出力水平、短期惯性和多场站同步变化，是模型读取未来气象时需要参照的运行状态约束。

说明：拆分长句，降低阅读负担；删去“记录了”的“了”，语气更稳。

引言中文原句：

> 本文关注的问题是：如何在未来气象主驱动预测中引入预测起点前可用的真实运行状态，使模型在读取未来气象时同时保持对当前场群状态的感知。

引言中文拟改：

> 本文旨在解决的问题是：如何在未来气象主驱动预测中引入预测起点前可用的真实运行状态，使模型在读取未来气象时同时保持对当前场群状态的感知。

说明：比“本文关注的问题是”略更主动，仍保持原意。

### 5. “参与未来气象读取”改为更明确的约束关系

位置：
- `sections_short/introduction_zh.tex`
- `sections_short/related_work_zh.tex`
- `sections_short/related_work_en.tex`

引言中文原句：

> 为使预测起点前的功率状态参与未来气象读取

引言中文拟改：

> 为使预测起点前的功率状态约束未来气象读取

相关工作中文原句：

> 本文则关注真实功率状态如何作为运行状态约束参与未来气象读取。

相关工作中文拟改：

> 本文则关注真实功率状态如何作为运行状态约束来约束未来气象读取。

英文原句：

> This paper instead focuses on how true power states participate in future-weather reading as operating-state constraints.

英文拟改：

> This paper instead focuses on how true power states constrain future-weather reading as operating-state constraints.

说明：用“约束/constrain”替代“参与/participate”，更贴合本文主线“运行状态约束”，也更准确描述 memory-level 注入的作用。

## 本轮不计划修改

- 不全篇统一“出力/功率”，避免破坏行业表达和现有语感。
- 不大幅削减“当前出力水平、短期惯性和多场站同步变化”这一核心三元组，只做局部长句和措辞优化。
- 不调整消融实验段落的结构和句式，避免引入额外版面变化。
- 不动作者、参考文献、字号、页边距和图表排版。
