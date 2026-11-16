# Gemini / 编程助手阅读入口

更新：2026-11-16。先读本文，再进入长篇历史交接文档。

先读新增的[跨分辨率 A 与状态条件兑现 B 提案](STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md)：整理另一助手的证据、完整对照表、判定门槛，并区分原提案与实际执行。2026-11-16 17:43 状态：30 项 B 及归因对照仍在运行，15 项教师梯度隔离版 R4-ISO 已在三机启动；独立时间分配 A / A+B 尚未进入清单。该页是方案与带时间的执行说明，不是新候选已获胜的结果报告。

随后读 [v2 15:28 历史结果快照](../project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS_CN.md)及[实验反馈分析](../project/runs/daily_state_representation_20261116_v2/analysis/LEARNINGS_20261116_CN.md)。在该快照时点，25 项阶段一全部完成，R4 三组各完成 seeds 42、43、45，共 34 项新训练完成；另有 10 项旧权重诊断完成。当时尚在训练的 6 项旧 R4 后续已保存退出并由 R4-ISO 替代，5 项 `LOSS_TF` 已完成；不要将快照中的“正在训练”当作实时状态。旧 v2 实现、门槛与调度见 [v2 执行入口](../project/runs/daily_state_representation_20261116_v2/README_CN.md)。

为避免只看到摘要，已发布 [当前任务完整证据 JSON](../project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS.json)：包含已完成任务的配置、逐轮记录、选中轮次、逐风场/时效指标、诊断及来源哈希；[05 当前结果 TXT](gemini_plain_text/05_current_experiment_results.txt)包含这些正文，无需解压。未完成任务的进度与曲线不是最终成绩。

当前有用反馈：R3 对无状态 R2 的 RMSE 改善 0.156%，但对原模型 R0 退化 0.244%；R4 三种子对 R0 的 RMSE 退化 0.695%。R0 对同编码 TF_E2L 的平均 MAE/RMSE 改善 0.157%/0.158%，但仅 1/5 seeds 双指标同时改善。不能据内部消融获益就宣称整体方法优越。此处 R0 是 ENC_neutral_E2L，R3 是新增状态生成器，不是旧输入配置名。

需要审阅结果的产生过程时，继续读[结果到代码的完整导航](RESULTS_TO_CODE_CN.md)，包含数据准备、模型/基线、调参、训练选轮、指标汇总及全部 2571 项任务的训练记录。附件解析失败时使用[未压缩 TXT 入口](gemini_plain_text/README.md)；阅读目录名或结果摘要不能代替读取源文件正文。

## 研究目标与约束

- 目标是单风场功率预测，预测未来 0–6 h 的 24 个 15 min 点；共享训练仍可以输出单个风场预测。
- 希望找到相对强 Transformer 有可复核优势的路线，不能只挑表现好的站点或某一个指标。
- 天气编码（含 E2L/E2C）只作实现选择与消融，不作为机制贡献点或独立贡献点。
- 共享训练、逐站独立训练、已见风场微调、完全未见风场迁移是不同设定。现有逐站微调不等于新风场迁移。
- 数据只有实测功率、Pangu 预报和 ERA5 再分析；没有场站实测天气。历史 Pangu 节点不是 ERA5。

## 当前结果的证据边界

当前协议为每日 00 UTC Pangu、15 h 后可用（北京时间 23:00 切换），气象时效 9–45 h；邻站/区域功率可用至预测起点。55 个固定风场，2022 训练、2023 选择。2024 曾在历史研究中被查看，后续只能称探索性跨年复评，不能声称完全盲测。

此前一轮已完成 2571 项，见[完成审计](../project/analysis/daily_protocol_review_20261115/COMPLETION_AUDIT.json)。以下表格属于该轮，相对匹配 Transformer 的误差变化中负值为改善；口径详见[完整报告](../project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)。已发布 v2 固定结果见页首快照，不要混用两轮的任务计数。

| 方案 | MAE 相对变化 | RMSE 相对变化 | 解释 |
| --- | ---: | ---: | --- |
| 共享 W0_R3 | −0.183% | +0.227% | 两项指标未同时胜出 |
| 独立 W0_R3 | −0.501% | +0.239% | MAE 优势不能覆盖 RMSE 退化 |
| 完整 W0 逐站微调 | +0.108% | +0.370% | 不支持作为当前最优路线 |
| 共享 E2L | −0.457% | −0.216% | 高功率 RMSE +0.153%，未满足原先不超过 +0.1% 的条件 |

E2L 是多层无激活线性映射，其复合仍为仿射映射。最新 v2 已完成 TF_E2L 五种子交叉骨干对照：相对原 TF_R3，MAE −0.3005%、RMSE −0.0582%、高功率 RMSE +0.2801%。编码收益也出现在 TF，不能归为本骨干独有或核心机制。早期[编码角色说明](../project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md)中的“缺少同编码 TF 对照”是当时状态，现已补上 E2L 对照。

E2L 是七种编码中按 2023 结果挑出的，存在选择偏差；原 R0 对 TF_R3 的 RMSE 在 seed 43、46 上落后。TF_E2L 已完成，但十种子与 2024 探索性复评尚未完成；十种子本身不能消除选择偏差。

## 关键实现与证据

- 共享模型：[model.py](../project/runs/daily_protocol_restart_20261114_v1/code/model.py)、[train.py](../project/runs/daily_protocol_restart_20261114_v1/code/train.py)、[data.py](../project/runs/daily_protocol_restart_20261114_v1/code/data.py)。
- 消融与独立训练：[model.py](../project/runs/daily_ablation_restart_20261114_v1/code/model.py)、[train.py](../project/runs/daily_ablation_restart_20261114_v1/code/train.py)、[实验清单](../project/runs/daily_ablation_restart_20261114_v1/manifest.json)。
- 微调：[train.py](../project/runs/daily_site_finetune_20261114_v1/code/train.py)、[site_data.py](../project/runs/daily_site_finetune_20261114_v1/code/site_data.py)。
- 汇总程序：[review_daily_results_20261115.py](../project/scripts/review_daily_results_20261115.py)。
- 逐站逐种子证据：[结果目录](../project/analysis/daily_protocol_review_20261115/)、[逐风场对比](../project/analysis/daily_protocol_review_20261115/per_farm_comparisons.csv)、[逐种子对比](../project/analysis/daily_protocol_review_20261115/per_seed_comparisons.csv)。
- 最新待检验方案：[跨分辨率 A 与状态条件兑现 B](STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md)。早期[状态条件中间表示计划](../project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md)用于了解来历；方案及其解释都需要实验验证。
- 新增零训练诊断：[RMSE 差距](../project/runs/rmse_gap_diagnostic_20261115/README_CN.md)、[爬坡信息](../project/runs/ramp_information_diagnostic_20261115/README_CN.md)。
- 全部历史索引：[PROJECT_HANDOFF_CN.md](../project/PROJECT_HANDOFF_CN.md)。旧协议结果只能解释来历，不应混入本轮成绩。

## 数据位置（原服务器，本仓库不含数据实体）

| 数据 | 路径 |
| --- | --- |
| 原始 Pangu 预报 | `/shared_data_hdd1/All_Datasets/ERA5_forecasting/Pangu/Forecast_data/` |
| 原始 ERA5 | `/shared_data_hdd1/All_Datasets/ERA5/` |
| 新协议 Pangu 缓存 | `/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1/` |
| 清洗后实测功率 | `/shared_data_ssd2/liying/wind_cluster_project_20261018/project/data/jilin_56farm_pad8.h5` |

## 可以直接发给 Gemini 的提问

```text
请阅读已通过“导入代码”附加的 GitHub 仓库或解压后的代码文件夹。先读 README.md 和
docs/GEMINI_READING_GUIDE_CN.md 和
docs/STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md，再读所链接的当前协议、结果报告、
编码角色说明和 STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md。
已发布的固定成绩请读取 project/runs/daily_state_representation_20261116_v2/analysis/
CURRENT_RESULTS_CN.md、CURRENT_RESULTS.json 和 LEARNINGS_20261116_CN.md，
不要沿用旧文档中“阶段一刚启动”“TF_E2L 尚缺结果”的历史状态；
新 B / R4-ISO 的执行情况与独立 A 尚缺的设计，见新增提案第 6 节。

项目任务是单风场 0–6 h、15 min 功率预测。我需要相对强 Transformer
有可信性能优势的研究路线，天气编码不作为机制贡献或独立贡献。

请基于实际可见文件：
1. 归纳已有结果、负结果与尚未完成的验证，区分共享、独立、微调和未见风场迁移。
2. 核对关键代码、数据可用时刻、同输入对照及潜在信息泄漏。
3. 评估最新状态条件表示计划的必要性、可辨识性与公平对照，给出优先级。
4. 提出三个候选贡献点，每个写清已有证据、缺失对照和失败后的备选路线。
所有结论引用仓库文件和具体位置；看不到的文件请明确说明。
不得混用旧协议成绩，不把计划当结果，不把已见风场微调写成新风场迁移，
也不要因为方案复杂就推断它会优于 Transformer。
```
