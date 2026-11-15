# Gemini / 编程助手阅读入口

更新：2026-11-15。先读本文，再进入长篇历史交接文档。

## 研究目标与约束

- 目标是单风场功率预测，预测未来 0–6 h 的 24 个 15 min 点；共享训练仍可以输出单个风场预测。
- 希望找到相对强 Transformer 有可复核优势的路线，不能只挑表现好的站点或某一个指标。
- 天气编码（含 E2L/E2C）只作实现选择与消融，不作为机制贡献点或独立贡献点。
- 共享训练、逐站独立训练、已见风场微调、完全未见风场迁移是不同设定。现有逐站微调不等于新风场迁移。
- 数据只有实测功率、Pangu 预报和 ERA5 再分析；没有场站实测天气。历史 Pangu 节点不是 ERA5。

## 当前结果的证据边界

当前协议为每日 00 UTC Pangu、15 h 后可用（北京时间 23:00 切换），气象时效 9–45 h；邻站/区域功率可用至预测起点。55 个固定风场，2022 训练、2023 选择。2024 曾在历史研究中被查看，后续只能称探索性跨年复评，不能声称完全盲测。

本轮已完成 2571 项，见[完成审计](../project/analysis/daily_protocol_review_20261115/COMPLETION_AUDIT.json)。以下为相对匹配 Transformer 的误差变化，负值为改善；口径详见[完整报告](../project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)。

| 方案 | MAE 相对变化 | RMSE 相对变化 | 解释 |
| --- | ---: | ---: | --- |
| 共享 W0_R3 | −0.183% | +0.227% | 两项指标未同时胜出 |
| 独立 W0_R3 | −0.501% | +0.239% | MAE 优势不能覆盖 RMSE 退化 |
| 完整 W0 逐站微调 | +0.108% | +0.370% | 不支持作为当前最优路线 |
| 共享 E2L | −0.457% | −0.216% | 高功率 RMSE +0.153%，未满足原先不超过 +0.1% 的条件 |

E2L 是多层无激活线性映射，其复合仍为仿射映射。当前尚缺 Transformer 使用相同 E2L/E2C 的交叉骨干对照，不能把其收益直接归因于主干或核心机制。详见[编码角色说明](../project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md)。

## 关键实现与证据

- 共享模型：[model.py](../project/runs/daily_protocol_restart_20261114_v1/code/model.py)、[train.py](../project/runs/daily_protocol_restart_20261114_v1/code/train.py)、[data.py](../project/runs/daily_protocol_restart_20261114_v1/code/data.py)。
- 消融与独立训练：[model.py](../project/runs/daily_ablation_restart_20261114_v1/code/model.py)、[train.py](../project/runs/daily_ablation_restart_20261114_v1/code/train.py)、[实验清单](../project/runs/daily_ablation_restart_20261114_v1/manifest.json)。
- 微调：[train.py](../project/runs/daily_site_finetune_20261114_v1/code/train.py)、[site_data.py](../project/runs/daily_site_finetune_20261114_v1/code/site_data.py)。
- 汇总程序：[review_daily_results_20261115.py](../project/scripts/review_daily_results_20261115.py)。
- 逐站逐种子证据：[结果目录](../project/analysis/daily_protocol_review_20261115/)、[逐风场对比](../project/analysis/daily_protocol_review_20261115/per_farm_comparisons.csv)、[逐种子对比](../project/analysis/daily_protocol_review_20261115/per_seed_comparisons.csv)。
- 最新待检验方案：[状态条件中间表示](../project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md)。该文是计划，提出的条件生成、教师/学生训练和增益都需要实验验证；其中诊断解释也应与实际证据交叉核对。
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
请阅读附加的私有 GitHub 仓库或代码 ZIP。先读 README.md 和
docs/GEMINI_READING_GUIDE_CN.md，再读所链接的当前协议、结果报告、
编码角色说明和 STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md。

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
