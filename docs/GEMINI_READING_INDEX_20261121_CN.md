# Gemini 完整阅读目录（截至 2026-11-21）

指定分支：`paper-evidence-20261119`。本目录把最新诊断、已完成主模型、原始结果证据、模型/数据源码和历史路线分开列出。每个文件均提供可浏览地址；纯文本另列 Raw 直链。不要仅导入默认 `main`。

## 最少先读这四份

1. [10：最新诊断和研究路线全文](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/10_latest_conclusions_20261121.txt)。
2. [08：已完成主模型的科学问题、基线与消融全文](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/08_paper_review_20261119.txt)。
3. [09：当前模型与训练／数据／评价源码](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/09_current_model_code_20261119.txt)。
4. [11：最新诊断源码和汇总结果](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt)。

如果 Gemini 无法直接读取链接，请下载相应 UTF-8 TXT 后导入。先要求它确认实际读取的文件，不要假定链接会被自动遍历。需要重算主模型配对指标，再追加 `METRIC_RECORDS.json` 和 `COMPARISONS.json`。

**口径优先级：**最新诊断读 10/11，已训练主模型读 08/09；01–07 是历史快照。55 站主模型与 56 站诊断、0–6 h 与日前任务、五种子与十种子、实际可用预报与未来再分析对照均须分开。方案不能替代结果；图不能证明每条子路径有效。

## 最新结论与研究方向

| 内容 | 文件 | 用途／边界 |
|---|---|---|
| 本次更新概览与口径说明 | [LATEST_CONCLUSIONS_20261121_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/LATEST_CONCLUSIONS_20261121_CN.md) | 区分已完成诊断、原模型结果与研究建议。 |
| 最新诊断报告 | [REPORT_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/REPORT_CN.md) | 2022–2023、56 站；包括 Pangu 误差、平流插值、历史功率、ERA5/CLDAS 与小时内信息。 |
| 跨时空分辨率研究路线 | [SINGLE_FARM_CROSS_RESOLUTION_RESEARCH_20261120_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/SINGLE_FARM_CROSS_RESOLUTION_RESEARCH_20261120_CN.md) | 11-20 的文献与方案讨论；不是新模型收益报告。后续可行性判断须同时读最新诊断。 |
| 最新结论全文 TXT | [10_latest_conclusions_20261121.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/10_latest_conclusions_20261121.txt) | 自包含：口径说明 + 最新诊断全文 + 研究路线全文。 |
| 新诊断源码与结果 TXT | [11_alignment_diagnostic_code_and_results_20261121.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt) | 14 个诊断脚本及 16 份原始汇总 JSON，逐文件带 SHA-256。 |
| 新诊断严格 JSON 汇总 | [ALIGNMENT_DIAGNOSTIC_RESULTS_20261121.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/ALIGNMENT_DIAGNOSTIC_RESULTS_20261121.json) | 供程序读取；一处 NaN 转为 null，并保留原路径及说明。 |

## 已完成主模型：科学问题、基线与消融

| 内容 | 文件 | 用途／边界 |
|---|---|---|
| 11-19 自包含论文证据包 | [GEMINI_REVIEW_20261119_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/GEMINI_REVIEW_20261119_CN.md) | 当前已完成神经网络实验的主要结论；151 项补充任务全部完成。 |
| 科学问题与贡献表述 | [SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS_20261119_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS_20261119_CN.md) | 与当时实际实现相对应；未被最新路线讨论自动替换。 |
| 输入、模块与表示说明 | [INPUTS_AND_REPRESENTATIONS_20261119_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/INPUTS_AND_REPRESENTATIONS_20261119_CN.md) | 历史功率、周边观测、运行统计、气象派生量与表示空间。 |
| 两年基线与消融总表 | [BASELINE_ABLATION_SUMMARY_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/BASELINE_ABLATION_SUMMARY_CN.md) | 共同五种子表、十种子复核和未通过项。 |
| 基线 CSV | [baseline_common5.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/baseline_common5.csv) | 相同五种子口径。 |
| 消融 CSV | [ablation_common5.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/ablation_common5.csv) | 相同五种子口径。 |
| 补充对照 CSV | [extra_common5.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/extra_common5.csv) | 额外对照模型。 |
| 带标准差的汇总表 | [READABLE_TABLES.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/READABLE_TABLES.json) | 54 条表格记录。 |
| 逐种子、逐风场与分时效指标 | [METRIC_RECORDS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/METRIC_RECORDS.json) | 322 条模型×种子×年份记录；较大文件，建议下载读取。 |
| 实际配置、完成标志与训练曲线 | [TASK_EVIDENCE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/TASK_EVIDENCE.json) | 151 项任务的配置、训练记录和来源校验。 |
| 配对比较与门槛 | [COMPARISONS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/COMPARISONS.json) | 54 项配对比较，保留未通过原因。 |
| 分组结果 | [GROUPS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/GROUPS.json) | 各实验组聚合指标。 |
| 两因素交互 | [INTERACTIONS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/INTERACTIONS.json) | 检查两项设计是否独立、是否存在交互。 |
| 起报日分块区间估计 | [ISSUE_BLOCK_UNCERTAINTY.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/analysis/ISSUE_BLOCK_UNCERTAINTY.json) | 按气象起报日处理样本相关性。 |
| 完成审计 | [COMPLETE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/COMPLETE.json) | 151/151；121 次训练（含 16 次调参）和 30 次已有权重复评。 |
| 冻结协议 | [PROTOCOL.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/PROTOCOL.json) | 训练和评价规则。 |
| 科学假设与判定冻结 | [SCIENCE_FREEZE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/SCIENCE_FREEZE.json) | 预先约定的比较与门槛。 |
| 任务清单 | [manifest.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/manifest.json) | 任务标识与实际配置入口。 |

## 当前模型代码、训练与数据处理

| 内容 | 文件 | 用途／边界 |
|---|---|---|
| 当前源码 TXT | [09_current_model_code_20261119.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/09_current_model_code_20261119.txt) | 完整当前 code/、数据准备与结果计算源码；不含模型权重。 |
| 模型入口和补充基线 | [model.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/model.py) | PaperModel、输入消融和同输入线性模型。 |
| 完整历史条件表示修正 | [history_model.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/history_model.py) | 轨迹卷积、状态/天气窗口条件与低维加性修正。 |
| 状态调制与加性对照 | [realization_model.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/realization_model.py) | 兑现模块的历史实现和相关对照。 |
| 共享主体与解码 | [parent_model.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/parent_model.py) | 天气、历史、上下文融合及输出路径。 |
| 表示与流匹配实现 | [representation.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/representation.py) | 包括历史生成式组件；存在代码不代表当前主模型采用它。 |
| 训练、选轮与评估 | [train.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/train.py) | 实际损失与选模逻辑。 |
| 样本和特征构造 | [data.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/data.py) | 当前训练数据接口。 |
| 气象派生特征 | [physics.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/physics.py) | 相关物理量的计算。 |
| 早期日协议数据准备 | [prepare.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_protocol_restart_20261114_v1/code/prepare.py) | 近邻、三环与相关预处理来源。 |
| 当前结果汇总 | [collect.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/operations/collect.py) | 读取完成任务并计算比较。 |
| 当前易读表生成 | [readable_report.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/operations/readable_report.py) | 基线、消融表的生成逻辑。 |
| 不确定性分析代码 | [uncertainty.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/operations/uncertainty.py) | 起报日分块计算。 |
| 日协议天气准备 | [prepare_operational_pangu.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/scripts/prepare_operational_pangu.py) | 业务可用性与天气缓存构造。 |
| 日协议天气审计 | [audit_operational_pangu.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/scripts/audit_operational_pangu.py) | 检查数据和时效。 |
| 数据、选模、评估历史 TXT | [04_data_selection_evaluation.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/04_data_selection_evaluation.txt) | 历史管线补充，当前模型以 09 为准。 |

## 数据来源、风场范围与论文图

| 内容 | 文件 | 用途／边界 |
|---|---|---|
| 服务器数据来源位置 | [SOURCE_LOCATIONS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/data/SOURCE_LOCATIONS.json) | 功率 CSV/HDF5、Pangu、ERA5 与缓存路径；这是路径记录，不是原始数据。 |
| 功率数据元信息 | [jilin_56farm_pad8.metadata.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/data/jilin_56farm_pad8.metadata.json) | 原始 56 站数据元信息。 |
| 每日起报业务配置 | [operational_pangu_d1_08_lag15h.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/configs/operational_pangu_d1_08_lag15h.json) | 每日 00 UTC 起报、15 h 后可用。 |
| 主模型固定风场集合 | [COHORT.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/assets/COHORT.json) | 55 个风场；不要与新诊断的 56 站混同。 |
| 主模型质量审计 | [QC_AUDIT.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/assets/QC_AUDIT.json) | 固定筛选规则及其审计。 |
| 逐风场质量表 | [qc_by_farm.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/assets/qc_by_farm.csv) | 逐场质量信息。 |
| 归一化参数 | [normalization.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/assets/normalization.json) | 当前共享模型的预处理参数。 |
| 业务协议说明 | [MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md) | 已完成 0–6 h 实验的协议。 |
| 英文论文框架图 | [framework_final.png](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/figures/framework_paper_v27_20261118/framework_final.png) | 当前已实现结构；不是 11-20 新提案的实现图。 |
| 论文图 PDF | [framework_final.pdf](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/figures/framework_paper_v27_20261118/framework_final.pdf) | 可下载版本。 |
| 论文图说明与来源 | [README_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/figures/framework_paper_v27_20261118/README_CN.md) | 图与实际实现的对应关系。 |

## 历史结果与路线演变

| 内容 | 文件 | 用途／边界 |
|---|---|---|
| 11-17 已完成实验（2023） | [EXPERIMENT_RESULTS_20261117_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/EXPERIMENT_RESULTS_20261117_CN.md) | v4–v7 共 105 次训练；含状态调制、加性修正和隔离教师梯度的负结果。 |
| 11-17 逐任务证据 | [EXPERIMENT_RESULTS_20261117_EVIDENCE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/EXPERIMENT_RESULTS_20261117_EVIDENCE.json) | 该批次的配对数据。 |
| 采样审计 | [SAMPLING_AUDIT_20261117_EVIDENCE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/SAMPLING_AUDIT_20261117_EVIDENCE.json) | 固定检查点采样检验。 |
| 11-17 跨年复评（2024） | [TEST2024_RESULTS_20261117_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/TEST2024_RESULTS_20261117_CN.md) | 120 个冻结检查点；五种子报告，不代替 11-19 十种子。 |
| 11-17 跨年逐任务证据 | [TEST2024_RESULTS_20261117_EVIDENCE.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/TEST2024_RESULTS_20261117_EVIDENCE.json) | 2024 逐风场、逐种子结果。 |
| 11-17 跨年总表 | [TEST2024_SUMMARY_20261117.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/TEST2024_SUMMARY_20261117.csv) | 历史五种子结果。 |
| 11-17 跨年配对表 | [TEST2024_COMPARISONS_20261117.csv](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/TEST2024_COMPARISONS_20261117.csv) | 历史配对比较。 |
| 11-15 日协议结果分析 | [DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md) | 原主干、编码和相关消融。 |
| 11-15 逐风场／种子／时效 | [FARM_AND_SEED_DETAILS_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/analysis/daily_protocol_review_20261115/FARM_AND_SEED_DETAILS_CN.md) | 逐场与稳定性证据。 |
| 天气编码角色澄清 | [ENCODING_ROLE_CLARIFICATION_20261115_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md) | 编码选择偏差和贡献边界。 |
| 跨分辨率／兑现早期提案 | [STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md) | 方案历史；实际结果以上述完成报告为准。 |
| 11-16 v3–v5 完整计划存档 | [STATE_REPRESENTATION_PLAN_V3_V5_ARCHIVE_20261121_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/STATE_REPRESENTATION_PLAN_V3_V5_ARCHIVE_20261121_CN.md) | 本次补齐此前本地修订；原文中的“在跑”是当时状态。 |
| 原 v2 计划 | [STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md) | 保留远程原快照，避免改写旧阅读包来源。 |
| v2 结果快照 | [CURRENT_RESULTS_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS_CN.md) | 历史局部结果；不能据此判断当前完成状态。 |
| 旧实验结果到代码导航 | [RESULTS_TO_CODE_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/RESULTS_TO_CODE_CN.md) | 2571 项早期任务的配置、训练记录与评估来源。 |
| 旧协议运行状态与天气机制 | [TWO_MECHANISM_EVIDENCE_UPDATE_20261027_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/TWO_MECHANISM_EVIDENCE_UPDATE_20261027_CN.md) | 旧协议机制讨论；不能与每日新协议直接拼接。 |
| 输出头 2×2 旧实验计划 | [PLAN_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/readout_factorial_20261028_v1/PLAN_CN.md) | 输出头当前功率输入的历史实验入口。 |
| 共享与逐场适配历史 | [FINAL_REVIEW_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/shared_site_finetune_20261104_v1/analysis/FINAL_REVIEW_CN.md) | 旧适配结果；不代表当前共享模型做了逐场微调。 |
| 完整历史交接文档 | [PROJECT_HANDOFF_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/PROJECT_HANDOFF_CN.md) | 仅按需追溯；日期较早的“当前”叙述可能已过时。 |

## 全部 11 份纯文本附件

| 附件 | 内容 | 下载 |
|---|---|---|
| [01_context_and_results.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/01_context_and_results.txt) | 早期背景、日协议结果与计划（历史） | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/01_context_and_results.txt) |
| [02_models_and_training.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/02_models_and_training.txt) | 早期模型和训练源码（历史，当前用 09） | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/02_models_and_training.txt) |
| [03_farm_seed_horizon_metrics.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/03_farm_seed_horizon_metrics.txt) | 早期逐风场、种子和时效指标（历史） | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/03_farm_seed_horizon_metrics.txt) |
| [04_data_selection_evaluation.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/04_data_selection_evaluation.txt) | 数据、选型、评估历史源码 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/04_data_selection_evaluation.txt) |
| [05_current_experiment_results.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/05_current_experiment_results.txt) | 11-16 v2 快照（历史；文件名中的 current 不表示现在） | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/05_current_experiment_results.txt) |
| [06_completed_experiments_20261117.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/06_completed_experiments_20261117.txt) | 11-17 完成的 2023 实验与负结果 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/06_completed_experiments_20261117.txt) |
| [07_test2024_results_20261117.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/07_test2024_results_20261117.txt) | 11-17 的 2024 五种子复评 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/07_test2024_results_20261117.txt) |
| [08_paper_review_20261119.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/08_paper_review_20261119.txt) | 当前已完成主模型的论文证据、基线与消融 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/08_paper_review_20261119.txt) |
| [09_current_model_code_20261119.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/09_current_model_code_20261119.txt) | 当前主模型、训练、数据和汇总源码 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/09_current_model_code_20261119.txt) |
| [10_latest_conclusions_20261121.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/10_latest_conclusions_20261121.txt) | 最新诊断全文与新研究路线全文 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/10_latest_conclusions_20261121.txt) |
| [11_alignment_diagnostic_code_and_results_20261121.txt](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt) | 最新诊断的全部源码和汇总结果 | [Raw](https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/paper-evidence-20261119/docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt) |

## 新诊断逐项证据与脚本

| 检验 | 原始结果 JSON | 代码 |
|---|---|---|
| 气象误差、平滑、位移和时间偏差 | [D1_D4_errors.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/D1_D4_errors.json) | [diag_errors.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/diag_errors.py) |
| ERA5 平流／线性重建 | [G0_era5.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/G0_era5.json) | [g0_interp.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/g0_interp.py) |
| Pangu 平流／线性重建 | [G0_pangu.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/G0_pangu.json) | [g0_interp.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/g0_interp.py) |
| 气象变量的功率信息 | [Q1_variables.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q1_variables.json) | [power_tests.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/power_tests.py) |
| 小时内功率信息 | [Q2_intrahour.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q2_intrahour.json) | [power_tests.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/power_tests.py) |
| 本站与邻站历史信息 | [Q3_history_neighbors.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q3_history_neighbors.json) | [power_tests.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/power_tests.py) |
| 空间邻域比较 | [Q4_spatial_ceiling.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q4_spatial_ceiling.json) | [spatial_ceiling.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/spatial_ceiling.py) |
| 平流路线分层检验 | [ROUTE1_stratified.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/ROUTE1_stratified.json) | [route1_stratified.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/route1_stratified.py) |
| 气象订正与功率转换解耦 | [Q5_decoupled.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q5_decoupled.json) | [decoupled_check.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/decoupled_check.py) |
| 不同时间尺度的波动信息 | [Q6_fluctuation.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q6_fluctuation.json) | [fluctuation_info.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/fluctuation_info.py) |
| 24 小时任务 | [Q7_horizon24.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q7_horizon24.json) | [horizon24.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/horizon24.py) |
| CLDAS 信息、空间均值与时效 | [Q8_cldas.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q8_cldas.json) | [cldas_check.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/cldas_check.py) |
| CLDAS 追加对照 | [Q8b_cldas_followup.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q8b_cldas_followup.json) | [cldas_followup.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/cldas_followup.py) |
| CLDAS 产品逐月核查 | [Q8c_cldas_product_by_month.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q8c_cldas_product_by_month.json) | 生成命令未随本批存档；勿假定已完整复现 |
| CLDAS 小时内细节检验 | [Q9_cldas_intrahour.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/Q9_cldas_intrahour.json) | [cldas_intrahour.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/cldas_intrahour.py) |
| ERA5 100 m 提取完整性 | [ERA5_100M_READY.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/ERA5_100M_READY.json) | [extract_era5_100m.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/extract_era5_100m.py) |

补充：[路径和共享计算](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/common.py)、[CLDAS 提取](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/extract_cldas.py)。所有原始结果保留，NaN 的处理说明见[本次概览](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/LATEST_CONCLUSIONS_20261121_CN.md)。

## 完整代码目录与来源审计

- [当前主模型全部源码（含实际导入的 legacy 依赖）](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/code/)。
- [当前结果收集、评估加载修复与审计](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/operations/)；[实际选择的基线配置](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_paper_evidence_20261118_v10/assets/selected/)。
- [最新诊断全部源码](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/)；[最新诊断全部汇总 JSON](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/output/)。
- [早期 2571 项任务的三组训练来源记录](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/analysis/daily_protocol_review_20261115/provenance/)。
- [早期日协议基线源码](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_protocol_restart_20261114_v1/code/)、[早期编码与输入消融源码](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_ablation_restart_20261114_v1/code/)、[早期逐场微调源码](https://github.com/LYingyx/wind-power-forecasting/tree/paper-evidence-20261119/project/runs/daily_site_finetune_20261114_v1/code/)。
- [机器可读完整文件目录](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/GEMINI_FILE_CATALOG_20261121.json)：本目录涉及的全部文件，加当前模型／诊断源码、配置与来源目录；逐文件提供字节数、SHA-256、GitHub 地址和 Raw 地址。
- [本次发布校验清单](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/GEMINI_PUBLICATION_MANIFEST_20261121.json)；[11-19 冻结证据哈希](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/PAPER_REVIEW_MANIFEST_20261119.json)；[01–07 历史附件哈希](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/gemini_plain_text/MANIFEST.json)。

服务器原始数据路径不会变成 GitHub 下载地址。证据 JSON 内的 `source` 可能指仅在服务器存在的 `jobs/` 原件；远程可读的指标和训练记录分别使用 `METRIC_RECORDS.json`、`TASK_EVIDENCE.json`。

## 可直接复制给 Gemini 的审阅要求

> 请先读本目录的 10、08、09、11 四份 TXT，再按需读结果 JSON。先列出实际读取的文件。区分已有模型净收益、内部机制证据、新诊断与待实现方案；解释最新的小时内／平流负结果如何限制旧研究路线，同时保留气象订正与小时级信息的正反馈。不要混用 55/56 站、0–6 h/24 h、五/十种子或预报/未来分析数据；不要将文献或代理投票当成实验验证，不把历史待办当成当前未完成。逐条引用文件与数字，不根据模型胜负筛选风场。

本次发布检查：`python3 project/scripts/export_gemini_review_20261121.py --check-only`。
已完成模型的独立表格复算：`python3 project/scripts/export_paper_review_20261119.py --check-only`。
