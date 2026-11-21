# 单风场 0–6 h 功率预测

GitHub 公开仓库：<https://github.com/LYingyx/wind-power-forecasting>，默认分支 `main`。

本仓库用于代码与研究结果审阅，包含现有源码、实验配置、每日起报协议的结果汇总及后续研究计划。实际工作目录为 `project/`。原始气象/功率数据、缓存、模型权重、逐样本预测和 Python 环境保留在研究服务器，未上传；克隆本仓库不能直接恢复完整训练环境。

2026-11-21 最新总入口：**[给 Gemini 的完整阅读目录](docs/GEMINI_READING_INDEX_20261121_CN.md)**。新增跨时空分辨率研究讨论、Pangu／ERA5／CLDAS 诊断全文及其 14 个脚本、16 份汇总结果。先读 [10 最新结论全文](docs/gemini_plain_text/10_latest_conclusions_20261121.txt)，再结合 [08 已完成模型证据](docs/gemini_plain_text/08_paper_review_20261119.txt)、[09 当前模型源码](docs/gemini_plain_text/09_current_model_code_20261119.txt)和 [11 新诊断源码与结果](docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt)。

**请使用 `paper-evidence-20261119` 分支。** 新诊断为 56 站、2022–2023，含 24 h 子任务；下文已完成主模型为固定 55 站、0–6 h，不能直接混用成绩。研究路线是方案，最新诊断包含对平流插值和小时内细节的负反馈，不能把推荐当成已验证收益。

2026-11-19 已完成主模型入口：**[科学问题、最终架构与完整基线／消融结果](docs/GEMINI_REVIEW_20261119_CN.md)**。本分支 `paper-evidence-20261119` 的最新补充批次已完成 **151/151 个任务**，包含 121 次训练（含 16 次调参）和 30 次已有权重复评。完整模型相对同骨干线性插值的十种子净收益在 2023/2024 均通过门槛，但左侧状态摘要的独立价值、完整窗口与历史顺序的必要性仍未确立；相对强 Transformer 也未全面通过高功率与稳定性要求。

给 Gemini 一次读取，优先使用 [08 自包含纯文本](docs/gemini_plain_text/08_paper_review_20261119.txt)；审阅实现再读 [09 当前源码纯文本](docs/gemini_plain_text/09_current_model_code_20261119.txt)。最新[论文总框架图](project/figures/framework_paper_v27_20261118/framework_final.png)和[输入编码说明](docs/INPUTS_AND_REPRESENTATIONS_20261119_CN.md)已附。图仍包含左侧状态摘要，不能把结构图误读为全部子路径均已验证有效。

以下 11-17 材料是历史五种子报告，最新十种子与补充消融以 11-19 入口为准。

2026-11-17 历史测试：[2024 跨年测试复评](docs/TEST2024_RESULTS_20261117_CN.md)，**120/120 个冻结检查点全部完成**。H-FULL 相对 R0 的 MAE/RMSE 改善 0.1994%/0.1433%，通过该对比门槛；B-additive 总体改善但高功率护栏失败，内部机制归因仍未闭合。2024 历史上曾被查看，本次不称完全盲测。

2026-11-17 验证集结果（2023）：[加性修正、完整历史条件与生成式复核总结](docs/EXPERIMENT_RESULTS_20261117_CN.md)。v4–v7 已完成 **105/105 次新训练**，另完成两个 seed 的固定检查点采样审计。B-additive 与 H-FULL 均通过相对 R0 的净收益门槛，但未同时通过全部内部归因与同输入强基线门槛；R4-ISO 仍输给 R0。正文附逐 seed、逐风场证据与未通过原因。

11-16 的 [v2 结果快照](project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS_CN.md)属于历史记录，其中“正在运行”不代表当前状态。以下 2571 项也属于此前实验，不能与新批次重复计数。

## 阅读顺序

需要理解“结果怎样得到”，先读[数据 → 模型 → 训练 → 选轮 → 指标的完整导航](docs/RESULTS_TO_CODE_CN.md)。其中链接了全部 2571 项任务的实际配置与逐轮训练记录。导入附件出现解析问题时，使用[未压缩纯文本入口](docs/gemini_plain_text/README.md)。

1. [给 Gemini / 编程助手的阅读入口](docs/GEMINI_READING_GUIDE_CN.md)：当前事实、代码地图、证据边界及可直接使用的提问。
   2024 测试读取 [07 测试结果纯文本](docs/gemini_plain_text/07_test2024_results_20261117.txt)；2023 验证读取 [06 完成结果纯文本](docs/gemini_plain_text/06_completed_experiments_20261117.txt)；[05 v2 历史快照](docs/gemini_plain_text/05_current_experiment_results.txt)用于追溯旧实验。
2. [最新协议](project/MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md)：每日 00 UTC、起报后 15 h 可用，0–6 h / 15 min。
3. [新协议完整结果分析](project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)与[逐风场、逐种子结果](project/analysis/daily_protocol_review_20261115/FARM_AND_SEED_DETAILS_CN.md)。
4. [编码角色澄清](project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md)：天气编码不作为机制贡献或独立贡献。
5. [状态条件中间表示计划](project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md)：待验证路线，不能写成已有收益。
6. [完整历史交接入口](project/PROJECT_HANDOFF_CN.md)：包含大量历史记录；日期较早的“当前”状态已经过时，旧协议结果不能作为新协议成绩。

截至 2026-11-15 的完成审计为 **2571 / 2571 项**：共享训练 180、微调 1100、补充消融与独立训练 1291。见[完成审计](project/analysis/daily_protocol_review_20261115/COMPLETION_AUDIT.json)。新协议下尚无全面胜过同输入 Transformer 的已确认路线，也未完成未见风场迁移验证。

## 代码入口

| 内容 | 位置 |
| --- | --- |
| 共享模型及 Transformer 对照 | [daily_protocol / code](project/runs/daily_protocol_restart_20261114_v1/code/) |
| 编码消融、补充输入消融、逐站独立训练 | [daily_ablation / code](project/runs/daily_ablation_restart_20261114_v1/code/) |
| 共享模型逐站微调 | [daily_site_finetune / code](project/runs/daily_site_finetune_20261114_v1/code/) |
| 通用包源码 | [project/src](project/src/) |
| 数据准备、评估、复核脚本 | [project/scripts](project/scripts/) |
| 当前业务配置 | [operational_pangu_d1_08_lag15h.json](project/configs/operational_pangu_d1_08_lag15h.json) |
| 依赖 | [pyproject.toml](project/pyproject.toml)、[环境记录](project/environment/) |
| 原手稿源码 | [la](la/) |

保留各轮 `code/legacy` 与其依赖源码；这些目录由当前训练代码直接导入。仓库仍有原服务器的绝对路径，运行前需要配置数据及输出目录。历史实验源码和文档用于追溯，不代表其旧成绩适用于当前业务协议。旧轮逐任务输出和大结果表未完整纳入本仓库，部分历史链接仅在原服务器有效。

## GitHub 与 Gemini

仓库已按用户要求设为 **Public**；默认分支使用 `main`。Gemini 网页版可通过“添加文件 → 更多上传 → 导入代码”直接导入本公开仓库，无需私有仓库访问授权；直接在聊天里粘贴 URL 不会读取仓库。请按[访问与配置说明](docs/GITHUB_GEMINI_ACCESS_CN.md)导入，或下载 ZIP 后解压并通过代码文件夹入口上传。
