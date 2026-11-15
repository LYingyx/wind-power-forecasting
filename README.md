# 单风场 0–6 h 功率预测

GitHub 私有仓库：<https://github.com/LYingyx/wind-power-forecasting>，默认分支 `main`。

本仓库用于代码与研究结果审阅，包含现有源码、实验配置、每日起报协议的结果汇总及后续研究计划。实际工作目录为 `project/`。原始气象/功率数据、缓存、模型权重、逐样本预测和 Python 环境保留在研究服务器，未上传；克隆本仓库不能直接恢复完整训练环境。

## 阅读顺序

1. [给 Gemini / 编程助手的阅读入口](docs/GEMINI_READING_GUIDE_CN.md)：当前事实、代码地图、证据边界及可直接使用的提问。
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

仓库已创建为 **Private**；默认分支使用 `main`。Gemini 网页版支持通过“添加文件 → 更多上传 → 导入代码”导入私有仓库，首次需关联有权限的 GitHub 账户；直接在聊天里粘贴 URL 不会读取仓库。请按[访问与配置说明](docs/GITHUB_GEMINI_ACCESS_CN.md)导入，或下载 ZIP 后解压并通过代码文件夹入口上传。
