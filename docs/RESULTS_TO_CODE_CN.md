# 从实验结果追溯到数据、模型和训练过程

本仓库包含当前每日起报协议的实现与结果证据，供代码审阅。请沿下面的链条阅读，不要只根据结果报告推测机制。原始数据、模型权重和全量预测仍在研究服务器，因此公开仓库支持方法与记录审查，不能单独重跑完整训练或从预测重新计算全部指标。

2026-11-16 更新：[v2 最新结果](../project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS_CN.md)已覆盖 34 项完成的新训练、10 项旧权重诊断，以及尚未完成任务的明确标记。阶段一 25 项全部完成，R4 三组各有 3 个种子完成；当前另有 11 项训练运行。下面的 2571 项完整证据仍指此前实验。

本轮从结果追溯实现的路径：先读 [CURRENT_RESULTS.json](../project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS.json) 的 `completed_training` / `reused_references` / `diagnostic_evaluations`，查看 `record.spec`、`history`、`checkpoint_epoch` 与 `source_files`；再对照 [train.py](../project/runs/daily_state_representation_20261116_v2/code/train.py)、[model.py](../project/runs/daily_state_representation_20261116_v2/code/model.py)、[representation.py](../project/runs/daily_state_representation_20261116_v2/code/representation.py) 和 [collect.py](../project/runs/daily_state_representation_20261116_v2/operations/collect.py)。本轮 JSON 包含完整原指标对象，含逐站/时效数组和诊断；导出脚本是 [export_current_results.py](../project/scripts/export_current_results.py)。

## 1. 数据怎样成为模型输入

| 环节 | 代码与配置 | 应检查的内容 |
| --- | --- | --- |
| 业务可用时刻 | [当前配置](../project/configs/operational_pangu_d1_08_lag15h.json)、[operational.py](../project/src/wind_forcing/data/operational.py) 的 `select_issue` / `plan_times` | `DAILY` 为 15 h 延迟、24 h 起报周期、最大 45 h；调用方必须传入当前配置，不能把兼容旧协议的默认值当成当前实验 |
| 天气裁剪与缓存 | [prepare_operational_pangu.py](../project/scripts/prepare_operational_pangu.py)、[merge_verified_pangu_crops.py](../project/scripts/merge_verified_pangu_crops.py)、[audit_operational_pangu.py](../project/scripts/audit_operational_pangu.py) | 原始预报读取、裁剪、旧裁剪帧验证合并、训练集归一化、时间与覆盖审计 |
| 实际执行入口 | [pipeline.py](../project/runs/daily_protocol_restart_20261114_v1/operations/pipeline.py) | 当前业务配置如何传给数据准备脚本，何时审计、冻结和启动训练 |
| 固定风场与窗口 | [prepare.py](../project/runs/daily_protocol_restart_20261114_v1/code/prepare.py)、[QC_RULES.json](../project/runs/daily_protocol_restart_20261114_v1/QC_RULES.json)、[COHORT.json](../project/runs/daily_protocol_restart_20261114_v1/assets/COHORT.json) | QC 后 55 站，功率归一化，2022/2023 起点和训练统计；邻站/区域功率可用至起点 |
| 批次特征 | [data.py](../project/runs/daily_protocol_restart_20261114_v1/code/data.py) 的 `SharedData.batch`、[physics.py](../project/runs/daily_protocol_restart_20261114_v1/code/physics.py) | 小时天气节点插值到未来 24 个 15 min 点；历史、物理/状态特征、邻站与区域信息；数据指纹核对 |

9–45 h 是裁剪覆盖范围，包含过去节点需求；不能把这个范围内的每个时效都说成模型的未来目标天气。天气数据是 Pangu 和 ERA5，没有场站实测天气。

## 2. 模型、基线和消融怎样实现

- [共享 model.py](../project/runs/daily_protocol_restart_20261114_v1/code/model.py)：`SharedModel.__init__` 按 `family` 构建 W0、TF、TiDE、TFT、B0；`forward` 定义历史编码、天气处理、区域输入、条件适配和解码。`NeighborReadout` 定义邻站读取。
- [基线源实现](../project/runs/daily_protocol_restart_20261114_v1/code/legacy/baselines_original.py)：`TransformerFutureCov` 等；[卷积依赖](../project/runs/daily_protocol_restart_20261114_v1/code/legacy/vendor/wind_forcing/models/p0i_legacy.py)：`ResidualConv`。当前 run 的整个 `code/legacy` 已上传。
- [消融 model.py](../project/runs/daily_ablation_restart_20261114_v1/code/model.py)、[消融 data.py](../project/runs/daily_ablation_restart_20261114_v1/code/data.py)：编码变体、输入屏蔽、历史/天气对照；指定 `station` 后仅取目标风场标签进行独立训练。
- [微调 train.py](../project/runs/daily_site_finetune_20261114_v1/code/train.py) 的 `load_model`：加载共享权重，区分完整微调与头部微调，约束目标风场 embedding 更新；[site_data.py](../project/runs/daily_site_finetune_20261114_v1/code/site_data.py) 定义本站数据。

冻结依赖中包含旧实验常量与兼容代码。判断实际行为须从当前 run 的入口、`settings.py`、`manifest.json` 和调用路径出发。天气编码不是独立机制贡献；新版状态条件表示已实现并取得部分实验结果，但其[研究计划](../project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md)中的预期收益不能当作已证实结论。

## 3. 参数、训练预算和选轮怎样确定

| 证据 | 位置 |
| --- | --- |
| 共享 180 项任务 | [manifest.json](../project/runs/daily_protocol_restart_20261114_v1/manifest.json)、[PROTOCOL.json](../project/runs/daily_protocol_restart_20261114_v1/PROTOCOL.json)、[plan.py](../project/runs/daily_protocol_restart_20261114_v1/code/plan.py) |
| 消融与独立训练 1291 项 | [manifest.json](../project/runs/daily_ablation_restart_20261114_v1/manifest.json)、[plan.py](../project/runs/daily_ablation_restart_20261114_v1/code/plan.py) |
| 微调 1100 项 | [manifest.json](../project/runs/daily_site_finetune_20261114_v1/manifest.json)、[prepare_daily.py](../project/runs/daily_site_finetune_20261114_v1/code/prepare_daily.py) |
| 调参后的实际配置 | [selected 目录](../project/runs/daily_protocol_restart_20261114_v1/assets/selected/)、[collect.py](../project/runs/daily_protocol_restart_20261114_v1/operations/collect.py) |
| 训练循环和验证选轮 | [train.py](../project/runs/daily_protocol_restart_20261114_v1/code/train.py) 的 `resolve` / `run` |

训练采用容量因子 MAE、AdamW、有效 batch 512、梯度范数裁剪 1，学习率常数；正式共享/独立任务一般最多 80 epoch、patience 10，具体以每项 manifest 为准。调参使用种子 40、41，正式比较使用 42–46。`collect.py` 按 2023 macro-MAE 的调参种子均值选配置，`resolve` 读取实际选中配置。

每项训练按验证 MAE 选一份权重，再用同一份权重报告 MAE、RMSE 等指标。微调把未修改的 epoch 0 纳入选择，同时另报预先固定的 epoch 1、6。B0 OOF 使用 2022 内部块作选择，不能误写成所有任务都用 2023 选轮。2023 同时用于选择与比较，不是独立测试集。

## 4. 每项实验的实际训练记录

新增[训练过程证据目录](../project/analysis/daily_protocol_review_20261115/provenance/)覆盖全部 2571 项任务，三个 JSON 分别对应上述三轮。每个任务包含：

- 请求参数与实际解析后的参数（`requested_spec` / `spec`），必要时含原共享模型的 `shared_spec`；
- 每轮训练 MAE、验证 MAE/RMSE、当时最佳 MAE及已有时间记录；
- 实际选中 epoch、参数量、指标摘要、权重哈希；
- 微调的 epoch 0 参照、固定 epoch 1/6 及已存在的 Z1 主口径摘要；
- 原服务器 `metrics.json`、`history.json` 等来源路径与原始文件 SHA-256。

这些是原记录的明确字段导出，不是新训练结果，也不是完整原 JSON/权重副本。导出时会逐项核对 manifest、数据指纹和成功状态，并验证选中轮次确为已记录验证 MAE 的最小值（微调包括 epoch 0）。原始逐任务 JSON 正文不全部上传；来源哈希用于在原服务器对应原件。

## 5. 指标和汇总表怎样计算

先读 [train.py 的 evaluate](../project/runs/daily_protocol_restart_20261114_v1/code/train.py)，再读 [review_daily_results_20261115.py](../project/scripts/review_daily_results_20261115.py)。

- MAE：逐风场累计绝对误差，对该站所有样本和 24 个时效平均，再做风场等权平均。
- RMSE：每站先对样本/时效平方误差求均值并开方，再做风场等权平均。不能把它改写成所有站混在一起的一次整体 RMSE。
- **高功率样本按预测起点历史末点 `h[:, -1] >= 0.75` 划分**，不是按未来实测功率分组；高功率也不等于大爬坡。
- 分时效区间是第 1–4、5–12、13–24 点，对应 0–1 h、1–3 h、3–6 h。
- 汇总器从逐任务指标提取风场 × seed 单元，形成 [all_farm_seed_metrics.csv](../project/analysis/daily_protocol_review_20261115/all_farm_seed_metrics.csv)，再计算 [group_summary.csv](../project/analysis/daily_protocol_review_20261115/group_summary.csv) 和 [comparison_summary.csv](../project/analysis/daily_protocol_review_20261115/comparison_summary.csv)。
- 相对变化是 `100 * (候选均值 / 参照均值 - 1)`，负值改善。5 个种子与 55 站构成的 275 个单元不是 275 个独立样本；报告中的 bootstrap 区间也是描述性的。

“MAE 改善、RMSE 退化”本身不能证明削峰填谷、爬坡欠响应或天气相位漂移。归因需要读[RMSE 诊断源码](../project/runs/rmse_gap_diagnostic_20261115/)和[爬坡诊断源码](../project/runs/ramp_information_diagnostic_20261115/)，并核对实际诊断输出。两个诊断 README 主要记录脚本用途，不包含全部诊断数值；未持久化的对话/控制台输出不能冒充仓库中的实证。

## 6. 阅读与后续更新检查

[纯文本备用入口](gemini_plain_text/README.md)提供五份未压缩 TXT：协议与旧轮结果、模型训练、旧轮逐站指标、数据准备与选型评估，以及新增的当前轮完整结果证据；所有来源文件正文及 SHA-256 可核对。第五份包含本轮完整训练记录与指标对象，旧轮逐任务记录仍保存在上面的三个 JSON 中。

更新源码或报告后，在仓库根目录执行：

```bash
python3 project/scripts/export_gemini_reading.py
```

原服务器需要刷新逐任务记录时执行：

```bash
python3 project/scripts/export_gemini_reading.py --refresh-training
```

刷新当前 v2 结果快照及明文副本：

```bash
python3 project/scripts/export_current_results.py
python3 project/scripts/export_gemini_reading.py
```

当前轮导出会验证完成标志、训练配置、检查点哈希与最小验证 MAE 选轮；发布前再核对逐风场/种子 CSV 和同种子配对数值。运行中的任务只导出明确标记的进度，不将某个中途 epoch 当成最终结果。

检查已提交版本：

```bash
python3 project/scripts/export_gemini_reading.py --check-only --revision HEAD
```

检查覆盖文本 UTF-8 解码、压缩字节/NUL、仓库文件数/体积、冻结训练源码哈希、训练记录选轮、训练记录与逐站表及汇总表的数值一致性，以及 TXT 是否含有对应源码全文、是否过期。它保证交付文件的可读性和一致性，不测试或保证 Gemini 网页端解析器。

原服务器已启用 [pre-push 检查](../.githooks/pre-push)：推送 `main` 前核验本次实际提交，文本副本过期或上述检查失败会阻止推送。其他机器克隆后需自行启用（Git 不自动复制本地配置）：

```bash
git config --local core.hooksPath .githooks
```

导入后可让 Gemini 先返回以下代码证据，再进行机制建议：

```text
请沿 docs/RESULTS_TO_CODE_CN.md 阅读整个实验实现，不只读结果摘要。
先引用并解释：SharedData.batch 如何构造天气输入；SharedModel.forward
如何输出功率；train.py 中 loss、选轮和高功率划分的原句；collect.py
如何选择配置；review_daily_results_20261115.py 如何计算相对变化。
再从 provenance 中选一项 W0_R3 和一项 TF_R3，报告实际配置、选中轮次
以及对应验证 MAE。所有回答给出原路径及函数/字段。
读不到文件时请指出具体路径和缺失环节，不要用结果摘要替代源码分析。
```
