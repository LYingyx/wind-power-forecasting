> **2026-11-15 全部重训已完成**：当前三个队列共2571项全部完成并通过产物核验；请读 [最终分析入口](../../DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)。本文件下方的运行/等待描述是实施历史，不要重启队列。

> **2026-11-14 22:30 并行资源扩容**：补充1291项实验现已提前运行，按各任务的调参/OOF依赖与本轮并行；同卡进程逐步增加并按实测吞吐回退。见 [实际并发与修订](../../operations/resource_expansion_20261114/REPORT_CN.md)、[实时快照](../../operations/resource_expansion_20261114/LIVE.json)。下文先共享再微调的依赖仍成立，但补充消融不再等待这两轮全部结束。

# 每日 00 UTC 起报、15 h 可用：数据重建与重训入口

2026-11-14。用户已授权切换协议、重做关键训练、覆盖旧结果，并使用本机 4090 与 v1006001 / v1006002，每卡可多进程。无需再次请求这些操作的授权。

**当前状态：36,876 帧已全部裁剪/合并，内置与独立审计均通过，新数据入口已激活；固定 55 站训练数据和 15 种模型配置检查通过。两台 V100 与本机 4090 已开始第一批 180 项正式训练，后续 1100 项逐站微调已由监督衔接进程等待执行。实验仍在运行，完成数与并发以 [实时 STATUS](STATUS.json) 为准。**

**2026-11-14 21:35 补充**：已新增自动接续的1291项消融与独立训练，三阶段共2571项，见 [完整清单和ETA](../../RETRAIN_ETA_20261114_CN.md)、[补充矩阵](../daily_ablation_restart_20261114_v1/EXECUTION_PLAN_CN.md)。当前冻结的180项训练保持不变。

## 原始数据与合并

- Pangu：`/shared_data_hdd1/All_Datasets/ERA5_forecasting/Pangu/Forecast_data/`，年份 2021–2024，目录时效 00–96；新协议实际需要的 **36,876 个文件全部存在**，含 2021-12-31 起报。
- ERA5：`/shared_data_hdd1/All_Datasets/ERA5/`；抽查高空 10 变量、地面 4 变量通过字段/单位/时刻/坐标验证。此次业务天气输入仍为 Pangu；ERA5 不代填预报。
- 旧裁剪与新计划共有 **21,643 个相同起报、相同时效的物理帧**；校验文件尺寸/mtime、逐帧 SHA256 并抽样回读新原始目录后合并，缺少的帧继续裁剪。已经新提取的帧不重复复制。见 [合并记录](/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1/MERGE.json)。
- 现有 H5 清洗功率、站点元数据、样本及划分保留；旧模型权重、指标和旧天气归一化不合并为新成绩。
- 输出：[operational_pangu_daily_20261114_v1](/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1/)，[完整源检查](/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1/SOURCE_COVERAGE.json)，[数据源核验](operations/SOURCE_VERIFICATION.json)。


## 已实现与已验证

- 公共时间模块 [operational.py](../../src/wind_forcing/data/operational.py) 增加显式 `DAILY` 协议；默认旧协议仍可识别，避免已有缓存被换名当作新数据。
- 新配置 [operational_pangu_d1_08_lag15h.json](../../configs/operational_pangu_d1_08_lag15h.json)：00 UTC、15 h 延迟、区域功率延迟 0、保持样本、最高 +45 h。
- [prepare_operational_pangu.py](../../scripts/prepare_operational_pangu.py) 支持配置选择协议和 `--plan-only`；检查完整源覆盖后才分配天气大数组；构建可断点恢复、逐帧记录清洗后裁剪哈希。
- [audit_operational_pangu.py](../../scripts/audit_operational_pangu.py) 可按 `--bundle` 独立复核新数据；[OperationalFarmData](../../src/wind_forcing/data/operational_farm.py) 接受显式新协议并继续拒绝跨数据指纹的统计量。
- 25 项旧/新协议与数据准入测试通过，包括 2022–2024 全部 15 min 钟点的独立日历核验、跨年/闰日、23:00 切换、0.25–6 h 插值、11 月边界。严格过去六节点实际时效为 +9～+37 h；提取 +9～+45 h 已覆盖。
- 网格缓存现有 train/val/test 的 **31,190 / 32,550 / 27,982** 个起点全部保留，新规则没有新增 11 月排除。
- 当前共享实验沿用 **55 站** QC 和 **31,190 / 29,647** train/val 起点（实验验证集另排除了 2023 年 11 月）。后续训练数据准备必须保留这些选择，不能直接把网格缓存的 32,550 条全部当作当前验证集合。

## 存储与训练读取

全球原文件只读 `[:, :, 171:195, 483:508]`，得到 `(14,24,25)` float32 吉林裁剪。唯一帧表约 **1.15 GiB**；一个起报时效只提取一次，索引将其映射到各功率样本。训练需要的局地 14 通道、历史预报节点、动力特征另行生成，天气统计只用 2022 的实际引用帧重算。

使用 `.npy` / mmap 支持顺序批量读取和操作系统页缓存；逐站模型可将本站张量加载到内存/GPU，共享模型按批装载，避免每个 worker 重复常驻全部网格。缓存中的 MW/CF、站序、origin、issue、available、valid time 均保留明确字段。

## 构建、审计和激活命令

从项目根目录运行：

```bash
.runtime/chronos/bin/python scripts/prepare_operational_pangu.py \
  --config configs/operational_pangu_d1_08_lag15h.json --plan-only

.runtime/chronos/bin/python scripts/prepare_operational_pangu.py \
  --config configs/operational_pangu_d1_08_lag15h.json --workers 8

.runtime/chronos/bin/python scripts/audit_operational_pangu.py \
  --bundle /shared_data_ssd3/liying/operational_pangu_daily_20261114_v1

.runtime/chronos/bin/python scripts/prepare_operational_pangu.py \
  --config configs/operational_pangu_d1_08_lag15h.json --activate
```

最后一条只在构建和独立审计均通过且指纹一致时更新 `data/ACTIVE_DATASET.json`。本轮 [流水线](operations/pipeline.py) 已启动：完成后自动重建训练张量、运行模型及时间核验、冻结文件、部署两台 V100 与本机 4090。查看 [流水线状态](operations/PIPELINE_STATUS.json)。

## 关键实验的迁移范围

| 阶段 | 必须重做的内容 | 实现起点/依赖 |
|---|---|---|
| 训练数据 | 固定 55 站、原实验起点，重算 SEL/DYN/STAB、E2C 柱坐标统计、近邻及三环 | 本轮 [prepare.py](code/prepare.py) 已迁移；使用固定起点选择、重映射新周期索引、重算训练统计 |
| 共享基线与条件机制 | TF、TiDE、TFT、W0，同输入调参及 5 seeds；B0/OOF、CB/CBH/anchor/错配/同容量 TF 控制重训 | 本轮 [plan.py](code/plan.py)、[manifest](manifest.json)：170 项重新训练，加 RM 10 项，共 180 项 |
| 输入与共享归因 | 同输入共享/独立；历史×天气四格；本站/区域对照 | 必须统一新天气和区域可用时刻，不能拼接旧成绩 |
| 个性化 | W0/TF 共享底座、全参数与输出层微调 | [新微调入口](../daily_site_finetune_20261114_v1/EXECUTION_PLAN_CN.md)：1100 项，共享完成后自动准备/审计/部署；保存统一 E1/E6 与逐站选择 |
| RM / Z1 | W0-RM、TF-RM；新模型预测上的严格因果滚动校正 | [RM 数据](code/prepare_rm.py) 已改为 24 点同刻、无首步补零；Z1 在新微调 P0/E1 预测上重做，所有观测延迟 0 |
| LM | 按每个历史时刻 τ 的新可用期次重建配对档案；依据新 Z1/RM 结果决定完整 LM 及匹配对照 | [小时档案构建](code/prepare_lm.py) 已迁移并安排在训练数据就绪后执行；LM 模型及匹配对照仍待新 Z1/RM 结果决定 |
| 2024 复评 | 冻结 2023 选择后，全场逐 seed、MAE/RMSE、高功率、时效/场景 | 仅报告新协议重训结果；2024 已被查看，仍为探索性跨年复评 |

区域功率延迟 0 的具体修改：原 24 点历史 `h` 的末点是 t；保持近邻输入 22 点长度时，从旧 `h[...,1:23]` 改为 `h[...,2:24]`，三环当前值使用下标 23，1 h 变化比较 23 与 19。RM 的逐时刻区域序列与本站历史同刻对齐，不再人为将首步补零。相关负对照/遮蔽及训练期标准化须同步。

## 机器与调度

已只读确认 v1006001 / v1006002 免密 SSH 可达，分别 8 张 V100 16GB；本机有 8 张 4090，按实时显存和算力余量准入。实际启动时重查负载，不能依据本次快照强占其他任务。

两台 V100 上限 8 worker/卡，本机上限 4 worker/卡；按显存预留和实测吞吐逐步增加，最少保留 max(4 GiB,20%) 显存。数据复制到远端独立磁盘后逐文件验证哈希，GPU 冒烟通过才启动新 worker。已运行后台衔接流水线；是否已进入正式训练请看实时状态，不以代码或队列准备完成代替训练完成。

所有旧结果均不作为新协议成绩。必须区分数据准备、队列部署、实际训练及完成；仅以本轮可验证产物报告进度。

## 本轮后台衔接与证据

- [裁剪合并记录](/shared_data_ssd3/liying/operational_pangu_daily_20261114_v1/MERGE.json)：复用 20,890 帧，另有 1,280 帧已新提取，合并时剩余 14,706 帧。
- [执行范围与判定](EXECUTION_PLAN_CN.md)、[数据源核验](operations/SOURCE_VERIFICATION.json)、[流水线日志](operations/pipeline.log)。
- [共享结果入口](analysis/RESULTS_CN.md)、[逐风场结果](analysis/per_farm_metrics.csv) 在首次训练回传后生成。
- [自动后续阶段](operations/followup.py) 在本轮 180 项完成后安排 1100 项逐站微调；不沿用旧机制的失败结论停止本次微调。
- 新风场迁移需目标站完全留出；本轮已见站微调不构成新风场迁移证据。共享/独立归因、历史×天气2×2、天气编码与输入消融已加入后续1291项自动队列；LM模型训练与2024复评仍不在这批训练队列，后续必须使用本轮产物。

本次启动的可核验证据：[LAUNCH_VERIFIED](operations/LAUNCH_VERIFIED.json)、[三机监督健康](operations/CONTROL_HEALTH.json)、[后续微调状态](operations/FOLLOWUP_STATUS.json)。数据源索引：[SOURCE_LOCATIONS](../../data/SOURCE_LOCATIONS.json)。
