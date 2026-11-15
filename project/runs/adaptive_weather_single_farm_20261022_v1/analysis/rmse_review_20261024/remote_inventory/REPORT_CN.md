# 跨服务器结果盘点与历史 RMSE 补充

日期：2026-10-24。目的：检查 P60002/P60003 是否保留遗漏结果，并判断它们是否改变当前独立单风场模型的 RMSE 结论。

## 1. 连接、存储与搜索范围

免密连接已验证：`P60002 → node1`、`P60003 → node2`，用户均为 `liying`。以后直接使用 `ssh P60002` 或 `ssh P60003`。

两机的 `/home` 是各自本地盘，但 `/shared_data_ssd1`、`/shared_data_ssd2`、`/shared_data_ssd3`、`/shared_data_hdd1` 与当前服务器挂载同一套共享存储。当前轮次 `COMPLETION_AUDIT.json` 在三机上的 SHA256 一致：`4288a2dc36749d8dce496a6d0f6ef4981b47922612a89f6dbb1018718fcd6ea4`。从不同 SSH 入口看到这些文件不代表新增实验或独立复验。

已检查两机 `/home/liying`、`/data`、`/mnt`、迁移前本地备份目录、`/data_large`、`/data_small`、`/data_ssd`、`/home_backup`、`/hj102_pack`，以及 node2 的 `/data_bgfs`、`/data_merge`、`/tmp_ssd`、`/ssd_test`。在这些位置未发现额外的本项目结果目录；未进入其他用户目录。检查时两机属于本用户的进程工作目录均为 `/home/liying`。这是所列位置的盘点，不宣称遍历了系统所有文件。环境、下载缓存及凭据目录不作为实验目录扫描。

原始盘点证据：[SERVER_AUDIT.json](SERVER_AUDIT.json)。没有读取或复制私钥、密码、令牌，也没有改动远端实验文件。

## 2. 找到的其他结果在哪里

最有用的历史工程位于：

`/shared_data_hdd1/liying/hj102/huawei/storage/liying/new_try/wind_cluster_forced_evolution/`

它保留了主项目轻量历史归档之外的原始运行目录、预测文件和恢复训练记录。这里的“补充”需要分清两种情况：**历史汇总已归档，但原始预测仍在旧盘；以及当前主项目指标目录中没有收录的更早运行文件。它们都不能称作刚完成的新实验。**

| 历史结果 | 可用证据 | 与当前实验的关系 |
| --- | --- | --- |
| `baselines_mse_station_20260911` | 13 类 MSE 基线、seed42–44，原配置及评估结果 | 旧共享训练；不能替换本轮独立训练基线 |
| `station_clean_20260912` | Transformer MAE / V1 Direct MAE，各10 seeds，从头训练；逐站指标与原预测 | 旧共享训练的结构对照 |
| `physics_gate_20260912` | Physics Gate，3 seeds，原预测可复算 | MSE训练的历史物理门控候选 |
| `level_shape_20260912` | Level/Shape，3 seeds，原预测可复算 | MSE训练的历史水平/形状分解候选 |
| `results_20260912` | 逐站指标、恢复训练修正及完整分析 | 包含 warm restart；不能与 clean 重训混称 |
| SSD1/SSD2 的 `new_try*` 与迁移工程 | E2C、共享基线、迁移/少样本等旧结果 | 与主项目逐文件核对，重复文件不计为新实验 |

已核验以下四张表在旧 HDD 与主项目 `historical_reports` 中逐字节相同：`station_clean_20260912/clean_per_station_metrics.csv`、`results_20260912/all_per_station_metrics.csv`、`physics_gate_20260912/metrics.csv`、`level_shape_20260912/metrics.csv`。因此这些表本身不是新增成绩。物理候选的轻量 `metrics.csv` 仅有 MAE；本次从旧盘原预测补算其逐站 RMSE。

另外找到 `roa_c2f_wind_forecasting`、ASTAFN 和早期 `results` 工程。ROA 任务是58站功率之和，ASTAFN包含California/GuangDong/YunNan数据，早期Paper Factory记录又有不同预测步长；这些结果的目标、数据或指标口径不同，列为历史入口，不加入当前56站独立训练排名。目录存在及 registry 状态不等于预测级审计通过。

## 3. 对 RMSE 结论的影响

以下全部为2024，在各自协议内按同一风场、seed配对。计数单位是“风场×seed”，不是独立风场数；只要误差严格降低就记改善，不设百分比门槛，也不跨风场或seed平均。不同协议的行不能按胜出数横向排名。

| 协议 | 候选 / 同期对照 | 配对数 | MAE更好 | RMSE更好 |
| --- | --- | --- | --- | --- |
| 当前独立单风场训练 | ST11 / Transformer | 560 | 385 | 208 |
| 历史共享训练，另有输入/精度差异 | E2C / 共享Transformer | 560 | 492 | 479 |
| 历史共享训练，另有输入/精度差异 | E2C / 共享TiDE | 560 | 508 | 487 |
| 历史共享、从头训练 | V1 Direct MAE / clean Transformer MAE | 560 | 216 | 212 |
| 历史共享，结构/损失/训练预算不同 | Physics Gate / clean Transformer MAE | 168 | 30 | 136 |
| 历史共享，结构/损失/训练预算不同 | Level/Shape / clean Transformer MAE | 168 | 15 | 131 |

**当前 ST11 相对 Transformer 的 RMSE 短板仍成立；“项目里所有模型都比基线 RMSE 差”则不成立。** 历史共享 E2C 和 MSE 物理候选都有 RMSE 改善，但不是当前独立 ST11 的结果。

E2C 是柱坐标天气MLP编码加冻结历史轨迹条件修正；这组历史模型还用了区域历史，且采用FP32，而共享基线为BF16。它不是当前“仅本站输入、逐站独立训练”的E2C或ST11。这里读取已保存的逐站逐seed表，没有用旧报告的跨站均值替代原值。

Physics Gate 是历史基准加天气修正与物理门控；Level/Shape 将输出分成水平项与零均值轨迹形状项；clean Transformer 以未来天气构造query、历史功率构造memory，使用encoder/decoder交互。历史配置中物理候选最多80 epochs，clean Transformer最多40 epochs，损失也不同。这组比较证明已有另一种MAE/RMSE取舍，**不能单独证明门控、分解或MSE损失中的哪一项带来收益**。

当前两轮模型及基线仍都是MAE训练、按验证MAE选择checkpoint。发现历史MSE基线并不改变这一事实。后续若验证MSE/混合损失，应在当前独立协议内固定结构、预算和checkpoint选择规则做受控对照，不能拿这些旧成绩替代。

### 历史物理候选逐seed方向

| 模型 | seed | 风场数 | MAE更好 | RMSE更好 |
| --- | --- | --- | --- | --- |
| Physics Gate | 42 | 56 | 9 | 41 |
| Physics Gate | 43 | 56 | 10 | 47 |
| Physics Gate | 44 | 56 | 11 | 48 |
| Level/Shape | 42 | 56 | 1 | 42 |
| Level/Shape | 43 | 56 | 10 | 44 |
| Level/Shape | 44 | 56 | 4 | 45 |

原先固定代表站之一双龙（station_index=14），seed42的实际值如下；完整56站见CSV，不能以该例替代全部结果。

| 模型 | MAE | RMSE |
| --- | --- | --- |
| clean Transformer MAE | 0.070203403 | 0.112991710 |
| Physics Gate | 0.072810157 | 0.112620639 |
| Level/Shape | 0.073919089 | 0.112747298 |

## 4. 核验与结果入口

本次直接读取旧盘9份预测归档：6份物理候选和3份同期Transformer参考。逐一确认origin、sample_index及真值完全相同，预测有限；重算56站MAE/RMSE，核对原MAE与已有参考指标。保留336条物理候选实际值，以及包含clean结构对照、共享E2C对照的3696条逐站逐seed配对。没有复制预测、checkpoint，也没有新增训练。

- [物理候选逐风场逐seed实际值](historical_physical_actual_per_farm_seed.csv)
- [全部历史配对实际值及来源](historical_paired_actual_values.csv)
- [各seed改善方向计数](historical_direction_counts_per_seed.csv)
- [9份原预测核验记录](HISTORICAL_CHECK.json)
- [主项目已归档的clean逐站原值](../../../../../historical_reports/station_clean_20260912/clean_per_station_metrics.csv)
- [主项目已归档的旧基线及恢复训练逐站原值](../../../../../historical_reports/results_20260912/all_per_station_metrics.csv)
- [原独立训练RMSE报告](../REPORT_CN.md)

旧盘部分顶层文件为指向迁移前`/storage/...`的绝对软链接。读取时定位实际`attempts/...`文件，未修改链接或重存数组。指标文件去重清单中的文件数包含不同split、attempt及诊断文件，不能作为已完成模型数。

## 5. 文件清单状态

上文9份预测归档的科学核验、逐风场逐seed表和服务器共享存储检查均已完成。另行尝试的8套旧工程指标文件递归总清单中途停止，尚未产出完整索引，因此不能把本报告解释为全部历史文件格式的穷尽盘点。相关脚本保留在 `maintenance/inventory_remote_results.py`，后续可按需续做；本次新物理机制任务不依赖该总清单。不存在的索引链接已移除。
