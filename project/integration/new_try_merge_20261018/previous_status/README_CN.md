# 合并后的风电预测项目

当前位置：`/shared_data_ssd2/liying/wind_cluster_project_20261018/project`。

已合并原 SLAST 项目与 `new_try_v2` 的迁移实验、column_weather 结果。导入的 9,263 个文件逐个校验 SHA256；71 个相同文件去重，4 个同名不同版本均保留。保留各实验自己的训练时源码快照，没有混合不同模型实现。原始实验记录中的旧绝对路径和哈希保持不变。

## 使用入口

在此目录运行：

```bash
./project status
./project check
./project slast-predict --seed 42 --split val --samples 2
./project slast-predict --seed 42 --split test --samples 16 --output runs/slast_seed42_test_sample.npz
./project fewshot --help
./project transfer --help
```

上面的预测命令只推理指定样本。需要训练时，显式运行：

```bash
./project slast-train --seed 42 --device cpu --output runs/slast_new_seed42
```

训练命令会实际开始训练；本次整合未执行训练。GPU 可用时可通过 `CUDA_VISIBLE_DEVICES=0` 配合 `--device cuda` 选择当前服务器的可用设备，不沿用旧服务器的固定多卡列表。新结果写入独立 `runs/` 或明确指定的新目录，不覆盖导入结果。默认 CPU；不启动 watchdog、不恢复旧队列。

请通过 `./project` 启动。它自动使用本目录 `.runtime/chronos` 独立环境，以及相应实验的源码版本；不依赖当前 shell 的 Conda 激活状态或工作目录。`artifacts/` 下旧的队列、freeze、迁移脚本是原始证据，仍带原实验的冻结约束，不应当作当前服务器启动入口。可执行路径适配放在 `runtime/`，原文件及其冻结哈希未被改写。

## 已验证

- 原环境包 SHA256 已核验，环境在 SSD2 新前缀重新解包并执行 conda-unpack；Python 3.11.0、PyTorch 2.11.0+cu128、NumPy 1.26.4、h5py 3.15.1。
- 10 个 SLAST checkpoint 严格加载，模型配置、数据 fingerprint、原验证/测试记录中的 checkpoint SHA256 均一致；冻结基线张量逐项相同；真实缓存样本输出形状为 2×56×24。
- 10 个 CBH source checkpoint 可加载；transfer T1/T2/T3/T4 在真实目标站样本上前向通过。
- 15 个 few-shot 模型族真实样本前向通过；433 个导入的 few-shot checkpoint 严格加载通过。
- 484 个 Python 文件语法检查通过，45 项现有模型/数据/预处理/分析测试通过，CUDA 基础张量运算通过。

这不代表重新完成全量实验或复算全量性能。检查结果见 `integration/COMPATIBILITY.json`、`integration/pytest.log`。`./project check` 可以复查当前环境。

## 内容入口

- `artifacts/p0i_summary_calibration_20260916_v1/`：SLAST 源码快照、配置、10 种子最终模型和基线、最终指标。
- `artifacts/cross_farm_shared_specific_20260920_v1/`：跨站迁移代码、源 adapter、30 站输入缓存和已有结果。
- `artifacts/selected30_fewshot_baselines_20260920_v1/`：few-shot 模型与已有结果。
- `artifacts/column_weather_20260928_v1/`：新服务器上传的 E2L 结果，当前仅作结果归档。
- `results/EXPERIMENT_INDEX.csv`：实验内容索引；`results/SLAST_MAIN_RESULTS.csv`：原论文 8 模型×2 年汇总。
- `paper/`、`figures/`：当前论文、说明和 v18 框架图。
- `provenance/conflicting_versions/`：同名文件的旧版本；`provenance/later_transfer_snapshot/`：另一时点的代码/报告；`provenance/queue_snapshots/`：仅存档的旧 SQLite 队列。
- `integration/`：来源、哈希、冲突处理、路径适配、环境与验证记录。

## 尚缺内容与依赖

`column_weather/E2L` 缺源码、原始配置、DEM 和归一化统计文件；seed44 缺 checkpoint/验证结果，seed47 目录缺失。不能据现有输出声称该实验已完整合并、可以重新推理或复现。详见 `integration/COLUMN_WEATHER_MISSING.json`。另一个正在接收的 `/shared_data_ssd2/liying/new_try` 未完成传输，本轮未把它当成稳定来源。

原 SLAST 的 6.92 GiB 运行缓存已完整放在本目录，并核验了 59 个文件；模型训练和预测可直接读取本地缓存。13 GiB 原始 HDF5 为节省空间保留指向 HDD 原始备份的软链接，仅重新构建缓存时需要；依赖清单见 `integration/EXTERNAL_DEPENDENCIES.json`。因此这不是脱离所有外部数据也能从零重建的独立数据包。

跨站实验原来仍有大量 pending 任务，本次未把它们改为完成，也未续跑。不同实验的年份划分、场站集合和指标定义继续按各自协议保留，未强行混成同一性能表。

## 迁移

SSD1 的本次整理副本在 SSD2 对应副本校验后删除。原 `new_try_v2`、`.conda`、`conda` 和 HDD 源备份未修改。上层 `migration/TRANSFER_VERIFIED.json`、`SSD1_CLEANUP.json` 记录迁移与清理。
