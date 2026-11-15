# 风电预测统一项目

**项目总入口（2026-10-19）：[框架、已有结果、代码地图与三项贡献计划](PROJECT_HANDOFF_CN.md)。** 接手项目请先读此文；包含全部实验索引、关键代码与权重、当前研究路线及待补对照。

唯一工作入口：`/shared_data_ssd2/liying/wind_cluster_project_20261018/project`。

这里合并了原 `/storage` 项目、之前的 `new_try_v2` 内容，以及本次迁回的 `new_try/wind_cluster_forced_evolution`。原始目录与已有环境保留；以后在这里查结果和开展新实验，避免把两个来源分别维护成新项目。

## 结果入口

2026-10-19 路线复核：[共同训练的已有风场跨年预测：性能优势与代表风场评价](runs/single_farm_route_audit_20261019/ROUTE_RECOMMENDATION_CN.md)。按当前目标，以 E2C-CBH 的已知风场预测为主线，新风场迁移退出主任务。已核对 157,313 条索引记录对应的 74,406 份原始指标文件，并补充全部15基线、验证集逐站选基线及分时效对比；本轮未新增训练。

2026-10-18 单风场方向重审：[研究结论、更新表格与图件](runs/single_farm_review_20261018/README_CN.md)。该目录重新区分全年共享模型、独立 pilot 和未见站适配，并给出训练描述量选站、分场景复算及贡献边界；原实验记录保持不变。

- `results/CATALOG.sqlite`：统一结果数据库。`observations` 存每次运行/划分的标量指标，`assets` 存原始报告及表格路径，`legacy_csv_rows` 存历史 CSV 原始行。
- `results/RUN_METRICS.csv`：157,313 条运行/划分指标，包括补回的 8,400 组全场站独立基线及 1,100 组早期扩展基线。
- `results/PROTOCOL_SUMMARY.csv`：按实验、模型、精度、场站组、训练预算、季节、划分分别汇总；先在站内平均种子，再等权平均场站。
- `results/EXPERIMENT_INDEX.csv`、`RESULT_ASSETS.csv`：39 个实验/资料目录及报告索引。
- `results/SLAST_MAIN_RESULTS.csv`：原 SLAST 论文汇总继续保留。

```bash
cd /shared_data_ssd2/liying/wind_cluster_project_20261018/project
./project status
./project results --experiment column_weather_20260928_v1 --split test
./project results --experiment selected30_fewshot_baselines_20260920_v1 --budget 7d --split test
./project results --experiment expanded_all_station_baselines_20260919_v1
```

结果库使用项目相对路径，每行可追溯原始指标文件。`checkpoint_exists=0` 表示历史轻量归档只保留指标/报告，不能把它当作已有可加载权重。`legacy_csv_rows` 中部分表存在历史重复，它是原始证据表；常规统计使用 `observations` 和协议汇总表。

重新扫描导入结果：`./project results --rebuild`。它只重建派生索引，不改写原始实验记录。新探索产物放入 `runs/`，不会自动混入已冻结的正式结果。

## 推理与继续探索

请通过 `./project` 启动。它使用当前项目 `.runtime/chronos` 环境和各实验训练时源码快照，不依赖 shell 中激活的环境。支持 SLAST、CBH、天气编码器、条件消融、matched direct/correction、共享/少样本基线、Transformer 迁移和 CBH/E2C 跨站适配的最终 checkpoint。

```bash
./project predict --checkpoint artifacts/column_weather_20260928_v1/runs/main/E2C/seed42/best_model.pt --samples 2
./project predict --checkpoint artifacts/p0i_condition_ablation_completion_20261014_v1/results/CBH_seed42/best_model.pt --samples 2
./project slast-predict --seed 42 --split test --samples 2
./project fewshot --help
./project transfer --help
./project source-train --help
```

`predict` 可加 `--device cuda`、`--split test` 和 `--output runs/example_predictions.npz`。它执行指定样本的 FP32 推理，不把这些样本当作全量指标，也不冒充原 FP16/BF16 评测。自定义新输出路径可通过 `--kind column` / `--kind condition` 等显式指定模型族。输入中的旧路径在内存中迁移，原 checkpoint、配置、冻结记录和哈希保持不变。T1–T4 的部分参数 checkpoint 会与对应源模型/adapter 组合后校验。

准备新的天气编码器实验配置（不训练）：

```bash
./project source-train --experiment column --variant E2C --seed 42 --output runs/e2c_plan --prepare-only
```

实际开展新实验时使用另一个新目录，省略 `--prepare-only`：

```bash
./project source-train --experiment column --variant E2C --seed 42 --device cuda --output runs/e2c_new
./project source-train --experiment condition --variant CBH --seed 42 --device cuda --output runs/cbh_new
```

上面两条命令会开始训练，本次整合没有执行训练。新输出必须使用未存在的目录；相对输出路径以项目根目录为基准。原 SLAST 训练入口是 `slast-train`；fewshot/transfer 通过相应子命令写入 `runs/`。GPU 编号通过当前机器的 `CUDA_VISIBLE_DEVICES` 选择。

## 实验边界

| 实验 | 本地正式结果 | 必须保留的区别 |
|---|---:|---|
| 天气编码器 E0/E1/E2/E2C/E2L/E2N/E3 | 70 组 | FP32；源 56 站，2022/2023/2024 |
| 新条件消融 CBH/CB/CH/CNEUTRAL | 40 组 | 全部新训 FP16，不能混入旧 BF16 CBH |
| matched direct/correction | 10 组 | 5 对种子，FP16；direct 使用专用重建模型 |
| 共享单站架构基线 | 150 组 | 15 个模型族 × 10 种子，共享参数 |
| 30 站少样本独立基线 | 63,000 组 | dev 21,000 / final 42,000 |
| 预训练 Transformer 迁移 | 1,400 组 | 20 站、5 种子、14 个预算/季节块 |
| E2C-T5 迁移 | 1,400 组 | FP32；不能宣称与旧 BF16 T5 精度完全配平 |
| 56 站独立基线明细 | 8,400 组 | 2023 pilot 验证，不能直接等同于全年评测 |

跨站任务的训练是 2023，选择集是 2024 H1，最终评测是 2024 H2。源 56 站任务和 held-out 目标站任务分开统计；种子、季节块不能冒充独立场站样本。`PROTOCOL.md`/`PROTOCOL_CN.md` 与原始报告保留了更具体的约束。

## 验证与存储

- 本轮 296,810 个导入/复用文件逐项 SHA256 核验，5,959 个相同文件复用，45 个不同版本的旧文件放入 `provenance/before_new_try_merge_20261018/`。
- 原 SLAST/CBH 兼容检查通过；新增 455 个代表性最终 checkpoint 严格加载并用真实样本前向通过，覆盖全部源模型/种子及目标模型族/协议/种子代表。全部导入文件另有字节完整性校验，并非逐个重跑 63,000 次实验。
- 110 份正式配置输入路径有效；70 组天气编码器、40 组条件消融的冻结权重/配置等哈希匹配；5 个存档队列 SQLite 完整。
- 45 项原有测试、2 项天气编码器测试通过，当前 RTX 4090 D 上 E2C 推理通过。
- 13 GiB 原始 HDF5 已是本地普通文件，核心缓存与目标站缓存也在项目内；已解除原 HDD 数据软链接依赖。
- 重复环境、迁移压缩包、训练日志、训练历史、smoke/benchmark/tmp 等临时内容未作为正式结果再次整批复制。保留最终模型、完整预测、结果和继续运行所需缓存。

环境：Python 3.11.0、PyTorch 2.11.0+cu128。迁回的 V100 实验原环境还使用过 PyTorch 2.5.1+cu121，版本清单存于 `integration/new_try_merge_20261018/SOURCE_ENVIRONMENTS.json`。检查证明当前主要入口可加载运行，不表示跨版本逐位数值一致或已经全量复算历史性能。旧探索的中间缓存和部分旧权重按原轻量归档策略未携带；相关报告仍在索引内。

完整复查：`./project check`。记录见 `integration/MERGE_STATUS.json`、`COMPATIBILITY.json`、`EXTENDED_COMPATIBILITY.json`、`FINAL_RECORD_AUDIT.json`、`LEGACY_CSV_VERIFIED.json`。本次明细、复制策略和测试日志在 `integration/new_try_merge_20261018/`。

`artifacts/` 内的旧绝对路径、原启动器、watchdog、冻结脚本和队列仅保留作历史证据。它们不作为当前运行入口，也不会自动恢复旧服务器任务。`paper/`、`figures/` 保留原内容；迁回服务器的论文闭环和图片在各自 `artifacts/paper_*` 目录下。
