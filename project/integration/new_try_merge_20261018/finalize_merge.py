from pathlib import Path
import json,shutil,sys,datetime
P=Path('/shared_data_ssd2/liying/wind_cluster_project_20261018/project');T=P/'integration/new_try_merge_20261018'
sys.path.insert(0,str(P/'runtime'))
from wind_paths import OLD_ROOTS

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def preserve(p):
 b=T/'previous_status'/p.relative_to(P)
 if p.exists() and not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
copy=read(T/'COPY_VERIFIED.json');catalog=read(P/'results/CATALOG_STATUS.json');core=read(P/'integration/COMPATIBILITY.json');extended=read(P/'integration/EXTENDED_COMPATIBILITY.json');audit=read(P/'integration/FINAL_RECORD_AUDIT.json');legacy=read(P/'integration/LEGACY_CSV_VERIFIED.json')
assert core['supported_profiles_pass'] and extended['passed'] and audit['passed'] and not catalog['errors'] and legacy['all_family_means_match_original_summaries']
for name in ('MERGE_STATUS.json','COLUMN_WEATHER_MISSING.json','EXTERNAL_DEPENDENCIES.json','PATH_AND_QUEUE_CHECK.json','RUNTIME_ADAPTATIONS.json'):
 preserve(P/'integration'/name)
preserve(P/'README_CN.md')
status={'status':'merged_and_verified','project_root':str(P),'incoming_source':'/shared_data_ssd2/liying/new_try/wind_cluster_forced_evolution','completed_at':datetime.datetime.now().astimezone().isoformat(),'imported_or_reused_files':copy['files'],'identical_files_reused':copy['counts']['identical'],'conflicting_versions_preserved':copy['counts']['replace_preserve_old'],'legacy_table_files_recovered':10,'artifact_groups':catalog['artifact_groups'],'run_split_metric_records':catalog['run_split_records'],'canonical_legacy_csv_runs':catalog['canonical_csv_runs'],'extended_checkpoints_strictly_loaded_and_forward_tested':sum(x['checkpoints'] for x in extended['checks'].values()),'core_checks_passed':True,'extended_checks_passed':True,'frozen_record_audit_passed':True,'existing_tests_passed':45,'column_tests_passed':2,'current_cuda_model_forward_passed':True,'raw_hdf5_local':True,'external_data_dependencies_for_supported_profiles':[],'source_directories_and_environments_preserved':True,'entrypoint':'./project','limits':['本次未重新训练或全量复算历史指标；跨 PyTorch/CUDA/精度不承诺逐位数值相同。','旧探索记录有仅保留报告/指标的项目；checkpoint_exists=0 表示未随轻量归档携带权重。','旧启动器、冻结脚本和队列属于历史证据；当前运行使用 ./project。旧探索的中间缓存未补复制。']}
save(P/'integration/MERGE_STATUS.json',status)
save(P/'integration/COLUMN_WEATHER_MISSING.json',{'status':'resolved_by_new_try_merge','variants':['E0','E1','E2','E2C','E2L','E2N','E3'],'seeds':list(range(42,52)),'complete_runs':70,'source_configs_dem_stats_checkpoints_predictions_present':True,'hash_verification':'integration/FINAL_RECORD_AUDIT.json','model_verification':'integration/EXTENDED_COMPATIBILITY.json','previous_report':str((T/'previous_status/integration/COLUMN_WEATHER_MISSING.json').relative_to(P))})
save(P/'integration/EXTERNAL_DEPENDENCIES.json',{'active_project_data_external_links':[],'raw_hdf5':'data/jilin_56farm_pad8.h5','raw_hdf5_is_local_regular_file':True,'source_environment_inventory':'integration/new_try_merge_20261018/SOURCE_ENVIRONMENTS.json','historical_provenance_paths_are_not_runtime_dependencies':True})
save(P/'integration/PATH_AND_QUEUE_CHECK.json',{'current':audit['checks']['active_project_external_links'],'archived_queues':audit['checks']['archived_queue_integrity'],'configuration_inputs':audit['checks']['runtime_config_inputs'],'historical_queues_are_not_resumed':True})
save(P/'integration/PATH_MAPPINGS.json',{'old_project_roots':list(OLD_ROOTS),'current_root':str(P),'runtime_rule':'Rebase configuration strings in memory; load experiment-specific source snapshots in isolated processes. Preserve archived bytes and their frozen hashes.','relative_result_paths':True,'new_outputs':'runs/ or an explicit new external path; imported evidence protected','independent_python':'.runtime/chronos/bin/python'})
old=read(P/'integration/RUNTIME_ADAPTATIONS.json');save(T/'RUNTIME_ADAPTATIONS_PREVIOUS.json',old)
save(P/'integration/RUNTIME_ADAPTATIONS.json',{'existing_adapters_preserved':old,'new_modules':['runtime/archive_loader.py','runtime/unified_predict.py','runtime/source_train.py','runtime/result_catalog.py','runtime/build_asset_index.py','runtime/merge_csv_results.py','runtime/check_extended.py','runtime/audit_final_records.py'],'archive_model_logic_changed':False,'new_runtime_path_maps':'integration/PATH_MAPPINGS.json','trained_checkpoint_checks':'integration/EXTENDED_COMPATIBILITY.json'})
readme='''# 风电预测统一项目

唯一工作入口：`/shared_data_ssd2/liying/wind_cluster_project_20261018/project`。

这里合并了原 `/storage` 项目、之前的 `new_try_v2` 内容，以及本次迁回的 `new_try/wind_cluster_forced_evolution`。原始目录与已有环境保留；以后在这里查结果和开展新实验，避免把两个来源分别维护成新项目。

## 结果入口

- `results/CATALOG.sqlite`：统一结果数据库。`observations` 存每次运行/划分的标量指标，`assets` 存原始报告及表格路径，`legacy_csv_rows` 存历史 CSV 原始行。
- `results/RUN_METRICS.csv`：{records:,} 条运行/划分指标，包括补回的 8,400 组全场站独立基线及 1,100 组早期扩展基线。
- `results/PROTOCOL_SUMMARY.csv`：按实验、模型、精度、场站组、训练预算、季节、划分分别汇总；先在站内平均种子，再等权平均场站。
- `results/EXPERIMENT_INDEX.csv`、`RESULT_ASSETS.csv`：{groups} 个实验/资料目录及报告索引。
- `results/SLAST_MAIN_RESULTS.csv`：原 SLAST 论文汇总继续保留。

```bash
cd /shared_data_ssd2/liying/wind_cluster_project_20261018/project
./project status
./project results --experiment column_weather_20260928_v1 --split test
./project results --experiment selected30_fewshot_baselines_20260920_v1 --budget 7d --split test
./project results --experiment expanded_all_station_baselines_20260919_v1
```

结果库使用项目相对路径，每行可追溯原始指标文件。`checkpoint_exists=0` 表示历史轻量归档只保留指标/报告，不能把它当作已有可加载权重。`legacy_csv_rows` 中部分表存在历史重复，它是原始证据表；常规统计使用 `observations` 和协议汇总表。

重新扫描结果：`./project results --rebuild`。它只重建派生索引，不改写原始实验记录。

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

`predict` 可加 `--device cuda`、`--split test` 和 `--output runs/example_predictions.npz`。它执行指定样本的 FP32 推理，不把这些样本当作全量指标，也不冒充原 FP16/BF16 评测。输入中的旧路径在内存中迁移，原 checkpoint、配置、冻结记录和哈希保持不变。T1–T4 的部分参数 checkpoint 会与对应源模型/adapter 组合后校验。

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

- 本轮 {files:,} 个导入/复用文件逐项 SHA256 核验，{duplicates:,} 个相同文件复用，{conflicts} 个不同版本的旧文件放入 `provenance/before_new_try_merge_20261018/`。
- 原 SLAST/CBH 兼容检查通过；新增 455 个代表性最终 checkpoint 严格加载并用真实样本前向通过，覆盖全部源模型/种子及目标模型族/协议/种子代表。全部导入文件另有字节完整性校验，并非逐个重跑 63,000 次实验。
- 110 份正式配置输入路径有效；70 组天气编码器、40 组条件消融的冻结权重/配置等哈希匹配；5 个存档队列 SQLite 完整。
- 45 项原有测试、2 项天气编码器测试通过，当前 RTX 4090 D 上 E2C 推理通过。
- 13 GiB 原始 HDF5 已是本地普通文件，核心缓存与目标站缓存也在项目内；已解除原 HDD 数据软链接依赖。
- 重复环境、迁移压缩包、训练日志、训练历史、smoke/benchmark/tmp 等临时内容未作为正式结果再次整批复制。保留最终模型、完整预测、结果和继续运行所需缓存。

环境：Python 3.11.0、PyTorch 2.11.0+cu128。迁回的 V100 实验原环境还使用过 PyTorch 2.5.1+cu121，版本清单存于 `integration/new_try_merge_20261018/SOURCE_ENVIRONMENTS.json`。检查证明当前主要入口可加载运行，不表示跨版本逐位数值一致或已经全量复算历史性能。旧探索的中间缓存和部分旧权重按原轻量归档策略未携带；相关报告仍在索引内。

完整复查：`./project check`。记录见 `integration/MERGE_STATUS.json`、`COMPATIBILITY.json`、`EXTENDED_COMPATIBILITY.json`、`FINAL_RECORD_AUDIT.json`、`LEGACY_CSV_VERIFIED.json`。本次明细、复制策略和测试日志在 `integration/new_try_merge_20261018/`。

`artifacts/` 内的旧绝对路径、原启动器、watchdog、冻结脚本和队列仅保留作历史证据。它们不作为当前运行入口，也不会自动恢复旧服务器任务。`paper/`、`figures/` 保留原内容；迁回服务器的论文闭环和图片在各自 `artifacts/paper_*` 目录下。
'''.format(records=catalog['run_split_records'],groups=catalog['artifact_groups'],files=copy['files'],duplicates=copy['counts']['identical'],conflicts=copy['counts']['replace_preserve_old'])
(P/'README_CN.md').write_text(readme)
parent=P.parent/'README_CN.md';backup=T/'parent_README.before.md'
if parent.exists() and not backup.exists():shutil.copy2(parent,backup)
parent.write_text('''# 风电项目 SSD2 统一归档

当前唯一工作项目为 `project/`，入口和使用说明见 [project/README_CN.md](project/README_CN.md)。已合并原 /storage 整理副本、new_try_v2 和本次迁回的 /shared_data_ssd2/liying/new_try 项目。

- `project/`：代码、最终模型、数据、统一结果索引和已验证运行入口。
- `archives/slast_essential/`：早期轻量归档。
- `archives/slast_full_incomplete/`：以前中断的大归档，不是完整实验副本，也不是运行入口。
- `migration/`：SSD1 → SSD2 转移、清理和校验记录。

SSD1 的本次整理副本已在 SSD2 校验后清理。源 new_try/new_try_v2、已有环境和 HDD 原备份继续保留。最新合并状态见 `project/integration/MERGE_STATUS.json`。
''')
print(json.dumps(status,ensure_ascii=False,indent=2))
