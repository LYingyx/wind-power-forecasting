# 物理与观测增强：独立单风场受控实验

> **2026-10-25 完成更新（优先）**：3432项2023筛选已于00:58:33完成，失败0；[完成审计](COMPLETION_AUDIT.json)逐一核验5286份选定权重、训练轨迹选轮、27456行实际指标及47520行原有配对。两台V100的1406/1129项结果全部归档，三台机器无遗留训练，远端空闲调度器已停止。旧完成标志中残留4个历史进程、pending=-4的显示错误已修正。阅读 [完整结果与缩写说明](analysis/SCREENING_RESULTS_20261025_CN.md)、[实际逐风场逐seed表](analysis/new_actual_per_farm_seed.csv)及[完成标志](FINISHED.json)。完成范围是2022训练/2023筛选，不包含2024复评或结构模块扩56站。

> **机器范围与调度更新**：按用户10月24日晚的新指示，仅使用当前4090与v1006001/v1006002。P60002/P60003已停用。两台V100免密连接已成功；不共享盘，使用精简数据副本、同版本V100兼容PyTorch和中央任务锁/回传校验。调度已改为逐卡资源扩缩容，详见 [运营说明](operations/README_CN.md)。最新实际启动/完成数量以STATUS及REMOTE_STATUS为准。

项目级判断与完整修订方案见 [PHYSICS_OBSERVATION_REVIEW_20261024_CN.md](../../PHYSICS_OBSERVATION_REVIEW_20261024_CN.md)。本目录是新实验，旧运行冻结。训练2022、选择2023，不读取2024；邻站输入独立标为扩展协议。

## 接手入口

- [完成状态](STATUS.json)：3432完成、0失败、0活跃、0待运行；[结果报告](analysis/SCREENING_RESULTS_20261025_CN.md)为本轮后续分析入口。
- [协议及全部缩写结构说明](PROTOCOL.json)、[逐任务清单](manifest.json)。没有训练结果前，不以假定收益重写论文贡献。
- [模型/物理正确性检查](CHECKS.json)、[实际优化与检查点恢复](SMOKE_CHECK.json)：小样本验收不计入实验成绩。
- [已有预测分组诊断](analysis/DIAGNOSTIC_REPORT_CN.md)、[逐站逐seed原值](analysis/existing_actual_by_farm_seed.csv)、[逐站逐seed配对](analysis/existing_paired_by_farm_seed.csv)、[诊断验收](analysis/DIAGNOSTICS_COMPLETE.json)。
- 新训练实际结果由 `code/collect_physics.py` 生成 `analysis/new_actual_per_farm_seed.csv` 与 `analysis/new_paired_per_farm_seed.csv`；没有已完成任务时不会伪造数据行。
- 各机调度状态在 `schedulers/主机名.json`；最新状态含GPU占用、等待原因和实际PID。

## 模型结构

两种骨干是W0（NEUTRAL_LAST的活跃部分：历史编码器、E2C点天气映射、查询适配器、历史交叉注意力解码器、末值残差头）和64维Transformer未来协变量基线。初始化从旧同seed实现复现。所有输入特征增加为64维查询中的零初始化线性投影；只有邻站、位移、响应曲线变体再增加对应机制。容量对照投影形状完全相同，但仅重复展开已有点天气、末值和时效。具体维数见检查记录与模型代码。

HIST读取六个历史天气节点及时间；SHIFT_ZERO/FIXED/SAMPLE三者具有同一位移网络，ZERO网络不活跃须在参数比较中说明，FIXED与SAMPLE只改变网络接收平均状态还是实际样本。PC_INPUT是REF+DENS输入控制；CURVE_FREE和CURVE_BIAS同参数，只区别自由/非负曲线增量。两种NEIGH也同参数，ADVECTION只加固定软先验。

## 文件与存储

`prepared`和`jobs`链接到 `/shared_data_ssd3/liying/physics_observation_single_farm_20261024_v1/`。新增缓存仅有历史点天气、空间导数量、跨周期差和时间/掩码，共约862MiB。所有原网格、功率和标签从旧路径直接读取。每任务最多两份不同选定epoch权重；训练临时保留一份可恢复状态，完成即删除；不保存新的全量预测。评估临时数组留在内存，结果保存为小型指标/逐seedCSV。

## 命令

从项目根目录执行，Python固定使用 `.runtime/chronos/bin/python`：

```bash
.runtime/chronos/bin/python runs/physics_observation_single_farm_20261024_v1/operations/audit_completed_screening.py
.runtime/chronos/bin/python runs/physics_observation_single_farm_20261024_v1/operations/summarize_completed_screening.py
.runtime/chronos/bin/python runs/physics_observation_single_farm_20261024_v1/operations/resource_scheduler.py --probe
```

正式worker通过 `operations/scheduled_worker.py --job-id <manifest中的id>` 包装原训练函数；由调度器设置CUDA_VISIBLE_DEVICES。每host一个调度器，4090通过中央任务锁和远端租约防止跨机重复训练，资源不满足时等待。先读SOURCE_FREEZE和CHECKS再改代码，已开跑后应另建版本，不直接变更同一manifest的模型定义。

3432项2023筛选现已全部完成；后续依据全部正负结果选定结构扩展和跨年复评清单，再生成新的选择冻结记录。结果报告已包含DENS/ALL、同输入TF、容量、损失、时间位移、曲线、邻站及延迟对照；本轮完成不等于后续完整论文验证完成。
