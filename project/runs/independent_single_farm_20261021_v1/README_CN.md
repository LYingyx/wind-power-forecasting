# 2026-10-21 独立单风场实验执行入口

完整审查及冻结计划：[EXPERIMENT_REVIEW_AND_EXECUTION_20261021_CN.md](../../EXPERIMENT_REVIEW_AND_EXECUTION_20261021_CN.md)。

本轮是 **56 站各自独立训练的 local-only E2C-CBH**。旧共享训练成绩不能移作本轮成绩。训练队列共 9500 阶段，含 528 个验证搜索和 572 个冻结历史参考训练。

## 实时状态

- [STATUS.json](STATUS.json)：约每 3 秒更新完成/运行/排队/失败数、GPU 分配和每个训练进度。
- [analysis/REPORT_CN.md](analysis/REPORT_CN.md)：每 10 分钟更新；不完整家族不进入正式主表。
- [analysis/CONTRIBUTION_EVIDENCE_CN.md](analysis/CONTRIBUTION_EVIDENCE_CN.md)：三项贡献对应证据的自动状态；未齐全比较明确标为待完成。
- [analysis/ANALYSIS_STATUS.json](analysis/ANALYSIS_STATUS.json)：汇总是否完整、可用家族和图表数。
- `FINISHED.json`：只有队列停止时才生成；须检查 `all_experiments_complete`，不能只检查文件存在。
- `COMPLETION_AUDIT.json`：全部任务完成后，重新核对 9500 个检查点散列、预测形状、MAE、冻结参考和选参来源；核验通过才标记完整成功。
- `scheduler.log`、`logs/<job_id>.log`：调度和训练日志。
- `jobs/<job_id>/SUCCESS.json`：单阶段成功标志；`FAILURE.json` 保留错误诊断，重试成功后以 SUCCESS 为准。

## 正确性与可复现性

- [PREFLIGHT.json](PREFLIGHT.json)：原 CBH 与缓存加速版的前向/梯度等价、风场参数隔离、匹配初始输出、优化器/RNG 精确续训、各基线输入敏感性。
- [CONCURRENCY_BENCHMARK.json](CONCURRENCY_BENCHMARK.json)：1/2/4/8 进程实测吞吐；最多 8 进程/卡，受可用显存限制。
- [prepared/AUDIT.json](prepared/AUDIT.json)：空间缓存逐项核对、时间插值边界、数据行数、固定统计散列。
- [INPUT_SHA256.json](INPUT_SHA256.json)：全部准备输入数组的内容散列。
- [ACTUAL_JOB_AUDIT.json](ACTUAL_JOB_AUDIT.json)：实际完成任务的权重重载/预测重算检查，及选参任务不读取 2024 的核对。
- [FORMAL_RESULT_AUDIT.json](FORMAL_RESULT_AUDIT.json)：正式完整数据任务的预测、修正量、指标和场景分组重算。
- [ABLATION_PIPELINE_AUDIT.json](ABLATION_PIPELINE_AUDIT.json)：主模型及七个匹配消融臂首组完整训练，冻结参考散列一致、全量预测与修正量核验；仅作工程检查，不作总体科学结论。
- [ANALYSIS_PREFLIGHT.json](ANALYSIS_PREFLIGHT.json)：已知效应下的区间计算、验证选择不受测试排名影响、不完整家族排除测试。
- [PROTOCOL.json](PROTOCOL.json)、[TRAINING_CODE_FROZEN.json](TRAINING_CODE_FROZEN.json)：冻结协议与执行代码散列。
- [BASELINE_IMPLEMENTATION_NOTES_CN.md](BASELINE_IMPLEMENTATION_NOTES_CN.md)：基线适配范围，不能把 adapted 写成官方复现。
- `selected/<family>.json`：只由该家族 48 个 2023 验证结果选出一个配置；保存全部验证分数与来源散列。
- `specs/`、`queue.sqlite`：具体任务配置与依赖关系。

所有正式阶段用完整 2022/2023 数据。`smoke_jobs/` 仅是 512 行的工程预检，**不入主表、不用于超参数选择**。

每个成功任务保存 `best.pt`、`history.json`、`metrics.json`、`val_prediction.npy`，正式任务另存 `test_prediction.npy`。B0 保存训练预测供同站同种子修正复用。`*_correction.npy` 定义为最终预测减 B0；在 DIRECT 和末值参考臂，它不是原始 readout。起报时间、样本索引、目标真值存于 `prepared/<split>/` 和各站目录。预测顺序严格对应这些数组，shape 为 `[origins,24]`。

## 后台执行

调度进程已与交互会话分离；`supervise.py` 在意外退出时只重启本轮调度器。已运行的本轮 worker 不会被重复启动；其检查点和进程会被重新接管。其他用户进程不修改、不终止。2、3 号卡显存不足时保持候补。

从项目根目录查看或手工复核：

```bash
cat runs/independent_single_farm_20261021_v1/STATUS.json
.runtime/chronos/bin/python runs/independent_single_farm_20261021_v1/code/analyze.py
```

需要恢复异常退出的调度器时，先检查 `scheduler.pid` 对应进程和 supervisor 是否仍存在，避免重复启动；文件锁也会拒绝重复实例。停止 supervisor 自动恢复可创建运行目录的 `STOP` 文件；这不终止正在运行的任务。变更训练协议需要新版本目录，不能在已有正式任务中静默改代码。

## 结果使用边界

CI 跨零不是“无效”的证据；误差交叉、收益随时效增长、哪种条件有效都等待本轮完整结果。代表站来自既有训练描述量；案例列表保留原窗口并按同一已观测历史规则补余字失败站。未来爬坡仅作事后诊断。当前命名基线为适配结构，不能宣称完成了官方 SOTA 复现或跨地区泛化验证。
