# 物理表示归因筛选：2026-10-25

> 2026-10-25 16:26：A阶段6944/6944全部完成，未解决失败0，9461份权重核验通过。优先阅读 [完成与归因审查](analysis/ATTRIBUTION_REVIEW_20261025_CN.md)、[逐风场逐seed原值](analysis/KEY_MODELS_PER_FARM_SEED_CN.md) 和 [完成审计](COMPLETION_AUDIT.json)。B1主表清单已在 [FROZEN_SELECTION.json](FROZEN_SELECTION.json)冻结；2024未启动，B2另需协议。下方描述保留A阶段执行规格，不应据此重新启动已完成队列。完成审计中的selection_frozen=false是16:16生成时的历史状态，B1冻结文件于16:25另行生成。

当前为 A 阶段：2022训练、2023选择。6272项新增训练，672项旧权重诊断，共6944项；不访问2024。科学解释、缩写和完整矩阵见 [审查与执行说明](../../NEXT_ROUND_REVIEW_AND_EXECUTION_20261025_CN.md)。

- [STATUS.json](STATUS.json)：中央完成数、活动任务、失败和各阶段状态。
- [manifest.json](manifest.json)、[PROTOCOL.json](PROTOCOL.json)：全部预声明任务与选择规则。
- [CHECKS.json](CHECKS.json)、[SOURCE_FREEZE.json](SOURCE_FREEZE.json)：科学检查及A阶段源码冻结。
- [operations/REMOTE_STATUS.json](operations/REMOTE_STATUS.json)：两台V100连接、租约和实际进程。
- [analysis/REPORT_CN.md](analysis/REPORT_CN.md)：结果报告；逐风场逐seed值及配对差链接在其中。
- [analysis/SELECTION_RECOMMENDATION.json](analysis/SELECTION_RECOMMENDATION.json)：A完成后的选择建议；未完成时建议为空，不是B冻结文件。
- 完成后 `COMPLETION_AUDIT.json` 核验全部权重、诊断与冻结源码；B须另有 `FROZEN_SELECTION.json`。

训练脚本 `code/train_physics.py`，变体 `code/physical_data.py`，骨干控制 `code/physical_models.py`，分组 `code/diagnostics.py`。只用本机4090与v1006001/6002；不使用P60002/P60003。调度器按实测余量增加任务，6002保持已授权的共享策略。

`prepared`链接旧轮缓存，`jobs`链接 `/shared_data_ssd3/liying/physical_representation_single_farm_20261025_v1/jobs`。保留逐站逐seed指标与必要选定权重，不保存预测数组，不复制旧权重。原3432项结果与源代码不改动。
