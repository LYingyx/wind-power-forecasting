# SEL基础上的邻站观测筛选

**2026-10-25 19:21 全部完成：1344/1344，失败0。** 6001/6002分别完成704/528项训练，本机CPU完成112项旧权重诊断；1874份选中权重审计通过。训练进程已退出，2024未开启。结果见 [自动报告](analysis/REPORT_CN.md) 和 [完成审计](COMPLETION_AUDIT.json)。下文为启动时协议记录，不要重新启动队列。

2026-10-25：用户授权执行。计划1232项新增训练、112项旧W0+SEL诊断，共1344项。2022训练、2023选择；2024未开启。完整矩阵、时间因果约束、对照含义及0.1%升阶规则见 [审查与执行说明](../../NEIGHBOR_SEL_REVIEW_AND_EXECUTION_20261025_CN.md)。

- [STATUS.json](STATUS.json)：中央任务计数与活动进程。
- [manifest.json](manifest.json)、[PROTOCOL.json](PROTOCOL.json)：冻结任务及实验规则。
- [CHECKS.json](CHECKS.json)、[SOURCE_FREEZE.json](SOURCE_FREEZE.json)：科学检查与源文件/索引冻结。
- [operations/REMOTE_STATUS.json](operations/REMOTE_STATUS.json)：两台V100租约、连通性、活动任务。
- [assets/READY.json](assets/READY.json)：邻站候选及错配时间检查；50km外孤立风场仅1个。
- `analysis/all_actual_per_farm_seed.csv`、`paired_per_farm_seed.csv`、`interventions_per_farm_seed.csv`、`diagnostics_per_farm_seed.csv`：完成后完整原值表。
- `analysis/NEIGHBOR_SELECTION_RECOMMENDATION.json`、`COMPLETION_AUDIT.json`：完成后的升阶建议与审计，不自动触发2024。

N0复用SEL；N1逐站最近8邻站；N2同月异日错配；N2_SELF同参数只读本站；N3同邻站容量加权汇总；N4_K4/K16为邻站数；N5_CORR仅按2022相关性选站；N7_D0/D30为匹配延迟训练；TF*仅训练N0/N1。全部无CAL，正常邻站延迟15min，所有延迟统一22个历史点。

主机只允许当前4090与v1006001/6002。复用上一轮prepared缓存，jobs链接SSD3。源码与旧结果不可混写；没有成功标记的任务不计完成。
