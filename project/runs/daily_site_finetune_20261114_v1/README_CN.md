> **2026-11-15 全部重训已完成**：当前三个队列共2571项全部完成并通过产物核验；请读 [最终分析入口](../../DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)。本文件下方的运行/等待描述是实施历史，不要重启队列。

# 新协议逐站微调入口

见 [执行计划](EXECUTION_PLAN_CN.md)。本轮等待 [共享阶段](../daily_protocol_restart_20261114_v1/README_CN.md) 的新模型完成；目前只有迁移后的代码，不存在新微调结果。调度后以 [STATUS](STATUS.json)、[结果](analysis/RESULTS_CN.md) 为准。

关键实现：[构建新矩阵](code/prepare_daily.py)、[逐站训练](code/train.py)、[严格因果校正](code/z1.py)、[汇总](operations/collect.py)。所有模型均用新协议，复用的数据只有同一新共享数据入口，旧模型和成绩不参与。
