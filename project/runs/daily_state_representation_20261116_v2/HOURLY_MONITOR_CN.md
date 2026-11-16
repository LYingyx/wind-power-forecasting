# 每小时检查与自动接续

本机 `liying` 用户的 cron 每小时整点执行一次 [巡检脚本](../../scripts/hourly_state_representation.py)，时区为北京时间（Asia/Shanghai）。cron 服务由服务器管理，退出当前对话或 SSH 不影响检查。最近一次结果见 [HOURLY_STATUS.json](HOURLY_STATUS.json)；逐次记录位于服务器的 `monitoring/YYYYMMDD_HHMMSS.json`，标准输出保存在 `monitoring/cron.log`。这些是服务器检查记录，不会自动发送聊天消息。

检查覆盖本机 4090 和 `v1006001` 的协调器、调度器、任务完成数、失败任务、心跳和阶段判定。现有 [coordinator.py](operations/coordinator.py) 每约 30 秒回收结果并按冻结规则判定，因此符合条件后无需等待下个整点。

- 25 项阶段一完成、R3 相对 R2 的 MAE 和 RMSE 均改善后，启动 15 项 R4/R4-noS/R4-wrongS。
- R4 同时通过主门槛和机制对照后，启动 5 项 TF_R4。
- 阶段二完成或按规则跳过后，按冻结顺序选择损失复核候选，启动候选与 TF 的 10 项混合损失训练。
- 全部应做任务及诊断结束后生成 [COMPLETE.json](COMPLETE.json)。之后巡检仅记录完成状态，不再恢复进程或启动训练。

巡检不直接写阶段启用文件，阶段决策仍由唯一持锁协调器执行。两台机器继续使用原多进程资源策略和固定任务分配。

2026-11-16 13:54 更新：按用户要求将剩余任务分配到 4090，优先 GPU 4–7，再使用 0–3；V100 调度器存活待命，但没有未完成任务的分配。巡检仍检查两端健康状态。它负责进程恢复，不自动进行跨机器任务迁移；当前本机容量可覆盖预定最大并发，资源不足时的后续转移仍须通过检查点交接和两端分配同步，见 [迁移记录](MIGRATION_4090_20261116.json)。

恢复保护：核对精确进程路径与用户，并使用互斥锁；缺失的管理进程只有在冻结文件哈希和启动检查通过后才可恢复。尊重 STOP、完成标志和失败熔断，每组件六小时内最多恢复两次。如果调度器退出但其训练子进程仍存活，先让已有训练完成，下次巡检再判断恢复；不会重复开进程挤占资源。失败任务、心跳陈旧或网络中断记录在 `warnings`，不会清除失败标志或降低实验门槛。保护策略的临时进程验证见 [test_hourly_state_representation.py](../../scripts/test_hourly_state_representation.py)。

巡检脚本独立于冻结训练代码。10 seeds、个性化和 2024 复评仍需本轮结论与最终候选冻结，不在本次自动接续范围内。

服务器查看方式：

```bash
crontab -l
python3 -m json.tool /shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/daily_state_representation_20261116_v2/HOURLY_STATUS.json
```
