# 调度与恢复入口

中央主机仅为当前4090机器。允许两台远端 `v1006001` / `v1006002`；禁止使用P60002/P60003。部署目录均为 `/home/liying/wind_physics_project/project/runs/physical_representation_single_farm_20261025_v1`，复用其上一轮环境和数据。

本地 `LAUNCHED.json` 记录进程PID；`resource_scheduler.log` 为4090日志；`remote_controller.log` / `REMOTE_STATUS.json` 为租约和回传；`reference_cpu_scheduler.log` 为旧权重CPU诊断；`finalize_when_done.log` 为完成后审计和归因汇总。远端 `bootstrap.log` 包含兼容检查和逐卡启动日志，`GPU_COMPATIBILITY.json` 存检查结果。

最多12进程/卡是安全上限，并不固定启动12个。每次扫描最多每卡增加1个进程，至少等待20秒观察负载；保留显存 max(4096MiB,20%)，按进程实测显存上调下一进程预算，目标总GPU占用低于92%。6002有明确授权共享算力；4090/6001对其他任务活跃时暂停增加，并在其算力负载升高时让本轮任务保存断点退让。CPU诊断最多4进程、1线程、nice10，并检查负载与可用内存。

每个任务有共享文件锁；远端每机最多持有48项租约。未启动的任务才会分配给远端，避免跨设备恢复训练轨迹；参考旧权重的诊断仅在中央进行。回传后核验spec和权重SHA，中央保存一份，远端移除已归档权重。全部任务完成后向远端发送本轮GLOBAL_COMPLETE标记，空闲调度器自动退出。

发生失败先读取 `jobs/<id>/FAILURE.json`；不删除成功结果以重跑，不修改已冻结科学代码。任务缺失或错误不能用“多数完成”代替完成审计。A阶段完成只生成选择建议，不触发2024。
