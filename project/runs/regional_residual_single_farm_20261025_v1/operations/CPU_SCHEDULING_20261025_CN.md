# 三机 CPU/GPU 调度升级

用户要求保留 4090 等待空闲、利用三台机器加速本批实验，同时避让其他人的负载。2026-10-25 22:29 启用本次调度。原科学代码、manifest、SOURCE_FREEZE、训练协议和 2024 开关保持原样。

## 执行方式

- 原 GPU 调度继续运行，4090 的 bootstrap 也保留。显存与计算余量同时允许时才新增训练；每卡保留至少 4 GiB 或 20% 显存。两台 V100 的既有 GPU 队列没有中断。
- 新增 cpu_scheduler_v2.py，三台机器各有独立的 CPU 调度器。6001 的并发上限为 56，6002 为 20，4090 主机为 32；这些只是上限，实际依据 /proc/stat 测得的 CPU 余量动态调整。
- 总 CPU 准入目标为逻辑核数的 85%，用最近三次采样中较高的“其他工作负载占用”计算本轮可用槽位；主存准入同时检查整机和当前进程 cgroup 的各级上限，按最紧有效余量保留 20% 或 8 GiB；为启动未满 60 秒的 worker 预留至 3 GiB 峰值。所有新 CPU worker 单线程、nice=15，不初始化 CUDA。
- 重放任务分别预留 6/4/4 个槽位（6001/6002/4090）。有重放等待时，可以暂停占用这些槽位的本轮 ARX，避免被长任务堵住。
- 负载升高只对本调度器登记的进程发 SIGSTOP，空闲时 SIGCONT；PID 启动时刻校验可避免误伤复用 PID。暂停保留正在计算的内存状态，不重启拟合，不对外部进程发送信号。
- 原本机 CPU 调度父进程已停止；原 4 个 ARX worker 原地接管，PID、内存、任务锁和已有计算均保留。旧 reference_cpu 状态文件已标记被替代。

## 跨机分配与结果返回

cpu_remote_controller.py 使用独立 CPU 租约和中央 worker.lock；与 GPU 控制器及本机 CPU 调度共享锁，防止同一任务重复运行。manifest 中原 local_only 字段继续阻止 GPU 控制器错误分配 CPU 任务；本 CPU 控制器依用户此次明确授权允许它们在 V100 主机的 CPU 上运行。

重放仅临时传输所选旧权重、normalization 和 metrics，不复制完整数据或预测。重放先校验源权重哈希及旧 MAE/RMSE 的一致性。回传后核验源权重，将来源路径规范化为中央路径，再删除远端本次临时权重；若该权重在部署前就存在，则保留。ARX 系数回传核对哈希后，删除远端重复系数。中央保留全部原始指标。

两台 V100 补齐与本机完全一致的 numpy/scipy/scikit-learn/joblib/threadpoolctl 版本，没有修改 numpy 或 torch。真实输入特征和映射摘要与本机完全一致；确定性稀疏求解器的系数最大差小于 9e−16。CPU_SCHEDULING_CHECKS.json 验证资源阈值、重放优先配额、PID 防复用及自身进程暂停/恢复。

## 当前入口

- CPU_REMOTE_STATUS.json：跨机 CPU 租约、容量、活动任务和异常。
- ../schedulers/cpu_scheduler_v2.json：本机测量、运行/暂停任务。
- ../schedulers/cpu_remote_v1006001.json、cpu_remote_v1006002.json：远端 CPU 任务。
- CPU_V2_LAUNCH.json：新调度进程及原地接管的 worker。
- ../STATUS.json：仍以核验完成结果计数，不将排队或暂停任务计为完成。

CPU 任务全部结束后，本控制器写 CPU_GLOBAL_COMPLETE.json，CPU 调度退出；原最终审计流程继续核验整轮任务并生成报告，不启动 2024。

## 账号内存限额修正与恢复

6001 的 user-1013.slice 硬上限为 101403095040 字节（约 94.44 GiB）。最初仅看整机 MemAvailable 的准入逻辑遗漏此限额，造成 11 个新增 ARX 被 OOM kill；为停止继续触顶，随后主动结束了本轮另外 18 个新增 ARX，以释放约 28 GiB 内存。原 4090 上的 4 个 ARX 没有重启，其他人的进程未被发送信号。全部 29 个受影响尝试保留在对应 jobs/*/attempts/ 下，已按同一科学 spec 重新排队，见 RESOURCE_RECOVERY.json。

resource_limits.py 现在遍历自身 cgroup 及祖先层级，用 limit−current 加可回收 inactive_file（扣除 dirty/writeback）与整机可用内存取更小值，再扣除本轮新 worker 的内存预留。CPU 逐进程启动前重查；内存不足时阻止新任务，避免以暂停现有任务代替释放内存。CPU 是否暂停仍由计算余量决定，且仅在有内存能接入重放时为其腾出配额。

6001 改用 gpu_scheduler_limits.py，在原冻结 GPU 调度器外只替换 RAM 准入函数；其 GPU 作业继续复用原训练代码。4090 的空闲等待入口更新为 bootstrap_limits.py，保留原 GPU 空闲监测并加入账号内存检查。6002 GPU 仍使用原队列，该机当前用户无单独内存硬限额；其 CPU 已升级为同样的层级内存检查。运行中的 CPU worker 在调度器更新时原地接管。
