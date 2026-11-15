# Farm interaction enhancement v1 启动前审计（2026-09-08）

## 科学比较修复

原 IAdd 的 correction logit 可分离，但最终 `sigmoid(baseline_logit + correction_logit)` 在 CF 输出空间不可分离。因此本轮 Int/IAdd 统一使用 `response_output_mode: cf_residual`：`station_factor = baseline_factor + correction_logit`。IAdd 的 baseline 和 history correction 只依赖 H，weather correction 只依赖 W（lead/site context 固定）；没有共同 sigmoid 或最终裁剪。越界比例单独报告，不根据测试结果改变输出形式。

旧 `legacy_logit` 默认、state-dict 和输出保持兼容。已用旧 M0 的 IAdd/Int seed42 实际 checkpoint strict loading，与修改前实现的 2023 数据输出逐项精确比对通过。旧 M0 不能用作新输出协议的公平 parent；在 `baseline/m0_{iadd,int}` 重新训练同配置 M0，已有 reused symlink 和旧结果原样保留，旧 baseline 单列历史参考。

B1 的 delta encoder 修正为真正的 linear；B2 的 TCN 保留 absolute forcing backbone residual。全部 A+B+C 组合同时验证参数相等和非零输出下最终 CF 的四点可分离性。

## 流程修复

- shell 和 worker 接收 9 个位置参数或包含 readout/output 的 11 个参数；配置 ID 不含 stage，formal symlink 复用相同不可变配置。
- 每个 run/总 launcher 均有进程锁。失败 attempt 保存在独立目录；训练成功、仅 validation export 失败时不重训。训练中断则从同 seed 完整 schedule 重试并保留原 attempt，不假称恢复了 optimizer epoch state。
- screening 提升为 formal 时校验源完整性和现有 link 目标；冲突即停止，不覆盖 artifact。analyzer 每个 method/seed 只统计一次。
- A/B 无 winner 时回退到 M0 或单方向 winner；A+B 也先做 5-seed gate，失败回退到较好的单方向；M0 上已有的 C screening 不重复运行。
- 合法 direction/context/ID diagnostic 才补足 10 seeds；ID 由 spec 排除在主模型选择之外。
- freeze 前选择函数只允许 val。freeze 登记 checkpoint/config/validation metric 哈希及 2022 q90。worker、直接 evaluation CLI 和 analyzer 均校验 freeze；冻结重启直接进入 inference，不能重新选型。
- scheduler 使用实时 free memory，扣除通过 /proc 子进程关系核算的尚未分配预算；每卡 8 slots，额外保留 1 GiB，每任务预算至少 2 GiB，OOM 禁止该卡继续接任务，失败最多重试一次。重启先等存活 worker 释放 run 锁，再发现已完成结果，避免重复训练。physical index 经 nvidia-smi 映射 UUID。
- protocol 3 总训练数依赖筛选，130–360 个；每个唯一训练 run 对应一次冻结测试 inference，机制诊断再增加 80–160 次 inference。初始阶段为 110 个训练（两套 M0 输出各 10 + 九组 pair 90）。完整加强计划见 `farm_interaction_enhancement_v1_strengthening_plan.md`。

## 验证

加强后的五组测试覆盖 selection gate 边界、重启/发布中断、freeze guard、protocol 2/3 完整 synthetic analyzer 报告、paired seed/day bootstrap、三种 ramp 间隔、attention archive roundtrip、legacy 输出兼容、D0/D1、bounded link 和干预。包括所有方向晋级并选择 D1 的完整 350-run 模拟，以及失败回退与冻结后幂等重启。部署日志记录最终测试数量与结果。既有短 horizon 单元测试有空 horizon 警告，与本轮固定 24 steps 无关。

CUDA smoke 使用真实 2022 数据，`cycles_per_batch=3`（12 samples），每个模型 3 次 backward，再测 6 cycles 的 evaluation batch。A2/B2/C2/完整 ABC/M0/D0/D1/ABCD1/bounded M0/bounded ABCD1 各自 IAdd 和 Int 共 20 模型均无 NaN/OOM；所有出现的新增模块都有参数更新。最高 reserved 484 MiB、allocated 429 MiB。20 组初验在 physical GPU 3 完成；部署后再次执行并校验 code snapshot hash 才启动，原始记录保存于 `artifacts/farm_interaction_enhancement_v1/cuda_smoke.json`。

140 份既有 baseline val/test 预测文件已核对存在；启动审计未读取任何 2024 结果内容。

## 分析产物与解释

六项要求产物全部由 analyzer 生成。完整 baseline 和 screening/formal 模型分别报告 station CF、physical-sum cluster MW；三种间隔的 RampMAE、方向准确率、极端 ramp precision/recall/F1；horizon MAE、per-farm coverage、paired seed bootstrap CI、station MAE 共享 UTC-day block CI、parent 对照、interaction DID、history attention。阈值只读取 freeze 中的 2022 q90，不能重新从 val/test 估计。

无 elevation、无 station selection、无 held-out-farm、无空间模型改动、无 cluster fine-tuning、无 ramp loss。预先规定的筛选条件不变，数值边界仅加入 1e-12 浮点容差。
