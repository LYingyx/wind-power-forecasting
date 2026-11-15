# Farm interaction v1：科学问题与完整加强实验（protocol 3）

## 已有证据与研究价值

已完成的 10-seed 历史结果：HW-Int 相比旧 IAdd，2023 station MAE 从 0.089537 降至 0.087629（2.13%），2024 从 0.086089 降至 0.084760（1.54%），两年均 10/10 seeds 改善。2023 有 44/56 farms 改善；跨年 38 farms 一致改善，7 farms 一致受损。固定风场的预测增益具有研究价值，但不等同于因果识别或新风场泛化。

2023 验证诊断显示，MSE 降低的 88.45% 来自每条轨迹平均误差的平方，11.55% 来自去均值形状误差。station 15 min ramp 略差，1 h 不确定，3 h 改善；cluster physical-sum ramps 三种间隔均改善。15 min 极端事件召回率仍很低（旧 IAdd 0.35%，Int 0.72%）。因此不能把整体 MAE 改善解释成全面解决快速爬坡问题。

旧 IAdd 的 correction logit 可加，但共同 sigmoid 使最终 CF 不严格可加。真实 checkpoint 的 CF 四角混合差 mean abs 为 0.001620，最大 0.112783。本轮重新训练公平 M0，修复识别对照。过去 FiLM/LoRA/attention-bias/operator 与更大空间搜索没有稳定增益，优先围绕时间表示、天气变化和 level/shape 读出验证具体假设。

## 固定数据与比较协议

- 56 farms 固定；2022 train，2023 所有筛选和 checkpoint 选择；2024 新模型推理仅在 validation freeze 后执行。
- Raw Bilinear Pangu；不做 GNN、4×4 搜索、cluster fine-tuning、station selection、ramp loss 或新风场泛化声明。
- 所有候选同时训练 Int / IAdd，参数量、初始化 seed、优化器和 schedule 匹配。参数匹配不代表计算量完全相等，另报训练耗时。
- 5-seed screening：42/44/46/48/50；合格候选补 43/45/47/49/51，formal 为 42–51。formal 含 screening seeds，属于扩展验证，不能宣称十个独立的事后确认 seeds。
- 固定 40 epochs、patience 12、bf16、cycles_per_batch=3，按 station MAE 选择 checkpoint。无 ramp loss。
- primary `cf_residual`：baseline CF + correction，无最终裁剪，报告 CF 越界率。IAdd 在最终 CF 空间严格可加。
- 输出稳健性 `bounded_cf_mix`：只依赖 lead 的学习权重混合两个 sigmoid 分支。IAdd 每个分支只依赖 H 或 W，最终 CF 严格可加且在 [0,1]；Int 天气分支允许 H/W 交互。该协议独立重训 M0 与最终架构，禁止改变 primary 选择。

## 实验矩阵与自动依赖

| 阶段 | 架构/对照 | Seeds/执行规则 |
|---|---|---|
| M0 | pooled history + absolute weather；两种输出协议 | 各 Int/IAdd，10 seeds |
| A1/A2 | lead-conditioned history summary / weather-query history cross-attention | 每组 Int/IAdd，先 5 后合格补 5 |
| B1/B2 | origin-referenced delta linear / residual delta TCN | 每组 Int/IAdd，先 5 后合格补 5 |
| C1/C2/C3 | metadata residual / metadata FiLM / ID FiLM | 每组 Int/IAdd；C3 仅诊断 |
| D0/D1 | 普通 auxiliary head / trajectory level + zero-mean shape head | 参数相等；D1 还必须优于 D0 与表示 parent |
| A+B | 合格 A 与 B 的组合 | 对较优单方向进行 5-seed gate，合格补 5 |
| (A+B)+C | 所选 A/B 表示上的 C1/C2/C3 | 物理 metadata 与 ID 分开筛选；ID 不能成为 primary |
| selected representation + D | 在所选 A/B/metadata 表示上重验 D0/D1 | D1 晋级时 D0 同步补足 10 seeds |
| final component removal | 分别移除 history/forcing/context/readout 的新增部件 | 每项两角色、5 seeds；与已有完全相同 Spec 去重 |
| final bounded link | 最终 CF-residual 架构的 bounded_cf_mix 复验 | 两角色、10 seeds；若 final=M0 则复用 |
| frozen interventions | M0 与 final 的 history reference、分层天气置换推理 | 两角色、10 seeds、val/test；final=M0 去重 |

初始 110 个训练；自适应总数 130–360 个唯一训练，每个训练对应一次冻结后的 2024 推理；机制干预另有 80–160 次推理。SCREEN 晋级通过 symlink 复用，不重复计数。任何失败方向也保留并报告，禁止仅呈现赢家。

D1 公式：令常规 decoder 输出为 r(h)，auxiliary 作用于 decoder 隐状态的时间均值，得到 a。输出 `r(h) - mean_h r(h) + a`。D0 使用完全相同的 auxiliary 参数，但输出 `r(h) + auxiliary(z(h))`。共享 lead-dependent baseline 保留，因此该模块是 correction 的 level/shape 归纳偏置，不能声称整个最终预测的两个独立物理机制已被识别。

## 预先固定的选择标准

5 seeds：Int 平均 station MAE 优于同配置 IAdd 与 parent Int，各至少 4/5 胜；station RMSE 不退化；0–1/1–3/3–6 h MAE 任一带宽退化不超过 0.0005 CF。D1 额外通过 D0 对照的相同 gate。

10 seeds：同配置 Int 对 IAdd 平均改善且至少 8/10 胜；非 M0 候选对 parent 同样要求，保留 RMSE/horizon guard；D1 同时检查 D0 与原表示 parent。最终从通过规则的模型选择验证 MAE 最低者；若不能优于 M0 IAdd，则主模型回退 M0 IAdd。ID、bounded-link 与事后 component ablation 不参与 primary 选择。筛选指标及大轴 reductions 使用 float64。

上述规则属于预定决策规则，8/10 胜不自动等于统计显著。统计区间与效应量一起报告；验证集搜索存在选择偏差，独立年度 2024 仅用于冻结后的评估，不能据其调整架构。已知的历史 2024 结果与新模型 2024 推理明确区分。

## 分析与可证伪结论

报告 station CF MAE/RMSE、per-farm/horizon、CF 越界率、MSE 的 trajectory mean/shape 精确分解；physical-sum cluster MW；15 min/1 h/3 h RampMAE、方向准确率、极端 precision/recall/F1 和极端条件误差。station ramp 同时分 forecast-internal 与 history-boundary；极端阈值固定为 2022 q90。

配对 seed bootstrap 与共享 UTC-day block bootstrap；主报告包含历史 baseline、新 M0、全部筛选失败项、parent 对照、Int–IAdd、interaction DID、组件移除、输出协议稳健性、history attention。Attention 仅为描述，reference/天气置换仅是模型输入依赖诊断，不能当作真实系统因果效应。单一 permutation seed=7301 不能估计置换分布。

history-reference 只替换 response-state 路径的历史表示，历史 baseline/当前功率锚点仍保留，因此检验的是交互分支对历史状态的依赖，不是完整 history removal。

关键可证伪问题：严格可加对照下交互优势是否保留；level/shape 是否优于等参数普通 head；提升是否超出 trajectory mean correction；组件移除是否破坏收益；收益能否在有界输出下保留。若不成立，报告负结果并缩小结论，不自动开启无限超参搜索。

## 资源、可恢复性与完成标准

使用 physical GPU 0–7，nvidia-smi index 映射 UUID 后设置 CUDA_VISIBLE_DEVICES。每卡配置 8 slots（64 总 slots），按 CUDA smoke 峰值、实时外部占用和本任务尚未分配的显存预算逐步派发；不停止其他任务，不伪称有足够显存运行 64 个 CUDA 进程。每卡留 1 GiB，每任务至少预留 2 GiB；资源释放后自动补充。OOM 停止该卡新派发并保留失败日志，单任务最多重试一次。

单一 persistent tmux launcher、总锁和 run 锁；独立 attempts、不覆盖原结果、不可变配置/code snapshot、freeze checkpoint/config/metric hashes。训练成功但验证导出失败时只重试导出。freeze 后自动运行全部测试、机制干预和分析；仅在所有步骤成功后写 COMPLETE。运行期间 status.json 提供 active/queued/failure 和 GPU 分配。
