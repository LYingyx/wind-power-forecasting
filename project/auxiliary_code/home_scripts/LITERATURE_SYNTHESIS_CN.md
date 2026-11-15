# 时间超分辨、扩散与流匹配路线调研

## 与当前任务最接近的工作

| 路线 | 代表工作 | 源/目标与监督 | 对本任务可迁移部分 | 不能直接迁移的地方 |
|---|---|---|---|---|
| 条件时间序列扩散 | CSDI, Tashiro et al. 2021, arXiv:2107.03502 | 从完整时间序列自监督遮盖，条件 score 只在 missing mask 上去噪 | observed/missing mask、任意缺失模式、多变量 attention、CRPS | 仍需要同分辨率完整训练序列；本数据没有真实 15min 天气 |
| 数据依赖 diffusion prior | TimeBridge, Park et al. 2025/2026, arXiv:2408.06672 | 用 spline、GP、pairwise 等 prior 代替标准 Gaussian；DDBM bridge | linear-spline prior 与整点 point-preserving sampler 很契合 | 论文的 imputation 实验有完整高频 ground truth；bridge time 不是物理时间 |
| paired Schrödinger bridge | I²SB, Liu et al. 2023, arXiv:2302.05872 | 清晰图像与退化图像成对；解析 bridge marginal/posterior | 把线性插值看成 degraded endpoint，missing residual 上建桥 | 图像退化的配对可得；当前 1h→15min 的 clean endpoint 不可得 |
| diffusion bridge | DDBM, Zhou et al. 2024, arXiv:2309.16948 | 两个端点分布间的 denoising diffusion bridge | 数据相关 prior、少步 bridge sampler | 仍依赖目标分布样本，不能从功率标签反推出唯一气象真值 |
| 时间聚合反演 | WIND, 2026, arXiv:2602.03924 | 日均约束下生成 4 个 6h ERA5 帧，使用 diffusion forcing | 聚合算子约束、逐帧噪声、时序联合 denoiser | 分辨率是 day→6h，数据和算子与 1h NWP→15min 不同 |

这里最重要的共识是：合理 prior 和 missing-only 更新比“从纯噪声生成整段序列”更适合插值。TimeBridge 的 point-preserving 设计最贴近“整点不能改变”；CSDI 的 mask training 最贴近自监督遮盖；I²SB/DDBM 给出了从数据依赖 prior 到目标的 bridge 机制。但这些工作都没有消除高频训练目标的需求。

## 流匹配家族并不是换几个名称

### Independent CFM / Rectified Flow

在 source `x0` 与 target `x1` 独立或已配对、路径为 `x_t=(1-t)x0+t x1` 时，监督速度为 `x1-x0`。初次训练的 Rectified Flow 与这种线性 I-CFM 在目标上高度重合；只有 coupling、时间采样、reflow/re-pairing 改变后才是不同实验。因此本批没有把“RF”和“I-CFM”重复计数。

### OT-CFM

Tong et al.（arXiv:2302.00482）用 2-Wasserstein coupling 代替 independent coupling，目的是降低 conditional velocity 的方差、缩短路径。官方 TorchCFM 在 minibatch 内计算 OT plan。当前每个天气残差对应特定站点、valid time、发行周期与两个小时端点；跨条件做 minibatch OT 会把 target 配给另一种气象条件。除非先定义严格的“同条件组”或采用 conditional OT cost，否则 OT-CFM 不应直接用于本任务。没有运行它是条件语义上的选择，不是把它与 I-CFM 混为一谈。

### SB-CFM

SB-CFM 使用熵正则 OT coupling 和 Brownian probability path，`sigma>0`；它学习 Schrödinger bridge 的 probability flow。它与 I-CFM 的差异同时包含 coupling、随机路径和正则强度。对于唯一条件样本，paired I²SB/DDBM 比跨样本 SB-CFM 更自然。

### Stochastic Interpolants

Albergo et al.（arXiv:2303.08797）允许 `x_t=α(t)x0+β(t)x1+γ(t)z`，同一 interpolant 可导出 deterministic probability-flow ODE 或带可调噪声的 SDE，并能学习 velocity、score 或 denoiser。本批 Sine-SI 使用非零中间噪声，区别于纯线性 I-CFM；它在功率 MAE 上有小幅优势，但天气遮盖恢复仍不及线性插值。

### Adaptive Flow Matching

Fotiadis et al.（ICML 2025, PMLR 267）先用 encoder 把低分辨率输入映射到靠近目标的 structured base distribution，再由 flow 添加小尺度随机细节，并按 encoder 不确定性自适应噪声。这个“两阶段 deterministic base + stochastic refinement”比纯 Gaussian source 更适合天气。其验证任务是空间 25km→2km 和 Kolmogorov flow，不是时间插值。PC-AFM（arXiv:2604.03459）进一步加入 coarse-scale conservation loss 和梯度冲突处理；对当前任务，结构性写回整点比 soft conservation loss更直接。

### MeanFlow

MeanFlow（arXiv:2505.13447）学习区间平均速度，通过含 Jacobian-vector product 的 identity 把多步瞬时速度积分压缩到 1 NFE，target 使用 stop-gradient。它主要解决采样成本，不自动改善条件正确性或数据不足。本批只把它作为同容量 1-NFE 对照，不能根据单次功率 MAE把 ImageNet 结论推广到天气。

## 扩散家族应按 forward path、target、prior 和 solver 分开

| 维度 | 选择 | 实验含义 |
|---|---|---|
| forward path | VP、VE/EDM sigma、bridge | 决定中间分布和端点；这是训练方法差异 |
| target | epsilon、score、x0/denoiser、v | 改变不同 SNR 区域的有效权重和数值条件 |
| prior | Gaussian、linear spline、GP、learned encoder output | 决定生成器需要跨越的距离；TimeBridge/AFM 的主要贡献在这里 |
| sampler | ancestral DDPM、DDIM、Euler、Heun、DPM solver | 主要是同一已训练模型的推理选择，不能重复计作模型 |
| constraint | soft loss、projection、missing-only subspace、point write-back | 对“整点不变”，结构性 subspace/write-back 最可审计 |

DDPM（arXiv:2006.11239）与 DDIM（arXiv:2010.02502）的关系说明了为什么不能把 DDIM 写成另一套训练路线。EDM（Karras et al., arXiv:2206.00364）系统改变了 sigma sampling、preconditioning、loss weighting 和 Heun schedule；本批按官方 `c_skip/c_out/c_in/c_noise` 与 rho=7 实现时序残差适配。VP-epsilon 与 VP-v 使用相同 path，只比较 target parameterization。

## 天气降尺度文献的证据边界

CorrDiff（arXiv:2309.15214）、AFM、PC-AFM、2026 weather flow-matching super-resolution（arXiv:2604.00897）和 CDSI（arXiv:2603.03838）都展示了生成模型在地球系统降尺度中的价值，但其主要任务是空间 coarse-grid→fine-grid，且训练阶段有高分辨率 ERA5/区域气候模拟配对。arXiv:2604.00897 甚至在各 lead 上独立做空间 SR，没有显式时序历史；不能把其结果当作 1h→15min 时间 SR 的实证。

这些工作仍提供三点可迁移经验：生成残差而非完整场；用数据依赖 base 缩短路径；用谱、守恒、CRPS和 ensemble calibration 共同评估。当前项目只有站点功率标签和小时 NWP，最多能验证下游任务价值与小时遮盖代理，无法验证 15min 天气的频谱、极值或物理一致性。

## 对当前项目的路线优先级

第一优先是 anchored deterministic residual 和 linear-prior bridge，两者都直接利用已有插值先验并把整点约束写进状态空间。第二优先是 CSDI/TimeBridge 风格的 mask-aware denoiser，但前提是获取真实 15min 天气或可信高频再分析。第三优先才是纯 Gaussian I-CFM/EDM，因为它们在本批天气代理上并未胜过线性插值。

如果继续只使用现有数据，模型应称为“weather-conditioned future-covariate adapter”，并以功率误差为主任务；如果要称“气象时间超分辨”，必须增加高频气象验证集。两种研究问题需要分开写假设、损失与结论。

原文和官方代码快照见 `references/fulltext`、`references/raw` 与各 `*_status.json`；TimeBridge 固定到 Git commit `023a0aa9f6dbd24f66bd84eb8983bb78affd4e27`，TorchCFM 为 1.0.7，EDM/CSDI/I²SB/DDBM 的下载来源和 SHA 已单独记录。
