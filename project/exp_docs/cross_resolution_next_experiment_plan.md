# 跨分辨率下一步实验与论文定位计划

日期：2026-09-02  
状态：方案已冻结；Stage A 已完成，后续严格按预注册 gate 执行

执行说明（在读取任何新模型的2024结果前记录）：若 Stage A 没有确认 T2，则 Stage B 不运行；Stage C 原本检验的是“最终 T2”的分辨率鲁棒性，在不存在确认胜出的最终 T2 时也停止，不能把鲁棒性作为事后挽救 T2 原生分辨率性能的依据。空间矩阵仍可在 Stage A 选出的时间结构上独立进行。

## 一、当前判断与论述边界

当前时间侧证据支持的是：原生小时节点先编码，再按目标时刻使用相邻节点做局部非线性响应映射。T2 相对 latent-linear 的三随机种子改善为 1.50 MW、3/3 获胜且配对区间不跨零；full-path attention 与 latent-linear 基本相同。因此目前不能把 full-path attention、continuous-time、非整点优势或高天气变化优势写成已经成立的贡献。

空间侧存在规则 NWP 网格到不规则场站位置的支持错配，但尚未证明 learned 4×4 优于固定投影。空间模块称为 `grid-to-station cross-support adaptation`，不称为气象空间超分辨率。

当前 T2 已输入目标到左右节点的实际时间距离，因此能够识别 bracket span，不再重复增加时间跨度输入。当前 anchor 为 `horizon_taper`，并非全时距固定 0.2。最终性能始终以独立单 checkpoint 为准，ensemble 只允许作为蒸馏 teacher。

## 二、Stage A：时间适配正式 10-seed 验证

比较 T0 raw-linear、T1 latent-linear、T2 bracket-nonlinear 和 T4 full-native-path，seeds 固定为 42--51。

- T0/T1/T2 复用已完成的 42、46、50，仅新增其余七个 seeds，共21个新训练任务。
- T4 复用已完成且配置、参数量和样本映射审计一致的十个 checkpoints。
- 统一固定 recent cluster 1 h、black-box 状态—天气融合、learned 4×4 空间前端、difference path、horizon-taper anchor、训练预算和 no-KD。
- 只使用2023 validation。T3 nearest 不扩展到10 seeds。

确认标准：

1. T2 vs T0：平均改善至少3 MW、10-seed paired 95% CI 上界小于0、至少7/10获胜。
2. T2 vs T1：平均改善至少1 MW、CI上界小于0、至少7/10获胜。
3. 上述两个主比较使用 Holm 校正。
4. T2 vs T4 使用1 MW非劣界限；单侧95% CI上界小于1 MW即可选择更简单的T2。

若 T2 vs T1 不成立，停止 nonlinear-bracket 贡献并选择 T1；若 T2 vs T0 也不成立，则将跨分辨率降级为任务背景。

## 三、Stage B：T2机制拆解

仅当 Stage A 确认 T2 后，先运行 seeds 42、46、50：

- `T2-PhaseOff`：局部 bracket mapper 的两个相对时间特征置零，输入维度和参数量不变；公共 decoder 中的 target-time token 保留。
- `T2-Linear`：局部 MLP 的 GELU 替换为 Identity，两层矩阵、dropout和参数量不变。
- `T2-NoDiff`：保留 difference encoder 完整结构，将显式差分通道置零，不使用参数更少的 independent encoder。

PhaseOff 或 Linear 相对 T2 恶化至少0.75 MW且3/3同方向时补齐10 seeds；正式机制贡献要求恶化至少1 MW、CI不跨零且至少7/10同方向。NoDiff 三 seed 平均退化不超过0.5 MW时补齐10 seeds，再以1 MW非劣界限决定是否删除显式 difference path。

## 四、Stage C：冻结模型后的分辨率鲁棒性

只在2023 validation上评估冻结 checkpoint：

- R0：完整8个小时节点。
- R1：只保留节点索引 `[0,2,4,6]`，形成固定2 h输入。
- R2：依次删除内部节点1--6，六种确定性场景等权汇总。

对 T0、T1、最终 T2 和 T4 报告各自的退化量：

\[
\Delta MAE=MAE_{perturbed}-MAE_{native}.
\]

只有最终 T2 的退化相对 T0 和 T1 均少至少1 MW且配对区间不跨零，才声称 resolution robustness。若 T2 在稀疏节点下明显退化，后续只测试 time-normalized difference \(\Delta z/\Delta t\)，不直接增加 ODE、连续 kernel 或任意输出网格 decoder。

## 五、Stage D：空间 same-support 最小矩阵

时间结构完全冻结后使用 seeds 42、46、50：

| ID | 空间映射 |
|---|---|
| S0 | 原始14变量先做2×2 bilinear，再计算风速派生量并经过共享点编码器 |
| S1 | 网格点先编码，再做2×2 latent bilinear |
| S2 | learned 2×2 attention，不使用 wind geometry |
| S3 | fixed 4×4 Gaussian distance kernel |
| S4 | learned 4×4 attention，不使用 wind geometry |
| S5 | learned 4×4 attention并加入 wind geometry，即当前前端 |

S3 对每个场站使用局部网格东西、南北间距几何平均作为固定 \(\sigma_i\)，不根据 validation 搜索。

核心比较为 S0--S1、S1--S2、S3--S4、S1--S3、S2--S4 和 S4--S5。只有 learned 方案相对同支持 fixed 方案改善至少1 MW、3/3获胜且任一时段不恶化超过1 MW，才补齐10 seeds。否则选择位于最佳结果1 MW非劣范围内的最简单 fixed 方案。本轮不测试6×6、deformable、top-k或状态控制空间 attention。

## 六、接口、统计和产物

新增向后兼容配置：

- `bracket_time_feature_mode: relative_distances | zero`
- `bracket_mapper_activation: gelu | identity`
- `forcing_difference_input: actual | zero`
- 空间模式增加 `raw_bilinear` 和 `fixed_gaussian`

旧配置缺少这些字段时保持当前行为。新产物统一保存到 `artifacts/cross_resolution_temporal_v2/`：

- `formal/run_manifest.json`：记录复用与新增的全部40个正式运行来源；
- `formal/analysis_val/`：10-seed主表、paired CI、胜场、逐lead/phase/horizon结果；
- `mechanism/`、`robustness/`、`spatial/`：后续阶段；
- 只有预期任务、checkpoint、指标和预测审计全部通过后才写完成标记。

统计以每个 checkpoint 的独立预测为单位，报告 mean±SD、same-seed差异和95% CI，不将多seed预测平均成最终成绩。launcher接受手动物理GPU列表，显示总任务和单卡训练 tqdm，并支持严格产物审计后的断点续跑。

## 七、测试、最终模型和论文出口

测试覆盖：

- T2、PhaseOff、Linear、NoDiff参数量和输出 shape 一致；
- 时间特征置零、Identity激活和zero-difference数值正确、梯度有限；
- 2 h与缺节点索引、边界处理和预测 shape 正确；
- S0/S1顺序正确，所有 fixed/learned 空间权重和为1；
- 旧配置和 checkpoint 兼容，2-epoch smoke、Python/shell语法和完整 pytest 通过。

结构全部冻结后，训练10个最终 Normal 模型，再用其共识和固定 KD 权重0.5训练10个单模型学生。最终只报告独立 KD checkpoint 的均值、波动和代表模型；代表 seed 在查看2024前按 validation MAE 最接近十个 seeds 中位数确定。相对当前 KD 单模型采用2 MW非劣界限，不将约242.9 MW的 ensemble 作为主性能参考。

“当前 KD 单模型”的参考在新模型2024推理前固定为 `artifacts/horizon_specialist_v1/groups/kd_l1_anchor_taper/`：它是此前已完成路线中 validation 均值更好的可部署10-seed KD 版本。比较使用 same-seed 的独立 checkpoint MAE，不使用该组十个模型的预测平均。

2024只在时间、空间、KD和代表 seed 全部冻结后统一评估一次。若 learned 空间方案不成立，最终论文定位为：

> 规则网格到场站的空间错配由简单固定投影处理；小时强迫先编码为任务相关表示，再通过透明的线性时间对齐进入状态条件化修正。时间和空间分辨率错配是任务背景，而不是任意分辨率重建或连续时间建模贡献。

## 八、已实现的执行入口

- 时间正式验证：`shell/launch_cross_resolution_formal_v2.sh GPU_CSV`
- T2机制筛选/正式扩展：`shell/launch_cross_resolution_mechanism_v2.sh` 与 `shell/launch_cross_resolution_mechanism_formal_v2.sh`（当前被 Stage A gate 停止）
- 冻结 checkpoint 的节点鲁棒性：`shell/launch_cross_resolution_robustness_v2.sh`（当前被 Stage A gate 停止；每个 checkpoint 只加载一次数据和模型，再完成七个扰动）
- 空间筛选/正式扩展：`shell/launch_cross_resolution_spatial_v2.sh GPU_CSV` 与 `shell/launch_cross_resolution_spatial_formal_v2.sh GPU_CSV`
- 最终 Normal、ensemble teacher 和十个 KD 学生：`shell/launch_cross_resolution_final_v2.sh GPU_CSV`
- 冻结后的统一2024评估：`shell/launch_cross_resolution_final_test_v2.sh GPU_CSV`

以上 launcher 均接受如 `0,2,5,7` 的物理 GPU 列表，并保留总任务及单卡训练/推理 tqdm。断点续跑只复用通过配置、checkpoint、metrics 和 predictions 审计的任务。
