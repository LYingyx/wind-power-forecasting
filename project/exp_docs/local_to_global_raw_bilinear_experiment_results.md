# Raw-Bilinear Local-to-Global 实验结果

## 完成与冻结状态

实验已全部完成。完成标记生成于 2026-09-04 19:39:58（Asia/Shanghai），当前没有残留的训练、评估或调度进程。

- 筛选：6 个 `screen_base` 和 9 个 `screen_core` checkpoint 全部完成；
- 正式实验：L1/L2/L3 各 10 seeds，共 30 个训练 checkpoint 和 30 个冻结测试全部完成；
- 机制检查：L3 的 3 种状态错配干预、10 seeds、validation/test，共 60 份评估全部完成；
- 模型选择只使用 2023 validation；冻结文件明确记录 `test_seen=false`，随后才统一读取 2024；
- 2023 validation 冻结的最终模型为 **L2 aggregate-state local**。

随机种子统一为 42–51。表中 `±` 为 10 个独立 checkpoint 的跨 seed 标准差，不是标准误。

## 模型定义与公平控制

| 模型 | 信息结构 |
|---|---|
| A0-FT aggregate | 聚合历史与区域聚合天气直接预测集群功率；同训练预算强控制 |
| L1 local no-state | 原始 14 变量固定双线性采样到 56 场站，共享局地天气—功率转换，最后按容量恢复 MW 并求和；无状态调制 |
| L2 aggregate-state local | 在 L1 上用共享的近期集群轨迹对各局地非零天气 latent 做乘法调制；无 state-only correction 路径 |
| L3 local-state shared | 在 L1 上用各场站自身近期功率状态调制本站天气 latent；状态编码器与调制器跨站共享 |

局地模型使用 `cluster loss + 0.5 × station capacity-factor loss`，没有 learned interpolation、GNN、KD、预测 ensemble 或 ramp loss。状态只允许乘法缩放天气表示；天气表示为零时，状态不能独立产生 correction。

## 2023 validation

| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-FT aggregate | 273.195 ± 2.435 | 391.839 ± 3.625 | 0.903 ± 0.002 | 60.102 ± 0.385 | 103.246 ± 0.894 | 253.228 ± 1.395 | 343.156 ± 3.987 |
| L1 local no-state | 269.437 ± 2.790 | 394.454 ± 5.053 | 0.901 ± 0.003 | 58.977 ± 0.390 | 109.084 ± 0.866 | 245.863 ± 2.629 | 338.604 ± 3.753 |
| **L2 aggregate-state local** | **268.792 ± 2.791** | 393.278 ± 4.653 | 0.902 ± 0.002 | **58.920 ± 0.403** | 109.708 ± 0.878 | 245.888 ± 2.635 | **337.089 ± 3.860** |
| L3 local-state shared | 269.781 ± 2.849 | 394.820 ± 4.824 | 0.901 ± 0.002 | 58.981 ± 0.382 | 109.170 ± 1.024 | 246.199 ± 2.751 | 339.039 ± 3.844 |

## 冻结后的 2024 test

| 模型 | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-FT aggregate | 270.325 ± 3.949 | 393.077 ± 6.612 | 0.893 ± 0.004 | 57.818 ± 0.465 | **97.372 ± 0.799** | 244.996 ± 2.437 | 344.862 ± 6.178 |
| L1 local no-state | 266.837 ± 1.996 | 392.888 ± 3.581 | 0.894 ± 0.002 | 56.546 ± 0.317 | 104.137 ± 0.735 | 239.981 ± 1.776 | 338.975 ± 2.836 |
| **L2 aggregate-state local** | **265.672 ± 2.200** | **390.879 ± 3.336** | **0.895 ± 0.002** | **56.486 ± 0.296** | 104.593 ± 0.649 | **239.771 ± 1.967** | **336.631 ± 3.300** |
| L3 local-state shared | 266.625 ± 1.396 | 392.448 ± 2.419 | 0.894 ± 0.001 | 56.520 ± 0.267 | 104.062 ± 0.644 | 239.804 ± 1.447 | 338.694 ± 2.179 |

## 配对比较

下表使用同 seed checkpoint 配对。`ΔMAE = candidate − reference`，负值表示候选更好；win 表示候选 MAE 更低的 seed 数。

| Candidate vs reference | 2023 ΔMAE / MW | 2023 wins | 2024 ΔMAE / MW | 2024 wins |
|---|---:|---:|---:|---:|
| L1 − A0-FT | −3.758 | 8/10 | −3.488 | 8/10 |
| L2 − A0-FT | **−4.403** | 8/10 | **−4.654** | 8/10 |
| L3 − A0-FT | −3.414 | 8/10 | −3.700 | 8/10 |
| L2 − L1 | −0.645 | 8/10 | −1.165 | 10/10 |
| L3 − L1 | +0.344 | 3/10 | −0.212 | 5/10 |
| L3 − L2 | +0.989 | 1/10 | +0.954 | 3/10 |

L2 相对 A0-FT 的总体 MAE 相对改善为 2023 的 1.61% 和 2024 的 1.72%。L1 已经取得主要增益，L2 的 aggregate recent state 在其上提供较小但跨年份一致的附加改善。相反，L3 没有稳定优于无状态 L1，并在两个年份都明显差于 L2。

## 时效与轨迹动态

L2 相对 A0-FT 的分项差值如下；负值表示改善。

| Split | Overall MAE | Ramp MAE | 0–1h | 1–3h | 3–6h |
|---|---:|---:|---:|---:|---:|
| 2023 validation | −4.403 | −1.182 | **+6.462** | −7.340 | −6.067 |
| 2024 test | −4.654 | −1.331 | **+7.221** | −5.224 | −8.231 |

局地模型的净收益主要来自 1–6 h，且 Ramp MAE 跨年份稳定改善；代价是 0–1 h 明显恶化。因此，结果支持局地天气转换改善中长时效和轨迹动态，但不支持它在所有时效上一致占优。若部署更重视超短期，应考虑保留 A0-FT 的近时效输出或单独研究预注册的 horizon fusion，不能忽略这一退化。

另一个需要保留的细节是：2023 上 L2 虽然 MAE 更好，但 RMSE 比 A0-FT 高 1.439 MW、R² 略低，说明 validation 的大误差尾部没有同步改善；2024 上 MAE、RMSE 和 R² 才一致改善。

## 局地状态对应关系干预

干预只施加于 L3，天气、模型参数和其他输入保持不变。`degradation = intervention MAE − original MAE`；如果模型真正使用了样本和场站对应的局地状态，应稳定为明显正值。

| 干预 | 2023 degradation / MW | 2024 degradation / MW |
|---|---:|---:|
| 场站维循环错配 `station_roll` | −0.043 ± 0.062 | +0.054 ± 0.120 |
| 样本维循环错配 `batch_roll` | −0.058 ± 0.059 | −0.036 ± 0.035 |
| 固定为 reference state | +0.141 ± 0.333 | −0.027 ± 0.255 |

所有效应都接近零、符号跨年份不一致，且最大平均绝对变化仅 0.141 MW。由此可判定：L3 几乎没有利用正确的“样本—场站局地状态对应关系”。它优于 A0-FT 的部分不能解释成局地 state-conditioned weather response 的证据。

## 筛选阶段与停止规则

筛选阶段仅使用 seeds 42/46/50：

| Cell | 2023 MAE / MW | 作用 |
|---|---:|---|
| L1-C | 276.425 | 仅 cluster supervision |
| L1-M | 272.649 | cluster + 0.5 × station CF supervision |
| L1-MFT | 268.638 | L1-M 同预算继续训练 |
| L2 aggregate-state | 267.490 | aggregate state control |
| L3 local-state | 269.274 | 主 local-state 候选 |

L1-M 相对 L1-C 改善 3.776 MW，说明局地辅助监督对局部分解的可辨识性有实际作用；这仍是 3-seed 筛选证据，不应当作 10-seed 主结论。L3 同时输给 L1-MFT 和 L2，因而预注册的 `local_state_gate=false`。按照停止规则，没有启动 station ID embedding 和 per-station affine adapter；这两个结构是“未测试”，不是“已证伪”。

## 假设判定

### 得到支持

1. **聚合前做局地天气—功率转换有价值。** 无状态 L1 相对 aggregate A0-FT 在 2023/2024 分别改善 3.758/3.488 MW，均为 8/10 seeds 获胜。
2. **共享 aggregate recent state 可提供小幅附加收益。** L2 相对 L1 在 2023 改善 0.645 MW（8/10），在冻结 2024 改善 1.165 MW（10/10）。
3. **局地分支需要辅助场站监督。** 3-seed 筛选中加入 station CF loss 改善 3.776 MW。
4. **局地建模的收益偏向中长时效和 ramp。** 这一方向在两个年份一致。

### 被当前实验否定

1. **“正确的场站自身状态应优于 aggregate state”不成立。** L3 相对 L2 在 2023/2024 分别恶化 0.989/0.954 MW。
2. **“L3 的增益来自局地状态—天气对应关系”不成立。** 三类错配/固定状态干预几乎不改变性能。
3. **“局地转换会改善所有 horizon”不成立。** 0–1 h 在两个年份分别恶化 6.462/7.221 MW。

### 尚未回答

由于 gate 未通过，本轮不能评价 station embedding、station affine adapter、独立场站模型或跨场站 GNN。也不能据此声称不存在任何形式的局地运行状态效应；准确说法仅限于：当前 3 h 场站功率编码与严格乘法调制没有提供可辨识增量。

## 解释边界

A0-FT 使用聚合历史/聚合监督，共 478,293 个参数；L1/L2/L3 使用 56 场站历史与场站标签，共 365,995 个参数。因此，L1 相对 A0-FT 的差值衡量的是完整的 local-to-global 信息结构——包括局地天气、局地历史、共享局地预测和辅助监督——不能严格归因于单一的“先转换还是先聚合”运算顺序。局地模型参数更少且仍取得更低 MAE，是工程优势，但不是隔离因果机制的参数匹配实验。

## 最终结论

本轮最稳妥、可用于论文的结论是：

> 对 6 h 集群风电预测，过早聚合空间天气与场站信息会损失局地非线性天气—功率转换所需的信息。采用固定双线性局地化、跨站共享的局地响应模型以及物理求和，在 2023 validation 和冻结后的 2024 test 上均优于强 aggregate 基线。最终 L2 分别降低 MAE 4.403 MW 和 4.654 MW，并稳定改善 ramp 与中长时效。然而，场站自身近期功率状态没有表现出可辨识的天气响应调制作用；一个共享的 aggregate recent trajectory 反而更好。

因此最终模型记录为 **L2 aggregate-state local**。论文主张应聚焦于 **local conversion before physical aggregation** 与共享局地建模，不应声称已验证 local state dependence。

## 机器记录

- 完成标记：`artifacts/local_to_global_raw_bilinear_v1/COMPLETE`
- validation 冻结：`artifacts/local_to_global_raw_bilinear_v1/selection/validation_freeze.json`
- 全量机器汇总：`artifacts/local_to_global_raw_bilinear_v1/analysis/summary.json`
- 实验方案：`exp_docs/local_to_global_raw_bilinear_experiment_plan.md`
