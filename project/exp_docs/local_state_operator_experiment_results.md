# Local State × Interaction Operator 实验结果

状态：全部完成。现有 L2/L3 被正确归类为 residual multiplicative；本轮没有训练重复的纯乘法 gate，也没有加入 station embedding、adapter 或 GNN。

2023 validation 冻结的 zero-invariant 模型为 **G-Int**。晋级的新候选：`['g_int', 'g_cat']`。Cat 诊断选择：`g_cat`。候选的 2024 只在冻结以后评估；该年份此前已用于旧实验报告，因此不表述为从未查看的纯盲测试。

完成审计：12/12 screening训练、20/20晋级候选的正式训练视图、20/20冻结测试、36份screening干预和120份新增formal validation/test干预均完整，并复用了L-RMul原有的60份正式干预；日志中没有Traceback或OOM，当前没有残留训练/评估进程。

## Executive conclusion

1. **Local operator-mismatch假设被否定。** L-Int和L-Cat在3-seed筛选中相对L1分别恶化0.157/0.169 MW，均只有1/3 seed获胜；station-roll和batch-roll也没有正退化。因此，更强MLP交互以及允许state-only路径都没有救活local state。
2. **Aggregate operator表达能力有小幅价值。** G-Int相对已有G-RMul在2023改善0.974 MW（8/10，配对95% CI不跨0），冻结2024改善0.420 MW（7/10，CI跨0）。它是本轮冻结的严格zero-weather-invariant模型。
3. **G-Cat是数值最优但仅作诊断。** 它达到267.655 MW/264.998 MW，但允许state-only路径；相对G-Int只再改善0.163/0.254 MW，配对CI均跨0，不能说明state-only shortcut是必要贡献。
4. **没有证据表明aggregate模型使用了正确的逐样本状态对应。** G-RMul/G-Int/G-Cat的batch-roll均没有造成退化，而固定reference造成约4/10/14 MW退化。准确解释是模型对状态表示的变化或分布敏感，但正确sample-state correspondence未被验证。
5. **新增收益集中在长时效。** G-Int和G-Cat相对G-RMul的3–6 h均改善，但0–1 h都略有恶化；G-Cat还恶化1–3 h。

## 3-seed screening

| Candidate | Reference | ΔMAE / MW | wins | gate-level state-sensitive | local correspondence | safety | promoted |
|---|---|---:|---:|---:|---:|---:|---:|
| G-Int | G-RMul (existing L2) | -1.303 | 3/3 | True | False | True | True |
| L-Int | L1 no-state | +0.157 | 1/3 | True | False | True | False |
| G-Cat | G-RMul (existing L2) | -1.106 | 3/3 | True | False | True | True |
| L-Cat | L1 no-state | +0.169 | 1/3 | True | False | True | False |

门槛固定为：ΔMAE≤−0.5 MW且至少2/3获胜；batch/reference至少一种 degradation≥0.5 MW且至少2/3为正；RMSE恶化≤3 MW，任一时效恶化≤5 MW。local correspondence另要求station-roll degradation≥0.25 MW且至少2/3为正。

这里的`gate-level state-sensitive=True`由预注册的“batch或reference任一满足”规则产生。事后完整干预表明，四个新模型全部是reference而非batch-roll触发该标记；因此它不能被解释成正确逐样本对应关系有效。

## 2023 validation（10 seeds）

| Model | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 no-state | 269.437 ± 2.790 | 394.454 ± 5.053 | 0.901 ± 0.003 | 58.977 ± 0.390 | 109.084 ± 0.866 | 245.863 ± 2.629 | 338.604 ± 3.753 |
| G-RMul (existing L2) | 268.792 ± 2.791 | 393.278 ± 4.653 | 0.902 ± 0.002 | 58.920 ± 0.403 | 109.708 ± 0.878 | 245.888 ± 2.635 | 337.089 ± 3.860 |
| L-RMul (existing L3) | 269.781 ± 2.849 | 394.820 ± 4.824 | 0.901 ± 0.002 | 58.981 ± 0.382 | 109.170 ± 1.024 | 246.199 ± 2.751 | 339.039 ± 3.844 |
| G-Int | 267.818 ± 2.960 | 391.641 ± 5.390 | 0.903 ± 0.003 | 58.786 ± 0.421 | 109.867 ± 0.914 | 245.772 ± 2.459 | 335.166 ± 4.512 |
| G-Cat | 267.655 ± 2.738 | 390.523 ± 4.334 | 0.903 ± 0.002 | 58.691 ± 0.429 | 110.395 ± 0.982 | 246.448 ± 2.396 | 334.212 ± 4.126 |

## 冻结后的2024（10 seeds）

| Model | MAE / MW | RMSE / MW | R² | Ramp MAE / MW | 0–1h | 1–3h | 3–6h |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 no-state | 266.837 ± 1.996 | 392.888 ± 3.581 | 0.894 ± 0.002 | 56.546 ± 0.317 | 104.137 ± 0.735 | 239.981 ± 1.776 | 338.975 ± 2.836 |
| G-RMul (existing L2) | 265.672 ± 2.200 | 390.879 ± 3.336 | 0.895 ± 0.002 | 56.486 ± 0.296 | 104.593 ± 0.649 | 239.771 ± 1.967 | 336.631 ± 3.300 |
| L-RMul (existing L3) | 266.625 ± 1.396 | 392.448 ± 2.419 | 0.894 ± 0.001 | 56.520 ± 0.267 | 104.062 ± 0.644 | 239.804 ± 1.447 | 338.694 ± 2.179 |
| G-Int | 265.252 ± 2.448 | 390.103 ± 4.147 | 0.895 ± 0.002 | 56.370 ± 0.291 | 104.953 ± 1.002 | 240.228 ± 2.271 | 335.367 ± 3.393 |
| G-Cat | 264.998 ± 2.436 | 389.525 ± 3.130 | 0.895 ± 0.002 | 56.295 ± 0.304 | 105.459 ± 0.912 | 240.928 ± 2.504 | 334.223 ± 3.194 |

## 配对比较

负的candidate-reference差值表示候选更好。

| Split | Candidate − reference | ΔMAE / MW | 95% paired CI | wins |
|---|---|---:|---:|---:|
| val | G-Int − G-RMul (existing L2) | -0.974 | [-1.761, -0.186] | 8/10 |
| val | G-Cat − G-RMul (existing L2) | -1.137 | [-1.781, -0.493] | 9/10 |
| val | G-RMul (existing L2) − L1 no-state | -0.645 | [-1.130, -0.161] | 8/10 |
| val | L-RMul (existing L3) − L1 no-state | +0.344 | [-0.154, +0.842] | 3/10 |
| val | L-RMul (existing L3) − G-RMul (existing L2) | +0.989 | [+0.389, +1.589] | 1/10 |
| test | G-Int − G-RMul (existing L2) | -0.420 | [-0.987, +0.147] | 7/10 |
| test | G-Cat − G-RMul (existing L2) | -0.674 | [-1.161, -0.187] | 8/10 |
| test | G-RMul (existing L2) − L1 no-state | -1.165 | [-1.883, -0.447] | 10/10 |
| test | L-RMul (existing L3) − L1 no-state | -0.212 | [-1.107, +0.684] | 5/10 |
| test | L-RMul (existing L3) − G-RMul (existing L2) | +0.954 | [-0.124, +2.032] | 3/10 |

### 关键补充比较

| Split | Candidate − reference | ΔMAE / MW | 95% paired CI | wins |
|---|---|---:|---:|---:|
| validation | G-Int − L1 | −1.619 | [−2.357, −0.881] | 10/10 |
| 2024 | G-Int − L1 | −1.585 | [−2.263, −0.908] | 10/10 |
| validation | G-Cat − G-Int | −0.163 | [−0.647, +0.320] | 7/10 |
| 2024 | G-Cat − G-Int | −0.254 | [−0.732, +0.224] | 6/10 |
| validation | G-Int − A0-FT | −5.377 | [−7.987, −2.767] | 8/10 |
| 2024 | G-Int − A0-FT | −5.074 | [−8.921, −1.226] | 8/10 |

因此，G-Int相对无状态local L1的增量稳定，但它相对G-RMul的额外优势在2024缩小；G-Cat相对G-Int没有稳定优势。与旧aggregate A0-FT相比，G-Int的总改善还包含此前已经确认的local-to-global信息结构收益，不能全部归因于本轮interaction operator。

## 状态干预

正 degradation 表示正确状态对应关系被模型利用。

| Split | Model | Intervention | degradation / MW | positive seeds |
|---|---|---|---:|---:|
| val | G-RMul (existing L2) | batch_roll | -0.192 ± 0.136 | 0/10 |
| val | G-RMul (existing L2) | reference | +4.066 ± 4.444 | 10/10 |
| val | L-RMul (existing L3) | station_roll | -0.043 ± 0.062 | 2/10 |
| val | L-RMul (existing L3) | batch_roll | -0.058 ± 0.059 | 0/10 |
| val | L-RMul (existing L3) | reference | +0.141 ± 0.333 | 6/10 |
| val | G-Int | batch_roll | -0.310 ± 0.213 | 0/10 |
| val | G-Int | reference | +10.492 ± 8.456 | 10/10 |
| val | G-Cat | batch_roll | -0.252 ± 0.370 | 2/10 |
| val | G-Cat | reference | +14.445 ± 6.923 | 10/10 |
| test | G-RMul (existing L2) | batch_roll | -0.134 ± 0.077 | 0/10 |
| test | G-RMul (existing L2) | reference | +3.974 ± 3.727 | 10/10 |
| test | L-RMul (existing L3) | station_roll | +0.054 ± 0.120 | 5/10 |
| test | L-RMul (existing L3) | batch_roll | -0.036 ± 0.035 | 0/10 |
| test | L-RMul (existing L3) | reference | -0.027 ± 0.255 | 2/10 |
| test | G-Int | batch_roll | -0.189 ± 0.212 | 1/10 |
| test | G-Int | reference | +9.793 ± 7.752 | 10/10 |
| test | G-Cat | batch_roll | -0.101 ± 0.370 | 3/10 |
| test | G-Cat | reference | +14.175 ± 7.122 | 10/10 |

`reference`把所有样本替换为统一参考轨迹，既消除了状态变化，也会改变state encoder的输入分布；它证明branch不是常数函数，但不能单独证明样本匹配。相反，`batch_roll`直接破坏样本对应：G-Int在validation/test分别为−0.310/−0.189 MW，G-Cat为−0.252/−0.101 MW，均没有出现预期的正退化。这是本轮最重要的机制限制。

## 时效与动态解释

相对G-RMul：

| 模型 | Split | Overall | Ramp | 0–1h | 1–3h | 3–6h |
|---|---|---:|---:|---:|---:|---:|
| G-Int | validation | −0.974 | −0.134 | +0.159 | −0.116 | −1.923 |
| G-Int | 2024 | −0.420 | −0.117 | +0.359 | +0.457 | −1.264 |
| G-Cat | validation | −1.137 | −0.229 | +0.688 | +0.560 | −2.877 |
| G-Cat | 2024 | −0.674 | −0.192 | +0.866 | +1.156 | −2.408 |

两个高表达算子的总体收益都由3–6 h主导。G-Int的约束代价较小；G-Cat虽然总体MAE最低，却通过更明显牺牲0–3 h换取长时效收益。

## 解释规则

- L-Int 优于L1且状态干预稳定为正：支持“此前失败来自operator mismatch”。
- L-Cat有效而L-Int无效：收益更可能来自state-only predictive path，不能称为weather-response modulation。
- L-Int/L-Cat均无效或干预接近零：停止local-state modulation。
- G-Int未超过G-RMul：保留简单的aggregate residual gain。
- Cat明确不是zero-weather-invariant主模型；即使数值更低也只作为表达能力上界。

## 最终判定

### 得到支持

- aggregate recent state与天气latent之间使用高表达、zero-invariant交互，比简单residual gain取得约1 MW validation增益。
- 不需要state-only shortcut也能获得绝大部分增益：G-Int与G-Cat差距很小且不稳定。
- 交互算子的贡献主要作用于3–6 h，而不是超短期。

### 被否定或未得到支持

- **Local state只是被乘法算子压死：否定。** L-Int和L-Cat都未通过筛选，且局地对应干预没有正信号。
- **正确逐样本aggregate state-weather correspondence有效：未得到支持。** batch-roll没有使任何aggregate模型恶化。
- **State-only直接路径带来必要增益：未得到支持。** G-Cat没有稳定超过G-Int。
- **Station embedding应立即启动：否定。** `station_identity_eligible=false`，继续遵守停止规则。

### 模型记录

- 机制约束下的正式模型：**G-Int，267.818 ± 2.960 MW（validation），265.252 ± 2.448 MW（2024）**。
- 纯数值最低但不可作状态调制主张的诊断模型：**G-Cat，267.655 ± 2.738 MW，264.998 ± 2.436 MW**。
- G-Int参数量390,763，比L1/G-RMul多24,768（约6.8%），仍少于A0-FT的478,293。

最稳妥的论文表述是：

> 在共享local-to-global模型中，提高aggregate state与局地天气latent交互的表达能力可以小幅改善总体预测，且zero-weather-invariant形式已足以获得该收益；但更强交互仍不能使farm-specific local state产生可辨识增量。现有干预只证明模型对aggregate state分布敏感，尚不能证明其利用了正确的逐样本state-weather correspondence。

完整机器结果：`artifacts/local_state_operator_v1/analysis/summary.json`。
