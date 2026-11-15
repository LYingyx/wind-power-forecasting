# 模块证据与取舍：允许小幅收益的修订判断

日期：2026-10-16。任务：为论文确定哪些设计有实验证据、收益多大、哪些仍缺直接消融。本文核对已有报告，不产生新的训练或评估结果。

## 1. 模块取舍原则

用户明确接受小幅性能收益。因此，旧实验使用的 0.0003 CF MAE 实用阈值不再作为后续模块保留或替换的硬门槛。

模块选择同时报告均值改善、配对不确定性、跨种子方向和额外成本。小幅收益可以支持保留；未显著不等于无效；单独一个较好的均值也不能自动称为稳定、必需或物理机制成立。已有报告的预设检验结果如实保留，新的取舍属于本次论文设计判断，不改写为原协议已经通过。

模块是否值得存在与模块是否值得成为论文贡献是两回事。一个零参数读出获得小幅收益，可以保留，但不必单列为方法创新。

## 2. 必须区分完整 memory 与额外 summary

当前模型的历史编码器产生 M，存在两条路径：

1. 完整历史路径：M → Transformer decoder 的 memory K/V。
2. 额外摘要路径：Pool(M) → condition encoder → weather adapter。

CBH 使用均值摘要，CB 仅中和第二条路径的摘要输入。两者均保留完整 decoder memory、冻结历史 baseline b、condition 中的 b/Δb、天气和公共 context。

因此 CBH−CB 回答“已有完整历史 decoder 时，再向 condition 提供均值摘要有没有增量”。它不回答“历史 memory 有没有用”，更不能据此删除整个 history encoder。

## 3. Summary 的明确结果

指标为 56 站等权 CF MAE；负差分表示改进。所有结果是十个 seed 的单模型指标均值，不是 ensemble。2023 用于验证和选择；2024 是已经查看过的跨年复评。

| 实现 | 2023 MAE | 2024 MAE | 相对 CBH 的额外参数 |
| --- | ---: | ---: | ---: |
| CB：无额外 summary | 0.083964 | 0.081797 | 0，模块形状保留 |
| CBH：mean(M) | 0.083814 | 0.081507 | 0 |
| SLAST：最后一个 contextual memory token | 0.083626 | 0.081351 | 0 |
| SRECENT：最近四个 token 均值 | 0.083672 | 0.081418 | 0 |
| STREND：均值加历史 token 趋势组合 | 0.083723 | 0.081452 | 0 |
| MFLAT：完整 memory 展平后 MLP | 0.083739 | 0.081417 | 102,656 |
| MPOOL：重复均值后同形状 MLP | 0.083750 | 0.081436 | 102,656 |
| HMLP：原始历史功率 MLP | 0.083893 | 0.081580 | 5,888 |

| 比较 | 2023 配对差分 [95% CI] | 胜出 seeds | 2024 配对差分 [95% CI] | 胜出 seeds |
| --- | --- | ---: | --- | ---: |
| CBH−CB | −0.000150 [−0.000307, +0.000028] | 8/10 | −0.000290 [−0.000517, −0.000078] | 6/10 |
| SLAST−CB | −0.000338 [−0.000508, −0.000165] | 9/10 | −0.000447 [−0.000678, −0.000234] | 8/10 |
| SLAST−CBH | −0.000188 [−0.000322, −0.000053] | 7/10 | −0.000157 [−0.000299, −0.000003] | 8/10 |
| MFLAT−CBH | −0.000076 [−0.000173, +0.000026] | 8/10 | −0.000090 [−0.000277, +0.000100] | 5/10 |
| MFLAT−MPOOL | −0.000011 [−0.000068, +0.000057] | 7/10 | −0.000019 [−0.000120, +0.000094] | 6/10 |

配对差分取原统计结果，可能与六位小数均值相减相差最后一位。

### 对这些结果的修订判断

- **额外 summary 分支有保留依据。** SLAST−CB 不仅均值更好，而且在其原预定次要比较中已通过统计和实际门槛。2023/2024 Holm p 分别为 0.008789/0.007812。不能把全部 summary 证据概括成“没有确认作用”。
- **均值读出有小幅正向信号，但不是唯一合理选择。** CBH−CB 两年均值改善；2023 CI 跨零，2024 为 6/10 seeds 改善，因此表述为“观察到小幅增益，种子间证据较弱”，而非无效或必需。
- **按当前允许小幅收益的选择标准，建议采用 SLAST 作为论文候选的 summary 读出。** 它在用于选择的 2023 年优于 CBH，零新增参数，并有同方向跨年结果。相对 CBH 的 2023 Holm p=0.049805；2024 p=0.131836，因此只能说小幅验证集优势与跨年同向，不能说两年均通过全部显著性标准。其相对 CB 的证据强于相对 CBH。
- SLAST 是经过历史 Transformer 上下文编码后的最后一个 token，不是仅使用最后一个原始功率值；完整 memory 仍进入 decoder。
- **目前没有优先采用大型 MLP summary 的理由。** 它们增加参数，尚未确认完整 token 输入优于均值输入；相对均值 CBH 的改善也较不确定。这里是现有成本—收益判断，不是否定所有 MLP 读出。

上述选择是论文实现候选的更新，不把历史 CBH 实验重命名为 SLAST。已有条件干预、Jacobian、共享/独立等 CBH 结果仍标注 CBH；未在 SLAST 上执行的诊断不得直接移植为其测量结果。summary 实验中的各模型都重新训练过，提取方式也可能改变共享 encoder 的优化，不能把差值解释为固定同一个 M 后的纯读出效果。

## 4. 其他关键模块的证据与决策

| 模块或设计 | 已有直接证据 | 当前取舍 | 结论边界 |
| --- | --- | --- | --- |
| 未来天气修正路径 | canonical W−History：2023/2024 为 −0.010897/−0.010383，均 10/10 | 保留 | 增加了天气修正网络及其容量/context，不是完全隔离的纯天气信息主效应 |
| 显式 baseline trajectory 条件入口 | 最终骨干 CB−CNEUTRAL：−0.004680/−0.004442，均 10/10；CBH−CH 同样显著改善 | 保留，作为主要设计依据 | CB 实际包含 b 和 Δb，首个差分还含当前功率；不能命名为严格 b-only |
| 原始历史在 correction 中的作用 | canonical J−T：−0.000918/−0.001857，均 10/10 | 支持保留历史访问路径 | 来自 canonical 网络；不是最终 P0I decoder memory 的单独删除实验 |
| 额外 history-summary 条件入口 | SLAST−CB：−0.000338/−0.000447，9/10、8/10 | 保留；推荐 SLAST 候选 | 支持当前候选的一条附加入口，不证明所有 summary 必需 |
| Δb | canonical LP−L：−0.000035/−0.000020；2023 CI 全负、8/10，2024 CI 跨零、7/10 | 可保留为低成本特征；不因低于旧阈值判无用 | 只有很小信号；最终骨干尚无严格 b-only 对照，不能声称已单独确认最终模型中的作用 |
| 显式乘法门控 | 匹配的 JM−G：+0.000014/+0.000127，无可靠改进 | 当前实现采用普通 MLP，不增加乘法门控 | G−J 本身容量不同；应使用 JM−G 判断额外乘法 |
| 联合 adapter 相对可分离 adapter | P0I PJ−PA 未显示可靠优势，PA 均值略低 | 两种均有依据，joint 形式不能列为必要创新 | PA 的后续 decoder 仍联合处理输入，不是严格输出 additive |
| 线性 target-time 天气对齐 | native、local/global 等替代未形成稳定优越性 | 保留简洁线性对齐 | 不讲 native retrieval 或高频真实天气重建创新 |
| 冻结历史 baseline 作为输出参照 | PJ/PA 整体优于 P0；PB 只替换参照却退化 | 保留已验证的参照加条件修正组合 | 不把单独使用 b 的输出 bypass 宣称为增益来源，也没有 frozen vs joint 的独立必要性证明 |

完整历史 memory 的独立价值、冻结训练本身、特定层数、各 context 字段和 weather bypass，不应因总体模型有效就逐一称为已证明必要。每个普通算子不需要被包装成科学贡献；针对需要解释的关键分支，应使用实际对应的消融。

## 5. 哪些还缺证据，哪些不需要重做

**summary 不是“没做对照”。** CBH/CB 以及 SLAST、SRECENT、STREND 和 MLP 对照已经完成。现在可据这些结果决定保留或替换，不需要为越过旧 0.0003 阈值重新训练。

如果论文要逐项确认最终模型所有关键分支，现有报告尚不能替代以下直接对照：

1. 最终骨干中仅中和 decoder memory 的样本特异内容，同时保留真实 summary、b、天气和 context，并重新训练。这隔离的是 decoder 历史入口，不是删除所有历史来源。
2. 最终骨干中仅去掉 condition 的 Δb，保留 b 及其他路径，并重新训练。canonical LP−L 的小效应不能直接充当这一实验。

这两项是明确的归因缺口，不是已有模块被证明无效；本记录没有伪称它们已经完成或启动。若执行，需固定参照 checkpoint、数据/切分、初始化、训练与评估协议。中和输入会改变有效函数类，参数注册数相同不能写成所有有效自由度完全一致。诊断时置零已有 checkpoint 的输入也不能替代重新训练的消融。

## 6. 论文可直接采用的措辞

> 在保留完整历史 decoder memory 的条件下，额外历史摘要入口提供了小幅预测增益，其效果取决于读出方式。最后一个 contextual history token 相对无摘要对照取得了两年一致的改进；相对均值读出则表现为更小的验证集优势及同方向跨年结果。由此支持保留简洁的历史摘要条件入口，但不将特定池化算子视为独立创新，也不把这些结果解释为完整历史 memory 不可删除的证明。

> 显式历史预测轨迹是当前证据最强的 correction 条件。原始历史与显式轨迹在 canonical 架构中的互补性有直接支持；最终骨干中的各条历史访问路径需分别归因。

## 7. 证据来源

- [最终骨干条件入口消融与审计](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_attribution_closure_review_20260916/REPORT_CN.md)
- [固定 summary 读出比较](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_summary_calibration_20260916_v1/analysis/REPORT_CN.md)
- [MLP summary 完整结果](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md)
- [canonical 归因结果](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/trajectory_attribution_20260914_v1/analysis/REPORT_CN.md)
- [共同协议迁移结果](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/p0i_condition_transfer_20260915_v1/analysis/REPORT_CN.md)
- [原 CBH 架构和实验索引](/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/main_model_architecture_review_20260919_v1/REPORT_CN.md)

统计范围：seed CI 主要描述固定数据上的训练随机性；跨年复评不是全流程盲测。允许小幅收益不改变这些统计含义。
