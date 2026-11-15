# 自适应天气提取：实时逐风场逐seed结果

更新时间：2026-10-23T20:04:14.411650+08:00。全时效MAE为预先指定主指标；不设置改善幅度门槛；不跨seed或风场平均。

[逐次实测值](results_per_farm_seed.csv) · [配对原值](paired_actual_results.csv) · [逐seed方向次数](improvement_counts_by_seed.csv) · [实际配置与成本](actual_configs_and_costs.csv) · [干预](intervention_results_per_seed.csv) · [历史×天气交互](history_weather_interaction_per_seed.csv)

[分时效配对](paired_horizons_per_seed.csv) · [逐lead配对](paired_leads_per_seed.csv.gz) · [分场景配对](paired_scenarios_per_seed.csv.gz) · [逐站逐seed退步清单](regressions_per_farm_seed.csv) · [干预分时效/场景](intervention_slices_per_seed.csv.gz) · [注意力物理关联](attention_associations_per_seed.csv) · [贡献证据台账](CONTRIBUTION_EVIDENCE_CN.md)

当前队列：`[{"kind": "evaluate", "status": "done", "count": 7840}, {"kind": "train", "status": "done", "count": 8016}]`。未完成项没有结果，不补值。旧W0和原Transformer/TiDE明确标为复用。2024仍为跨年复评。

ST00/10/01/11分别表示空间/时间的静态或自适应；ST_NO_H仅冻结权重生成中的历史，保留解码器真实历史；H2保留网络层，以训练期历史替代真实历史，末值锚点仍使用当前功率。完整结构与实验协议见[运行入口](../README_CN.md)。

所有gain列为参考误差减候选误差，正值表示改善；干预increase列为干预误差减匹配误差。缺失场景写unavailable，不计为持平。退步清单收录MAE或RMSE任一变差的配对，保留另一指标，避免隐藏指标间取舍。

注意力关联只在同一模型的起报样本/lead内统计，分别列全年和各季节，不跨风场/seed平均。upwind为天气风矢量反方向；时间差为所读取节点时刻减目标时刻。注意力对应candidate分支；最终前端是point+tanh(gate)×(candidate−point)，gate可能为负，因此权重和空间重心不能直接解释为最终天气输入的正向物理贡献。关联不代表因果；需同时检查天气与权重交换、gate和误差变化。
