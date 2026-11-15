# Aggregate 天气驱动功率机制实验计划

本轮固定3 h aggregate history、R2 Regional CNN、静态capacity pooling和原生小时latent插值，只比较天气latent到24步集群功率的结构。唯一正式参照是同训练预算A0-FT；旧R2-W仅提供同seed初始化。

候选包括baseline/current headroom、递归headroom、固定/自适应weather-equilibrium dynamics、Q0.9 meteorological envelope × realization和固定/自适应weather-error decay。M1–M4以拓扑保证capacity factor位于 `[0,1]`；M4使用15 min解析relaxation更新，rate限制在 `[1/12,4] h⁻¹`。

2022训练，2023使用42/46/50完成筛选。数值最好的结构必定补齐42–51；另外保留达到预注册MAE或Ramp规则的候选，正式候选最多3个。最终按10-seed validation平均MAE选择，任何平均改善都可替代A0-FT，显著性只作描述。2024只在结构冻结后统一推理。

M5使用5折、6 h embargo的purged chronological cross-fitting生成2022 realization factor；M6使用独立R2 weather-mean模型。当前天气不拿未来首节点冒充，而从预测时刻可获得的forecast issue中取左右小时节点，在R2 latent空间插值，并审计 `issue ≤ origin`。

训练采用4 epoch head warm-up及最多16 epoch joint continuation，总预算不超过A0-FT的20 epoch；Regional CNN、capacity pooling和R2 interpolation始终冻结。主指标为cluster MAE，辅助报告RMSE、R²、Ramp MAE、三个horizon band、参数量和延迟。

全部产物位于 `artifacts/aggregate_weather_dynamics_v1/`，精简结果写入 `exp_docs/aggregate_weather_driven_dynamics_experiment_results.md`。统一启动命令：

```bash
bash shell/launch_aggregate_weather_dynamics_v1.sh 0,1,2,3,4,5,6,7 auto
```
