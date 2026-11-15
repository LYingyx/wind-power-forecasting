# 历史结果入口（不同协议单独保留）

本次完整重制的是2026-10-21起的9500个独立单场训练阶段。历史39个实验/资料目录已在原项目索引中登记；这里提供入口，不把其共享模型、迁移预算和旧独立试验误记为本轮结果，也不把原报告均值替代逐seed值。

- [原始逐run索引](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/results/RUN_METRICS.csv)：保留seed与指标来源，但部分指标本身是多站汇总；不能冒充单站值。
- [历史模型结构与协议说明](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/PROJECT_HANDOFF_20261019_ARCHIVE_CN.md)
- [39目录索引](../tables/historical_experiment_index.csv)

以下是已有逐站逐seed历史表；列名与协议以各文件为准：

- [prediction_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/prediction_station_seed.csv)
- [scenario_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/scenario_station_seed.csv)
- [condition_ablation_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/condition_ablation_station_seed.csv)
- [matched_output_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/matched_output_station_seed.csv)
- [independent_baselines_pilot_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/independent_baselines_pilot_station_seed.csv)
- [sharing_2x2_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/sharing_2x2_station_seed.csv)
- [source_full_year_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/source_full_year_station_seed.csv)
- [target20_common_seed_metrics.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/target20_common_seed_metrics.csv)
- [shared_baselines_station_seed.csv](/shared_data_ssd2/liying/wind_cluster_project_20261018/project/runs/single_farm_review_20261018/tables/shared_baselines_station_seed.csv)
