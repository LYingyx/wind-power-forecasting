# 物理表示筛选完成：结果入口

2026-10-25：本轮A阶段6944/6944项全部完成（6272训练+672旧权重诊断），未解决失败0，9461份选定权重核验通过。

阅读 [归因审查与已冻结B1清单](runs/physical_representation_single_farm_20261025_v1/analysis/ATTRIBUTION_REVIEW_20261025_CN.md)，然后查看 [逐风场逐seed实测表](runs/physical_representation_single_farm_20261025_v1/analysis/KEY_MODELS_PER_FARM_SEED_CN.md)、[全部原值CSV](runs/physical_representation_single_farm_20261025_v1/analysis/all_actual_per_farm_seed.csv) 和 [完成审计](runs/physical_representation_single_farm_20261025_v1/COMPLETION_AUDIT.json)。

主输入选SEL=DENS+DYN+STAB+STATE；2023相对W0 BASE的补充等权MAE/RMSE为−2.3525%/−2.0335%，逐seed改善站数分别53/55和49/51。主体收益不应归给密度或v³；优化TF后骨干差距缩小。B1选型已冻结，但还缺TF*+SEL的112个既有seed格子及2688个新seed训练；2024尚未运行，不能把A完成写成全项目完成。
