# W0 与 TF 的 RMSE 差距诊断（2026-11-15，零训练）

数据：`daily_site_finetune_20261114_v1` 保存的共享模型预测（`p0_prediction.npy`，即 W0_R3 / TF_R3）。55 站 × 5 seeds，2023 年，每日协议。只读 CPU，不训练，不读 2024。

- `diag.py`：误差分布、预测与观测的回归斜率、按实际变化幅度 |y − 当前功率| 分解 ΔMSE / ΔMAE，以及 W0 与 TF 平均。
- `ens.py`：5-seed 集成后的差距，以及 seed 方差占 MSE 的比例。
- `ramp.py`：按"预测变化 × 实际变化"二维分箱的 ΔMSE；实际变化 > 0.3 时，模型捕获的变化比例。

主要结论见对话记录与 `HISTORICAL_WEATHER_AND_W0_PLAN` 之后的分析。
- `phase.py`：同一时效段内，目标落在整点与 :15 / :30 / :45 时的 MSE 对比（检验线性插值是否造成小时内部的损失）。
