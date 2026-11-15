# 区域可靠性与距离验证
完成 4256 个结果；2023；每站 2 个种子。
指标先站内平均种子，再风场等权；负值改善。完整逐站逐种子值、失败工况和详细分组见同目录 CSV。
距离匹配比较只用所有圈均有候选的共同支持站；FULL/MATCH 均为等权、同一摘要槽位，N3固定。
历史/天气真实状态不在错配中替换；C2仅将全部邻站观测共同换到2022同月同钟点档案，仍有年际分布混杂。

|骨干/候选|整体对N0 MAE/RMSE|高功率3–6h对N0 RMSE|全部门槛通过|
|---|---:|---:|---|
|W0/G_FREE|-0.7678% / -0.7174%|+0.2280%|False|
|W0/G_LEAD|-0.7196% / -0.7490%|-0.0358%|True|
|W0/G_STATE|-0.7790% / -0.7361%|-0.0068%|False|
|W0/G_STATE_LEAD|-0.7184% / -0.6985%|-0.1318%|True|
|W0/R3_MEAN_DELTA|-0.8649% / -0.7815%|+0.0692%|True|
|TF_z1_lr0.0002_dp0.05/G_FREE|-0.8699% / -0.7472%|+0.6456%|False|
|TF_z1_lr0.0002_dp0.05/G_LEAD|-0.9005% / -0.8626%|-0.1470%|True|
|TF_z1_lr0.0002_dp0.05/G_STATE|-0.8320% / -0.7820%|+0.2484%|False|
|TF_z1_lr0.0002_dp0.05/G_STATE_LEAD|-0.8735% / -0.8660%|-0.2083%|True|
|TF_z1_lr0.0002_dp0.05/R3_MEAN_DELTA|-1.0615% / -0.9565%|+0.0860%|True|

自动扩种子清单：[["TF_z1_lr0.0002_dp0.05", "C2_G_FREE"], ["TF_z1_lr0.0002_dp0.05", "C2_G_LEAD"], ["TF_z1_lr0.0002_dp0.05", "C2_G_STATE"], ["TF_z1_lr0.0002_dp0.05", "C2_G_STATE_LEAD"], ["TF_z1_lr0.0002_dp0.05", "C2_R3_MEAN_DELTA"], ["TF_z1_lr0.0002_dp0.05", "G_FREE"], ["TF_z1_lr0.0002_dp0.05", "G_LEAD"], ["TF_z1_lr0.0002_dp0.05", "G_STATE"], ["TF_z1_lr0.0002_dp0.05", "G_STATE_LEAD"], ["TF_z1_lr0.0002_dp0.05", "R3_MEAN_DELTA"], ["W0", "C2_G_FREE"], ["W0", "C2_G_LEAD"], ["W0", "C2_G_STATE"], ["W0", "C2_G_STATE_LEAD"], ["W0", "C2_R3_MEAN_DELTA"], ["W0", "EXCLUDE20_KEEP"], ["W0", "EXCLUDE20_REFILL"], ["W0", "FULL_ALL"], ["W0", "FULL_INNER"], ["W0", "FULL_MIDDLE"], ["W0", "FULL_OUTER"], ["W0", "G_FREE"], ["W0", "G_LEAD"], ["W0", "G_STATE"], ["W0", "G_STATE_LEAD"], ["W0", "MATCH_INNER"], ["W0", "MATCH_MIDDLE"], ["W0", "MATCH_OUTER"], ["W0", "MATCH_RANDOM"], ["W0", "R3_MEAN_DELTA"]]

不以点估计非劣判定等价，不开启2024。
