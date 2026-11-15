# 历史状态可辨识性、聚合尺度与路线停止实验（预注册）

## 目标与边界

主任务始终是56场、总装机6827.01 MW的集群预测。2022训练，2023用于结构选择，2024仅在结构冻结后检查；鉴于2024已在既往实验中被查看，不称其为纯盲测试。禁止根据2023/2024结果删除或挑选场站。

本实验只判断：历史在既有局地历史 baseline 与当前功率锚点之外，能否提供可辨识的 weather-response state。所有 strict 模型满足状态不能在零天气 latent 下产生 correction。

## 关键修正

- baseline 输入与 response-state 输入分离；donor control只错配后者。
- 状态编码器和交互头使用完整 head learning rate，旧天气/基线 backbone 使用0.25倍学习率。
- continuation固定40 epochs、patience 12；所有候选和控制同预算。
- 强样本错配按UTC月份、origin phase和NWP age分层后固定无放回错排；不再用相邻batch roll。
- 历史Pangu固定使用源HDF5前6个节点，并逐样本审计 `issue <= valid <= origin`。
- loss为 `2 × cluster + 1 × station-CF`，等价于整体尺度不变的 `cluster + 0.5 × station-CF`。

## 3-seed screening

Phase A：`B0`、`B-Static`、`ID-only`、`P-NoID`、`P-Static`、`P-Direct`、`G-Static`、`GL-Static`及所有状态单元的matched train-shuffle control。

Phase B：`P12`、`Physical`、`Hierarchical`、`IO-Pair`、`IO-Pair-ID`及matched controls。IO-Pair token为 `[z, Δz, p, Δp]`；IO-Pair-ID以前3节点为support、后3节点为historical query，识别损失固定0.2，未来状态使用全部6节点。

ID-only先独立screen；只有learned identity显示增量时，才运行state+ID，以及带匹配no-state affine control的共享backbone+小型场站affine。禁止56套独立模型。

Screen gate同时要求：相对同identity no-state控制的集群MAE改善至少0.5 MW且2/3 seed获胜；相对matched shuffled-state控制至少改善0.5 MW且2/3为正；RMSE及任一时效段平均恶化不超过3 MW；local claim还要求farm macro capacity-normalized MAE至少2/3获胜。最多晋级两个strict候选，另可晋级一个direct diagnostic。

## 10-seed正式判据

- 同identity no-state配对95% CI完全小于0，至少8/10获胜，并通过Holm校正；
- 对matched train-shuffle control的配对95% CI小于0；
- 强推理错配的按日block-bootstrap 95% CI完全大于0；
- local claim还要求远距离station-state错配同时恶化farm和K=8子集群normalized MAE，至少8/10为正；
- 2024平均同方向；
- strict模型零天气不变误差小于1e-6 CF。

## 尺度与路线决策

同一组预测固定做farm、K=8/4/2静态坐标K-means子集群和全56场聚合，报告MW MAE、capacity-normalized MAE、RampMAE、状态增益、shuffle degradation、误差相关与抵消率。

- strict local/hierarchical通过：继续局地状态路线；
- 只有direct通过：历史是直接预测信息，不能称天气响应调制；
- 只有IO-Pair通过：转向forcing-realization response-state identification；
- 只有aggregate通过：保留global common-mode calibration；
- 只有identity有效：研究场站异质性；
- 全部失败：停止response-state路线；
- 只在farm/subcluster有效：报告尺度效应，不把它包装成集群状态模型。
