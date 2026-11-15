# Aggregate 严格状态—天气调制实验计划

## 要回答的问题

现有 R2-Full 把状态和天气一起送入黑盒修正网络，状态可以绕开天气单独产生修正。因此它只能回答“黑盒融合是否有效”，不能严格回答“状态是否改变天气到功率的响应”。本轮固定已经验证的 R2 天气路线和3 h历史基线，只检验：

> 当前发电状态能否在不单独预测功率的前提下，改善未来天气到集群功率的映射？

## 硬约束

把原 R2-W 天气解码器写成：

\[
D_W(z,\tau)=D_W(0,\tau)+E_W(z,\tau).
\]

状态只能调制天气效应 \(E_W\)，或通过零保持变换调制天气latent。所有结构必须满足：

\[
C(0,s_1,\tau)=C(0,s_2,\tau).
\]

这里不要求 \(C(0,s,\tau)=0\)，因为原解码器可能包含随预测时距变化的偏置。

## 候选矩阵

| 编号 | 状态 | 调制对象 |
|---|---|---|
| B1/B2 | 当前功率 / baseline latent | 整体天气效应强弱 |
| C1/C2 | 当前功率 / baseline latent | 天气latent逐通道强弱 |
| D1/D2 | 当前功率 / baseline latent | rank-4低秩天气latent变换 |
| E1 | baseline latent | 三个rank-4天气响应专家的路由 |
| F1 | baseline latent | 天气效应轨迹的15 min差分 |

当前功率和baseline latent两套adapter在每个模型中同时存在，保证同一算子下参数量一致。状态控制量相对2022训练集参考状态居中；scale/channel/ramp幅度固定0.25，低秩rank固定4，专家数固定3、残差系数固定0.1。

## 训练和选择

1. 加载同seed完整R2-W checkpoint，新模型第0步逐元素等于R2-W。
2. seeds 42、46、50先冻结历史基线、天气编码器和天气解码器，只训练状态adapter与调制器，最多30 epoch、patience 6、学习率2e-4。
3. 前三名进入联合微调：历史基线和天气编码器仍冻结，天气解码器学习率2e-5，调制器2e-4，最多20 epoch、patience 5。
4. 同时训练A0-FT和每个候选的固定参考状态控制Candidate-Const。
5. 至少最佳严格候选扩展到seeds 42–51；最多保留两个正式候选。最终把R2-W、正式严格候选和可部署控制A0-FT一起按2023 validation的10-checkpoint平均MAE选择；任何平均改善都可以替代R2-W，精确相同时选择结构最简单者。
6. 冻结后统一评估2024，不依据2024结果追加结构。

## 必做诊断

- 全数据集固定随机置乱状态；
- 把状态固定为2022训练参考状态；
- 零天气latent下改变状态，correction必须数值不变；
- 与A0-FT、Candidate-Const和原R2-W比较；
- 当前功率五分位×未来10 m风速下降/稳定/上升×0–1/1–3/3–6 h的5×3×3误差矩阵；
- 总体MAE为选择指标，Ramp MAE为正式辅助指标。

产物统一保存在 `artifacts/aggregate_strict_modulation_v1/`，精简结果写入 `exp_docs/aggregate_strict_state_weather_modulation_results.md`。启动命令为：

```bash
bash shell/launch_aggregate_strict_modulation_v1.sh 0,1,2,3,4,5,6,7 auto
```
