# 最终状态

- 正式原结构：交接文档确定的完整 CBH；
- 原 CBH 直接评测：10/10 seeds；
- 适配家族：LINEAR、DET、I-CFM、VP-CFM、Sine-SI、VP-epsilon、VP-v、EDM、MeanFlow；
- 正式适配结果：900/900 station×seed；
- 站点：10；种子：42–51；
- 爬坡筛选/损失/指标：未使用；
- 真实 15 分钟天气：不存在，也未被当成监督；
- 主评价：未来 24 个 15 分钟功率点的 station CF MAE；
- 原 CBH 均值±SD：0.082244 ± 0.000549；
- 最佳适配控制 LINEAR：0.086943 ± 0.000684；
- 最佳非 LINEAR 代理 DET：0.086976 ± 0.000746；
- 非 LINEAR 方法相对 LINEAR 的确认性改善：无；
- 适配路线相对原 CBH：全部退化；
- 预测/样本/指标重算、checkpoint、协议、数据与源码哈希审计：通过；
- 小时节点保持与隐藏目标隔离：继承的逐方法审计通过且源码 hash 一致；
- 只用功率 MAE 的 CBH→生成器梯度审计：八种方法全部通过；
- 最终报告：`analysis/FINAL_REPORT_CN.md`；
- 最终审计：`status/FINAL_AUDIT.json`。

这些是固定 2023 report split 的探索结果。原 CBH checkpoint 曾使用完整 2023 validation 选模，不能称为盲测。
