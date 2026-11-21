# 最新结论与阅读口径（2026-11-21）

本次发布新增的研究方向文档、截至 11-21 的诊断报告、14 个诊断脚本和 16 份汇总 JSON，并补齐 11-16 的 v3–v5 计划存档。没有在本次发布中重新训练模型或改变已有实验成绩。

完整阅读目录：[GEMINI_READING_INDEX_20261121_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/GEMINI_READING_INDEX_20261121_CN.md)。远程分支为 `paper-evidence-20261119`；默认 `main` 上的旧报告不能代替该分支的新材料。

## 三套材料分别回答什么

| 材料 | 范围与状态 | 如何使用 |
|---|---|---|
| [已完成主模型实验](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/GEMINI_REVIEW_20261119_CN.md) | 55 站共享模型，0–6 h / 15 min，2022 训练、2023 选轮、2024 跨年复评；补充批次 151/151 完成 | 当前基线、消融、十种子和方法有效性证据 |
| [11-20 研究方向讨论](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/SINGLE_FARM_CROSS_RESOLUTION_RESEARCH_20261120_CN.md) | 文中改为单风场日前问题，提出平流插值与生成式后验等方案；文档本身没有新实验结果 | 研究建议、文献线索；不表示新路线已经实现或通过 |
| [截至 11-21 的诊断](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/REPORT_CN.md) | 56 站，2022–2023，不读 2024；功率评价通常剔除 2023 年 11 月，含小时内、1/3/6 h 和 24 h 子任务 | 检查数据中有什么信息，以及新路线在已检验设置下的可行性；不能直接与 55 站主模型成绩比 |

报告所称“零训练诊断”是未重训主神经网络，其中仍包含在 2022 年拟合的梯度提升树和单调回归。原文“等温曲线”对应代码 `IsotonicRegression`，应理解为单调回归曲线。

## 最新诊断值得先看的结论

- 平流插值的总体与分层结果没有支持预期收益。CLDAS 的细尺度结构在所检验的尺度上并不按风速平移；这限制了该具体“空间换时间”方案的依据，不能外推为所有时空方法都无效。
- 在使用同期 ERA5/CLDAS、甚至已知未来整点功率的诊断设置下，小时内细节仍只有很小的可恢复增益。它们是额外信息／神谕对照，不能作为正常部署的输入成绩。
- 气象侧订正再转换到功率，在 Q5 的有限模型比较中出现正反馈，尤其在功率训练数据减少时。它仍需要与同协议强基线比较，不能据此宣称新主模型已胜出。
- CLDAS 周边区域均值和 1–6 h 信息有价值，但未来分析值不是业务预报；报告发现部分产品约 48 h 延迟，起点可用性需要单独满足。
- 11-20 路线文档中的推荐必须与这些后续负结果一起阅读；小时内功率的时间戳究竟代表瞬时值还是区间平均，最新报告仍标为未核实。文献调查记录与代理投票不替代独立的文献或实验验证。

## 已完成模型的结论保持独立

主模型相对同骨干线性插值，在 2023/2024 十种子比较中通过原门槛；相对强 Transformer、部分内部机制必要性仍未全部通过。当前实现为确定性低秩加性特征修正，不能称为已验证的生成式天气超分辨率或局地气象重建。新诊断没有替代或重新计算该组实验。2024 历史上曾被查看，仍称跨年复评，不称盲测。

## 数据和可复核范围

功率、Pangu 与 ERA5 服务器位置见 [SOURCE_LOCATIONS.json](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/data/SOURCE_LOCATIONS.json)；CLDAS 来源为 `/shared_data_hdd1/All_Datasets/CLDAS_2.0`，读取逻辑见 [extract_cldas.py](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/project/runs/alignment_error_diagnostic_20261120/code/extract_cldas.py)。原始数据、裁剪数组、训练权重与逐样本预测保留在服务器，本仓库提供来源位置、文本源码和汇总指标。

新诊断原始 `Q9_cldas_intrahour.json` 中 `T2.catmull_rom_from_power.R2` 是 `NaN`，不是有效成绩。原文件按字节保留；[严格 JSON 副本](https://github.com/LYingyx/wind-power-forecasting/blob/paper-evidence-20261119/docs/ALIGNMENT_DIAGNOSTIC_RESULTS_20261121.json)将这一项记为 `null` 并标注来源。它不影响正文报告的其他有效对照。`Q8c_cldas_product_by_month.json` 的单独生成命令未随本批脚本保存，作为结果存档发布，不宣称其执行链已完整复现。

本次检查验证发布内容、哈希、JSON 转换和原有结果表计算；不等于重新运行这些诊断或独立核验研究路线中的全部文献。
