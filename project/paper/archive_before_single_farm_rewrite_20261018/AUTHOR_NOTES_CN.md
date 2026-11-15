# 写作与实现核对记录

初稿核对日期：2026-10-16；中文与结果整合：2026-10-17。仅用于作者编辑，不进入编译正文或 arXiv 源码包。

## 当前稿件如何定位

Introduction 先说明已经实现的历史功率与未来 NWP 在预测中的不同角色，再引出显式 power-space reference 与条件修正。贡献集中于信息组织及可检验的表示作用，不把最后 token、乘法、Δb 或时间插值单独列为创新。

Method 描述真实训练完成的 SLAST：`transfer_variant=CBH`、`condition_mode=full`、`summary_mode=last`。保留完整 decoder memory、b/Δb/lead 条件、learned per-lead bias、普通 concat MLP、天气 bypass 和 b+c 输出。

本次不做相关工作综述，也没有凭空引用论文。正式投稿前需要为背景事实和采用的标准架构补充文献、核对方法新颖性，但该工作不混入本次正文。

## 关键实现核对

| 内容 | 已核对的实现事实 |
| --- | --- |
| SLAST summary | `memory[:,-1]`，不是均值或原始最后功率 |
| History memory | 24×64；2 层、4 heads、FFN 128、pre-norm Transformer；无历史内 causal mask |
| Transformer FFN 激活 | PyTorch 默认 ReLU；不要与 condition/adapter/readout 的 GELU 混淆 |
| Weather | 缓存可能含 8 个节点，但实际当前 forward 只使用前 7 个节点；14 channels |
| Temporal alignment | 实际 valid-time offset 线性插值；区间外 endpoint hold |
| Weather projection | point mask 仅保留一个 key，因此有效映射为仿射；底层是因子化线性投影，不宣称 query-conditioned retrieval |
| Correction context | lead/T、sin(2πlead/24)、cos(2πlead/24)、纬度 z-score、经度 z-score、capacity/mean(capacity)、phase/3；共 7 维 |
| Baseline target embedding | 使用另外的 5 项 lead 特征、learned phase embedding 与 static farm embedding，不与 correction context 共享 |
| Condition | 67→64 Linear–GELU–LayerNorm，再加每个 lead 的 learned bias |
| Adapter | 128→64→64，GELU；末层零初始化 |
| Correction head | 64→32→1，GELU；末层零初始化；初始化时最终预测等于 b |
| Baseline | 两路历史 causal TCN；future trajectory conv 为对称 padding 的 temporal residual blocks，不将其称为 causal forced dynamics |
| Baseline output | inverse softplus 的当前功率输入 clamp_min(1e-4)；softplus 仅属于 B0 内部 |
| Training | 当前冻结 baseline checkpoint 的 loss_mode=mae、protocol_lock=canonical_mae_20260914；当前 Stage 2 也是 MAE |
| Stage 2 | BF16、AdamW、lr 2e-4、weight decay 1e-4、cosine、80 epochs 上限、patience 10、梯度范数1；baseline 固定 eval |
| Station objective | 等权 station CF MAE，aggregate loss weight=0；cluster MW 来自容量加权求和 |

没有将旧的 Smooth L1 运行写入当前模型训练说明。参数量没有采用包含未激活遗留模块的注册总数来宣称有效模型大小。

天气的 channel normalization 在正文中作为缓存输入约定。此次没有重查原始 HDF5 中 mean/std 的估计样本范围，因此正文不无依据声称该预处理已通过 training-only 审计；数据章节需要另行核对这一来源。

## 结果解释边界

- SLAST 作为当前结构选择，不能把此前 CBH mean 模型上的 Jacobian/置换诊断改名为 SLAST 结果。
- 额外 summary 有小幅预测收益，但均值池化并非被证明必需，完整 memory 的单独路径必要性也不能由 summary 消融推出。
- b 来自历史，是明确预测参照，不是无天气物理反事实；C 可能同时包含天气依赖和参考校准。
- Δb 是未除以时间步长的邻步差，不是动力学状态或已确认的物理趋势响应。
- 2024 已被开发过程查看，只称跨年复评；2023 也已反复用于开发，不声称整个研究流程盲测。
- 原先 0.0003 实用阈值不能事后改写成原协议通过；当前接受小效应属于后续模型取舍。Introduction 不用该阈值过滤小收益，也不把小收益包装成“不可替代”。

## 资料来源

实验项目根目录：`/storage/liying/new_try/wind_cluster_forced_evolution`。

- `artifacts/p0i_summary_calibration_20260916_v1/configs/SLAST_seed42.yaml`
- `artifacts/p0i_summary_calibration_20260916_v1/snapshot/src/wind_forcing/models/p0i_transfer.py`
- `artifacts/p0i_summary_calibration_20260916_v1/snapshot/src/wind_forcing/models/p0i_legacy.py`
- 同 snapshot 下 `models/forced_evolution.py`、`models/components.py`、`training.py`
- `artifacts/p0i_summary_calibration_20260916_v1/inputs/baselines/seed42.pt` 的配置、loss_mode 与 protocol_lock 元数据
- `artifacts/p0i_summary_calibration_20260916_v1/analysis/REPORT_CN.md`
- `artifacts/p0i_attribution_closure_review_20260916/REPORT_CN.md`
- `artifacts/trajectory_attribution_20260914_v1/analysis/REPORT_CN.md`
- `artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md`

图像来源：`/home/liying/codex/output/imagegen/concept_v11/modules/`。
正文分别使用 overview 和 conditional-correction 两幅独立矢量图，删除只对完整六面板编辑画布有意义的面板跳转标记；不改变模型计算图。

## 2026-10-17 更新

正文已改为中文，并增加实验结果章节。数据来源和原始目录当前不可访问的限制见 RESULTS_PROVENANCE_CN.md。未新增训练或重新计算预测。

主图改用 gpt-image-2.5 请求生成的功能拓扑预览，文件为 figures/architecture_overview.png。此前 v11/v12 矢量图仅保留为历史文件，不再嵌入本版正文。模型生成图保留原 prompt、两轮编辑 prompt 及 SHA256 记录；未用代码补画连接线。

中文编译改为 XeLaTeX + ctex + Fandol。上一版英文稿备份在 archive/english_v1。

## 2026-10-17 框架图布局优化

当前正文主图已替换为 v14。上方面板只展示总体依赖，下方面板展开轨迹条件特征融合；基准预测输出支路走上方，完整历史记忆支路走下方。已逐项视觉核对两条历史路径、两条基准路径、天气直接路径、特征三输入相加和输出两输入相加，没有改变模型定义。

v14 首轮编辑漏掉必要支路，已弃用；最终采用独立生成并核对通过的版本。图片未经代码补画结构或箭头，生成记录见 `../output/imagegen/framework_model_v14/`。替换前的中文 PDF、源码包和检查记录保存在 `archive/chinese_figure_v13/`。

## v15 框架视觉优化

按用户更新后的模型选择，使用 imagegen 技能自带 CLI 请求 `gpt-image-2.5`（edit、1536×1024、quality=high），在 v14 上增强模块分组和语义图标。保留两个面板和既有全部计算依赖；风机、天气网格、编码器与解码器图形、时间和位置图标仅辅助解释，不新增模型操作。逐项视觉核对基准双路径、历史双路径、天气直接路径及两个求和节点的输入数量。

正文使用 v15，图注所描述的计算关系不变。完整 prompt、请求参数与文件校验记录见 `../output/imagegen/framework_model_v15/`。v14 中文稿与源码包保存在 `archive/chinese_figure_v14/`。未重新训练模型或修改实验结果。

## v16 概念框架重绘

当前主图改为单幅概念框架，不再显示上下两面板或逐层操作框。使用 imagegen CLI 请求 gpt-image-2.5 从头生成，再通过两次模型编辑校正标签和连线。以历史证据、冻结参考、未来天气、条件特征调整和最终预测表达方法思想。

图中 History-aware readout 概念性合并未来特征融合、读取完整记忆的解码器和功率读出，不是新增网络层。蓝色历史带与彩色等高线均为抽象示意；不表示注意力权重、置信区间或物理天气变化。保留基准双路径、历史双路径、直接天气路径和独立目标上下文。最终求和仅接收基准和功率修正。

完整生成/编辑 prompt 及概念对应关系位于 `../output/imagegen/framework_concept_v16/`。正文图注已改为概念框架说明，方法公式和实验结果未改动。v15 中文稿和源码包保存在 `archive/chinese_figure_v15/`。

## v17 模块与连线规则化

根据用户要求，将 v16 调整为介于实现图与纯概念图之间的框架：使用对齐的矩形功能模块，内部保留历史曲线、天气切片和抽象特征场；模块间信息流全部采用水平、垂直或直角折线及一致的三角箭头。NWP 图内的小风向矢量属于输入示意，不是模块连线。

使用 imagegen CLI 请求 gpt-image-2.5，两次 edit。首轮出现一条无来源的目标上下文箭头，第二轮已删除，并移除两个抽象特征图内的风向标记。已核对基准双路径、历史双路径、直接天气、独立目标上下文、读出的四类概念输入以及最终两输入求和。模型定义与实验数据不变。

当前图及 prompt 位于 `../output/imagegen/framework_structured_v17/`；v16 中文稿、源码包和检查记录保存在 `archive/chinese_figure_v16/`。

## v18：查看近期会议方法图后重绘

2026-10-18 实际查看 NeurIPS 2025 TiRex 论文第 4 页 Figure 2，以及 ICLR 2026 CoRA、STORM 官方海报中的方法图。参考其白底、细线、小型图示和浅色分组，重排整体框架。详细链接、访问范围和设计取舍见 `../output/imagegen/framework_research_v18/STYLE_REVIEW_CN.md`。

使用 imagegen CLI 请求 gpt-image-2.5；首次以三张方法图裁剪作为风格参考生成新图，第二次将历史摘要改接至参考条件并简化读出文字。没有复制参考论文的模型部件，也未将参考原图嵌入正文。最终图片仅裁去外围白边，未通过代码补画连线或修改文字。历史、参考、天气直连、独立目标上下文和最终相加关系已核对，方法定义与结果不变。

v17 中文稿、源码包和检查记录保存在 `archive/chinese_figure_v17/`。本次只作图形设计参考，不把这些论文当作方法新颖性比较或实验基准。
