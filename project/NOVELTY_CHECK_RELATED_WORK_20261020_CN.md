# 三项贡献的新颖性核查与相关工作（deep-research 结果整理，v2）

> 更新：2026-10-20（v2，合并第二轮）。
>
> 来源与核验流程：
> 1. deep-research 第一轮 `wf_d22ba401-40e`：23 个一手来源 → 107 条论断 → 25 条 3 票对抗核验（9 成立 / 9 被否 / 7 因额度中断未核）。
> 2. 第二轮（同一脚本续跑，实际为新检索；16 个抓取因服务端 503 失败）：7 个来源 → 25 条核验（23 成立 / 2 被否），并产出综合摘要。
> 3. 补充检索代理：中文期刊、经典持续法/NWP 融合与 Kalman 修正、基线原文、“上游轨迹作条件”的最近工作；英文条目以 Crossref 核对，中文条目以 ChinaDOI 解析＋出版方页面核对。
> 4. 补核代理：第一轮未完成的 7 条论断，全部已对照一手来源给出结论。
>
> 标记：**【已核实】**＝3 票对抗核验通过，或补核代理对照一手原文确认；**【元数据已核】**＝Crossref/DOI/出版方页面确认文献存在与书目信息，方法描述来自摘要；**【未核验】**＝仅抽取，引用前须看原文；**【表述被否/需修正】**＝文献真实，但某种解读不成立，下文给出正确用法。

---

## 0. 结论速览

| 贡献 | 风险 | 对新颖性威胁最大的文献 | 可主张的差异点 |
|---|---|---|---|
| 贡献一：历史预测轨迹条件化的气象修正 | **中–高** | Yang M. et al. 2024（Energy 312:133580，两阶段，吉林）；杨茂等 2024（高电压技术，“预测模板＋NWP 修正”，吉林超短期）；ORCA 2026；UniWind 2026；FiLM 2018；Nielsen 1998 / Thordarson 2010 / PRWind 2026（按时效权衡持续法与 NWP） | 由**冻结的纯历史功率模型**给出 24 步参考轨迹 b 与 Δb，并与提前期、历史记忆一起作为**可学习的条件**去组织 NWP 天气表示，输出相对 b 的修正；0–6 h、15 min、单场轨迹 |
| 贡献二：非线性局地气象编码 | **中–高** | PSE-WPF 2026（height-aware 垂直廓线嵌入）；Hardy & Finney 2025（AI 气象模型输出→100 m 风的 CNN 映射）；FuXi-2.0（AI 气象输入优于 HRES）；Higashiyama 2018（网格 NWP 3D-CNN） | Pangu-Weather 气压层输出 × 站点相对的可逆垂直差分柱坐标（含 DEM）× 逐目标时刻对齐 token；以同参数 E2N/E2L 对照证明**非线性激活**的增益 |
| 贡献三：共享训练、逐场评价、跨年验证 | **高（方法）／中（实证协议）** | Montero-Manso & Hyndman 2021；Cohen et al. 2025（74 场共享 iTransformer 后校准）；UniWind；TimeXer；吉林 18 场 ASTGCN 2025 | 同一深度结构下**共享×区域输入 2×2** 量化、56 场、2022/2023/2024 全年跨年、逐场胜率；定位为系统实证，不是新算法 |

---

## 1. 贡献一：历史预测轨迹条件化的气象修正框架

### 1.1 威胁最大的直接相关工作（风电、两阶段/模板修正）

1. **Yang M., Li X., Fan F., Wang B., Su X., Ma C.** Two-stage day-ahead multi-step prediction of wind power considering time-series information interaction. *Energy*, 2024, 312: 133580. DOI 10.1016/j.energy.2024.133580【元数据已核；方法来自摘要，未读全文】
   - 第一阶段用历史功率＋0–24 h NWP 预测次日功率；对一阶段输出做深度分解，提取“可预测分量”，与 24–48 h NWP 一起送入第二阶段。数据来自吉林、内蒙古。
   - 差异：日前尺度；第一阶段已用 NWP（不是纯历史冻结模型）；传递的是分解分量而非原始轨迹 b/Δb＋提前期条件；输出为重新预测而非 b＋残差。
   - **同一课题组、同省数据，审稿人最可能引用，需在相关工作中正面对比。**

2. **杨茂, 王达, 王小海, 范馥麟, 高博, 王勃.** 基于数据物理混合驱动的超短期风电功率预测模型. 高电压技术, 2024, 50(11): 5132–5141. DOI 10.13336/j.1003-6520.hve.20230401【出版方 PDF 已核】
   - BiGRU 残差网络输出作为“预测模板”，双向循环残差网络修正 NWP 风速（多高度 NWP＋实测风速）；风速波动超阈值的时点切换为功率曲线预测并替换模板值。吉林 400.5 MW 风电场，4 h、16 步。
   - 差异：基于阈值的逐点切换/替换，模板不作为可学习网络的条件输入；单场。
   - 同组前身：杨茂, 许传宇, 王凯旋. 基于切换输出机制的超短期风电功率预测. 高电压技术, 2022, 48(2): 420–430. DOI 10.13336/j.1003-6520.hve.20201712【出版方 PDF 已核】——时序外推模型与风速-功率模型按波动门限切换。

3. **Feng et al.** An Ultrashort-Term Wind Power Prediction Method Based on a Switching Output Mechanism. *International Transactions on Electrical Energy Systems*, 2023, 5514460. DOI 10.1155/2023/5514460【已核实 3-0】
   - 按修正后风速的波动程度在 WSPC（功率曲线）与 STAN（时空注意力网络）之间硬切换，结果沿时间轴拼接，修正风速低于切入风速时功率置 0；130.5 MW 单场 4 h NRMSE 0.0907（归一化基准与是否为 0–4 h 平均，摘要未说明）。同类：Yang, Wang & Zhang, *Energy* 289 (2024) 130075, DOI 10.1016/j.energy.2023.130075。
   - 规则切换，非学习型条件修正，重叠风险低，但代表中国超短期“组合/切换”路线。

4. **Jonas S., Winter K., Brodbeck B., Meyer A.** Bias correction of wind power forecasts with SCADA data and continuous learning. *J. Phys.: Conf. Ser.* 2767 (2024) 092061 (TORQUE 2024). DOI 10.1088/1742-6596/2767/9/092061；arXiv:2402.13916【已核实 3-0 / 2-1】
   - 用 GB/NN/LSTM/CNN 修正 65 台机组风电场基于 NWP 的 48 h 功率预报，CNN 使平均 NRMSE 35%→22%；作者称网络结构变化影响很小，应改进“模型流水线”；持续学习收益最大。
   - 差异：基线是未修正的 NWP 功率预报；日前 48 h；不以基线轨迹调制天气特征。

5. **Cohen J., Xu T., Jang Y., Du P., Byon E., Huan X.** Inverted Transformers for Effective Post-Calibration in Multisite Wind Power Forecasting. *Wind Energy*, 2025, 28(10): e70057. DOI 10.1002/we.70057【元数据与摘要已核】
   - 用 iTransformer 预测 NWP 物理预报（ERCOT HSL）的未来误差做后校准，74 个风电场共享。方向相反（被修正的是 NWP 预报），但会被用来质疑“深度后修正”与贡献三。

6. **Jia Z., Shang L., Jia Y.** Ultra-short-term wind power prediction for enhanced reliability considering error distribution characteristics and guided correction. *Energy*, 2026, 360: 141397. DOI 10.1016/j.energy.2026.141397【仅元数据；Crossref/OpenAlex 无摘要，检索亦未找到摘要】
   - 题名含“guided correction”，可能与贡献一接近，**需人工读全文**。

7. 其他“预测＋误差补偿”传统对照：
   - 刘雨佳, 樊艳芳, 白雪岩, 宋雨露, 郝瑞鑫. 基于特征交叉机制和误差补偿的风力发电功率短期预测. 电工技术学报, 2023. DOI 10.19595/j.cnki.1000-6753.tces.220477【期刊页已核，卷期页码未确认】
   - Li Z., Ye L. et al. Short-term wind power prediction based on extreme learning machine with error correction. *Protection and Control of Modern Power Systems*, 2016, 1: 1. DOI 10.1186/s41601-016-0016-y【Crossref 已核】——对 ELM 短期预测误差用持续法外推得到超短期预测（方向相反：NWP 模型为基础、近期误差修正）。
   - Ye L. et al. An ensemble method for short-term wind power prediction considering error correction strategy. *Applied Energy*, 2022, 322: 119475. DOI 10.1016/j.apenergy.2022.119475【Crossref 已核】
   - Liu, Duan, Chen & Wu. *Energy Conversion and Management* 199 (2019) 111975. DOI 10.1016/j.enconman.2019.111975【已核实 3-0】——WPD＋CNN＋自适应多重误差修正，仅作用于风速序列自身残差，无 NWP。
   - Li J. et al. SSRN 7081161（2026-07，预印本；修订版 SSRN 7493966）【表述被否：是先修正 NWP 风速再预测功率（correct-then-forecast），不是 forecast-then-correct】；修订版称 LRE-Net“用近期功率历史调节潜在路由与气象聚合”，与“历史状态组织天气信息”部分重叠。24 步 nRMSE 0.0927。
   - 只修正 NWP 风速、不修正功率轨迹（风险低）：Yang M. et al., *Energy* 2024, 302: 131797（DOI 10.1016/j.energy.2024.131797）；Yang M. et al., *Energy* 2023, 282: 128947（DOI 10.1016/j.energy.2023.128947）。

### 1.2 按提前期融合历史与 NWP（经典与最新）

- **Nielsen T.S., Joensen A., Madsen H., Landberg L., Giebel G.** A new reference for wind power forecasting. *Wind Energy*, 1998, 1(1): 29–34. DOI 10.1002/(SICI)1099-1824(199809)1:1<29::AID-WE10>3.0.CO;2-B【Crossref 已核】——持续法与气候均值按提前期加权的参考模型；“短时效信历史、长时效信天气”的经典出处，**必引**。
- **Nielsen H.A. et al.** Optimal combination of wind power forecasts. *Wind Energy*, 2007, 10(5): 471–482. DOI 10.1002/we.237【Crossref 已核】
- **Thordarson F.Ö., Madsen H., Nielsen H.A., Pinson P.** Conditional weighted combination of wind power forecasts. *Wind Energy*, 2010, 13(8): 751–763. DOI 10.1002/we.395【Crossref 已核】——组合权重以气象条件/提前期为条件，是“条件组织修正”的线性前身。
- **Zhao X. et al.** Physics-Guided Deep Learning for Short-Term Probabilistic Offshore Wind Power Forecasting (PRWind). *Atmosphere*, 2026, 17(10): 915. DOI 10.3390/atmos17100915【Crossref 含摘要已核】——Transformer 内融合持续法、功率曲线、风速三次方律三类基线，权重随提前期自适应＋在线偏差修正；海上 2 场、1 h、23 步。基线只做加权融合，不调制 NWP 编码。
- **Fu W. et al.** Dual-path ultra-short-term wind power forecasting based on NWP and multi-order temporal dynamic gating fusion. *Energy*, 2025, 335: 138327. DOI 10.1016/j.energy.2025.138327【Crossref 已核，摘要来自检索片段】——历史功率路径（xLSTM）与 NWP 路径（XGBoost）并行＋动态门控融合；NWP 路径不以历史预测轨迹为条件。**适合作为贡献一的直接对照方法**。
- **ORCA**：Peduto, Yang, Giezendanner, Tuia & Wang. Observation-Driven Correction of Numerical Weather Prediction for Marine Winds. *JGR: Machine Learning and Computation* 3(4), e2025JH001174, 2026. DOI 10.1029/2025JH001174；arXiv:2512.03606【已核实，两轮共 8 条 3-0】
  - Transformer encoder-decoder 通过 cross-attention 读取近期 {观测, GFS} 对，学习 GFS 10 m 风的加性修正；坐标感知位置编码使单一共享模型服务任意站点。
  - 修正增益随提前期单调递减：1 h 45%、2 h 34.6%、4 h 27.3%、8 h 20.9%、48 h 13%。
  - 角色相反：ORCA 中 NWP 被修正、观测驱动修正；本文中 b 被修正、NWP 被条件组织。消融中显式“残差 GFS 连接”几乎无增益【未核验】→ 本文须明示 matched direct vs correction 对照。
- Kalman 类：Louka P. et al. Improvements in wind speed forecasts for wind power prediction purposes using Kalman filtering. *J. Wind Eng. Ind. Aerodyn.*, 2008, 96(12): 2348–2362. DOI 10.1016/j.jweia.2008.03.013；Cassola F., Burlando M. Wind speed and wind energy forecast through Kalman filtering of NWP model output. *Applied Energy*, 2012, 99: 154–166. DOI 10.1016/j.apenergy.2012.03.054【均 Crossref 已核】
- 综述：Giebel G. et al. *The State-Of-The-Art in Short-Term Prediction of Wind Power: A Literature Overview*, 2nd ed., ANEMOS.plus, 2011. DOI 10.11581/DTU:00000017【DOI 未独立解析】

### 1.3 机制层面的通用先例

- **FiLM**：Perez, Strub, de Vries, Dumoulin & Courville. AAAI 2018. arXiv:1709.07871. DOI 10.1609/aaai.v32i1.11671【已核实 3-0 ×2】——条件向量生成逐特征 scale/shift；原文指出**拼接条件或加条件偏置等价于 γ≡1 的 FiLM**。谱系：Conditional Normalization → FiLM → Hypernetworks（Ha et al. 2016）。
- **UniCA**（arXiv:2506.22039）【已补核，部分确认】：冻结 TSFM＋pre/post-fusion 协变量模块；原文称 TimesFM“uses an auxiliary regressor for residual correction”、ChronosX“injects covariates through linear transformations but limits its application to only point-wise TSFM”，批评点是**难以处理异构协变量**（不是笼统的“线性/加性不足”）；Frozen＋Adapter 优于全量微调仅见于 Chronos-Bolt 的 Fig. 4b。基线含 PatchTST、DeepAR、TFT、TiDE、N-BEATSx、TimeXer。无风电目标与 NWP 网格（EPF 协变量含 wind generation 预测）。
- **ECTO**：Cao Y., Wang J. Exogenous-Conditioned Temporal Operator for Ultra-Short-Term Wind Power Forecasting. arXiv:2605.12196, 2026【预印本】——以外生变量为条件的 MoE 做 gain-bias 校准与分 horizon 修正，3 个风电场、15 min；无独立参考轨迹。
- **SADR-PAEF**：Zhang W., Liu Y., Li H. *Mathematics*, 2026, 14(17): 3076. DOI 10.3390/math14173076【Crossref 含摘要已核】——以持续法为锚＋神经残差专家，限制偏离锚点；不用 NWP。形式上接近 y = b + correction。
- **Park J., Hwang E.** A Two-Stage Multistep-Ahead Electricity Load Forecasting Scheme Based on LightGBM and Attention-BiLSTM. *Sensors*, 2021, 21(22): 7697. DOI 10.3390/s21227697【Crossref 已核】——“上游预测作为下游输入”的通用先例（负荷）。
- **UniWind**（arXiv:2607.01670，预印本）【未核验】——物理功率曲线先验轨迹＋以历史偏差与未来天气为条件的 MoE 修正；日前；ECMWF/GFS 最近格点。

### 1.4 重叠评估：**中–高**

- 已被占据：
  - 基线＋学习修正（ORCA、Jonas、UniWind、SADR-PAEF、大量误差补偿文献）；
  - 两阶段、上游输出作为下游输入（Yang M. 2024 Energy、Park 2021）；
  - 模板＋NWP 修正（杨茂 2024）；
  - 按提前期权衡历史与 NWP（Nielsen 1998、Thordarson 2010、PRWind 2026、Fu 2025 门控）；
  - 条件调制机制（FiLM）、冻结骨干＋适配器（UniCA）、encoder memory＋decoder cross-attention 修正（ORCA）。
- 检索范围内未见：**冻结的纯历史功率模型 → 24 步原始轨迹 b、Δb＋提前期＋历史摘要 → 作为可学习条件组织未来 NWP 表示 → 输出相对 b 的修正**，并用于 0–6 h、15 min、多场共享、逐场评价。Jia et al. 2026 尚待确认。

### 1.5 建议表述

> 提出一种历史参考轨迹条件化的天气修正框架：由冻结的历史功率模型给出未来 24 步参考轨迹及其逐步变化，与历史记忆和提前期共同作为条件组织未来格点天气信息，学习相对参考轨迹的修正量。

### 1.6 应避免的表述

- “首次提出残差/两阶段/误差修正风电预测”。
- “首次实现历史功率与 NWP 随提前期自适应融合”——Nielsen 1998、Thordarson 2010、PRWind、Fu 2025 均可反驳；新意必须落在“**以轨迹为条件组织天气特征**”，而非“短期靠历史、长期靠 NWP”。
- “提出新的条件调制机制”；若 adapter 无乘性/门控交互，不强调“非加性”。
- 把收益归因于 `y = b + correction` 结构本身——以 matched direct/correction 与 CBH/CB/CNEUTRAL 对照说话（2024 MAE：CBH 0.081471 / CNEUTRAL 0.086151 / CB 0.081731）。
- 物理因果、反事实解释。

### 1.7 建议补充的对照/讨论

- 在相关工作中与 Yang M. 2024 Energy、杨茂 2024 高电压技术逐条对比差异（同省数据，审稿人必然注意）。
- 若算力允许，加入 Fu 2025 式“历史路径＋NWP 路径＋门控融合”对照，证明“轨迹条件化”优于“并行门控”。

---

## 2. 贡献二：面向功率预测的非线性局地气象编码

### 2.1 最接近的已有工作

1. **PSE-WPF**：Zheng L., Ma H., Wang B., Xiao S., Ma F., Gao D.W. Spatiotemporal Collaborative Multi-Step Wind Power Forecasting Driven by Physics-Structured Embedding. *IET Renewable Power Generation* 20(1), e70357, 2026. DOI 10.1049/rpg2.70357【已核实，两轮共 5 条 3-0】
   - height-aware vertical-profile embedding 保留相对高度、学习跨高度依赖；批评展平多高度气象信息；方向/距离条件化图注意力（机组级）；级联 TCN-Informer＋power-evolution consistency loss，24/48/96 步；私有多高度 NWP 数据＋SDWPF（SDWPF 无多高度廓线，公开数据仅验证水平空间与时间组件）。
   - 【表述被否】不能说它与本文 C2“是同一论点”：它讲结构保留，不讲非线性 vs 线性。
   - **反向风险**：E2C 是 15 维拼接向量上的 flat MLP（`column_model.py` 中 `nn.Linear(15,64)+GELU`），正是 PSE-WPF 批评的“展平”。C2 不应主张“保留垂直结构”。
   - 其 consistency loss 涉及步间变化，但只是损失项，不是条件化修正的 Δb，对 C1 威胁低。

2. **Hardy L., Finney I.** Leveraging state-of-the-art AI models to forecast wind power generation using deep learning. *Meteorological Applications* 32(2), e70038, 2025. DOI 10.1002/met.70038【已核实，5 条 3-0】
   - 多数 AI 气象模型不输出 100 m 风；CNN（12 年 ERA5 训练）由 AIFS 变量子集诊断 100 m 风，RMSE 0.18 vs 幂律 0.63 m/s。
   - 转换为英国风电功率（功率曲线，未与 SCADA 比对）后，AIFS 驱动的预报**仅在提前期 > 3 天时**优于 IFS；作者指出 AI 模型普遍过度平滑。
   - 核验提醒：CNN 输入变量更多且以 ERA5 自身为真值，增益不能全归于“非线性”；AIFS Single 1.1.0 已可直接输出 100 m 风。
   - 含义：审稿人可能据此质疑在 0–6 h 使用 Pangu 的合理性，本文需以实验自证。

3. **FuXi-2.0 / FuXi-Energy**：Zhong et al. arXiv:2409.07188 (2024)【已核实 3-0】；*Applied Energy* 412 (2026) 127657, DOI 10.1016/j.apenergy.2026.127657
   - 同一功率模型下 FuXi-2.0 输入在 Kelmarsh 与 Yeongheung 两场优于 ECMWF HRES 输入；日前 28–51 h、1 h。
   - 特征为 3×3 网格×前后各 1 h 的 10 m/100 m 风共 162 维、展平进 5 层 ReLU MLP【未核验】——可作为本文编码的对照基线方案。
   - 【表述被否】不能写“FuXi-Energy 普遍优于 HRES”：天气评分以 ERA5（FuXi 训练数据）为参照，对 HRES 不公平；两场、作者自评。

4. **Pangu-Weather**：Bi K. et al. Accurate medium-range global weather forecasting with 3D neural networks. *Nature* 619, 533–538, 2023. DOI 10.1038/s41586-023-06185-3；arXiv:2211.02556
   - 【已核实 3-0】0.25°；3D Earth Specific Transformer；13 个气压层（含 1000、925 hPa）的 Z/Q/T/U/V＋地表 MSLP/U10/V10/T2M；1/3/6/24 h 模型分层时间聚合。→ 本文“1000 hPa−地表”“925−1000 hPa”差分只能对同类量配对（U/V 对 U10/V10、T 对 T2M）；无地表气压、100 m 风、降水；原生步长 ≥ 1 h，需自行插值到 15 min。
   - 【表述被否 ×4】不能写“变量正是风电所需”（为对标 TIGGE/IFS 而选）；不能引用预印本“首次在所有时效（1 h–1 周）全面优于 IFS”（Nature 版删去，1 h 未与 IFS 比较，地表 IFS 分数取自 FourCastNet 图读数）；“U10 仅 0–3 天优于 IFS”是把 RQE 极端分位结论误当 RMSE；训练数据规模等细节以 Nature 版为准。
   - 【未核验】GMD 17, 7915 (2024)：数据驱动模型系统性低估极端风速、随时效模糊化；中国东部沿海站点评估：FengWu 风速 RMSE 最低（1.17 m/s），Pangu 非最优。

5. **Higashiyama, Fujimoto & Hayashi.** Feature Extraction of NWP Data for Wind Power Forecasting Using 3D-Convolutional Neural Networks. *Energy Procedia* 155 (2018) 350–358. DOI 10.1016/j.egypro.2018.11.043【已核实：方法；补核：结果需修正】
   - 3D-CNN 从风场周围 NWP 网格自动提取时空特征——点插值路线的对立方案。
   - 【需修正】摘要只称优于“several benchmark approaches”，未点名；据引用文献（Shin et al., arXiv:2304.01545），对比主要是 3D-CNN vs 2D-CNN。**不能引用为“非线性优于线性/简单基线”的证据**。建议写法：“Higashiyama et al. (2018) 报告 3D-CNN 在单一风电场 NWP 网格上优于若干基准（据引用文献，包括 2D-CNN）。”

6. **TimeXer**（见 §3）【已补核】：气象实验用站点周围 3×3 ERA5 网格（3 h）作为 36 个外生变量预测 3,850 站逐时气温，基线需最近值插值到逐时 → 本文不应声称首个处理“网格天气输入”或“频率不对齐”。

7. 其他：
   - 杨茂, 白玉莹. 基于多位置 NWP 和门控循环单元的风电功率超短期预测. 电力系统自动化, 2021, 45(1): 177–183. DOI 10.7500/AEPS20200521007【DOI 已解析】——随机森林评估多位置 NWP 重要度、加权后与历史功率拼接进 GRU；C2 的基线类做法。
   - Wind-Topo：Dujardin & Lehning, *QJRMS* 148 (2022) 1368–1388, DOI 10.1002/qj.4265【未核验】——DEM 与粗 NWP 的非线性交互学习，降尺度近地面风。
   - WindDragon：Keisler & Le Naour, *Environmental Data Science* 4:e19 (2025), DOI 10.1017/eds.2025.10【表述被否：无点插值基线，不能用于“网格 vs 点”的论证】；NMAE 按平均发电量归一化。
   - Andrade & Bessa 2017, *IEEE TSTE* 8(4):1571–1580, DOI 10.1109/TSTE.2017.2694340（单场网格 vs 点特征）——核验代理建议补读。

### 2.2 重叠评估：**中–高**

- 已被占据：
  - 网格 NWP 深度特征提取（Higashiyama、WindDragon、FuXi-2.0 的 3×3 patch）；
  - 多高度结构嵌入（PSE-WPF）；
  - AI 气象模型输出经非线性映射进入风电链路（Hardy & Finney）；
  - AI 气象模型作风电输入（FuXi-2.0）；
  - 多位置 NWP 加权（杨茂 2021）。
- 相对空白：Pangu-Weather 气压层 × 0–6 h/15 min 单场轨迹 × 站点相对可逆柱坐标（含 DEM）× 逐目标时刻对齐 token，并以 E0/E1/E2L/E2N/E2/E2C/E3 七臂匹配对照归因。

### 2.3 建议表述

> 针对 AI 气象模型（Pangu-Weather）的气压层输出，构建站点相对的可逆垂直差分表示（含地形高程）与逐时刻对齐的非线性局地编码，并在输入、深度和参数量匹配的条件下验证非线性表示的增益及其作用边界。

### 2.4 应避免的表述

- “保留/学习垂直结构”“首次利用风切变/多层信息”（PSE-WPF）。
- “首次将 AI 气象大模型用于风电预测”（FuXi-2.0、Hardy & Finney）。
- “Pangu-Weather 全面优于 IFS”“在 0–6 h 优于 NWP”。
- “点插值优于网格提取”或反之——本文无网格 patch 对照；若要主张，补 3×3 patch（FuXi-2.0 式）或 TimeXer 式变量 token 对照。
- “恢复轮毂高度风/真实垂直廓线”。

---

## 3. 贡献三：共享训练、逐场评价与跨年验证

### 3.1 最接近的已有工作

1. **Montero-Manso P., Hyndman R.J.** Principles and algorithms for forecasting groups of time series: Locality and globality. *International Journal of Forecasting* 37(4) (2021) 1632–1653. DOI 10.1016/j.ijforecast.2021.03.004；arXiv:2008.00444【已核实：命题 1（2-0）；补核确认 §3.4、§5.4、§6.5】
   - 命题 1：局部方法能产生的预测，单一全局函数也能产生，无需序列相似假设 → 共享训练本身不能主张为新颖。
   - §3.4：“the complexity of local algorithms grows with the size of the set … while it remains constant for global methods”；§5.4：“Local models have better average in-sample loss … Local models suffer from over-fitting”——审稿人期待的“共享为何有益”的标准解释。
   - §6.5：“globality and multivariate are kind of orthogonal concepts … one can have a multivariate global model”→ 正好对应本文 2×2（共享 × 区域输入）。但泛化保证“assumes groups of independent time series, under heavy dependence global models loose their beneficial performance guarantees”——吉林邻场强相关，**不能作为形式保证**。
   - 仅用其他序列/协变量而不输入目标序列自身历史时“there is no guarantee”；加入主序列即恢复等价能力 → 支持本文共享天气修正网络以本场历史与 b 为条件。
   - 其实证只比较经典局部统计模型与朴素全局模型，**未比较同一深度结构＋NWP 输入下的逐序列 vs 共享训练**——本文实证价值的落点。

2. **Cohen et al. 2025**, *Wind Energy* 28(10): e70057（见 §1.1）——74 场共享 iTransformer 后校准，与“多场共享深度模型”直接重叠。

3. **UniWind**（arXiv:2607.01670）【未核验】——单一共享 checkpoint 跨 24 场、逐场容量归一化评价、跨场/跨区零样本；无跨年测试，日前。

4. **TimeXer**：Wang Y., Wu H., Dong J., Qin G., Zhang H., Liu Y., Qiu Y., Wang J., Long M. *NeurIPS 37*, 2024: 469–498. DOI 10.52202/079017-0015；arXiv:2402.19072【已补核，部分确认】
   - 每条外生序列（整个回看窗口）经线性投影成一个 variate token；内生序列的 learnable global token 与 patch tokens 自注意力，再以 Exogenous-to-Endogenous Cross-Attention 读取外生 token。通道独立、共享层 → “共享模型＋其他序列作外生输入”的通用先例。
   - 形式设定只用回看期外生值；预测期外生值只在附录 H.1 做定性展示（无数值表）→ 作基线时须说明如何喂入 Pangu 未来预报。
   - EPF（Table 2，168→24）平均 MSE/MAE：TimeXer 0.307/0.265、PatchTST 0.330/0.282、iTransformer 0.335/0.289、TiDE 0.412/0.338（RLinear 0.412/0.339，勿混淆）。
   - 【需修正】缺失值实验（§4.3.1，Table 5）：作者称外生置零/随机替换“no significant decline”，但平均 MSE 由 0.307 升至 0.330/0.333（约 +8%），恰与 PatchTST 持平——可用作“通用外生 Transformer 对未来协变量利用有限”的动机，但须如实引用数字。

5. **吉林 18 场自适应时空图卷积**：*Renewable Energy* 249 (2025) 123321（S0960148125009838）【未核验；作者未抽取】——单一多任务模型同步输出所有风场多步功率，自适应图结构；4 h NRMSE 0.1161（持续法约 0.1334）。

6. **Li M., Yang M., Yu Y.** Short-term regional wind power forecast method based on hierarchical graph neural network and extreme value theory. *ECM* 341 (2025) 120002. DOI 10.1016/j.enconman.2025.120002【未核验】——区域总功率；邻场时空依赖显著提升精度 → “区域历史输入有益”不是新发现。

7. 中文集群/漂移：
   - 杨茂, 彭天, 苏欣. 基于预测信息二维坐标动态划分的风电集群功率超短期预测. 中国电机工程学报, 2022, 42(24): 8854–8863. DOI 10.13334/j.0258-8013.pcsee.212388【DOI 已解析，方法未核】
   - Yang M., Guo Y. et al. Error Correction Method of Ultra-Short-Term Prediction Based on Load Peak-Valley Characteristics for Wind Farm Cluster. *CSEE JPES*, 2026, 12(1). DOI 10.17775/cseejpes.2022.06180【Crossref 有记录，作者字段缺失】——东北三省集群超短期误差修正。
   - 潘春阳, 文书礼, 朱淼, 等. 基于概念漂移监测与增量更新机制的超短期风电功率在线预测. 中国电机工程学报, 2025, 45(6): 2133–2145. DOI 10.13334/j.0258-8013.pcsee.231834【DOI 已解析】——与跨年分布漂移相关。
   - 崔杨等. 基于多元注意力框架与引导式监督学习的闭环风电功率超短期预测策略. 中国电机工程学报, 2023, 43(4): 1334–1346. DOI 10.13334/j.0258-8013.pcsee.220912【DOI 已解析】——模型引导训练过程，与 C1 弱相关。

8. **SDWPF**：Zhou J. et al. arXiv:2208.04360；*Scientific Data* 11:649 (2024)，https://www.nature.com/articles/s41597-024-03427-5【已补核，部分确认】
   - 单风场、134 台 SL1500/82（1.5 MW）、10 min SCADA；KDD Cup 2022 为 48 h（288 步）、仅历史信息；完整版 2020-01 至 2021-12 含机组 ASTER GDEM v3 高程（仅 SciData 版）。
   - 划分：arXiv 版为 245 天中前 214 天训练、余 31 天验证；214/16/15 天划分出自 SciData Supplement §C。两版均为单次时间顺序划分 → 本文全年跨年协议更严格，但“在后续时段测试”本身不新。
   - 基线：GRU、LightGBM、N-BEATS、DeepAR、Informer、Autoformer、Reformer、SARIMAX（Table S2），不含 PatchTST/iTransformer/TimeXer/TiDE。
   - 天气消融：加入未来 48 h **ERA5（oracle，非 NWP）**天气后 Informer 48 h RMSE 49.368→19.828、MAE 31.862→13.268（主文 Fig. 2）；Table S5 中第一个 6 h 区间 MAE 仅 19.756→14.800，其余区间降约 20 点（注意 Table S5 无天气基线为 47.836/31.766，与主文数值不一致）。→ 0–6 h 历史信息占主导，间接支持“历史基线锚定＋天气修正”。

### 3.2 审稿人期待的基线（原文书目已核）

| 模型 | 引用 | 状态 |
|---|---|---|
| TiDE | Das A., Kong W., Leach A., Mathur S.K., Sen R., Yu R. Long-term Forecasting with TiDE: Time-series Dense Encoder. *TMLR*, 2023. https://openreview.net/forum?id=pCbC3aQB5W | TMLR bib 已核 |
| N-BEATSx | Olivares K.G., Challu C., Marcjasz G., Weron R., Dubrawski A. *IJF* 39(2): 884–900, 2023. DOI 10.1016/j.ijforecast.2022.03.001 | Crossref 已核 |
| PatchTST | Nie Y., Nguyen N.H., Sinthong P., Kalagnanam J. A Time Series is Worth 64 Words. *ICLR 2023*. arXiv:2211.14730 | arXiv 已核 |
| iTransformer | Liu Y., Hu T., Zhang H., Wu H., Wang S., Ma L., Long M. *ICLR 2024* | 论文集页面已核 |
| TimeXer | Wang Y. et al. *NeurIPS 37*: 469–498, 2024. DOI 10.52202/079017-0015 | Crossref 已核 |
| TFT | Lim B., Arık S.Ö., Loeff N., Pfister T. *IJF* 37(4): 1748–1764, 2021. DOI 10.1016/j.ijforecast.2021.03.012 | Crossref 已核 |
| DLinear | Zeng A., Chen M., Zhang L., Xu Q. Are Transformers Effective for Time Series Forecasting? *AAAI* 37(9): 11121–11128, 2023. DOI 10.1609/aaai.v37i9.26317 | Crossref 已核 |
| GEFCom2014 | Hong T., Pinson P., Fan S., Zareipour H., Troccoli A., Hyndman R.J. *IJF* 32(3): 896–913, 2016. DOI 10.1016/j.ijforecast.2016.02.001 | Crossref 已核 |

UniCA（2025/2026）的基线集合同样为 PatchTST、DeepAR、TFT、TiDE、N-BEATSx、TimeXer【已补核】，与项目已有 15 类共享基线基本一致；TimeXer 若作基线须说明未来 NWP 的输入方式。

### 3.3 重叠评估：**高（方法）／中（实证协议）**

- 已被占据：共享/全局训练（Montero-Manso & Hyndman）；多场共享深度模型（Cohen 2025、UniWind、吉林 18 场 GNN）；邻场信息有益（Li et al. 2025、TimeXer 通用机制）。
- 相对空白：同一深度结构＋NWP 输入下的**共享 vs 独立 × 区域 vs 本站历史 2×2**；56 场、0–6 h/15 min、2022/2023/2024 全年跨年；对 15 类共享基线的逐场胜率（E2C 对共享 Transformer 50/56 站；SL 对 IL 54/56 站）。

### 3.4 建议表述

> 在 56 个风场的跨年协议下，通过训练共享与区域历史输入的 2×2 对照，量化共享训练与区域信息对单风场预测的各自贡献，并在逐场、分时效层面检验模型的适用范围。

### 3.5 应避免的表述

- “提出共享训练/全局模型”“首次多场站联合建模”“首次利用邻场信息”。
- 把 2024 称为“盲测/独立验证”（2023 已用于选择；真正独立需 2025 数据）。
- 把 Montero-Manso & Hyndman 的泛化界当作本文的理论保证。
- 在输入未对齐时宣称优于 TimeXer/TiDE 等（项目 P0：同区域输入强 Transformer 对照尚未完成）。

---

## 4. 精度背景（容量归一化为主，供引言/讨论参考）

| 来源 | 场景 | 指标 | 数值 | 状态 / 可比性 |
|---|---|---|---|---|
| Renewable Energy 249 (2025) 123321 | 吉林 18 场集群，4 h | NRMSE | 0.1161（持续法约 0.1334） | 未核验；同省，最接近 |
| Feng et al. 2023, ITEES | 中国 130.5 MW 单场，4 h | NRMSE | 0.0907 | 已核实；归一化基准与时段口径未说明 |
| SSRN 7081161（预印本） | 24 步（6 h，15 min） | nRMSE | 0.0927 | 未同行评审 |
| UniWind（arXiv:2607.01670） | 中国 SD_A/AH_A，日前 | NMAE | 约 8.1% / 8.2% | 未核验；日前，松散参照 |
| Jonas et al. 2024 | 单场 65 台，48 h | NRMSE | 35%→22% | 已核实；日前，NWP 修正 |
| WindDragon（EDS 2025） | 法国全国/12 区，1–6 h，小时 | NMAE（按平均发电量） | 全国 7.7% | 口径不同，不可直接比 |

本文 2024 单场 MAE 约 0.081（CF），与上表 NRMSE 指标口径不同；**本轮未找到“中国 0–6 h、15 min、容量归一化 MAE”的系统性典型范围**，引言中宜只给零散参照并注明口径。

---

## 5. 必引文献清单（按贡献）

**贡献一（必引）**：Yang M. et al. 2024 *Energy* 312:133580；杨茂等 2024 高电压技术 50(11)；杨茂等 2022 高电压技术 48(2)；Feng et al. 2023 ITEES；Jonas et al. 2024；Cohen et al. 2025 *Wind Energy*；ORCA 2026；FiLM 2018；Nielsen et al. 1998；Thordarson et al. 2010；Fu et al. 2025 *Energy* 335:138327；PRWind 2026。
**贡献一（建议引）**：UniCA；UniWind；ECTO；SADR-PAEF；Liu et al. 2019 ECM；Li & Ye 2016 PCMP；Ye et al. 2022 Applied Energy；Louka et al. 2008；Cassola & Burlando 2012；Giebel et al. 2011。

**贡献二（必引）**：Pangu-Weather（Nature 2023，仅事实性内容）；PSE-WPF 2026；Hardy & Finney 2025；FuXi-2.0 / FuXi-Energy；Higashiyama et al. 2018（按修正后的表述）。
**贡献二（建议引）**：杨茂、白玉莹 2021 电力系统自动化；Wind-Topo 2022；WindDragon 2025；Andrade & Bessa 2017；TimeXer（网格天气实验）。

**贡献三（必引）**：Montero-Manso & Hyndman 2021；TimeXer 2024；Cohen et al. 2025；吉林 18 场 ASTGCN 2025；Li, Yang & Yu 2025 ECM；SDWPF（arXiv 2022 / SciData 2024）。
**贡献三（基线原文）**：TiDE；N-BEATSx；PatchTST；iTransformer；TFT；DLinear；GEFCom2014（若讨论基准数据集）。

---

## 6. 仍未完成 / 需人工处理

1. **Jia, Shang & Jia 2026, *Energy* 360:141397**（“guided correction”）：Crossref、OpenAlex、网页检索均无摘要，**需人工读全文**，可能是贡献一的近邻。
2. **Yang M. et al. 2024, *Energy* 312:133580**：方法来自摘要，需读全文确认第二阶段输入的具体形式。
3. 杨茂等 2022 中国电机工程学报（集群二维坐标动态划分）方法未核；刘雨佳 2023 卷期页码未确认。
4. 中文期刊中“按预测步长对持续法与 NWP 模型时变加权”的论文未检到，建议在 CNKI 以“持续法 数值天气预报 预测步长 权重”人工检索。
5. 标【未核验】的条目（UniWind、吉林 18 场 ASTGCN、Li et al. 2025 ECM、Wind-Topo、GMD 2024、中国东部站点评估）引用前须打开原文。
