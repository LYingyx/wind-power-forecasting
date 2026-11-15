> **2026-11-15 用户最新路线约束（优先）**：不把天气编码作为机制贡献点或独立贡献点；E2L/E2C仅作实现选择与消融。现有7种编码在本框架内的对照已完成，但TF＋同一E2C/E2L的交叉对照未做，不能将前端差异直接归因于骨干或核心机制。先读 [编码角色澄清](ENCODING_ROLE_CLARIFICATION_20261115_CN.md)。已完成2571项结果不变，本次未启动新增训练。

> **2026-11-15 19:41 全部重训完成与路线审查（当前入口）**：2571/2571项、失败0，2571份选中权重＋2200份E1/E6权重和25份B0预测核验通过。共享、微调、全部消融及55站独立训练均完成。原W0独立R3对TF的MAE−0.501%、RMSE＋0.239%；E2L共享对TF为−0.457%/−0.216%，但高功率RMSE＋0.153%，尚非全面胜出。先读 [完整分析与路线判断](DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md)、[逐风场逐seed明细](analysis/daily_protocol_review_20261115/FARM_AND_SEED_DETAILS_CN.md)、[完成审计](analysis/daily_protocol_review_20261115/COMPLETION_AUDIT.json)。三机本轮进程已退出；下方排队和ETA记录均为历史，不要重启已完成队列。2024复评/LM模型/未见风场迁移不在已完成范围。

> **2026-11-15 14:53 重训ETA更新（当前入口）**：已完成2178/2571，剩余393项全部为逐风场独立训练；共享180项、逐站微调1100项及补充消融已完成，失败0。按本轮独立训练实测吞吐，预计还需约2.9小时，预留2–4小时（今天17:00–19:00左右）。见 [最新ETA](RETRAIN_ETA_20261115_CN.md) 和 [机器可读快照](RETRAIN_ETA_20261115.json)。下方凌晨估算及调度修复是历史记录，不能作为当前剩余时间。

> **2026-11-15 01:42 ETA核查与停滞恢复（当前入口）**：已完成180/2571，共享170/180；本次将占满的6002上34项暂停任务核验检查点后迁到6001，实际恢复22个训练进程，失败0。新并行情景预计共享剩余约2–4小时、全部约14–30小时；资源收缩和较晚早停时约2–3天，单站耗时待实测。先读 [最新ETA与修复证据](RETRAIN_ETA_20261115_CN.md) 和 [机器可读快照](RETRAIN_ETA_20261115.json)。下方11月14日串行ETA和旧分池为历史，不能直接沿用。

> **2026-11-14 22:30 并行扩容（最新调度）**：补充消融已提前开始，与共享训练按各任务调参/OOF依赖并行；三机同卡并发改为稳健吞吐窗口逐级扩容。旧串行ETA已标过时。详见 [资源报告](operations/resource_expansion_20261114/REPORT_CN.md) 与 [实际并发/逐卡降速](operations/resource_expansion_20261114/LIVE.json)。模型、数据、每个任务的种子/预算均未改。

> **2026-11-14 21:38 重训ETA与消融补齐（最新入口）**：当前三机首批180项持续运行；已补充1291项天气编码、区域/物理输入、历史×天气和55站独立训练，连同原1100项微调共2571项。见 [完整清单与ETA](RETRAIN_ETA_20261114_CN.md) 和 [新增队列状态](runs/daily_ablation_restart_20261114_v1/operations/QUEUE_STATUS.json)。预计首批剩余约6–13小时，全部训练约20–38小时，取决于早停与可用算力；2024复评和条件性LM模型不在此ETA内。新队列已通过CPU/恢复/隔离检查并由后台等待前序完成自动部署。

## 2026-11-14：每日起报协议切换（当前优先任务）

> **2026-11-14 21:07 当前任务入口**：新协议 36,876 帧已完成裁剪/合并、通过独立审计并激活；固定 55 站与原实验起点保留。两台 V100 和本机 4090 正在重跑首批 180 项，之后自动接续 1100 项逐站微调（含统一 E1/E6 与 Z1）。全部模型和指标重新产生。优先阅读 [本轮入口](runs/daily_protocol_restart_20261114_v1/README_CN.md) 与 [启动核验](runs/daily_protocol_restart_20261114_v1/operations/LAUNCH_VERIFIED.json)。下方等待旧挂载/旧队列的记录均为历史；实际完成数看本轮 STATUS。

用户已授权按 `la` 手稿切换为每天 00 UTC Pangu、起报后 15 h 可用（北京时间 23:00 切换），保持 0–6 h / 15 min 任务，邻站及区域功率可用至起点；关键实验全部重训，旧结果不作为新协议成绩，可覆盖，无须额外备份。先读 [协议与实施进展](MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md) 和 [重训执行入口](runs/daily_protocol_restart_20261114_v1/README_CN.md)。

早期的原始路径不可见问题已解决：从 `/shared_data_hdd1/All_Datasets` 找到原始Pangu/ERA5，新天气已完成合并、审计及激活，训练正在运行。下方旧协议结果仅解释代码来历，不能重标为新协议成绩。

### 23:55 导出修复并恢复

首批34个任务已训练/进入训练，旧结果写出把`data_fingerprint`从spec和显式关键字重复传入导致TypeError。训练数学未改，latest/选中epoch完整保留；错误归档为`FAILURE_EXPORT_INITIAL.json`。删除重复关键字、用包含正式fingerprint的spec重跑完整恢复检查后通过；初始freeze保存在`operations/SOURCE_FREEZE_INITIAL.json`，当前freeze重新部署并核对172文件。34个已开始任务保留原远端归属继续恢复，不能清除REMOTE_RESERVATION或另起重复训练。三机新版启动记录为`operations/LAUNCH_1.json`。

## 2026-10-27 23:45：业务协议实验重启（新任务优先）

- 当前实验：`runs/operational_restart_20261027_v1`。新数据bundle fingerprint `1052d8456c9b950e91ad2a15813c6ceffc541fe7ddba7450c5abd7cc892eef80`。
- 14配置×56站×2种子=1568次P0/年龄训练，持续法已计算；正向配置及配套对照按冻结规则自动补到10种子，**不自动打开2024评价**。
- P1诊断：误差持续相关中位约0.449，但本站误差在历史/NWP/年龄之外MAE+0.0405%、RMSE+0.0782%；区域误差仍劣于原始三圈MAE+0.512%、RMSE+0.312%。未过门槛，暂不启动P1 GPU任务。完整证据位于新run `analysis/P1_GATE.json` 和 `P1_DIAGNOSTIC.json`。
- 新run独立保存代码、训练归一化统计、同一期预报历史映射、配对错配表。旧checkpoint不复用。时区/时延/30小时覆盖已审计。
- 本轮2×2保留当前实际功率锚点，固定历史与STATE、固定天气与天气派生SEL均一致处理；解释范围是当前功率条件下历史上下文×天气。
- 旧`regional_reliability_10seed_20261027_v1`协调器已经排空已领取队列并退出；两台V100旧调度器已停止，本机旧bootstrap/finalizer已停止。已完成结果保留，不要自行重启旧协议。
- 只允许本机4090、v1006001、v1006002；V100可自由使用，本机按其他用户算力/显存余量等待；上限12进程/卡但实际以epoch效率和资源决定。
- 管理：`operations/manage.py start`，状态`STATUS.json`；自动回收/10seed扩展/报告由`operations/finalize.py`负责，完成需`operations/COMPLETE_AUDIT.json`。

# 单风场功率预测：框架、全部结果入口、代码地图与贡献计划

更新日期：2026-10-27。适用对象：接手项目的编程助手、实验执行者及论文撰写者。

> **2026-10-27 23:20：业务可用 Pangu 数据重建与独立审计完成（最高优先级数据协议）**：用户要求北京时间08/20起报、12小时后可用，功率北京时间、Pangu文件名UTC。源目录为`/hj102_data_large/ERA5_forecasting/Pangu/Forecast_data`；按UTC00/12选择最新可用起报，同轮所有天气，年龄12–23.75h，6h预测需+30h端点。测试明确剔除2024年11月及受影响边界，最终train/val/test=31190/32550/27982，test排除800。见[数据说明](OPERATIONAL_PANGU_DATA_20261027_CN.md)、[当前入口](data/ACTIVE_DATASET.json)、[业务配置](configs/operational_pangu_08_20_lag12h.json)。实体`/shared_data_ssd3/liying/operational_pangu_20261027_v1`；cache链接`data/model_cache_operational_08_20_lag12h_v1`。23:18构建审计、23:20独立审计通过：49179裁剪哈希、18原文件复读、HDF5功率值与时间、730全年切换点、BASE/SEL加载及旧统计拒绝检查；11项测试通过。天气归一化只用16626个train引用帧；56站动力计算与原函数差1.2e−7。构建进程已正常退出，不要重复启动。新单站入口`wind_forcing.data.operational_farm.OperationalFarmData`；物理、局地天气与起报参照均已重建。旧HDF5天气/派生缓存及冻结实验保持旧协议；旧十种子队列未切换，用户未明确停止，不能将其结果重标为业务成绩。下一轮业务训练须新建实验、读取ACTIVE_DATASET并重训，旧残差/周期差/错配索引不得直接沿用。本次按授权重组2024测试数据，没有模型测试评价。

> **2026-10-27 晚：两种子可靠性筛选完成，十种子恢复运行（最新入口）**：见[结果复核](REGIONAL_RELIABILITY_RESULTS_20261027_CN.md)。筛选4256/4256、4968选权重核验于21:21通过；独立复核4256份原始指标、17024行总体/时效表，904组比较，596行与自动表交叉核验一致。G_LEAD/G_STATE_LEAD及均值＋变化量在两骨干均通过点估计筛选，但高功率对N0区间仍跨零；门控相对RAW总体MAE小幅退化，状态门控无明确额外价值。彻底删除近圈已覆盖N3及摘要两路径，整体约不变、高功率略改善，尚未通过总体非劣区间；41共同支持站中远圈匹配摘要有增量，增加汇总站数也有增量，不能写成因果传播。十种子队列regional_reliability_10seed_20261027_v1于21:21自动激活，因缺少中央JOBS父目录一直未分发；21:36调用原settings.setup()恢复，未改冻结代码/清单、未重启重复任务。21:45回传135/17024、失败0，6001/6002实际19/16工作进程，4090仍监测空余。见[实时STATUS](runs/regional_reliability_10seed_20261027_v1/STATUS.json)、[恢复审计](runs/regional_reliability_10seed_20261027_v1/operations/RECOVERY_RUNNING_AUDIT_20261027.json)。扩展13440新训练+3584重放，另复用4256筛选结果，目标21280=38×56×10；不要把两种子完成当成十种子完成，也不要重开筛选。2024保持关闭，最终模型/TF*的H×W尚未做。未来启用新轮队列必须在分发前调用该轮settings.setup()；本轮存储已补齐，无须重复部署。

> **2026-10-27 17:39 区域可靠性验证已实际启动（最新执行入口）**：用户授权完成删近圈/匹配距离/门控/简化验证。见[执行说明](REGIONAL_RELIABILITY_EXECUTION_20261027_CN.md)、[新轮运行入口](runs/regional_reliability_validation_20261027_v1/README_CN.md)、[实时STATUS](runs/regional_reliability_validation_20261027_v1/STATUS.json)和[启动审计](runs/regional_reliability_validation_20261027_v1/operations/START_AUDIT.json)。4256项=3248新训练+1008旧权重重放，56站×42/43，2022拟合/2023筛选；834文件冻结，两V100六组数值差最大1.49e−7。17:39中央已回传12项、失败0；6001=19进程、6002=20进程，多卡2–3并发，正式训练最高epoch21；4090监测等待外部计算/显存空余。主机监控/中央控制/完成审计PID在LAUNCH_LOCAL.json，勿重复启动。GPU多进程受计算/显存/CPU/cgroup及1.5倍epoch效率回退约束，仅本轮检查点退让。完整两种子审计后按SELECTION_RULES自动成组扩44–51并复用42/43，目标目录regional_reliability_10seed_20261027_v1；现在尚未激活十种子，不能写成完成。门控高功率保护同时相对N0/N3，实际未来下降只作诊断；不打开2024、不改旧冻结候选。

> **2026-10-27 距离圈与高功率机制诊断（最新分析入口）**：见[代码核查、联合诊断及后续设计](REGIONAL_MECHANISM_DIAGNOSTIC_20261027_CN.md)。三圈 <20/20–60/60–150 km，风场数中位 2/8/30；R3_NO_INNER 只删摘要，8近邻通路仍含近站，尚无彻底删<20km的训练对照。十种子高功率RAW对N0在0–1h改善、3–6h退化（W0 RMSE−0.7676%/+0.5818%，TF*−0.7895%/+0.3632%）。新增56站×seed44/45×四既有模型的CPU重放完成，448权重记录哈希和指标复现通过；仅2023诊断、没有新训练/2024访问。两种子W0长时效高功率的净MSE增量约86%来自实际较起报下降>20容量百分点的样本，NWP已报风速下降时仍有退化；区域近期已下降时则改善。优先假说为状态转折/区域信息持续时间与幅度失配；远圈距离与更多风场降噪尚未分离，不能写成已证实的传播或纯天气误差校准。保持原冻结候选；新诊断建议不等于新训练已启动。全部诊断CPU进程完成后退出。

> **2026-10-27 两个机制证据更新（当前论证入口）**：见[最新判断](TWO_MECHANISM_EVIDENCE_UPDATE_20261027_CN.md)。旧W0四格原值复算确认2024天气增益真实/固定历史比值MAE2.008、RMSE1.623，正交互站数54/56与53/56（纠正此前RMSE也写54）；保留真实持续法锚点，只支持预测交互，尚非物理定律。机制二D关键对照和十种子复验已补齐，但2024未做，中/远环有边际价值、近环冗余；不能写三圈必需或纯预报误差校准。新56站ARX中REG_RAW对LOCAL的MAE/RMSE为+0.0405%/+0.6960%，未复现双指标收益，残差也无优势。最终W0/TF*的H×W复验及2024主表仍缺；本次只评估已有结果，未开新任务。

> **2026-10-27 16:17 ARX恢复全部完成（当前状态）**：56/56 ARX成功，最后任务15:18:11结束，区域残差旧轮2968/2968于15:20:07最终审计通过，0未解决失败、0运行/排队。新增[恢复完成审计](runs/regional_residual_single_farm_20261025_v1/operations/arx_recovery_20261027/RECOVERY_COMPLETION_AUDIT.json)独立核验56份系数哈希、26恢复站全部312个特征组×alpha的原精度收敛证明及原科学冻结文件。三机本轮CPU工作/调度/控制进程均退出，不要再重启。D与十种子扩展的完成状态不变；2024和最终H×W析因仍未执行。下方14:38“正在恢复”是历史记录。

> **2026-10-27 14:38 ARX 等价加速恢复（当前运行入口）**：用户要求核查并重启。原15个慢任务确实在两V100的CPU运行（GPU监控不可见），另11项达到迭代上限；不是失效活动标记。已验证完整增广QR压缩与原多任务ElasticNet等价，四组全年s00结果及断点恢复一致、154冻结文件未改；同alpha/l1_ratio/tol下扩大迭代预算。30成功站保留，26未完成任务旧尝试归档后切至cpu_scheduler_qr_v3.py，已实查三机26个不同任务均在求解（本机6、6001=12、6002=8），新增失败0，单线程nice15与原资源保护保留。先读[恢复说明](ARX_RECOVERY_20261027_CN.md)和[实时状态](runs/regional_residual_single_farm_20261025_v1/STATUS.json)。失败归零是恢复尝试重排，不是删除原未收敛证据；旧求解器与全部原始结果保留。无需再启动队列，未开启2024。

> **2026-10-27 D 与十种子扩展全部完成（当前优先入口）**：先读[完成与结果复核](REGIONAL_COMPLETION_AND_10SEED_REVIEW_20261027_CN.md)。D 2800/2800、十种子新增5376/5376，均0失败，分别于02:16/08:07审计通过；复用1344原结果后共12配置×56站×10seed=6720格，仍为2023。W0 R3_RAW相对N3的MAE/RMSE为−0.5109%/−0.3879%，TF*为−0.6025%/−0.5238%；新增8种子复现收益。ALLPOOL未通过D预设非劣上界，继续推荐RAW。高功率W0 RAW相对N3为−0.0109%，但相对N0（SEL）RMSE为+0.2601%，不可写成全面无退化。三机D/扩展进程均已退出，不要重启队列；下方“运行/挂起”和PID均为历史记录，PENDING_STATUS中的activated是历史状态，以完成审计为准。旧ARX在14:18为30/56成功、15运行、11未收敛；最终模型2024主表和H×W析因尚未执行，E仍需完整比较与输入冻结。本次仅核验及写报告，未开新训练或2024。

> **2026-10-27 单卡多进程补位修复（最新调度）**：用户再次明确每卡可多进程。实查 v2 已能跑到3进程，但任务退出会重置60秒整卡校准，短任务空等；现新增操作层 v3，按并发数变化校准，记录已验证安全并发，完成后及时补位，增加更高并发才观察至少20秒/两个稳定轮次。单任务耗时1.5倍回退、GPU/CPU/cgroup保护、仅本项目检查点退让均保留。D新调度本机PID3356776、6001=1806495、6002=3405838；扩展条件守护入口 watch_selection_v3.py，PID3357685。原v2和冻结科学文件保持原样，两机v3文件哈希已核验，补位/换任务不重置/资源否决/超并发回退测试通过。规则同样部署至10种子队列；不要启动旧入口。实时进程数和失败数看STATUS及各机schedulers，不依据这些PID历史数当成当前并发。

> **2026-10-26 10 种子扩展已挂起（最新）**：用户要求有效两种子实验扩展到十种子，已建立 [执行说明](TEN_SEED_EXTENSION_20261026_CN.md) 和条件队列 `regional_observation_10seed_20261026_v1`；先看 [等待状态](runs/regional_observation_10seed_20261026_v1/PENDING_STATUS.json)。D 当前完成 502/2800，失败 0；D 两种子完整审计后，按冻结门槛自动扩展通过的 R3_RAW / R3_ALLPOOL（各骨干分别判断），以及配对 N0/N3 和通过候选的 C2。复用42/43、新增44–51，新增1792–5376次训练；仍只评2023，不能表述为开启2024或新增8种子已训练。两V100已部署743冻结文件及448个新种子初始化权重，条件守护PID3076460。D和扩展均使用带效率反馈的操作层v2（具体hash见各自 operations/OPERATIONAL_OVERLAY_V2.json）；最高每卡12只是上限，实际受GPU/CPU/cgroup及同任务轮次耗时约束；超过1.5倍持续回退，效率回退补位10秒，外部负载避让120秒。三机调度PID本机3075454、6001=1438072、6002=2661346；本机仍等待安全空余。原D完成审计曾因Path索引错误退出，已保留原冻结脚本并用 finalize_when_done_fixed.py 修复，当前PID3030625；原记录中的2993319已失效。12组新种子训练、5组已学权重一致性、断点续训、2240结果的自动激活/分组汇总验证通过。不要启动第二份队列，不要改写旧冻结文件。

> **2026-10-26 21:39 D阶段已实际启动（最新执行入口）**：用户授权评价并执行剩余清单，当前新轮为 [regional_observation_completion_20261026_v1](runs/regional_observation_completion_20261026_v1/README_CN.md)，先读 [修订与E边界](D_REVIEW_AND_EXECUTION_20261026_CN.md)、[实时STATUS](runs/regional_observation_completion_20261026_v1/STATUS.json)、[启动审计](runs/regional_observation_completion_20261026_v1/operations/START_AUDIT.json)。2240新训练+560旧权重重放=2800项；新增同容量本站/统计量匹配控制，并补齐简单单池候选的TF*与错配。R3_RAW旧权重预测逐位一致，真实训练/断点恢复检查通过，213个冻结文件核验，两V100数值差最大1.35e-7。21:39有6001的4–7卡各4进程（16），6002共30进程，已有epoch≥15；11项重放回传通过，失败0；4090监测PID2993318继续等待空余。中央控制PID2993317、最终审计PID2993319；只用这三机，动态计算/显存/cgroup内存准入，保留GPU4GiB或20%、CPU85%准入，其他负载变忙时本轮检查点让出，不动他人进程。旧ARX仍在旧轮，不重复、不算新D失败、不阻塞D。D主候选R3_RAW、简化候选R3_ALLPOOL；双指标对N0/N3至少0.1%、各≥29站、优于自身C2、高功率退化≤0.1%；简单候选还需预设+0.1%非劣上界。D完成自动输出建议与审计；E依赖D结果和比较/析因输入冻结，当前未投递2024任务。不要重复启动D，不要把D源码冻结当作E选型冻结。

> **2026-10-26 20:49 区域残差神经部分复核（最新）**：实际尚未整轮完成。2352项神经训练和560项重放全部成功；ARX为16/56成功、33运行、7未收敛失败（s09/33/37/45/50/52/53，均alpha=0.0003），不是新的OOM。先读[神经结果复核](runs/regional_residual_single_farm_20261025_v1/analysis/NEURAL_REVIEW_20261026_CN.md)、[子集审计](runs/regional_residual_single_farm_20261025_v1/analysis/NEURAL_SUBSET_AUDIT_20261026.json)及[时效曲线](runs/regional_residual_single_farm_20261025_v1/analysis/NEURAL_LEAD_EFFECTS_20261026.png)。4091份选定权重、154项冻结文件及360行独立重算效应通过核验。预设主方案R1未过RMSE与高功率门槛；次要方案R3通过神经部分所有门槛，对N3的MAE/RMSE为−0.3497%/−0.2060%，高功率RMSE+0.0048%；但R3_RAW更好（对N3为−0.5178%/−0.3893%），不支持显式残差必要性。TF* R3对N3为−0.6332%/−0.5329%。保持运行中的ARX和全部失败记录，不重开神经任务，不把16站ARX当成完整56站结论，不覆盖冻结求解器或直接放宽收敛条件。整轮最终审计仍等待全部任务成功；2024未开，尚未作新一轮选型冻结。

> **2026-10-25 22:56 调度内存限额修正（最新）**：6001账号有约94.44GiB硬限额，整机空闲不能代表可分配内存；现已加入cgroup各级上限、共享启动内存预留及CPU/GPU逐项准入。先前11个OOM和18个主动释放内存的本轮新增ARX已归档并按原spec重排，原4090上4个长ARX保留进度，当前未解决失败归零。6001 GPU用gpu_scheduler_limits.py；4090仍等待GPU空余，入口为bootstrap_limits.py；三机CPU调度按实际计算/内存余量运行，本机上限调整为32。查看[恢复记录](runs/regional_residual_single_farm_20261025_v1/operations/RESOURCE_RECOVERY.json)和[调度说明](runs/regional_residual_single_farm_20261025_v1/operations/CPU_SCHEDULING_20261025_CN.md)，不要按22:29的启动上限直接开满进程。

> **2026-10-25 22:29 三机调度升级（最新运行入口）**：按用户要求保留4090 GPU空闲监测，新增独立CPU租约及重放预留槽位，三机并行执行ARX和旧权重重放。原4个ARX已原地接管，未重启；原GPU训练继续运行。新CPU上限6001/6002/4090分别56/20/16，实际按85% CPU准入及主存余量动态调整，只暂停/恢复本轮自己的进程。详见[调度说明](runs/regional_residual_single_farm_20261025_v1/operations/CPU_SCHEDULING_20261025_CN.md)及[CPU状态](runs/regional_residual_single_farm_20261025_v1/operations/CPU_REMOTE_STATUS.json)。科学代码/协议未变，2024保持关闭；不要再启动旧reference_cpu_scheduler.py。

> **2026-10-25 区域残差筛选（最新执行入口）**：用户已授权优化并启动本轮，入口为[方案与边界](REGIONAL_RESIDUAL_REVIEW_AND_EXECUTION_20261025_CN.md)、[运行说明](runs/regional_residual_single_farm_20261025_v1/README_CN.md)、[实时进度](runs/regional_residual_single_farm_20261025_v1/STATUS.json)。共2352次神经训练、560项旧权重诊断、56项CPU稀疏ARX任务。按月OOF功率映射、完整C1/C2及同信息表示对照已实现；154项源码/资产冻结，两台V100兼容检查通过并已实际进入epoch；21:13中央确认77项新训练、6项旧权重诊断完成，失败0，首批权重回传/去重核验通过，实时状态继续更新。本轮2022拟合、2023选择，不开启2024。主机制固定R1，R2–R4为次要候选；主门槛为相对N1/N3双指标≥0.1%、各至少29站改善、优于C1/C2、高功率RMSE对N3退化≤0.1%。残差不是纯NWP误差；同协议8近邻边中位距离31.34km。只用4090、v1006001/6002；不要重开旧队列。是否已进入epoch及完成数以实时状态为准。

> **2026-10-25 邻站结果及上游诊断复核（当前分析入口）**：见[综合复核与修订方案](NEIGHBOR_UPSTREAM_REVIEW_20261025_CN.md)。已完成1344项筛选中，N1对SEL的MAE/RMSE补充等权改善0.2795%/0.2816%，但对区域汇总N3仅0.0211%/0.0009%，不能再写“加总无效、注意力才有效”。收益在1–6h更明显，高功率组仍略退化。旧上游诊断有全量读取后筛日期、非共同样本、UTC年界、只预测编号较大风场及滞后搜索窗等限制；原文件保留。新增[严格train/val缓存线性复核](runs/neighbor_upstream_review_20261025/CORRECTED_DIAGNOSTIC_AUDIT.json)补齐2128有向对、56目标、共同样本，得到35304条记录；完整本站历史下上风向RMSE中位数1/2/3h改善0.1753%/0.1556%/0.1259%，MAE为0.0606%/0.1573%/0.1871%，平均效应仍退化。支持方向性条件关联，尚不证明因果传播。下一轮建议保留区域通路、软方向门控、因果短时权重、邻站NWP和完整对照；本次未启动新神经网络队列或2024。

> **2026-10-25 19:21 SEL邻站筛选全部完成（当前优先入口）**：1344/1344项，失败0；v1006001完成704项训练，v1006002完成528项训练，本机CPU完成112项旧权重诊断，4090 GPU未参与。1874份不同选中权重核验通过，远端重复权重已清除；19:25核查两台V100训练/调度进程均已退出，本机残留等待资源的bootstrap已停止。先读[完成审计](runs/neighbor_observation_sel_single_farm_20261025_v1/COMPLETION_AUDIT.json)、[自动结果表](runs/neighbor_observation_sel_single_farm_20261025_v1/analysis/REPORT_CN.md)及[升阶判定](runs/neighbor_observation_sel_single_farm_20261025_v1/analysis/NEIGHBOR_SELECTION_RECOMMENDATION.json)。尚需完整解释对照结果；本轮未读取2024，B1仍延后，不要重启本轮队列。

> **2026-10-25 18:13 SEL邻站筛选已部署启动（当前优先入口）**：用户要求执行56站×2seed的邻站观测复验。新轮[方案与对照](NEIGHBOR_SEL_REVIEW_AND_EXECUTION_20261025_CN.md)、[运行入口](runs/neighbor_observation_sel_single_farm_20261025_v1/README_CN.md)、[实时进度](runs/neighbor_observation_sel_single_farm_20261025_v1/STATUS.json)。共1232新增训练+112原W0/SEL权重重放，全部无CAL；加入N2_SELF同容量本站控制、同邻站容量加权汇总控制、固定22点历史的延迟重训。两台V100已通过108项冻结文件与GPU兼容检查并开始启动worker；当前4090及本机CPU忙，等待空余，不抢占。原B1冻结文件保留，B1尚未开始并[延后至邻站审查后](runs/physical_representation_single_farm_20261025_v1/operations/B1_DEFERRED_NEIGHBOR_SCREEN.json)。2023筛选不读取2024。主检验固定N1，按用户此次明确要求MAE相对N0改善≥0.1%且优于N2/N3才满足升阶规则，小于门槛的全部实际结果仍报告。112个TF*+SEL缺项包含在新轮中。只用4090、v1006001/6002；旧轮完成事实不变，不重开旧队列。

> **2026-10-25 16:26 本轮归因筛选全部完成（当前优先入口）**：6944/6944项已完成（6272训练+672旧权重诊断），未解决失败0，9461份选定权重核验通过；CPU重放的一项数值差异已用CUDA补齐，未重训或改容差。本轮及两台V100没有遗留训练/调度进程，远端重复权重已清除。先读 [最新结果入口](PHYSICAL_REPRESENTATION_RESULTS_20261025_CN.md)、[归因审查](runs/physical_representation_single_farm_20261025_v1/analysis/ATTRIBUTION_REVIEW_20261025_CN.md)、[逐风场逐seed表](runs/physical_representation_single_farm_20261025_v1/analysis/KEY_MODELS_PER_FARM_SEED_CN.md)。2023选出SEL=DENS+DYN+STAB+STATE，W0相对BASE的MAE/RMSE补充等权效应为−2.3525%/−2.0335%，seed42/43改善站数分别53/55与49/51。主要收益来自显式风速表示；密度耦合增量较小，v³单独不如模长。优化TF为零输出末层、lr2e−4、dropout0.05，显著缩小原始骨干差距；TF*+SEL的112项仍缺，不能宣称已完成同输入优化基线比较。[FROZEN_SELECTION](runs/physical_representation_single_farm_20261025_v1/FROZEN_SELECTION.json)已冻结B1六模型比较，尚需2800训练、3360模型×站点×seed复评；B1及2024尚未启动，B2干预另需冻结。不要重启已完成的A队列。

> **2026-10-25 02:27 新轮归因筛选启动（当前优先入口）**：已实现6272项新增训练及672项旧权重诊断，2022拟合、2023选择；旧3432项完成事实不变。先读 [方案审查与执行](NEXT_ROUND_REVIEW_AND_EXECUTION_20261025_CN.md)，再看 [运行README](runs/physical_representation_single_farm_20261025_v1/README_CN.md) 和 [实时STATUS](runs/physical_representation_single_farm_20261025_v1/STATUS.json)。两台V100通过85项冻结文件与GPU兼容核验并已启动调度；4090按其他任务余量准入，CPU四进程重放参考权重。新增DENS解除耦合、SEL留组/替换、TF零初始化×学习率×dropout完整对照。默认实现与断点恢复检查通过。A源码冻结不等于B选型冻结；当前不访问2024，完成后自动生成归因表、选择建议和审计。不得把2024称为完全未见测试集，项目早期已分析过。不要重复启动同一队列；不访问P60002/P60003。

> **2026-10-25 物理与观测增强筛选已完成（优先）**：3432/3432项于00:58:33完成，失败0；[完成审计](runs/physics_observation_single_farm_20261024_v1/COMPLETION_AUDIT.json)通过，5286份选定权重哈希、逐轮选权重和全部原值表一致。v1006001/v1006002分别回传1406/1129项，当前无遗留训练，远端空闲调度器已停止。接手先读 [结果报告与缩写说明](runs/physics_observation_single_farm_20261024_v1/analysis/SCREENING_RESULTS_20261025_CN.md)：DENS特征组合使W0在seed42/43的51/52个风场MAE改善、50/48个风场RMSE改善（相对各自BASE，56站，MAE训练/MAE选轮，2023全时效）；ALL也有广泛改善。DENS同时含密度、校正风速和风能代理，尚不能把收益单独归给密度。自适应时间位移、曲线约束及传播先验未得到一致优势。保留同输入TF及所有负结果；本轮不含2024复评或结构扩56站。旧完成标志残留active/pending显示已修正，不要依据下方历史排队记录重启训练。

> **2026-10-24 22:57 更新**：用户要求6002直接利用剩余资源，v1006002已通过兼容校验，8张卡均已启动训练并产出epoch进度。该机启用共享算力策略，每20秒按资源逐步增加进程，保留4GiB显存；4090与v1006001继续原调度。策略与记录见 [运营入口](runs/physics_observation_single_farm_20261024_v1/operations/README_CN.md)。

> **2026-10-24 晚机器切换（最新）**：用户停用P60002/P60003，改用当前4090和v1006001/v1006002。两台V100均已免密验证，原两台调度已停止且无遗留训练。新的逐卡资源调度、V100环境和非共享盘防重/回传机制见 [运营入口](runs/physics_observation_single_farm_20261024_v1/operations/README_CN.md)。下方“三机调度”旧名称以此为准；实际进度仍读取STATUS，不按已部署/已排队推断完成。

> **2026-10-24 物理与观测增强实现（最新）**：新任务按“评估七条路径、修订后实现”完成代码、数据核查、零训练诊断和7项优化/恢复验收。阅读 [路线审查与修订](PHYSICS_OBSERVATION_REVIEW_20261024_CN.md) 和 [新实验入口](runs/physics_observation_single_farm_20261024_v1/README_CN.md)。3432项2023筛选已加入本机/P60002/P60003调度，启动时GPU由其他用户作业占用，需以 [实时STATUS](runs/physics_observation_single_farm_20261024_v1/STATUS.json) 确认实际训练数，不能把排队或小样本验收记为正式完成。2023已有预测的逐站逐seed诊断不支持把高功率/晚间当作唯一差距来源；最高稳定度代理分组存在较一致差距，但尚不是因果证据。仅本站输入与邻站观测扩展协议分开；新增训练不读取2024、保留逐seed原值、不设百分比提升门槛。代码使用独立新run，复用旧网格/预测，新增数据存SSD3，完成后不保存新全量预测或重复检查点。

> **2026-10-24 完成状态与证据更新（优先）**：本轮8016次训练和7840次评价已于2026-10-23 21:00:35全部完成，[FINISHED](runs/adaptive_weather_single_farm_20261022_v1/FINISHED.json)与[完成审计](runs/adaptive_weather_single_farm_20261022_v1/COMPLETION_AUDIT.json)均通过。下方10月23日“评价继续排队”的文字是恢复过程记录。当前独立训练ST11相对Transformer的2024 RMSE在352/560个风场×seed配对上更差，不能写成全面优于强基线；详见[RMSE复核入口](RMSE_REVIEW_20261024_CN.md)。P60002/P60003免密连接已验证，两机共享盘与当前机器相同；历史共享训练、MSE物理候选和迁移结果须与当前独立训练分开解释。逐风场逐seed原值及远端盘点从RMSE入口进入，不以跨风场/seed均值替代实测值。

> **2026-10-23 执行状态更新（优先）**：新一轮8016次训练已经全部完成。2024评价在凌晨因SSD2账户配额用尽中断，18:08完成恢复核验：2998次评价有效，4842次继续排队。评价已恢复，最终完成仍须查看FINISHED及审计。评价文件整合到SSD3，原evaluation路径通过链接保持可用；重复预测/置换索引共享存储，注意力无损压缩，成功后删除重复步骤指标和废弃临时文件。详见[存储规则与恢复记录](runs/adaptive_weather_single_farm_20261022_v1/STORAGE_POLICY_20261023_CN.md)，实时数量见下方STATUS。不要把“训练完成”写成“评价和贡献审查完成”。

> **2026-10-22 新一轮受控实验已实现并启动**：用户已授权执行修订后的天气提取/历史状态实验。入口为[实施与执行说明](ADAPTIVE_WEATHER_EXECUTION_20261022_CN.md)，实时进度见[新轮次 STATUS](runs/adaptive_weather_single_farm_20261022_v1/STATUS.json)，完整结构见[运行 README](runs/adaptive_weather_single_farm_20261022_v1/README_CN.md)。计划8016次新增训练、7840次标准/干预评价；先2023冻结选择，再2024复评，主指标全时效MAE。新轮次正在运行，不能把队列任务当作已完成结果。原9500阶段结果继续保留，以下入口仍有效。

> **2026-10-22 结果与统计口径更新（优先阅读）**：本轮9500个独立单风场训练阶段已全部完成，2026-10-21 23:03:42完成审计通过。最新入口为[逐随机种子结果总入口](EXPERIMENT_RESULTS_PER_SEED_20261022_CN.md)，含Excel、56站逐seed表、真实模型结构、场景/时效/逐日明细及设计审查。按用户要求，不以跨seed/风场均值替代实测结果；候选误差严格降低即记改善，无百分比门槛，同时完整保留退化。新表取代旧分析中的均值展示口径，不修改原始指标或旧证据。CB、DIRECT、NEUTRAL_LAST、E0等设计也存在改善；不要预设完整E2C-CBH必须最优。下面10月21日“任务已启动”的描述属于执行历史，当前完成状态以上述审计与新表为准。

> **2026-10-21 优先级最高的路线更新**：当前任务改为 **56 个风场分别独立训练、仅本站输入的 E2C-CBH**。贡献为方法、受控归因分析、逐风场/分时效适用边界；E2C 编码与共享训练均不作为独立贡献。完整审查、修正及 9500 阶段实验矩阵见 [EXPERIMENT_REVIEW_AND_EXECUTION_20261021_CN.md](EXPERIMENT_REVIEW_AND_EXECUTION_20261021_CN.md)，实时执行入口见 [本轮 README](runs/independent_single_farm_20261021_v1/README_CN.md)，进度见 [STATUS.json](runs/independent_single_farm_20261021_v1/STATUS.json)，自动结果见 [REPORT_CN.md](runs/independent_single_farm_20261021_v1/analysis/REPORT_CN.md)。任务已启动，但只能以实时完成标志确认结果，不能把排队当完成。
>
> **下文是 10 月 19 日的历史证据与代码地图**，其中“当前决定/主贡献/待补实验”的旧表述均被上面更新取代；共享成绩不能写成独立训练成绩。完整原版另存为 [PROJECT_HANDOFF_20261019_ARCHIVE_CN.md](PROJECT_HANDOFF_20261019_ARCHIVE_CN.md)。接手模型应先读本次审查与执行 README，再按下文查历史资产。

唯一工作根目录：`/shared_data_ssd2/liying/wind_cluster_project_20261018/project`。本文所有相对文件链接均相对此目录。本文是当前任务与证据的总入口；原始实验、配置、权重和历史报告保持原样。历史文件中的旧路径、运行中状态和研究建议，以其记录日期理解。

**10 月 19 日历史决定（已被取代）：以 E2C-CBH 为主模型，研究“多风场共同训练的已有风场跨年单场功率预测”。新风场迁移退出主任务与优先补实验，不作为贡献点。当前独立实验不以必须获胜为前提。**

已有核心证据：2024 跨年复评中，E2C 的 56 站等权 MAE 为 **0.080703**，共享 Transformer 为 **0.084699**，改善 **4.72%**，50/56 站更好。E2C 使用区域历史，当前 Transformer 基线仅使用本站输入，因而这是完整系统比较；严格同输入对照仍待补齐。本站输入版本 CBH-SL 已有独立支持：相对同为本站输入的共享 Transformer 改善 **3.42%**。

## 导航

- [1. 接手时先读：目标、状态与证据优先级](#start)
- [2. 数据、任务、指标与比较协议](#protocol)
- [3. 当前模型的真实计算路径](#architecture)
- [4. 当前框架的主要结果与消融](#results)
- [5. 逐风场、分时效、场景与案例](#farms)
- [6. 迁移和历史探索：结果保留、路线边界](#history)
- [7. 计划的三个贡献点](#contributions)
- [8. 关键代码地图与修改入口](#code)
- [9. 权重、预测文件与资产可用性](#assets)
- [10. 已验证命令与复现方式](#commands)
- [11. 下一步实验与完成标准](#next)
- [12. 结果解释中的已知问题](#pitfalls)
- [13. 全部 39 个实验/资料目录索引](#experiments)
- [14. 更早历史报告、论文与交付物](#older)
- [15. 交接检查单与本次文档范围](#handoff)

<a id="start"></a>
## 1. 接手时先读：目标、状态与证据优先级

### 1.1 已确定的研究路线

| 项目 | 当前决定 |
|---|---|
| 预测对象 | 单个风场的未来功率轨迹；每站单独评价 |
| 训练方式 | 允许源 56 站共同训练、共享参数 |
| 时间泛化 | 2022 训练、2023 选择、2024 跨年复评；风场集合不变 |
| 完整主模型 | E2C-CBH，含区域历史辅助，历史摘要为均值 |
| 本站输入对照 | CBH-SL；不能把已有 E2C 改名为 local-only E2C |
| 性能基线 | 15 个共享命名家族，以验证集最佳 Transformer 为关键对照，保留 TiDE/N-BEATSx 等 |
| 展示风场 | 六个训练期描述量代表站，完整 56 站结果保留；失败案例单列 |
| 新风场迁移 | 原始结果保留，退出主任务与优先补实验 |
| 完全独立训练 | 已有对照，不是当前首选主线 |
| 论文贡献 | 核心条件修正框架＋非线性天气编码设计＋共享学习实证验证 |

“单风场预测”指输出与评价对象，不自动意味着只能使用一个风场训练，也不自动禁止区域历史输入。用户尚未提出严格本站输入的硬性部署约束。模型用了什么输入必须如实标记。

### 1.2 三个优先入口

1. 本文：目标、贡献、实现与全部证据导航。
2. [2026-10-19 路线与原始指标复核](runs/single_farm_route_audit_20261019/ROUTE_RECOMMENDATION_CN.md)：最新主线、完整系统优势和待补对照。
3. [2026-10-18 预测级重审](runs/single_farm_review_20261018/REVIEW_CN.md)：逐样本复算、场景、选站、条件输入核查和统计权重修正。该文较早的 local-only/迁移优先级已被本次路线决定更新，数值证据继续保留。

出现冲突时：当前用户确定的目标决定工作优先级；具体实验的冻结配置、源码、原始指标和校验记录决定实验事实；新审查对旧报告的解释纠错优先于旧叙述。不要依据旧 `STATUS` 中的“运行中”自行恢复队列。

### 1.3 已完成核查的范围

- 统一索引：39 个实验/资料目录，157,313 条运行/划分记录，83,511 个实验内唯一 run；run 不等于独立风场，也不等于全部权重都可恢复。
- 2026-10-19：74,406 份原始指标 JSON/CSV、469,775 个标量核对，无不一致；无重复 `experiment/run_id/split` 键。最大浮点差约 `9.7e-17`。
- 本轮逐站原始指标与上次预测级复算的最大差约 `2.75e-10`；四组目标方案对齐 5,600 条 H2 记录。
- 2026-10-18：40 份 E0/E2C 预测归档重算，MAE 最大差约 `5.4e-10`；448 行逐站季节结果最大差约 `9.9e-10`。
- 本会话完成的是审查、重汇总和文档，没有新增训练。历史兼容检查不等于在当前 GPU/软件版本上重新跑过全部实验。

核验文件：[指标审计](runs/single_farm_route_audit_20261019/audit.json)、[输入哈希](runs/single_farm_route_audit_20261019/input_hashes.json)、[预测核验](runs/single_farm_review_20261018/prediction_verification.json)、[导入状态](integration/MERGE_STATUS.json)。

<a id="protocol"></a>
## 2. 数据、任务、指标与比较协议

### 2.1 源风场主任务

- 过去 6 小时：24 个 15 分钟历史功率点。
- 未来 6 小时：24 个 15 分钟目标点，lead 为 0.25–6 小时。
- NWP：14 通道小时级格点气象；缓存保留 8 个节点，当前 P0I/CBH 修正分支实际使用前 7 个，按真实有效时间差插值。
- 原始 HDF5：[data/jilin_56farm_pad8.h5](data/jilin_56farm_pad8.h5)；元数据：[data/jilin_56farm_pad8.metadata.json](data/jilin_56farm_pad8.metadata.json)。
- 统一缓存：[data/model_cache_pad5_f8/](data/model_cache_pad5_f8/)，[manifest](data/model_cache_pad5_f8/manifest.json) 保存划分、形状和预处理指纹。
- 训练/验证/跨年有效起报窗口数：**31,190 / 32,550 / 28,782**。这里的“全年”是全年有效缓存，不是全年无缺测；月份覆盖见 [valid_origin_coverage.csv](runs/single_farm_review_20261018/tables/valid_origin_coverage.csv)。
- 源正式比较通常为 seeds **42–51** 共 10 个；matched output 为 42–46 共 5 对。2024 已反复查看，只能称跨年复评。

| 缓存位置或字段 | 含义 |
|---|---|
| `static/station_id.npy`、`station_name.npy` | 风场 ID 与名称；逐站数组顺序以此为准 |
| `static/station_capacity_mw.npy` | 容量；CF 与 MW 转换必须使用对应站容量 |
| `static/nwp_channel_name.npy`、`nwp_mean.npy`、`nwp_std.npy` | 气象通道顺序与训练期标准化统计 |
| `train/val/test/station_history.npy` | 历史 CF，单样本 `[56,24]` |
| `station_future.npy` | 未来 CF 标签，单样本 `[56,24]` |
| `cluster_history.npy` | 区域聚合历史的缓存输入；本站版本须从计算路径移除其信息 |
| `weather.npy` | 按气象周期保存；单周期 `[8,14,24,25]`，不能直接以功率样本索引代替气象周期索引 |
| `weather_cycle_index.npy`、`weather_time_offset_hours.npy` | 样本到气象周期的映射及有效时间偏移 |
| `origin_unix.npy`、`sample_index.npy`、`origin_phase.npy` | 起报时间、样本身份、15 分钟相位；用于预测对齐 |

14 通道顺序：`msl,u10,v10,t2m,z1000,z925,q1000,q925,t1000,t925,u1000,u925,v1000,v925`。预处理缓存中的“不进行空间/时间插值”不意味着模型内部不插值。

### 2.2 三种不同协议，不能直接混排

| 协议 | 训练/选模/评价 | 规模与用途 |
|---|---|---|
| 源 56 站全年 | 2022 / 2023 / 2024 | 当前主线、共享基线、CBH 2×2、天气/条件消融 |
| 源站独立 pilot | 2022 月抽样 / 2023 月抽样 / 未开 2024 | 3,156 / 3,349 origins；15×56×10＝8,400 次独立训练 |
| 新增站适配 | 2023 / 2024 H1 / 2024 H2 | dev10＋final20；7/30/90d 各四季节块，180d/full 各一块；历史扩展 |

源/目标站划分见 [selection_protocol.json](artifacts/cross_farm_shared_specific_20260920_v1/data/selection_protocol.json)。目标训练预算不包含完整 H1 选 checkpoint 的标签，不能解释为严格总标注预算。原目标队列资格筛选曾使用 2024 覆盖率等质量信息，不能称整个站点筛选过程完全未查看 2024。

### 2.3 指标与统计单位

主指标为各站容量归一化 MAE（CF MAE）：每站先对有效 origin 与 24 leads 计算误差，再平均种子，最后对风场等权平均。`0.001 CF` 是额定容量的 **0.1 个百分点**；相对改善另用 `(基线−模型)/基线` 计算。

- 不是先将 56 站功率相加再算 MAE；`cluster_mae_mw` 为另一个次级指标。
- 主表是单模型误差的 seed 均值，不是先平均预测再算的集成指标。
- macro RMSE：每站先开平方、再平均站点和种子；不能替换为 `sqrt(mean station MSE)`。
- macro R²：先逐站计算，再等权平均；不是把所有风场拼接后的 R²。
- 10 月 19 日配对区间：站内先平均种子/季节块，20,000 次风场重采样。它不覆盖全部时间、空间相关和模型选择不确定性。
- E2C/E0 时间敏感性采用共享七日块、seed、farm 的交叉重采样，见 [source_e2c_week_bootstrap.csv](runs/single_farm_review_20261018/tables/source_e2c_week_bootstrap.csv)。不要将每个重叠 15 分钟窗口视作独立样本。

<a id="architecture"></a>
## 3. 当前模型的真实计算路径

### 3.1 完整 E2C-CBH

记本站历史为 H，区域历史为 G，气象为 W，冻结历史模型为 B₀。模型构造历史基准轨迹 b，再输出修正后的单站轨迹：

```text
本站历史 H + 区域历史 G
  → 冻结的 ForcedEvolution history-only 模型 B₀
  → 24 点基准轨迹 b；Δb 首点相对最后观测功率，其余相对前一步 b

本站历史 H + 历史时间位置
  → 修正分支的 Transformer history encoder
  → 完整历史 memory M；额外条件摘要 s = mean(M)

格点 NWP
  → 固定双线性取本站小时天气
  → 使用前 7 个小时节点，按真实时间偏移插值到 24 个目标时刻
  → E2C native_projection：反标准化 → 可逆 15 维柱坐标 → 训练期标准化
  → MLP(15,64,64,64)，两个 GELU
  → P0I point 对齐：每个目标只允许读取对应的一个插值天气 token

[s, b, Δb, lead/6] → condition_projection
天气表示与条件表示 → transfer_adapter
目标时刻/位置/容量/相位 context + 天气 + adapter 输出
  → Transformer decoder，持续读取完整 M
  → readout 得到 correction
  → 输出 y_hat = b + correction，逐风场 CF
```

关键实现事实：

1. **先时间插值，后非线性投影。** 当前框架不是先编码所有小时再学习全时域天气检索。
2. `point` mask 每个目标只保留一个天气 token；不要将其描述成已学得的多小时天气注意力或物理响应核。
3. 历史基准 B₀ 与 correction 内部的 history encoder 是两条不同路径；冻结 B₀ 不等于整个历史分支不训练。
4. 当前摘要是 `memory.mean(1)`；没有“E2C＋SLAST”的已完成匹配实验。
5. `b` 与 `Δb` 来自历史模型预测，不读取未来功率标签；decoder 始终保留完整历史 memory。
6. E2C 直接替换原 `native_projection`，没有旧天气编码器并联回退。其余条件修正主体保留。
7. 配置为 `output_clamp_min: null`；不能把额外裁剪后的分数冒充原始预测分数。
8. 同一批次可输出 `[B,56,24]`；单站评价不要求实现只能一次返回一个风场。完整 E2C 的历史基准仍依赖区域 G。

实际类：[ColumnWeatherCBH / WeatherProjection](artifacts/column_weather_20260928_v1/code/column_model.py) 继承 [P0IConditionTransfer](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_transfer.py)，后者继承 [TemporalAlignedBaseline](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_legacy.py)。

### 3.2 编码器与模型代号

| 代号 | 定义 |
|---|---|
| E0 | 原 14 通道单线性天气投影，FP32 重训控制 |
| E1 | 原 14 通道＋DEM，单线性层 |
| E2L | 原 14 通道，三层 affine、无激活；E2N 的等参数控制 |
| E2N | 原 14 通道，三层 MLP＋GELU |
| E2 / E2raw | 原 14 通道＋DEM，flat MLP；代码、CLI 和 CSV 名称为 `E2` |
| E2C | 15 维场站相对柱坐标，flat MLP；当前完整主模型 |
| E3 | 相同柱坐标，surface/lower/upper 三分组编码 |
| CBH / CB / CH / CNEUTRAL | 条件字段消融；与天气编码 E0/E2C 是不同维度的实验 |
| SC / SL / IC / IL | 共享/独立训练 × 区域/本站历史的 2×2 控制 |
| F0–F4 / T0–T5 | 历史源站个性化 / 新站适配协议；不等于天气编码的 E 系列 |

E2C 的 15 维包含地表量、场站地形高程、1000 hPa 相对地表差与 925 hPa 相对下层差。它是可逆的天气＋高程重新参数化；不等于恢复了轮毂高度风或真实转子垂直廓线。

<a id="results"></a>
## 4. 当前框架的主要结果与消融

### 4.1 全部 15 类共享基线与主模型

以下每个共享家族为 10 seeds、同源站年度有效窗口。H 表示仅历史功率，W 表示仅未来天气，FC 表示历史＋未来天气。命名模型是项目内相应实现，不自动等同于文献官方实现与最佳调参。

| 模型 | 训练 | 输入范围 | 精度 | 2023 MAE ± seed SD | 2024 MAE ± seed SD |
| --- | --- | --- | --- | ---: | ---: |
| Persistence | 无训练 | 本站历史 | — | 0.117542 | 0.112763 |
| GRU_H | 共享 | 本站历史 | BF16 | 0.111366 ± 0.000035 | 0.106572 ± 0.000078 |
| LSTM_H | 共享 | 本站历史 | BF16 | 0.111398 ± 0.000058 | 0.106594 ± 0.000104 |
| PATCHTST_H | 共享 | 本站历史 | BF16 | 0.111493 ± 0.000025 | 0.106675 ± 0.000054 |
| ITRANSFORMER_H | 共享 | 本站历史 | BF16 | 0.111667 ± 0.000032 | 0.106771 ± 0.000034 |
| MODERNTCN_H | 共享 | 本站历史 | BF16 | 0.111295 ± 0.000062 | 0.106458 ± 0.000085 |
| CNN_W | 共享 | 本站天气 | BF16 | 0.122563 ± 0.000627 | 0.115232 ± 0.000379 |
| VIT_W | 共享 | 本站天气 | BF16 | 0.121657 ± 0.000551 | 0.114404 ± 0.000898 |
| SWIN_W | 共享 | 本站天气 | BF16 | 0.121518 ± 0.000529 | 0.114129 ± 0.000749 |
| DLINEAR_FC | 共享 | 本站历史＋天气 | BF16 | 0.112123 ± 0.000140 | 0.108997 ± 0.000187 |
| NBEATSX_FC | 共享 | 本站历史＋天气 | BF16 | 0.089040 ± 0.000248 | 0.085961 ± 0.000360 |
| TCN_FC | 共享 | 本站历史＋天气 | BF16 | 0.090184 ± 0.000228 | 0.086435 ± 0.000316 |
| TFT_FC | 共享 | 本站历史＋天气 | BF16 | 0.089270 ± 0.000452 | 0.085779 ± 0.000561 |
| TIMEXER_FC | 共享 | 本站历史＋天气 | BF16 | 0.089149 ± 0.000444 | 0.085834 ± 0.000506 |
| TIDE_FC | 共享 | 本站历史＋天气 | BF16 | 0.088862 ± 0.000221 | 0.085663 ± 0.000329 |
| TRANSFORMER_FC | 共享 | 本站历史＋天气 | BF16 | 0.087732 ± 0.000364 | 0.084699 ± 0.000506 |
| CBH_SL | 共享 | 本站历史＋天气 | BF16 | 0.084140 ± 0.000179 | 0.081799 ± 0.000261 |
| CBH_IL | 独立 | 本站历史＋天气 | BF16 | 0.088450 ± 0.000129 | 0.086998 ± 0.000111 |
| CBH_SC | 共享 | 本站＋区域历史、本站天气 | BF16 | 0.083814 ± 0.000303 | 0.081507 ± 0.000318 |
| CBH_IC | 独立 | 本站＋区域历史、本站天气 | BF16 | 0.088445 ± 0.000101 | 0.087228 ± 0.000116 |
| E0 | 共享 | 本站＋区域历史、本站天气 | FP32 | 0.083827 ± 0.000220 | 0.081528 ± 0.000242 |
| E2C | 共享 | 本站＋区域历史、本站天气 | FP32 | 0.082667 ± 0.000129 | 0.080703 ± 0.000126 |

来源：[source_model_summary.csv](runs/single_farm_route_audit_20261019/source_model_summary.csv)、[主表及 RMSE/R²](runs/single_farm_review_20261018/tables/paper_source_main.csv)、[全部 15 基线相对改善/区间](runs/single_farm_route_audit_20261019/source_advantage_vs_all15.csv)。完整逐站：[source_all56.csv](runs/single_farm_route_audit_20261019/source_all56.csv)。

完整 E2C 在 2024 相对 Transformer、TiDE、TimeXer、TFT、N-BEATSx 的 MAE 改善分别为 **4.72%、5.79%、5.98%、5.92%、6.12%**。这些是完整系统差异，E2C 额外用了区域历史，且 FP32 与历史基线 BF16 不同。

新增稳健性检查：每站只依据 2023 验证集从 15 类中选最强家族，得到 53 站 Transformer、2 站 N-BEATSx、1 站 TCN；其 2024 MAE 为 **0.084705**。E2C 仍改善 **4.73%，50/56 站胜出**，站配对差 −0.004002，95% 风场区间 [−0.005022,−0.002861]。这是回顾性验证集选择分析，不冒充预注册主实验。见 [逐站选择结果](runs/single_farm_route_audit_20261019/per_farm_validation_selected_baseline.csv)。

| 2024 macro 指标 | Transformer | E2C |
|---|---:|---:|
| MAE | 0.084699 | 0.080703 |
| RMSE | 0.133015 | 0.127069 |
| R² | 0.661216 | 0.689828 |

来源：[source_mae_rmse_r2.csv](runs/single_farm_route_audit_20261019/source_mae_rmse_r2.csv)。

### 4.2 共享训练 × 在线历史输入：2×2

| 模型 | 训练方式 | 历史输入 | 2023 MAE ± seed SD | 2024 MAE ± seed SD |
| --- | --- | --- | ---: | ---: |
| SC | 共享 | 本站＋区域 | 0.083814 ± 0.000303 | 0.081507 ± 0.000318 |
| SL | 共享 | 本站 | 0.084140 ± 0.000179 | 0.081799 ± 0.000261 |
| IC | 独立 | 本站＋区域 | 0.088445 ± 0.000101 | 0.087228 ± 0.000116 |
| IL | 独立 | 本站 | 0.088450 ± 0.000129 | 0.086998 ± 0.000111 |

- 2024 `SL−IL = −0.005199`：共享训练改善 **5.98%，54/56 站**；富裕、查干湖为该比较中的两站退化。
- 2024 `SC−SL = −0.000291`：区域历史平均改善约 **0.36%，30/56 站**。
- 2024 `SL−Transformer = −0.002900`：同为本站输入，共享 CBH 改善 **3.42%，49/56 站**。
- SC/SL/IC/IL 历史路径为 BF16；不要拿另批 FP32 E0 代替 SC 做严格匹配差值。
- 轻量归档提供逐站 MAE；无法恢复的 RMSE/R² 留空，不从 MAE 推造。

来源：[原始 station×seed CSV](artifacts/cross_farm_shared_specific_20260920_v1/analysis/two_by_two_station_seed.csv)、[配对统计](runs/single_farm_route_audit_20261019/paired_comparisons.csv)、[原协议](artifacts/cross_farm_shared_specific_20260920_v1/PROTOCOL_AND_STATUS_CN.md)。

### 4.3 条件信息消融：新 FP16 四臂

全部为 2026-10-14 同批重新训练的 40 个模型，不能混入旧 BF16 CBH 或 FP32 E2C。

| 条件臂 | 额外历史摘要 | b | Δb | 2023 MAE ± seed SD | 2024 MAE ± seed SD |
| --- | --- | --- | --- | ---: | ---: |
| CBH | 有 | 有 | 有 | 0.083781 ± 0.000182 | 0.081471 ± 0.000223 |
| CB | 无 | 有 | 有 | 0.083961 ± 0.000318 | 0.081731 ± 0.000317 |
| CH | 有 | 无 | 无 | 0.088499 ± 0.000679 | 0.086065 ± 0.000752 |
| CNEUTRAL | 无 | 无 | 无 | 0.088549 ± 0.000624 | 0.086151 ± 0.000606 |

**CB 保留 b 和 Δb。** 旧补全报告中“CB 不含 increment”的文字不正确；源码和实际 checkpoint 输入核查已确认。所有臂均保留公共 decoder history memory、输出基准及 lead，CNEUTRAL 不等于整个模型没有历史信息。

2024：`CB−CNEUTRAL ≈ −0.004419`，支持 b/Δb 条件信息包；`CBH−CB ≈ −0.000260`，是额外均值摘要的小增量；`CBH−CH ≈ −0.004593`，支持在摘要之外增加轨迹信息包。尚无单独开关 b 与 Δb 的完整因子对照。

来源：[四臂均值](runs/single_farm_review_20261018/tables/condition_ablation_fp16.csv)、[实际输入核验](runs/single_farm_review_20261018/tables/condition_actual_inputs.csv)、[逐站配对](runs/single_farm_review_20261018/tables/condition_farm_contrasts.csv)。

### 4.4 天气编码器：同组 FP32 七臂

| 变体 | 天气编码 | 2023 MAE ± seed SD | 2024 MAE ± seed SD |
| --- | --- | ---: | ---: |
| E0 | 14 通道，单线性 | 0.083827 ± 0.000220 | 0.081528 ± 0.000242 |
| E1 | 14 通道＋DEM，单线性 | 0.083729 ± 0.000265 | 0.081422 ± 0.000302 |
| E2L | 14 通道，三层 affine，无激活 | 0.083811 ± 0.000186 | 0.081590 ± 0.000260 |
| E2N | 14 通道，三层 MLP＋GELU | 0.082787 ± 0.000132 | 0.080892 ± 0.000227 |
| E2 | 14 通道＋DEM，flat MLP | 0.082766 ± 0.000203 | 0.080804 ± 0.000268 |
| E2C | 15 维柱坐标，flat MLP | 0.082667 ± 0.000129 | 0.080703 ± 0.000126 |
| E3 | 柱坐标，三分组编码 | 0.082734 ± 0.000200 | 0.080725 ± 0.000324 |

- E2C 对 E0：2023/2024 分别改善 **1.38% / 1.01%**，两年均 10/10 seeds 同向，逐站为 51/56 与 47/56。
- 最干净的非线性控制为 E2N−E2L：同输入、同深度、同参数数目，2023 ΔMAE **−0.001025**，归档区间 [−0.001550,−0.000518]；2024 均值为 −0.000698，但归档等周区间跨零。不可扩大为所有时间/风场条件下都显著。
- DEM、柱坐标、三分组没有被确认独立稳定增益；E2C 比 E2raw 的差仅约 `1e-4`。
- 旧天气归档的点估计按有效 origin、区间按等周，统计目标不同。新 E2C/E0 的有效-origin 加权交叉 bootstrap：2024 Δ **−0.000825**，95% 区间 **[−0.001358,−0.000292]**；等周敏感性仍保留。不是按显著性选择区间。

来源：[七臂表](runs/single_farm_review_20261018/tables/weather_ablation_fp32.csv)、[原始因子控制](runs/single_farm_review_20261018/tables/weather_factorial_contrasts.csv)、[新权重区间](runs/single_farm_review_20261018/tables/source_e2c_week_bootstrap.csv)、[等周敏感性](runs/single_farm_review_20261018/tables/equal_week_sensitivity.csv)、[完整消融解释](artifacts/final_e2c_extensions_20260930_v1/analysis/FINAL_EXTENSION_ANALYSIS_CN.md)。

### 4.5 输出形式与历史摘要

matched direct/correction：5 对 seeds、同 FP16、保留完整条件输入与容量，仅改变最终输出锚点。

| 输出形式 | 2023 MAE ± seed SD | 2024 MAE ± seed SD |
| --- | ---: | ---: |
| direct | 0.083900 ± 0.000272 | 0.081631 ± 0.000179 |
| correction | 0.083754 ± 0.000162 | 0.081535 ± 0.000238 |

2024 correction 的平均增量仅约 **0.000096**，风场区间跨零；完整方法价值不能简化为“加一个残差”。direct 必须使用专用 `MatchedDirect` 类恢复，不能按普通 CBH 前向读取其 checkpoint。

旧 BF16 摘要实验中 SLAST 的 2023/2024 MAE 为 **0.083626/0.081351**，对应均值摘要 CBH 为 **0.083814/0.081507**。这是另一条受控历史分支，不是当前 E2C 已采用 SLAST。MLP 摘要 MFLAT/MPOOL/HMLP 未形成满足原实际门槛的主模型替代证据，当前保留均值。

来源：[matched output 表](runs/single_farm_review_20261018/tables/matched_output_fp16.csv)、[配对区间](runs/single_farm_review_20261018/tables/matched_output_contrasts.csv)、[旧摘要全部结果](results/SLAST_MAIN_RESULTS.csv)、[MLP 摘要最终报告](artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md)。

<a id="farms"></a>
## 5. 逐风场、分时效、场景与案例

### 5.1 六个正文代表风场

仅用源站 2022 年经纬度、log 容量、CF 分布与历史一小时变化等描述量，四类特征等权，采用确定性 PAM 中心站选择。未依据预测误差挑选，但这是回顾性确定的规则，不是原实验前预注册。完整 56 站总体统计不被六站替代。

下表为 2024 跨年 MAE 的 10 seeds 均值；相对改善以 Transformer 为参照。

| 风场 ID | 名称 | 容量 MW | Transformer MAE | E2C MAE | E2C 相对改善 | CBH-SL MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1043008 | 大岗风电 | 240 | 0.067920 | 0.062521 | 7.95% | 0.063201 |
| 1043010 | 马力风电 | 99 | 0.073643 | 0.071312 | 3.17% | 0.070548 |
| 1043016 | 双龙风场 | 99 | 0.072848 | 0.070154 | 3.70% | 0.071129 |
| 1043018 | 王府风电 | 49.5 | 0.093561 | 0.087643 | 6.33% | 0.090055 |
| 1043048 | 到保风场 | 49.5 | 0.077873 | 0.074553 | 4.26% | 0.074904 |
| 1043051 | 协和新发风场 | 97.8 | 0.118263 | 0.107387 | 9.20% | 0.111037 |

龙湖（1043056）单列为事后失败诊断：2024 E2C **0.115545**，Transformer **0.101018**；不得删去该边界或把它混入“事先代表性选站”过程。

来源：[选站规则](runs/single_farm_review_20261018/farm_selection.json)、[全部描述量](runs/single_farm_review_20261018/tables/source56_descriptors.csv)、[六站性能](runs/single_farm_route_audit_20261019/representative_performance_advantage.csv)、[六站及失败站完整指标](runs/single_farm_route_audit_20261019/representative_source_farms.csv)。

### 5.2 分时效与运行场景

| 2024 预测时效 | E0 MAE | E2C MAE | Transformer MAE | E2C 相对 Transformer 改善 |
| --- | ---: | ---: | ---: | ---: |
| 0-1h | 0.045081 | 0.044922 | 0.046179 | 2.72% |
| 1-3h | 0.078560 | 0.077941 | 0.081207 | 4.02% |
| 3-6h | 0.095656 | 0.094471 | 0.099867 | 5.40% |

分时效来源：[source_horizon_advantage.csv](runs/single_farm_route_audit_20261019/source_horizon_advantage.csv)，完整 24 leads：[all_models_lead_seed.csv](runs/single_farm_review_20261018/tables/all_models_lead_seed.csv)。

已有完整场景复算是 **区域 E2C 对 E0**；不能将其改写为 E2C 对 Transformer 的全年场景优势。2024 的主要观察：

- 春季 MAM：E2C−E0 约 −0.001355，47/56 站改善；季节按起报 UTC 月分组。
- 起报北京时间 00–06/06–12/12–18/18–24：差值约 −0.000442/−0.000714/−0.001018/−0.001131。
- 高功率 CF≥0.7 状态只有 55 个风场出现；每组须报告实际支持数。
- 根据未来真值标记的爬坡窗口：差值 −0.001892，非爬坡约 −0.000085。未来真值分类仅作事后诊断，不能用来部署时选择模型或门控。
- 场景定义、逐站数据与区间：[scenario_summary.csv](runs/single_farm_review_20261018/tables/scenario_summary.csv)、[scenario_per_station.csv](runs/single_farm_review_20261018/tables/scenario_per_station.csv)、[scenario_e2c_minus_e0.csv](runs/single_farm_review_20261018/tables/scenario_e2c_minus_e0.csv)。

### 5.3 已有图件

图均含 PNG 与矢量 PDF，数值图由真实结果生成。

| 图 | 文件 | 使用边界 |
|---|---|---|
| 训练描述量选站 | [01_farm_selection.pdf](runs/single_farm_review_20261018/figures/01_farm_selection.pdf) | 六个源站＋四个历史目标代表站 |
| 源站基线比较 | [02_source_baseline_comparison.pdf](runs/single_farm_review_20261018/figures/02_source_baseline_comparison.pdf) | 输入设置要明确分组 |
| 24-lead 曲线 | [03_forecast_horizons.pdf](runs/single_farm_review_20261018/figures/03_forecast_horizons.pdf) | 单模型 seed 平均 |
| 场景差异 | [04_scenario_contrasts.pdf](runs/single_farm_review_20261018/figures/04_scenario_contrasts.pdf) | E2C 对 E0 |
| 源站预测案例 | [06_source_forecast_examples.pdf](runs/single_farm_review_20261018/figures/06_source_forecast_examples.pdf) | 固定规则选起报时刻；不是按模型误差挑曲线 |
| 龙湖失败案例 | [07_failure_forecast_examples.pdf](runs/single_farm_review_20261018/figures/07_failure_forecast_examples.pdf) | 保留失败边界 |
| 历史迁移曲线 | [05_transfer_learning_curve.pdf](runs/single_farm_review_20261018/figures/05_transfer_learning_curve.pdf) | 已退出当前主线 |
| 历史目标代表站曲线 | [08_target_medoid_learning_curves.pdf](runs/single_farm_review_20261018/figures/08_target_medoid_learning_curves.pdf) | 已退出当前主线 |

案例固定 seed42，E2C 阴影为 10 seeds 的 10–90% 分位，不是预测置信区间。Transformer 案例来自 FP32 CPU 小规模推理，未替换历史 BF16 全年分数。规则及推理记录：[forecast_case_selection.csv](runs/single_farm_review_20261018/tables/forecast_case_selection.csv)、[case_inference.json](runs/single_farm_review_20261018/case_inference.json)。

<a id="history"></a>
## 6. 迁移和历史探索：结果保留、路线边界

### 6.1 迁移完整主比较，当前不作为性能主张

下表对齐 final20、seeds 42–46、相同预算和季节块。Scratch 家族只在 dev10/H1 选择；先站内平均再平均 20 站。不同来源精度仍未完全配平。

| 目标训练预算 | 从零训练 MAE（dev 选家族） | 所选从零家族 | 预训练 Transformer | CBH-T5 | E2C-T5 |
| --- | ---: | --- | ---: | ---: | ---: |
| 7d | 0.123650 | GRU_H | 0.094832 | 0.096048 | 0.095585 |
| 30d | 0.124890 | TFT_FC | 0.093827 | 0.095112 | 0.095062 |
| 90d | 0.120178 | TIMEXER_FC | 0.092754 | 0.093741 | 0.093435 |
| 180d | 0.108321 | TRANSFORMER_FC | 0.092510 | 0.092669 | 0.092528 |
| full | 0.096610 | TIDE_FC | 0.090381 | 0.090350 | 0.089853 |

结论：相对从零训练，预训练与适配有明显收益；但预训练 Transformer 的前四档均值更低，E2C 全年略优而区间跨零，不能支持本模型普遍更可迁移。30d E2C−Transformer 约 +0.001235，风场区间在零以上。按当前用户目标，迁移退出主任务与贡献点，不优先增加适配实验。

保留全部结果入口：[target_budget_comparison.csv](runs/single_farm_route_audit_20261019/target_budget_comparison.csv)、[逐目标风场](runs/single_farm_route_audit_20261019/target_all20.csv)、[共同五种子全部方案](runs/single_farm_review_20261018/tables/target20_all_baselines_common_seeds.csv)、[独立基线全部十种子](runs/single_farm_review_20261018/tables/target20_scratch_all10seeds.csv)、[原迁移报告](artifacts/cross_farm_shared_specific_20260920_v1/analysis/FINAL_REPORT_CN.md)。

### 6.2 其他方向与停止依据

| 方向 | 已有结果与当前判断 | 原始入口 |
|---|---|---|
| 56 站独立命名模型 | 8,400/8,400，最佳 Transformer 的 2023 pilot MAE 0.099661；约 10% 时间覆盖，不能与全年直接混排 | [pilot 报告](artifacts/expanded_all_station_baselines_20260919_v1/analysis/REPORT_CN.md)、[全部明细](artifacts/expanded_all_station_baselines_20260919_v1/analysis/all_seed_metrics.csv) |
| 独立 CBH 与共享 CBH | 独立完整年度 IL 结果可用；独立天气修正有收益，共享总体更强 | [独立 CBH 分析](artifacts/cbh_per_station_independent_20260918_v1/analysis/DETAILED_ANALYSIS_CN.md) |
| Canonical 轨迹归因 J/H/T/A 等 | 轨迹与历史信息互补；J 的 2023/2024 为 0.084760/0.082758；协议早于当前 E2C，不作单模块跨协议归因 | [轨迹复核](artifacts/trajectory_review_20260915/REPORT_CN.md)、[原始报告](artifacts/trajectory_attribution_20260914_v1/analysis/REPORT_CN.md) |
| P0I/PA/PJ 条件接入 | P0→PJ 有系统收益；PA/PJ 未证明联合 adapter 操作独立领先；不是未见风场迁移 | [研究总结](artifacts/final_research_summary_20260915/KEY_CONCLUSIONS_CN.md)、[条件接入](artifacts/p0i_condition_transfer_20260915_v1/analysis/REPORT_CN.md) |
| F4 低秩个性化 | 源站有小增益，full T4 0.094301 劣于容量控制 T2 0.090733 与 T5 0.090350；不继续作为主机制 | [源确认与迁移](artifacts/cross_farm_shared_specific_20260920_v1/analysis/FINAL_REPORT_CN.md) |
| 高频天气代理接入原 CBH | 原 CBH 0.082244；最佳非线性代理 DET 0.086976；同样微调的 LINEAR 控制下无稳定改善 | [原 CBH 生成接入](artifacts/temporal_sr_original_cbh_20260917_v1/analysis/FINAL_REPORT_CN.md) |
| 命名模型＋高频生成 | 20 家族×10 站×10 seeds＝2,000；含负结果与已排除的早期泄漏批次，不恢复被拒绝数据 | [正式生成结果](artifacts/temporal_sr_named_models_20260917_v2/analysis/FINAL_REPORT_CN.md)、[统一复核](artifacts/temporal_sr_unified_review_20260918_v1/analysis/UNIFIED_REPORT_CN.md) |
| 显式恢复与宽上下文 | 同 10 站三种子 pilot：R0 0.081916、R5 0.085041、R6 0.083548；后两者 0/3 seeds 胜出，触发停止 | [context vs recovery](artifacts/context_vs_recovery_20260918_v1/KEY_CONCLUSIONS_CN.md) |
| 小时残差代理 | 重建改善未转化为功率收益，无真实高频天气监督，不优先扩大 flow/diffusion | [hourly residual](artifacts/hourly_residual_proxy_20260918_v1/KEY_CONCLUSIONS_CN.md) |
| SLAST/MLP 摘要 | 有小效应或未达实际门槛；不拼接成未经验证的 E2C＋SLAST | [摘要结果](results/SLAST_MAIN_RESULTS.csv)、[MLP 结论](artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md) |
| 更早区域/集群动力学 | 目标、输入、损失或统计量常不同；作为设计历史，不能替代当前单站主表 | [历史报告](historical_reports/)、[实验文档](exp_docs/) |

<a id="contributions"></a>
## 7. 计划的三个贡献点

**定位：一个核心方法贡献、一个模块设计贡献、一个实证贡献。现有证据不支持把三项都写成独立的新算法创新。**

### 贡献一：历史预测轨迹条件化的气象修正框架

拟用表述：**提出历史预测轨迹条件化的天气修正框架，使历史功率生成的未来参考轨迹参与天气信息组织，并保留完整历史记忆，实现单风场多步功率预测。**

- 核心是 `[历史摘要,b,Δb,lead]` 条件与气象的联合组织，不是简单残差相加。
- 已有证据：新 FP16 四臂；CBH/CNEUTRAL 的 2024 MAE 为 0.081471/0.086151；CB 保留 b/Δb 时为 0.081731。matched direct/correction 排除了把全部收益归于输出锚点的说法。
- 关键代码：[p0i_transfer.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_transfer.py) 中 `forward`、`condition_projection`、`adapter_response`。
- 边界：b 与 Δb 尚未单独因子化；“neutral”仍有 decoder 历史记忆；不是物理因果/反事实识别。

### 贡献二：面向功率预测的非线性局地气象编码

拟用表述：**构建面向功率修正任务的非线性局地气象编码，并通过输入、深度与参数量控制验证非线性表示的预测增益及作用边界。**

- 完整实现为 E2C；七臂控制区分 DEM、深仿射、GELU、坐标和分组。
- 已有证据：E2C/E0 的两年改善 1.38%/1.01%；E2N/E2L 的匹配控制在验证期支持非线性激活作用。
- 关键代码：[column_model.py](artifacts/column_weather_20260928_v1/code/column_model.py) 中 `WeatherProjection`、`column_coordinates`、`ColumnWeatherCBH`。
- 边界：普通 MLP/GELU 本身不构成强算法新颖性；DEM、柱坐标、垂直分组不能分别列为已成立贡献，也不能声称恢复真实高频天气。

### 贡献三：共享训练的单风场跨年性能验证与收益归因

拟用表述：**建立共享训练、逐场评价的跨年验证体系，通过训练共享与在线区域输入的交叉对照量化共享收益，并验证模型对不同风场和预测时效的适用性。**

- 已有证据：SL 对 IL 改善 5.98%，54/56 站；E2C 对共享 Transformer 改善 4.72%，50/56 站；六个非误差选取的代表站均有改善；0–1/1–3/3–6h 分段结果完整。
- 关键代码：[train_sl_seed.py](artifacts/cross_farm_shared_specific_20260920_v1/code/train_sl_seed.py)、[共享基线 train_one.py](artifacts/shared_all_station_baselines_20260922_v1/code/train_one.py)、[审计汇总](runs/single_farm_route_audit_20261019/audit.py)、[性能复核](runs/single_farm_route_audit_20261019/performance_route_evidence.py)。
- 边界：普通参数共享不是新算法；完整 E2C 对命名基线的输入尚不完全相同；需补同区域输入强基线。贡献三是系统实证贡献，不是未见风场迁移机制。

建议论文结果顺序：全部共享基线主表 → 六站逐场表与完整 56 站附录 → 条件/天气消融 → 共享×输入 2×2 → 时效、场景、预测案例与失败边界。迁移不进入主贡献叙事。

<a id="code"></a>
## 8. 关键代码地图与修改入口

### 8.1 运行调用关系：不要只改根 `src/`

```text
./project
  → .runtime/chronos/bin/python -B runtime/project_cli.py
  → runtime/wind_paths.py 选择对应实验 snapshot/src
  → predict: runtime/unified_predict.py
      → ColumnWeatherCBH（天气实验 code/column_model.py）
      → P0IConditionTransfer（归因实验 snapshot/src）
  → source-train: runtime/source_train.py
      → 导出重定位配置到新 runs 目录
      → 注册天气模型 builder
      → 归因实验 snapshot/scripts/train_forced_evolution.py
```

`wind_forcing` 存在多个历史版本。`runtime/wind_paths.py::source('cbh')` 当前选择 **p0i_attribution_closure_20260915_v1/snapshot/src**；`source('slast')` 选择旧摘要实验快照。根 [src/wind_forcing/](src/wind_forcing/) 主要保留更早模型，直接修改它不一定影响当前 E2C。

同一个 Python 进程不要混装多个同名 `wind_forcing` 快照；`sys.modules` 缓存会导致源码版本混用。按模型族使用独立 `./project` 进程。历史绝对路径由 `wind_paths` 和 `archive_loader` 在内存重定位，原文件哈希保持不变。

### 8.2 实现与职责地图

| 要理解或修改的部分 | 文件入口 | 关键符号/职责 |
|---|---|---|
| 统一命令入口 | [project](project)、[project_cli.py](runtime/project_cli.py) | 命令分发、项目 Python 环境 |
| 路径与输出保护 | [wind_paths.py](runtime/wind_paths.py) | `source`、`select_source`、`rebase_tree`、`new_output` |
| 旧模块载入 | [archive_loader.py](runtime/archive_loader.py) | `load_module`，AST 内存路径重写 |
| 当前 checkpoint 重建 | [unified_predict.py](runtime/unified_predict.py) | `kind_of`、`predict_checkpoint`；strict state loading、指纹和形状核验 |
| 新源站训练入口 | [source_train.py](runtime/source_train.py) | `main`、`execute`，支持 column/condition，输出新目录 |
| E2C 与其他天气编码器 | [column_model.py](artifacts/column_weather_20260928_v1/code/column_model.py) | `WeatherProjection`、`ColumnWeatherCBH`、`install_builder` |
| 当前条件修正主体 | [p0i_transfer.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_transfer.py) | `P0IConditionTransfer.forward` |
| 空间读取、时间插值、点对齐 | [p0i_legacy.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_legacy.py) | `station_weather`、`interpolate`、`TemporalAlignedBaseline._align_masked` |
| 区域历史基准 | [forced_evolution.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/forced_evolution.py) | `ForcedEvolutionModel`、`build_forced_evolution_model` |
| 模型注册 | [models/__init__.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/__init__.py) | `build_forecast_model`；column builder 在运行时扩展 |
| 局地采样/基础组件 | [components.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/components.py) | `build_station_patch_metadata` 等 |
| 缓存数据载入 | [dataset.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/data/dataset.py) | `ForecastDataset`、`ForecastCollator`、`CycleGroupedBatchSampler` |
| 损失、训练、指标 | [training.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/training.py) | `deterministic_loss`、`fit_deterministic`、`evaluate_deterministic`、`MetricAccumulator` |
| 训练/完整评估脚本 | [train_forced_evolution.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/scripts/train_forced_evolution.py)、[evaluate_forced_evolution.py](artifacts/p0i_attribution_closure_20260915_v1/snapshot/scripts/evaluate_forced_evolution.py) | 实验底层训练、导出完整预测；原路径需适配 |
| E2C 训练配置 | [E2C_seed42.yaml](artifacts/column_weather_20260928_v1/configs/E2C_seed42.yaml) | 主配置模板；完整 seeds 42–51 同目录 |
| 条件消融配置 | [CBH_seed42.yaml](artifacts/p0i_condition_ablation_completion_20261014_v1/configs/CBH_seed42.yaml) | FP16 四臂模板 |
| DEM 与训练统计 | [prepare_dem.py](artifacts/column_weather_20260928_v1/code/prepare_dem.py)、[prepare_features.py](artifacts/column_weather_20260928_v1/code/prepare_features.py) | 既有 DEM、2022 训练气象归一化；不要用验证/跨年数据重算 |
| 天气编码器测试 | [test_column_model.py](artifacts/column_weather_20260928_v1/code/test_column_model.py) | 坐标可逆性与模型检查 |
| 本站输入 SL 复现参考 | [train_sl_seed.py](artifacts/cross_farm_shared_specific_20260920_v1/code/train_sl_seed.py)、[其 forced_evolution.py](artifacts/cross_farm_shared_specific_20260920_v1/snapshot/src/wind_forcing/models/forced_evolution.py) | `local_history_only=True`；第二历史编码器改读本站历史 |
| 15 家族实际模型 | [runtime/experiments/fewshot/models.py](runtime/experiments/fewshot/models.py) | `TransformerFutureCov`、`TiDEFutureCov`、`build_model`、`input_group` |
| 共享基线数据/训练 | [共享 train_one.py](artifacts/shared_all_station_baselines_20260922_v1/code/train_one.py) | `SourceSplit`、`predict`、`evaluate`；origin×station 展开 |
| 严格 direct 输出控制 | [matched_direct.py](artifacts/pretrained_transformer_transfer_20260927_v1/code/matched_direct.py) | `MatchedDirect`、`build_matched_model` |
| 历史目标适配运行代码 | [transfer_experiment.py](runtime/experiments/transfer/transfer_experiment.py)、[fewshot train_one.py](runtime/experiments/fewshot/train_one.py) | 已退出主线，保留复现 |
| 结果数据库构建/查询 | [result_catalog.py](runtime/result_catalog.py)、[merge_csv_results.py](runtime/merge_csv_results.py) | 标量索引、pilot CSV 导入、协议分组 |
| 本轮全指标审计 | [audit.py](runs/single_farm_route_audit_20261019/audit.py) | 核对所有索引标量，重建逐站/配对结果 |
| 本轮性能路线汇总 | [performance_route_evidence.py](runs/single_farm_route_audit_20261019/performance_route_evidence.py) | 15 基线、验证选家族、六站、分时效 |
| 上轮选站/表格汇总 | [audit_tables.py](runs/single_farm_review_20261018/scripts/audit_tables.py) | 描述量、PAM、主表、条件和迁移汇总 |
| 预测级复算与场景 | [analyze_predictions.py](runs/single_farm_review_20261018/scripts/analyze_predictions.py)、[finalize_review.py](runs/single_farm_review_20261018/scripts/finalize_review.py) | 对齐原始预测、指标、场景及图件 |
| 条件字段核查 | [audit_condition_inputs.py](runs/single_farm_review_20261018/scripts/audit_condition_inputs.py) | 实际 checkpoint 条件输入 hook，纠正 CB 解释 |
| 权重统计敏感性 | [audit_interval_weighting.py](runs/single_farm_review_20261018/scripts/audit_interval_weighting.py) | 有效 origin 与等周 bootstrap |

**local-only 特别提醒：**当前 `source-train` 没有 `--local-only` 选项，默认使用既有区域历史基准权重。支持 SL 的 `local_history_only` 分支位于 cross-farm 快照；不能只给当前配置加一个字段就声称已构建本站模型。需要核对所载入快照、基准权重、实际输入依赖和对应训练配置。

### 8.3 当前关键训练配置

E2C：`architecture=p0i_condition_transfer`、`alignment_variant=P0I`、`native_scope=native7`、`transfer_variant=CBH`、`condition_mode=full`、`d_model=64`、`history_visible_steps=24`、冻结历史基准、FP32、AdamW、lr `2e-4`、weight decay `1e-4`、80 epochs 上限、patience 10、以 `station_mae_cf` 选 checkpoint。站点 MAE 权重为 1，集群损失与 ramp loss 权重为 0。

E2C 可训练 correction 参数约 **278,729**；这不包含整个冻结历史基准的训练成本，不能当作完整系统参数量或总训练成本。精确参数和实际执行以保存配置/metrics 为准，旧 YAML 中保留的其他阶段参数不等于每次运行都执行了对应预训练阶段。

<a id="assets"></a>
## 9. 权重、预测文件与资产可用性

| 资产 | 实际例子/目录 | 状态 |
|---|---|---|
| E2C seed42 最终权重 | [best_model.pt](artifacts/column_weather_20260928_v1/runs/main/E2C/seed42/best_model.pt) | 本地存在；其余正式变体/种子同级 |
| E2C 原始指标 | [metrics_test.json](artifacts/column_weather_20260928_v1/runs/main/E2C/seed42/metrics_test.json) | 可追溯主表 |
| E2C 完整预测 | [predictions_test.npz](artifacts/column_weather_20260928_v1/runs/main/E2C/seed42/predictions_test.npz) | `[28782,56,24]` 的逐站 MW 预测/目标 |
| E2C 历史基准权重 | [seed42.pt](artifacts/p0i_attribution_closure_20260915_v1/inputs/baselines/seed42.pt) | `source-train` 默认使用该族区域基准 |
| DEM/天气统计 | [station_dem_srtm30m.json](artifacts/column_weather_20260928_v1/data/station_dem_srtm30m.json)、[column_train_stats.json](artifacts/column_weather_20260928_v1/data/column_train_stats.json) | 固定站序、训练统计、哈希 |
| 共享 Transformer 权重 | [seed42/best_model.pt](artifacts/shared_all_station_baselines_20260922_v1/results/main/TRANSFORMER_FC/seed42/best_model.pt) | 本地存在；家族/种子按同模式定位 |
| 新条件 CBH 权重 | [CBH_seed42/best_model.pt](artifacts/p0i_condition_ablation_completion_20261014_v1/results/CBH_seed42/best_model.pt) | FP16 训练，当前 sample predict 为 FP32 |
| SL/IC 与独立 CBH | [2×2 原始指标](artifacts/cross_farm_shared_specific_20260920_v1/analysis/two_by_two_station_seed.csv)、[独立 CBH 目录](artifacts/cbh_per_station_independent_20260918_v1/) | 指标存在；本地 SL 目录缺失，独立 CBH 目录未找到 `.pt`，不能承诺直接恢复全部推理 |
| 独立 8,400 pilot | [all_seed_metrics.csv](artifacts/expanded_all_station_baselines_20260919_v1/analysis/all_seed_metrics.csv) | 明细已补回；旧报告“CSV 未迁入”已过时；权重不能据训练完成数推断存在 |

原始 E2C NPZ 的关键字段：`station_prediction_mw`、`station_target_mw`、`origin_unix`、`sample_index`，另外含 cluster/baseline/correction MW 字段。复算 CF 时须除以对应 `station_capacity_mw`。统一 `./project predict --output` 导出的字段是 **`station_factor`**，与原始评测 NPZ 格式不同，不能直接互换。

源模型 checkpoint 通常含 `model_config`、`model_state`、数据指纹；迁移部分 checkpoint 只保存可训练目标参数，需与源模型/adapter 组合，不能简单 `load_state_dict` 当成完整模型。

结果数据库：[CATALOG.sqlite](results/CATALOG.sqlite)，CSV 镜像：[RUN_METRICS.csv](results/RUN_METRICS.csv)、[PROTOCOL_SUMMARY.csv](results/PROTOCOL_SUMMARY.csv)、[RESULT_ASSETS.csv](results/RESULT_ASSETS.csv)。`checkpoint_exists=0` 表示本地缺权重；`legacy_csv_rows` 有历史重复，不能与 `observations` 简单相加计算实验量。

<a id="commands"></a>
## 10. 已验证命令与复现方式

### 10.1 只读查看与少量推理

从项目根执行；统一入口使用 `.runtime/chronos/bin/python`，不依赖 shell 已激活环境。

```bash
cd /shared_data_ssd2/liying/wind_cluster_project_20261018/project
./project status
./project results --experiment column_weather_20260928_v1 --split test --limit 20
./project results --experiment shared_all_station_baselines_20260922_v1 --model TRANSFORMER_FC --split test
./project source-train --help
./project predict --help
./project predict --checkpoint artifacts/column_weather_20260928_v1/runs/main/E2C/seed42/best_model.pt --split test --samples 2 --device cpu
```

少量推理只验证加载/前向，不产生全年性能结论。统一推理为 FP32，原评测精度保留在结果中。共享基线的 `--start/--samples` 对应展平的 `(origin,station)` 样本，E2C 对应包含 56 站的 origin 样本；比较时须重新按 origin/station 对齐。

### 10.2 重汇总与验证

以下写入对应 `runs/` 下派生文件；不修改原实验，也不训练。全指标审计会读取七万余份文件，预测复算会读取数 GiB 数据，文档编辑时无需重复全跑。

```bash
.runtime/chronos/bin/python runs/single_farm_route_audit_20261019/audit.py
.runtime/chronos/bin/python runs/single_farm_route_audit_20261019/performance_route_evidence.py
```

完整旧图表复算顺序与命令见 [重审 README](runs/single_farm_review_20261018/README_CN.md)。`./project check` 会运行项目兼容/完整性检查并更新验证产物，不是只读查询，也不等于训练或全数据复评。`./project results --rebuild` 会重建派生索引，做文档阅读无需运行。

### 10.3 新实验配置与训练

仅准备配置、会创建新目录但不训练：

```bash
./project source-train --experiment column --variant E2C --seed 42 --output runs/example_e2c_prepare_v1 --prepare-only
```

确实需要开展新训练时，使用另一个未存在的新目录：

```bash
./project source-train --experiment column --variant E2C --seed 42 --device cuda --output runs/example_e2c_train_v1
```

后一条会实际训练，不应作为文档校验命令。`--epochs` 缩短训练产生的是新探索结果，不能冒充原 80-epoch 协议复现；准备过的输出目录不能直接复用为训练输出。该入口复现现有区域 E2C，**并未实现下节的同区域输入 Transformer 或 local-only E2C**。

旧 `artifacts/**/code/run_queue.py`、controller、watchdog 与历史 shell 可能含旧绝对路径、GPU 编号并向冻结目录写入；保留作证据，不作为当前统一启动入口。新的训练/推理/分析产物放新 `runs/` 目录，配置、代码与数据身份一并记录。

<a id="next"></a>
## 11. 下一步实验与完成标准

| 优先级 | 工作 | 状态 | 完成标准/代码起点 |
|---|---|---|---|
| P0 | 固定已有风场跨年预测主线，以 E2C 为主模型 | 已确定 | 本文与路线报告保持一致；迁移退出主贡献 |
| P0 | 15 类共享基线、56 站总体、六站展示、三个时效段 | 已汇总 | 本文第 4–5 节及 CSV；不用重跑全部旧实验 |
| P0 | 同区域输入强 Transformer 对照 | **未实现/未训练** | 从 `TransformerFutureCov`、`SourceSplit` 出发；提供同区域历史信息，统一精度、窗口与合理优化/选模预算 |
| P0 | E2C 与强基线共同场景复评 | **部分已有** | E2C/E0 全量场景已完成；Transformer/TiDE/N-BEATSx 的全年共同场景预测仍待补齐 |
| P0 | 按最新三贡献与去迁移主线同步论文 | **未同步本次决定** | 当前 `paper/tex` 已是 10 月 18 日单风场稿，但仍含迁移叙事；不要回退旧 SLAST 稿 |
| P1 | 真正独立的后续年份确认 | 待数据与协议 | 优先同一风场新年份，如 2025；冻结模型后评价，不把现有 2024 再称盲测 |
| 条件项 | local-only E2C/E2N | 未运行；非当前默认前置 | 仅在明确纯本站输入要求时推进；基准 B₀ 与 correction 都须满足本站输入 |
| 条件项 | 单独 b-only / Δb-only 控制 | 未运行 | 只有拟声称两字段独立贡献时需要；否则表述为信息包 |
| 低优先级 | SL 权重恢复 | 指标可用，权重缺失 | 需要 SL 新场景曲线时再恢复/复现，不能把缺权重说成缺指标 |
| 退出主线 | 新风场迁移、低秩适配、高频生成新搜索 | 历史结果保留 | 不自动恢复旧队列或为选择有利新站增加筛选 |

同输入对照应在查看跨年结果前固定区域输入接入方式、搜索范围、精度与模型选择规则；记录各模型总参数/可训练参数、历史基准预训练成本和实际训练预算。不能只给弱基线极少调参，也不能仅凭名称宣称超过官方 SOTA。

运行前冻结验证选择和效果报告规则：报告有效-origin 加权主指标、seed 差、站点胜率与时间块敏感性，保存失败/退化风场。若新强对照消除优势，应如实更新路线判断；现有 4.72% 系统优势不保证未运行的严格对照胜出。

<a id="pitfalls"></a>
## 12. 结果解释中的已知问题

| 容易误读的内容 | 正确解释 |
|---|---|
| 单风场任务＝完全独立训练 | 输出/评价粒度与训练数据共享是两回事 |
| E2C 是纯本站输入 | 现有完整模型的冻结历史基准用区域历史 |
| 修改根 `src/` 即修改当前主模型 | 实际由 runtime 选择实验快照 |
| E2C 先编码小时气象再插值 | 当前 forward 是先插值，再 `native_projection` |
| `CB` 没有 Δb | CB 同时保留 b、Δb；新四臂旧报告的该段解释有误 |
| CNEUTRAL 完全无历史 | decoder memory 与输出基准仍在 |
| `E2` 与 `E2raw` 是两组独立实验 | 是同一变体的代码名与报告名 |
| E2C＋SLAST 已验证 | 当前 E2C 用均值摘要，没有该组合的配对实验 |
| 0.099661 对 0.082667 证明共享增益 17% | 前者是独立 pilot，后者是全年 E2C，数据与评价窗口不匹配 |
| 7d 迁移只需七天全部标签 | 原实验还用完整 H1 标签选 checkpoint |
| 2024 是新的盲测 | 已经多轮查看，只能称跨年复评 |
| 所有 CI 都对应同一统计目标 | 有效-origin、等周、seed-only、farm-only 不同 |
| 场景优势已经胜 Transformer | 现有全年场景主要是 E2C 对 E0；Transformer 只有逐 lead 与部分案例 |
| seed 阴影是预测概率区间 | 是训练随机性分布，未校准预测覆盖率 |
| 高未来 ramp 可用于选择模型 | 用未来真值定义，只能事后诊断 |
| 训练成功数等于本地可用权重数 | 轻量归档可能只保留指标；先查资产 |
| 旧 2026-10-01 报告说独立 CSV 缺失 | 10 月 18 日已补回 8,400＋1,100 条独立明细 |
| 根目录旧论文等于当前稿 | 当前实际稿是 `paper/tex/main.tex`；仍需同步 10 月 19 日新路线 |

<a id="experiments"></a>
## 13. 全部 39 个实验/资料目录索引

由 [EXPERIMENT_INDEX.csv](results/EXPERIMENT_INDEX.csv) 与 [RESULT_ASSETS.csv](results/RESULT_ASSETS.csv) 导航。下列“记录”是进入 `observations` 的运行/划分标量记录数，**0 不代表没做实验**，可能是只保留报告、旧 CSV 或交付资料。不同目录可能复用历史结果，不可直接把各目录规模相加当成独立训练数。

| 实验/资料目录 | 标量记录 | 定位 | 已索引报告（可点击） |
| --- | ---: | --- | --- |
| [cbh_per_station_independent_20260918_v1](artifacts/cbh_per_station_independent_20260918_v1/) | 0 | 独立 CBH；全年逐站对照 | [STATUS_CN.md](artifacts/cbh_per_station_independent_20260918_v1/STATUS_CN.md)<br>[analysis/DETAILED_ANALYSIS_CN.md](artifacts/cbh_per_station_independent_20260918_v1/analysis/DETAILED_ANALYSIS_CN.md)<br>[analysis/REPORT_CN.md](artifacts/cbh_per_station_independent_20260918_v1/analysis/REPORT_CN.md) |
| [column_weather_20260928_v1](artifacts/column_weather_20260928_v1/) | 140 | 主模型 E2C；七种天气编码控制 | [PROTOCOL.md](artifacts/column_weather_20260928_v1/PROTOCOL.md)<br>[analysis/FINAL_MODEL_ANALYSIS_CN.md](artifacts/column_weather_20260928_v1/analysis/FINAL_MODEL_ANALYSIS_CN.md)<br>[analysis/TYPICAL_STATION_REPORT_CN.md](artifacts/column_weather_20260928_v1/analysis/TYPICAL_STATION_REPORT_CN.md) |
| [context_vs_recovery_20260918_v1](artifacts/context_vs_recovery_20260918_v1/) | 0 | 宽上下文/显式恢复 pilot；未胜出 | [EXPERIMENT_PLAN_REVIEW_CN.md](artifacts/context_vs_recovery_20260918_v1/EXPERIMENT_PLAN_REVIEW_CN.md)<br>[KEY_CONCLUSIONS_CN.md](artifacts/context_vs_recovery_20260918_v1/KEY_CONCLUSIONS_CN.md)<br>[README_CN.md](artifacts/context_vs_recovery_20260918_v1/README_CN.md) |
| [cross_farm_shared_specific_20260920_v1](artifacts/cross_farm_shared_specific_20260920_v1/) | 14,898 | 共享×输入 2×2；F/T 个性化与迁移 | [EXPERIMENT_EVALUATION_AND_EXECUTION_CN.md](artifacts/cross_farm_shared_specific_20260920_v1/EXPERIMENT_EVALUATION_AND_EXECUTION_CN.md)<br>[PROTOCOL_AND_STATUS_CN.md](artifacts/cross_farm_shared_specific_20260920_v1/PROTOCOL_AND_STATUS_CN.md)<br>[analysis/BASELINE_BEST_AND_ABLATION_REPORT_CN.md](artifacts/cross_farm_shared_specific_20260920_v1/analysis/BASELINE_BEST_AND_ABLATION_REPORT_CN.md)<br>[analysis/FINAL_REPORT_CN.md](artifacts/cross_farm_shared_specific_20260920_v1/analysis/FINAL_REPORT_CN.md)<br>[analysis/KEY_CONCLUSIONS_CN.md](artifacts/cross_farm_shared_specific_20260920_v1/analysis/KEY_CONCLUSIONS_CN.md) |
| [expanded_all_station_baselines_20260919_v1](artifacts/expanded_all_station_baselines_20260919_v1/) | 8,400 | 56 站独立 pilot；8,400 组 | [README_CN.md](artifacts/expanded_all_station_baselines_20260919_v1/README_CN.md)<br>[analysis/REPORT_CN.md](artifacts/expanded_all_station_baselines_20260919_v1/analysis/REPORT_CN.md) |
| [expanded_single_station_baselines_20260919_v1](artifacts/expanded_single_station_baselines_20260919_v1/) | 1,100 | 早期独立命名基线；1,100 组 | [README_CN.md](artifacts/expanded_single_station_baselines_20260919_v1/README_CN.md)<br>[analysis/REPORT_CN.md](artifacts/expanded_single_station_baselines_20260919_v1/analysis/REPORT_CN.md) |
| [final_e2c_extensions_20260930_v1](artifacts/final_e2c_extensions_20260930_v1/) | 2,800 | E2L/E2N 控制分析与 E2C-T5 迁移 | [PROTOCOL.md](artifacts/final_e2c_extensions_20260930_v1/PROTOCOL.md)<br>[analysis/ABLATION_AUDIT_CN.md](artifacts/final_e2c_extensions_20260930_v1/analysis/ABLATION_AUDIT_CN.md)<br>[analysis/FINAL_EXTENSION_ANALYSIS_CN.md](artifacts/final_e2c_extensions_20260930_v1/analysis/FINAL_EXTENSION_ANALYSIS_CN.md) |
| [final_research_summary_20260915](artifacts/final_research_summary_20260915/) | 0 | 早期核心结论与轨迹解释 | [KEY_CONCLUSIONS_CN.md](artifacts/final_research_summary_20260915/KEY_CONCLUSIONS_CN.md) |
| [framework_handoff_20260916](artifacts/framework_handoff_20260916/) | 0 | 早期框架交接资料；路线已更新 | [README_CN.md](artifacts/framework_handoff_20260916/README_CN.md) |
| [hourly_residual_proxy_20260918_v1](artifacts/hourly_residual_proxy_20260918_v1/) | 0 | 小时残差代理探索；未形成主线 | [KEY_CONCLUSIONS_CN.md](artifacts/hourly_residual_proxy_20260918_v1/KEY_CONCLUSIONS_CN.md)<br>[README_CN.md](artifacts/hourly_residual_proxy_20260918_v1/README_CN.md) |
| [main_model_architecture_review_20260919_v1](artifacts/main_model_architecture_review_20260919_v1/) | 0 | 历史架构审查与图文说明 | [REPORT_CN.md](artifacts/main_model_architecture_review_20260919_v1/REPORT_CN.md) |
| [mechanism_audit_20260914](artifacts/mechanism_audit_20260914/) | 135 | 早期机制核验 | [EXECUTION_AUDIT.md](artifacts/mechanism_audit_20260914/EXECUTION_AUDIT.md)<br>[PROTOCOL.md](artifacts/mechanism_audit_20260914/PROTOCOL.md)<br>[RESULT_STATUS.md](artifacts/mechanism_audit_20260914/RESULT_STATUS.md)<br>[analysis_final_20260914/REPORT_CN.md](artifacts/mechanism_audit_20260914/analysis_final_20260914/REPORT_CN.md)<br>[analysis_final_20260914/source_snapshot/PROTOCOL.md](artifacts/mechanism_audit_20260914/analysis_final_20260914/source_snapshot/PROTOCOL.md)<br>[canonical_mae_20260914/incident_audit_20260914_2030/EXECUTION_ACCOUNT_CN.md](artifacts/mechanism_audit_20260914/canonical_mae_20260914/incident_audit_20260914_2030/EXECUTION_ACCOUNT_CN.md)<br>[canonical_mae_20260914/results_snapshot_20260914_2034/RESULTS_CN.md](artifacts/mechanism_audit_20260914/canonical_mae_20260914/results_snapshot_20260914_2034/RESULTS_CN.md) |
| [p0i_attribution_closure_20260915_v1](artifacts/p0i_attribution_closure_20260915_v1/) | 80 | 当前 CBH 源码/基准权重来源；归因控制 | [PROTOCOL.md](artifacts/p0i_attribution_closure_20260915_v1/PROTOCOL.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_attribution_closure_20260915_v1/analysis/REPORT_CN.md) |
| [p0i_attribution_closure_review_20260916](artifacts/p0i_attribution_closure_review_20260916/) | 0 | 归因控制复核 | [REPORT_CN.md](artifacts/p0i_attribution_closure_review_20260916/REPORT_CN.md) |
| [p0i_condition_ablation_completion_20261014_v1](artifacts/p0i_condition_ablation_completion_20261014_v1/) | 80 | 同批 FP16 条件四臂；当前关键证据 | [PROTOCOL.md](artifacts/p0i_condition_ablation_completion_20261014_v1/PROTOCOL.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_condition_ablation_completion_20261014_v1/analysis/REPORT_CN.md) |
| [p0i_condition_transfer_20260915_v1](artifacts/p0i_condition_transfer_20260915_v1/) | 80 | 条件接入形式；非新风场迁移 | [PROTOCOL.md](artifacts/p0i_condition_transfer_20260915_v1/PROTOCOL.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_condition_transfer_20260915_v1/analysis/REPORT_CN.md) |
| [p0i_learned_summary_20260916_v1](artifacts/p0i_learned_summary_20260916_v1/) | 113 | 历史摘要候选探索 | [PROTOCOL.md](artifacts/p0i_learned_summary_20260916_v1/PROTOCOL.md)<br>[README_CN.md](artifacts/p0i_learned_summary_20260916_v1/README_CN.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_learned_summary_20260916_v1/analysis/REPORT_CN.md) |
| [p0i_mlp_final_20260917_v1](artifacts/p0i_mlp_final_20260917_v1/) | 7 | MLP 摘要最终控制与停止依据 | [PROTOCOL.md](artifacts/p0i_mlp_final_20260917_v1/PROTOCOL.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_mlp_final_20260917_v1/analysis/REPORT_CN.md)<br>[inputs/handoff_before.md](artifacts/p0i_mlp_final_20260917_v1/inputs/handoff_before.md)<br>[inputs/summary_review_before_final.md](artifacts/p0i_mlp_final_20260917_v1/inputs/summary_review_before_final.md) |
| [p0i_summary_calibration_20260916_v1](artifacts/p0i_summary_calibration_20260916_v1/) | 100 | SLAST/均值等摘要；独立历史分支 | [PROTOCOL.md](artifacts/p0i_summary_calibration_20260916_v1/PROTOCOL.md)<br>[README_CN.md](artifacts/p0i_summary_calibration_20260916_v1/README_CN.md)<br>[analysis/REPORT_CN.md](artifacts/p0i_summary_calibration_20260916_v1/analysis/REPORT_CN.md) |
| [p0i_summary_diagnostic_seed49_20260916_v1](artifacts/p0i_summary_diagnostic_seed49_20260916_v1/) | 0 | seed49 摘要诊断 | [PROTOCOL.md](artifacts/p0i_summary_diagnostic_seed49_20260916_v1/PROTOCOL.md)<br>[analysis/DIAGNOSTICS_CN.md](artifacts/p0i_summary_diagnostic_seed49_20260916_v1/analysis/DIAGNOSTICS_CN.md) |
| [p0i_summary_recovery_20260916_v1](artifacts/p0i_summary_recovery_20260916_v1/) | 0 | 摘要实验恢复记录 | [PROTOCOL.md](artifacts/p0i_summary_recovery_20260916_v1/PROTOCOL.md) |
| [p0i_summary_review_20260916_v1](artifacts/p0i_summary_review_20260916_v1/) | 0 | 摘要实验复核 | [analysis/REPORT_CN.md](artifacts/p0i_summary_review_20260916_v1/analysis/REPORT_CN.md) |
| [paper_closure_20261003_v1](artifacts/paper_closure_20261003_v1/) | 0 | 旧论文闭环计划；未完成项须重核 | [OLD_SERVER_RECOVERY_PROMPT_CN.md](artifacts/paper_closure_20261003_v1/OLD_SERVER_RECOVERY_PROMPT_CN.md)<br>[PAPER_EXPERIMENT_PLAN_CN.md](artifacts/paper_closure_20261003_v1/PAPER_EXPERIMENT_PLAN_CN.md) |
| [paper_figures_20261003_v1](artifacts/paper_figures_20261003_v1/) | 0 | 示意图资产、提示词和绘图说明 | [DRAWING_COMMANDS.md](artifacts/paper_figures_20261003_v1/DRAWING_COMMANDS.md)<br>[E2C_CBH_IMAGEGEN_EDIT_PROMPT.md](artifacts/paper_figures_20261003_v1/E2C_CBH_IMAGEGEN_EDIT_PROMPT.md)<br>[E2C_CBH_IMAGEGEN_PROMPT.md](artifacts/paper_figures_20261003_v1/E2C_CBH_IMAGEGEN_PROMPT.md)<br>[E2C_CBH_IMAGEGEN_WIRING_FIX_PROMPT.md](artifacts/paper_figures_20261003_v1/E2C_CBH_IMAGEGEN_WIRING_FIX_PROMPT.md)<br>[FIGURE_NOTES.md](artifacts/paper_figures_20261003_v1/FIGURE_NOTES.md)<br>[MAIN_ARCHITECTURE_DRAW_PROMPT.md](artifacts/paper_figures_20261003_v1/MAIN_ARCHITECTURE_DRAW_PROMPT.md) |
| [pretrained_transformer_transfer_20260927_v1](artifacts/pretrained_transformer_transfer_20260927_v1/) | 2,820 | Transformer 迁移与 matched output 控制 | [PROTOCOL.md](artifacts/pretrained_transformer_transfer_20260927_v1/PROTOCOL.md)<br>[analysis/FINAL_ANALYSIS_CN.md](artifacts/pretrained_transformer_transfer_20260927_v1/analysis/FINAL_ANALYSIS_CN.md)<br>[analysis/PAPER_KEY_RESULTS_AND_CONTRIBUTION_GAP_CN.md](artifacts/pretrained_transformer_transfer_20260927_v1/analysis/PAPER_KEY_RESULTS_AND_CONTRIBUTION_GAP_CN.md) |
| [selected30_fewshot_baselines_20260920_v1](artifacts/selected30_fewshot_baselines_20260920_v1/) | 126,000 | 30 目标站独立少样本基线；63,000 组 | [PROTOCOL_CN.md](artifacts/selected30_fewshot_baselines_20260920_v1/PROTOCOL_CN.md)<br>[analysis/MASTER_EXPERIMENT_SUMMARY_CN.md](artifacts/selected30_fewshot_baselines_20260920_v1/analysis/MASTER_EXPERIMENT_SUMMARY_CN.md)<br>[analysis/REPORT_CN.md](artifacts/selected30_fewshot_baselines_20260920_v1/analysis/REPORT_CN.md) |
| [shared_all_station_baselines_20260922_v1](artifacts/shared_all_station_baselines_20260922_v1/) | 300 | 15 共享家族×10 seeds；主要强基线 | [PROTOCOL_CN.md](artifacts/shared_all_station_baselines_20260922_v1/PROTOCOL_CN.md)<br>[analysis/REPORT_CN.md](artifacts/shared_all_station_baselines_20260922_v1/analysis/REPORT_CN.md) |
| [single_station_baselines_sr_20260917_v1](artifacts/single_station_baselines_sr_20260917_v1/) | 0 | 单站命名基线与天气超分辨率探索 | [README_CN.md](artifacts/single_station_baselines_sr_20260917_v1/README_CN.md)<br>[analysis/REPORT_CN.md](artifacts/single_station_baselines_sr_20260917_v1/analysis/REPORT_CN.md) |
| [temporal_alignment_20260912](artifacts/temporal_alignment_20260912/) | 0 | 早期天气时间对齐 | [EXPERIMENT_MATRIX.md](artifacts/temporal_alignment_20260912/EXPERIMENT_MATRIX.md)<br>[MATRIX_RESULTS.md](artifacts/temporal_alignment_20260912/MATRIX_RESULTS.md) |
| [temporal_alignment_followup_20260913](artifacts/temporal_alignment_followup_20260913/) | 0 | 时间对齐后续资料 | 索引未列报告，直接进入目录查看 |
| [temporal_retrieval_20260913](artifacts/temporal_retrieval_20260913/) | 0 | 早期天气时序读取/检索 | [FINAL_REPORT_CN.md](artifacts/temporal_retrieval_20260913/FINAL_REPORT_CN.md)<br>[README_CN.md](artifacts/temporal_retrieval_20260913/README_CN.md)<br>[SCREEN_ANALYSIS_CN.md](artifacts/temporal_retrieval_20260913/SCREEN_ANALYSIS_CN.md) |
| [temporal_retrieval_confirmation_20260913](artifacts/temporal_retrieval_confirmation_20260913/) | 0 | 天气读取确认资料 | 索引未列报告，直接进入目录查看 |
| [temporal_sr_exploration_20260916](artifacts/temporal_sr_exploration_20260916/) | 0 | 高频天气代理前期探索 | [README_CN.md](artifacts/temporal_sr_exploration_20260916/README_CN.md)<br>[analysis/LITERATURE_REVIEW_CN.md](artifacts/temporal_sr_exploration_20260916/analysis/LITERATURE_REVIEW_CN.md)<br>[analysis/STATION_SELECTION_CN.md](artifacts/temporal_sr_exploration_20260916/analysis/STATION_SELECTION_CN.md)<br>[analysis/temporal_sr_research_report_CN.md](artifacts/temporal_sr_exploration_20260916/analysis/temporal_sr_research_report_CN.md) |
| [temporal_sr_named_models_20260917_v2](artifacts/temporal_sr_named_models_20260917_v2/) | 0 | 命名模型高频代理；正式与排除批次分开 | [README_CN.md](artifacts/temporal_sr_named_models_20260917_v2/README_CN.md)<br>[STATUS.md](artifacts/temporal_sr_named_models_20260917_v2/STATUS.md)<br>[analysis/FINAL_REPORT_CN.md](artifacts/temporal_sr_named_models_20260917_v2/analysis/FINAL_REPORT_CN.md)<br>[analysis/LITERATURE_SYNTHESIS_CN.md](artifacts/temporal_sr_named_models_20260917_v2/analysis/LITERATURE_SYNTHESIS_CN.md)<br>[analysis/METHOD_MATRIX_CN.md](artifacts/temporal_sr_named_models_20260917_v2/analysis/METHOD_MATRIX_CN.md)<br>[analysis/RESEARCH_AND_PROTOCOL_CN.md](artifacts/temporal_sr_named_models_20260917_v2/analysis/RESEARCH_AND_PROTOCOL_CN.md) |
| [temporal_sr_original_cbh_20260917_v1](artifacts/temporal_sr_original_cbh_20260917_v1/) | 0 | 高频代理接入原 CBH；未稳定改善 | [README_CN.md](artifacts/temporal_sr_original_cbh_20260917_v1/README_CN.md)<br>[STATUS.md](artifacts/temporal_sr_original_cbh_20260917_v1/STATUS.md)<br>[analysis/FINAL_REPORT_CN.md](artifacts/temporal_sr_original_cbh_20260917_v1/analysis/FINAL_REPORT_CN.md)<br>[analysis/LITERATURE_SYNTHESIS_CN.md](artifacts/temporal_sr_original_cbh_20260917_v1/analysis/LITERATURE_SYNTHESIS_CN.md)<br>[analysis/METHOD_MATRIX_CN.md](artifacts/temporal_sr_original_cbh_20260917_v1/analysis/METHOD_MATRIX_CN.md)<br>[analysis/RESEARCH_AND_PROTOCOL_CN.md](artifacts/temporal_sr_original_cbh_20260917_v1/analysis/RESEARCH_AND_PROTOCOL_CN.md) |
| [temporal_sr_unified_review_20260918_v1](artifacts/temporal_sr_unified_review_20260918_v1/) | 0 | 高频代理统一复核与负结果 | [README_CN.md](artifacts/temporal_sr_unified_review_20260918_v1/README_CN.md)<br>[analysis/UNIFIED_REPORT_CN.md](artifacts/temporal_sr_unified_review_20260918_v1/analysis/UNIFIED_REPORT_CN.md) |
| [trajectory_attribution_20260914_v1](artifacts/trajectory_attribution_20260914_v1/) | 260 | J/H/T/A 等早期轨迹归因 | [PROTOCOL.md](artifacts/trajectory_attribution_20260914_v1/PROTOCOL.md)<br>[analysis/ASSESSMENT_CN.md](artifacts/trajectory_attribution_20260914_v1/analysis/ASSESSMENT_CN.md)<br>[analysis/REPORT_CN.md](artifacts/trajectory_attribution_20260914_v1/analysis/REPORT_CN.md) |
| [trajectory_conditioned_correction_20260913](artifacts/trajectory_conditioned_correction_20260913/) | 0 | 早期轨迹条件修正资料 | 索引未列报告，直接进入目录查看 |
| [trajectory_review_20260915](artifacts/trajectory_review_20260915/) | 0 | 轨迹归因复核 | [REPORT_CN.md](artifacts/trajectory_review_20260915/REPORT_CN.md) |

精确追溯任意指标时，在 `RUN_METRICS.csv` 或 SQLite `observations` 读取 `experiment,run_id,model,seed,station,cohort,stage,budget,season,precision,split,evaluation_period,mae_key,metric_file,checkpoint,checkpoint_exists`。先按协议、精度、cohort、stage 分组，再汇总；`PROTOCOL_SUMMARY.csv` 的季节块没有自动合并为全部预算曲线。

<a id="older"></a>
## 14. 更早历史报告、论文与交付物

### 14.1 更早的 41 个历史报告目录

这些目录主要保留更早集群预测、区域交互、状态机制和初期单站探索的报告/表格，不保证全部中间缓存或权重可运行。它们不包含在上节“39 个当前 artifact/资料目录”计数中，不应用其 MW 指标替换当前 CF 主表。

- [aggregate_cluster_r2_state_v1](historical_reports/aggregate_cluster_r2_state_v1/)
- [aggregate_cluster_v1](historical_reports/aggregate_cluster_v1/)
- [aggregate_cluster_v1_smoke_v2](historical_reports/aggregate_cluster_v1_smoke_v2/)
- [aggregate_response_state_v1](historical_reports/aggregate_response_state_v1/)
- [aggregate_strict_modulation_v1](historical_reports/aggregate_strict_modulation_v1/)
- [aggregate_weather_dynamics_v1](historical_reports/aggregate_weather_dynamics_v1/)
- [audit_workspace_20260913](historical_reports/audit_workspace_20260913/)
- [baselines_mse_station_20260911](historical_reports/baselines_mse_station_20260911/)
- [conditional_forcing_validation_v1](historical_reports/conditional_forcing_validation_v1/)
- [cross_farm_context_v1](historical_reports/cross_farm_context_v1/)
- [cross_resolution_temporal_v2](historical_reports/cross_resolution_temporal_v2/)
- [dynamic_response_feasibility_20260913](historical_reports/dynamic_response_feasibility_20260913/)
- [ensemble_stabilization_v1](historical_reports/ensemble_stabilization_v1/)
- [explicit_interaction_matrix_v1](historical_reports/explicit_interaction_matrix_v1/)
- [farm_conference_analysis_cache](historical_reports/farm_conference_analysis_cache/)
- [farm_conference_runtime_v2](historical_reports/farm_conference_runtime_v2/)
- [farm_conference_v1](historical_reports/farm_conference_v1/)
- [farm_interaction_enhancement_v1](historical_reports/farm_interaction_enhancement_v1/)
- [farm_level_history_weather_v1](historical_reports/farm_level_history_weather_v1/)
- [final_route_pad5_v1](historical_reports/final_route_pad5_v1/)
- [horizon_control_matrix_v1](historical_reports/horizon_control_matrix_v1/)
- [incremental_mechanism_v2](historical_reports/incremental_mechanism_v2/)
- [incremental_predictability_v1](historical_reports/incremental_predictability_v1/)
- [level_shape_20260912](historical_reports/level_shape_20260912/)
- [local_state_cluster_transfer_v2](historical_reports/local_state_cluster_transfer_v2/)
- [local_state_identifiability_v2](historical_reports/local_state_identifiability_v2/)
- [local_state_operator_v1](historical_reports/local_state_operator_v1/)
- [local_state_spatial_interaction_v1](historical_reports/local_state_spatial_interaction_v1/)
- [local_state_transfer_multistage_v1](historical_reports/local_state_transfer_multistage_v1/)
- [local_to_global_raw_bilinear_v1](historical_reports/local_to_global_raw_bilinear_v1/)
- [multiscale_state_pad5_v1](historical_reports/multiscale_state_pad5_v1/)
- [paper_ablation_completion_v1](historical_reports/paper_ablation_completion_v1/)
- [paper_experiments_pad5_v1](historical_reports/paper_experiments_pad5_v1/)
- [physics_gate_20260912](historical_reports/physics_gate_20260912/)
- [regional_composition_v1](historical_reports/regional_composition_v1/)
- [results_20260912](historical_reports/results_20260912/)
- [state_forcing_diagnostics_frozen_v1](historical_reports/state_forcing_diagnostics_frozen_v1/)
- [state_forcing_diagnostics_v1](historical_reports/state_forcing_diagnostics_v1/)
- [state_forcing_ladder_v1](historical_reports/state_forcing_ladder_v1/)
- [station_clean_20260912](historical_reports/station_clean_20260912/)
- [station_followup_20260911](historical_reports/station_followup_20260911/)

补充文字导航：[exp_docs/](exp_docs/)、[docs/](docs/)、[早期总体实验总结](EXPERIMENT_SUMMARY_AND_NEXT_STEPS.md)、[早期研究方案](RESEARCH_PROBLEM_AND_TECHNICAL_PLAN.md)、[增量可预测性结果](INCREMENTAL_PREDICTABILITY_RESULTS.md)。这些保留形成过程，当前目标和贡献以本文为准。

项目外同级 [archives/](../archives/) 是早期归档，其中 `slast_full_incomplete` 明确不完整；唯一新工作入口仍是当前 `project/`，不在多份副本中分散新实验。

### 14.2 当前论文与图件

- 当前 TeX 主文件：[paper/tex/main.tex](paper/tex/main.tex)，说明：[paper/tex/README_CN.md](paper/tex/README_CN.md)，证据映射：[RESULTS_PROVENANCE_CN.md](paper/tex/RESULTS_PROVENANCE_CN.md)。
- 2026-10-18 已重写为单风场工作稿，含 CBH-SL、区域 E2C、消融、场景及迁移。**它不是仍未重写的旧 SLAST 稿，但也尚未同步当前“迁移退出主线＋三贡献”的最新决定。**
- 当前 PDF：[paper/tex/manuscript_cn.pdf](paper/tex/manuscript_cn.pdf)；交付包：[paper/delivery_single_farm_20261018/](paper/delivery_single_farm_20261018/)。
- 更新表格：[update_single_farm_tables.py](paper/tex/tools/update_single_farm_tables.py)；排版检查：[verify_manuscript.py](paper/tex/tools/verify_manuscript.py)；构建：[build.sh](paper/tex/build.sh)。现有脚本从 10 月 18 日重审目录取数，新增 10 月 19 日表格需显式接入，不能以重新编译代替正文更新。
- 作者信息、文献综述和参考文献仍待完善，详见 [AUTHOR_NOTES_CN.md](paper/tex/AUTHOR_NOTES_CN.md)。稿件不能视为可直接投稿终稿。
- 框架示意图是示意资产；数值图应使用 CSV/预测程序生成。后续改图必须按本文真实计算路径核对，尤其是插值先后、完整 memory、输出锚点和区域输入。

### 14.3 环境与整合审计

当前入口环境 Python 3.11、PyTorch 2.11.0+cu128；历史 V100 运行还使用过 PyTorch 2.5.1+cu121。版本变化与 FP16/BF16/FP32 会影响数值，不能保证逐位一致。

整合记录：[MERGE_STATUS.json](integration/MERGE_STATUS.json)、[COMPATIBILITY.json](integration/COMPATIBILITY.json)、[EXTENDED_COMPATIBILITY.json](integration/EXTENDED_COMPATIBILITY.json)、[FINAL_RECORD_AUDIT.json](integration/FINAL_RECORD_AUDIT.json)、[历史环境](integration/new_try_merge_20261018/SOURCE_ENVIRONMENTS.json)。已有 455 个代表 checkpoint 的严格载入/真实样本前向核验；这是已有整合验证，不是本次文档编写重新运行了全部检查。

<a id="handoff"></a>
## 15. 交接检查单与本次文档范围

后续助手开始具体任务时：

1. 先阅读本文第 1、7、11 节，确认当前任务是已有风场跨年单场预测；不要从旧报告恢复迁移或生成主线。
2. 查第 8 节确定实际加载源码、模型类和配置；检查 checkpoint、数据指纹、精度与输入设置。
3. 复用已有表格/权重。仅因需要新主表，不必重复训练 8,400 个 pilot 或 63,000 个目标基线。
4. 新方法/严格对照写入新 `runs/` 目录，记录代码/config/data 身份、训练与选模标签、checkpoint、完整预测和失败项；原始 `artifacts/` 保留证据。
5. 完成新结果后更新本入口的状态、比较边界、贡献证据与链接；未运行计划保持“待运行”，不写进完成结果。

本次文档汇总了当前主框架结果、历史扩展与负结果导航、39 个实验目录、41 个更早报告目录、代码/资产地图和三贡献计划。文档编写没有启动新训练，没有修改模型或原始实验记录。README 已链接本入口，后续阅读可从本文件开始。

文档校验：354 个链接目标有效，15 个内部导航锚点有效，八张主要结果表的 53 行数值与所链接 CSV 按展示精度一致；实验索引记录数合计 157,313。此项校验针对本文的链接与转录，不代替上文已有的原始指标和预测级审计。
