# 独立单风场：自适应天气提取与历史状态受控实验

日期：2026-10-22。用户已授权实现并执行修订方案。当前运行状态以[STATUS.json](STATUS.json)为准，全部结束必须有[FINISHED.json](FINISHED.json)及通过的完成审计，不能把已排队写成已完成。

## 入口

- [2026-10-23存储去重与配额恢复](STORAGE_POLICY_20261023_CN.md)：评价输出移到SSD3，原路径继续有效；相同预测/置换索引共享实际文件，注意力无损压缩，清理废弃临时文件。
- [冻结训练协议](PROTOCOL.json)、[任务队列](queue.sqlite)、[实时结果报告](analysis/REPORT_CN.md)
- [逐站逐seed Excel](analysis/EXPERIMENT_RESULTS_PER_SEED.xlsx)、[原始逐次结果](analysis/results_per_farm_seed.csv)
- [模型验收](PREFLIGHT.json)、[数据视图审计](prepared/AUDIT.json)、[断点恢复检查](RESUME_CHECK.json)、[流水线检查](PIPELINE_CHECK.json)、[分析检查](ANALYSIS_CHECK.json)、[并发基准](CONCURRENCY_BENCHMARK.json)
- [执行计划与结果解释](../../ADAPTIVE_WEATHER_EXECUTION_20261022_CN.md)

## 科学问题与边界

远时效骨干成绩接近只构成动机，未证明瓶颈一定在天气提取。主指标为全时效MAE；同时报告RMSE、0–1/1–3/3–6小时、场景、失败风场。任何严格正改善均记录，不设百分比或等效容差门槛。结果按风场×seed分别展示；调参内部沿用站点/seed/匹配arm等权2023 MAE，不用该汇总数替代正式实测值。

2022训练，2023选择，2024仅跨年复评，不能重新称盲测。训练worker从不加载2024；评价worker与Data入口都要求ALL_SELECTIONS_FROZEN.json。测试视图只在该文件生成后准备。六个代表站沿用训练描述量选定站，调参seed40/41，正式42–46；扩展47–51。FP32/no-TF32、AdamW、80epoch、patience10、batch256、梯度裁剪1、余弦退火、无后处理裁剪。最优epoch用2023 MAE选择；新训练采用float64误差归约，明确区别于旧训练的部分float32归约，保存预测后审计同为float64。

## 实际结构

公共主骨干沿用旧NEUTRAL_LAST：本站24点历史经两层Transformer encoder形成完整memory；E2C列坐标天气编码与时间/位置/phase上下文形成查询；两层Transformer decoder读取memory；两层GELU读出c，最终输出最后观测o+c。中性条件仍保留lead条件、偏置和adapter；不存在学习型空间/时间检索的旧单token部分只保留v/out线性映射。旧完整结构见上一轮seedwise_review的ARCHITECTURES_CN.md。

前端对全部骨干统一输出[B,24,14]标准化天气值。新前端独立RNG初始化，不改变骨干初值；输出为point+tanh(gate)×(candidate−point)，gate初始0，初始化时与W0天气完全相同。gate实际数值在逐job指标front_gate_tanh中记录，注意力不能脱离gate解释。

空间与时间评分均为单头宽度32：query把24历史+3个lead特征+phase共28维映射28→32→32，GELU；空间key为14天气+二维几何+节点时刻共17维，时间key为14天气+节点时刻共15维，分别输入→32→32，GELU；最后key线性无bias，避免softmax中恒抵消的共享偏置。softmax(query·key/sqrt(32))生成权重；value是对应样本原始14通道天气。历史固定描述为本站训练历史逐位置均值；天气固定描述为本站训练期唯一cycle逐节点逐格点均值，仅使用2022。权重生成器无dropout。

静态权重可依赖lead、节点时间、phase以及训练期固定描述，不依赖当前样本的历史或天气内容。所有value仍来自当前样本。静态与自适应对照的模块、参数规模、初始化、输入邻域、训练预算相同，不增加无效参数凑数。相同参数规模不代表两种函数具有相同的样本条件自由度；该差异正是被检验的因素。

| arm | 当前结构/输入差别 |
|---|---|
| W0 | 固定双线性取点，按真实时刻线性插值；原NEUTRAL_LAST；前五seed复用旧权重 |
| S1 | 每格点先按时刻插值；逐lead邻域展平14k²→32→14，GELU |
| S2 | 空间注意力；query历史和key天气取训练固定描述，value为实际天气 |
| S3 | 空间注意力；query历史、key天气均为当前样本 |
| S4 | 与S3同规模；几何(dx,dy)换成沿上风向/横风向坐标；风矢量用反标准化u10/v10，低风速分母clamp1e-3 |
| T1_7/T1_8 | 点位置七/八原生节点时间注意力；权重用训练固定描述和时间信息 |
| T2_7/T2_8 | 与T1同结构，权重改为当前历史与天气；七/八节点单独配对 |
| ST00 | 逐目标lead先空间汇聚原生节点，再时间汇聚；两种权重均静态 |
| ST10 | 空间自适应、时间静态 |
| ST01 | 空间静态、时间自适应 |
| ST11 | 空间与时间均自适应 |
| ST_NO_H | 两种权重使用当前天气、训练固定历史；解码器仍读取真实历史 |
| H1 | 旧CB条件[0,b,Δb,lead]，输出改为o+c；b是同站同seed旧冻结B0预测缓存 |
| H2 | 与W0同结构，历史encoder输入换为训练固定历史；decoder层全部保留；o仍为当前末次观测 |
| CLIM_LAST | W0天气改为本站训练期逐lead均值天气 |
| H2_CLIM | 同时使用固定历史和训练均值天气；与W0/H2/CLIM_LAST构成2×2 |
| TRANSFORMER_FC__S1/ST00/ST11 | 原Transformer未来协变量适配骨干，d64/lr0.001，接对应前端 |
| TIDE_FC__S1/ST00/ST11 | 原TiDE未来协变量适配骨干，d128/lr0.0002，接对应前端 |

ST的时间评分key始终来自固定双线性点天气，不来自前一级空间汇聚结果，避免ST10中空间自适应使“静态时间权重”间接随样本变化。时间权重对应的value仍为前一级空间汇聚值，因此实际输出实现分解时空汇聚。

## 选型与任务数

ST00/ST11共同搜索k=3/5/7和前端lr=0.0002/0.001，6站×2seed，共144任务；选择后锁定全局k。SPACE组S2/S3共享lr选择（48任务），S4沿用；S1单独选择lr（24）；TIME四组共同选择lr（96）；Transformer/TiDE各对S1/ST00/ST11共同选择lr（各72）。其他ST组沿用ST选择；H组与W0保持旧超参数。相同分数时取更小k、更低lr。共456次调参。

23个新增arm×56站×5seed=6440次正式训练。W0、ST00、ST11及GSTAR各补五seed，共1120次扩展。GSTAR从Transformer/TiDE的S1/ST00四组中按调参时2023分数选定，平局按arm名称排序。共8016次新增训练；前五seed旧W0不重复训练。

训练结束的非调参job生成7560个评价任务，另加旧W0的280个干预评价任务，总7840个评价任务。正式42–46中所有气象前端及W0做五次天气交换；有注意力的前端再做五次权重交换。扩展47–51与H组仅标准评价。所有对比保留实际可用共同seed，不补值。

## 数据、干预和解释

缓存原始天气[cycle,8,14,24,25]不变；新视图按最邻近格点中心取真正7×7，3/5×5为其中心子块，全部56站均在边界内。实际时间offset随phase变化，范围从0到7.75小时；严格通过cycle映射及valid_unix−origin校验。前七节点已经覆盖6小时目标区间，线性W0中额外第八节点不改变预测值；注意力是否受益由T1/T2七八节点对照回答。

天气/权重交换在同站、同月、同phase及相同NWP age组内构造无放回错排，并排除同日、同weather cycle，固定随机数1001–1005，不读标签决定交换。空间和时间权重一起交换，当前天气值保持原样。原始/干预预测、donor索引、分时效和场景指标完整保存；静态权重在相同时间条件下应不受权重交换影响。

每个注意力模型保存32个固定等距起报样本的权重，以及全部起报×lead的空间重心、时间期望/峰值、u10/v10诊断数组。attention_diagnostics.npy最后一维依次为dx_km,dy_km,time_expected_hours,time_peak_hours,u10,v10；无对应维度的模型该部分为0，需结合arm解读。注意力/风向相关性只作旁证，不作为物理因果证明。

自动分析按模型×风场×seed，分别输出全年/各季节及全时效/三个时效段的上风向投影、重心与风速关系、所读节点与目标时刻差。这里只在同一次模型的起报样本与lead内求描述统计，不混合风场或seed。注意力描述candidate分支，最终输入还受tanh(gate)控制；gate允许为负，不能把权重重心直接解释为最终输入的正向物理贡献。门控实测值保存在actual_configs_and_costs.csv及注意力表中。

H2变差只证明历史有用；历史×天气交互另逐seed计算：(CLIM_LAST−W0)−(H2_CLIM−H2)。其方向不预设。ST11对ST_NO_H才直接回答在提取权重中加入历史状态的增量价值。任何单一实例改善不自动推广为普遍优势。

## 运行与恢复

调度器使用SQLite事务、进程身份校验、训练断点、三次重试、失败依赖可见化和独立supervisor。训练与评价分开标记。调度全8卡，按并发基准上限、空闲显存和实时外部负载放任务，保留4GiB；初始无本项目任务且利用率≥95%的卡暂缓新任务，每3秒重查。OOM只降低本项目并发，不终止外部任务。

开始前通过24种结构梯度/隔离检查、旧W0预测一致性、分段RMSE检查、模型保存恢复、真实短训练、11组验证期干预、8016任务DAG模拟和全验证期错排检查。训练源码冻结后不得改动；分析/调度修复单独记录，不篡改指标。

每十分钟自动刷新分析。最终检查所有8016权重、全部实际和干预预测、MAE/RMSE重算、配置冻结指纹及source哈希，全部通过才写FINISHED。

## 自动结果出口

- results_per_farm_seed.csv：每个风场、seed、年份的实际MAE/RMSE，保留调参与正式阶段标签。
- paired_actual_results.csv及paired_horizons/paired_leads/paired_scenarios_per_seed：共同风场、seed、年份和切片下的候选/参考原值；gain=参考误差−候选误差。
- regressions_per_farm_seed.csv：MAE或RMSE任一变差的全部配对，保留另一指标，展示指标间取舍。
- intervention_results_per_seed.csv及intervention_slices_per_seed.csv.gz：每次置换的实测值与匹配值、分时效/场景差，不平均五次置换。
- attention_associations_per_seed.csv：上述物理关联诊断，按正式评价完成情况增量生成。
- CONTRIBUTION_EVIDENCE_CN.md：按年份和seed列改善/退步/持平的风场数；明确配对是否已覆盖56站，未完成时不提前确定贡献结论。

以上文件位于analysis目录；大表采用gzip CSV，主要表另存Excel。分析进程有独立锁，CSV与Excel原子替换。分析检查覆盖极小正改善、缺失场景不计持平、共同seed配对、上风向符号、时间差及真实验证期注意力数组。
