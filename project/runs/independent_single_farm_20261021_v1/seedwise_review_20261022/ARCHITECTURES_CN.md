# 实验缩写与实际结构（2026-10-21 独立训练轮次）

下列名称依据本轮冻结源码，不依据名称推测官方结构。所有命名基线都是任务适配版，不是官方复现。

## 记号与共同协议

h：本站过去24个15分钟功率点；o：最后观测；M：历史编码token；m：M沿token维的均值；b：同站同seed冻结B0的24步预测；Δb=b−[o,b前23步]；c：decoder读出。输出24步（0.25–6小时），功率以本站容量归一化为CF。

2022训练，2023选模型，2024跨年复评（非未查看的盲测）。各站模型独立训练；FP32且无TF32；AdamW+余弦退火，最多80epoch，patience=10，batch=256，梯度裁剪1，无输出后处理裁剪。实际width/lr/weight_decay、参数量、最优epoch见逐job配置表；调参width可能不同于正式配置。

E2C的15维物理柱坐标依次为：msl,u10,v10,t2m,高程,u1000−u10,v1000−v10,θ1000,q1000,z1000/g−高程,u925−u1000,v925−v1000,θ925−θ1000,q925−q1000,(z925−z1000)/g，其中θ925=t925×(1000/925)^0.2854。坐标使用训练统计标准化。E2C使用的原有统计是训练期统计，不代表所有预处理统计都按站重新估计。

## 缩写结构字典

| family | name_cn | inputs | condition | output | structure | caveat | source |
|---|---|---|---|---|---|---|---|
| B0 | 冻结历史参考模型 | 仅本站24步功率、静态站点、时刻与phase；无天气/邻站 |  | softplus(inverse_softplus(o)+delta) | forced_evolution/history_only,d=64：两套独立3层因果历史TCN(dilation=1/2/4)，两套均接收本站同一历史；站点embedding+时间MLP(5,64,64)/phase embedding；4头相对时间交叉注意力；[历史响应,第二TCN末态,站点token]→MLP(3d,2d,d)+LayerNorm；3个轨迹残差卷积块(dilation=1/2/4)；Conv1d(64,1)头；dropout=.1。 | 内部旧变量名cluster仍只输入本站历史。softplus是结构内映射；B0独立训练后对同站同seed修正模型冻结。存储总参数包含未激活模块。 | code/reference_original.py; code/vendor/wind_forcing/models/forced_evolution.py; code/vendor/wind_forcing/models/components.py; code/train.py |
| E2C_CBH | 完整轨迹/历史条件修正 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [m,b,Δb,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 完整条件主模型；不能预设为最优。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| CB | 仅轨迹条件修正 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [0,b,Δb,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 只去掉额外条件摘要m；decoder仍读取完整历史M。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| CH | 仅历史摘要条件修正 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [m,0,0,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 去掉条件中的b、Δb，输出仍使用冻结B0。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| CNEUTRAL | 中性条件+B0锚点 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [0,0,0,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 保留lead条件、逐lead偏置、adapter和完整历史decoder；并非无条件网络。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| DIRECT | 直接输出 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [m,b,Δb,lead/6] | c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | B0仍进入条件；输出初始为0，故比较同时改变输出参数化及优化起点。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| NEUTRAL_LAST | 中性条件+末值锚点 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [0,0,0,lead/6] | o+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 有效前向不使用B0数值，但仍保留共同的冻结参考模块/生成流程。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| E0 | 原始天气线性编码 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [m,b,Δb,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E0天气：标准化原始14通道→Linear(14,d)。 | 与E2C同时改变坐标、地形输入、非线性、深度和参数量，不能归因于单一因素。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| CLIMATOLOGY | 训练期均值天气对照 | 本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0 | [m,b,Δb,lead/6] | b+c | 历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；m为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。 | 输入天气换为本站训练期逐lead天气均值；不是无天气通路。均值是模型输入定义，不是跨seed结果平均。 P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。 | code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py |
| TRANSFORMER_FC | Transformer未来协变量适配版 | 历史功率+相对时刻、未来14天气+4维lead/phase |  | o+MLP | 2层历史encoder+2层未来decoder，4头、FF=2d、GELU、dropout=0；sinusoidal位置编码；[decoder,o]→MLP(d+1,d,1)。 |  | code/baselines_original.py:TransformerFutureCov |
| TIDE_FC | TiDE未来协变量适配版 | 历史功率+相对时刻、未来天气+lead/phase |  | o+history_skip+decoder | 历史48→d→d；未来432→2d→2d；拼接3d后3个残差MLP块(3d,3d,3d)；decoder(3d,2d,24)；中心化历史Linear(24,24) skip。 | 紧凑TiDE风格，无官方完整temporal decoder。 | code/baselines_original.py:TiDEFutureCov |
| NBEATSX_FC | N-BEATSx未来协变量适配版 | 中心化历史24、天气336、phase1 |  | o+四块forecast和 | 4个backcast/forecast块；各块361→d→d→d→48，ReLU；前24维从剩余历史扣除，后24维累加；每块重新接收外生天气。 | generic块，无interpretable basis stack。 | code/models.py |
| TIMEXER_FC | TimeXer未来协变量适配版 | 24历史、24×14未来天气；不显式用phase |  | 预测反归一化 | 历史实例归一化→6个长度4不重叠patch Linear(4,d)+位置编码+1个global token；14个外生变量token各Linear(24,d)；2轮内生self-attention(4头,FF2d,GELU,dropout0)+global token对外生token的4头cross-attention/LayerNorm；展平Linear(7d,24)。 |  | code/models.py |
| TFT_FC | TFT未来协变量适配版 | 历史功率+相对时刻、14天气；不显式用phase |  | o+head | 14个天气变量向量嵌入，14→d→14/ELU/softmax变量选择；历史Linear(2,d)→2层LSTM；历史state初始化2层未来LSTM；4头因果attention读取48个历史/未来token；两组GLU门控残差+LayerNorm，含d→2d→d/ELU；Linear(d,1)头。 | 无完整官方静态GRN/变量选择；点MAE训练。 | code/models.py |
| TCN_FC | TCN未来协变量适配版 | 全部24步历史+时刻、天气、lead、phase |  | o+head | 历史Conv(2,d,k3,dil1)→GELU→Conv(d,d,k3,dil2)→GELU→展平24d→d；天气14+历史摘要d+lead+phase经1×1Conv→d；4个残差块Conv(k3,dil1/2/4/8)→GELU→Conv(k1)；1×1头。 | 已发布未来天气允许双向时序卷积。 | code/models.py |
| DLINEAR_FC | DLinear未来协变量适配版 | 24历史、全未来336天气、phase |  | trend+season+weather | 历史5点复制边界滑动均值分解trend/season；两条Linear(24,24)；天气+phase Linear(337,24)；三项求和。 | 纯线性，无额外末值锚点；config.width字段不参与结构。 | code/models.py |
| GRU_H | 纯历史GRU | 历史功率+相对时刻；无天气/phase |  | o+MLP | Linear(2,d)→2层GRU；末态拼o→MLP(d+1,d,24)。 |  | code/baselines_original.py:HistoryLSTM |
| PATCHTST_H | 纯历史PatchTST适配版 | 24历史；无天气/phase |  | 预测反归一化 | 历史实例归一化→6个长度4不重叠patch Linear(4,d)+位置编码；3层Transformer encoder(4头,FF2d,GELU,dropout0)；展平Linear(6d,24)。 |  | code/models.py |
| CNN_W | 纯天气CNN | 24×14未来天气；无功率历史/phase |  | head | Conv(14,d,k3)后直接接Conv(d,d,k5)→GELU；Conv(d,d,k3,dil2)→GELU；Conv(d,d,k3,dil4)→GELU；Conv(d,1,k1)。 |  | code/baselines_original.py:CNNWeather |
| PERSISTENCE | 末值持续性 | 末次功率o |  | o | 未来24步均重复o；没有训练。 | 确定性结果，无随机种子；不能复制十遍计作十次训练。 | code/analyze.py |
| SELECTED_BASELINE | E10：验证期选定基线 | 由该站选定family决定 |  | 实际family输出 | 不是新网络；沿用原2023验证协议冻结的每站最优family，并取它每个seed的原始结果。 | 旧选择协议使用2023跨seed验证均值；本次不改选、不呈现该均值、不计作新增训练。actual_family和source_job_id可追溯。 | analysis/validation_selected_baselines.csv |

## 正式超参数

| family | selected_config_index | width | lr | weight_decay | selection_split |
|---|---|---|---|---|---|
| CNN_W | 2 | 128 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| DLINEAR_FC | 2 | 64 | 0.00020000000000000001 | 0.01 | 2023_validation_only |
| E2C_CBH | 0 | 64 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| GRU_H | 2 | 128 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| NBEATSX_FC | 0 | 64 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| PATCHTST_H | 0 | 64 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| TCN_FC | 3 | 128 | 0.001 | 0.0001 | 2023_validation_only |
| TFT_FC | 1 | 64 | 0.001 | 0.0001 | 2023_validation_only |
| TIDE_FC | 2 | 128 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| TIMEXER_FC | 2 | 128 | 0.00020000000000000001 | 0.0001 | 2023_validation_only |
| TRANSFORMER_FC | 1 | 64 | 0.001 | 0.0001 | 2023_validation_only |

B0固定d=64、lr=0.0002、weight_decay=0.0001；所有CBH消融继承E2C_CBH正式配置。DLINEAR_FC的width字段不参与模型结构。所有调参试验的实际配置另见all_tuning_results.csv。

total_parameters是模块全部参数；trainable_parameters是requires_grad参数；active_gradient_parameters是训练结束时实际存在梯度的参数数目。CBH检查点含冻结B0及未激活源模块，三项不可混称“有效参数量”。seconds_training仅对应该job阶段，CBH阶段不含前置B0成本；不能把它当总部署成本。

注意：DIRECT/NEUTRAL_LAST保存的*_correction.npy定义为最终预测−B0，并非这里的原始读出c；解读输出消融请按上述公式。
