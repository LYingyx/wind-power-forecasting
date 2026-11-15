"""Architecture descriptions for the frozen 2026-10-21 independent round."""
CATALOG = []
def add(family, name, inputs, structure, condition='', output='', caveat='', source='code/models.py'):
    CATALOG.append(dict(family=family, name_cn=name, inputs=inputs, structure=structure,
                        condition=condition, output=output, caveat=caveat, source=source))

COMMON = ('历史[功率,相对时刻] Linear(2,d)→2层Transformer encoder(4头,FF=2d,ReLU,dropout=.05,norm_first)得到M；'
          'm为M的token均值；条件[d+3]→Linear(d)→GELU→LayerNorm并加逐时效偏置；'
          '天气经align_v/align_out两线性映射；[天气,条件]→Linear(2d,d)→GELU→Linear(d,d) adapter；'
          '查询=7维时间/站点/phase投影+天气+adapter；2层Transformer decoder(4头,FF=2d,ReLU,dropout=.05)交叉读取完整M；'
          'readout: d→d/2→GELU→1；adapter末层和readout末层零初始化。')
E2C = ('E2C天气：14物理天气量+本站地形高程转换为15维柱坐标，按训练统计标准化，'
       'Linear(15,d)→GELU→Linear(d,d)→GELU→Linear(d,d)。')
add('B0', '冻结历史参考模型', '仅本站24步功率、静态站点、时刻与phase；无天气/邻站',
    'forced_evolution/history_only,d=64：两套独立3层因果历史TCN(dilation=1/2/4)，两套均接收本站同一历史；'
    '站点embedding+时间MLP(5,64,64)/phase embedding；4头相对时间交叉注意力；'
    '[历史响应,第二TCN末态,站点token]→MLP(3d,2d,d)+LayerNorm；3个轨迹残差卷积块(dilation=1/2/4)；Conv1d(64,1)头；dropout=.1。',
    output='softplus(inverse_softplus(o)+delta)',
    caveat='内部旧变量名cluster仍只输入本站历史。softplus是结构内映射；B0独立训练后对同站同seed修正模型冻结。存储总参数包含未激活模块。',
    source='code/reference_original.py; code/vendor/wind_forcing/models/forced_evolution.py; code/vendor/wind_forcing/models/components.py; code/train.py')
arms = [
 ('E2C_CBH','完整轨迹/历史条件修正','[m,b,Δb,lead/6]','b+c',E2C,'完整条件主模型；不能预设为最优。'),
 ('CB','仅轨迹条件修正','[0,b,Δb,lead/6]','b+c',E2C,'只去掉额外条件摘要m；decoder仍读取完整历史M。'),
 ('CH','仅历史摘要条件修正','[m,0,0,lead/6]','b+c',E2C,'去掉条件中的b、Δb，输出仍使用冻结B0。'),
 ('CNEUTRAL','中性条件+B0锚点','[0,0,0,lead/6]','b+c',E2C,'保留lead条件、逐lead偏置、adapter和完整历史decoder；并非无条件网络。'),
 ('DIRECT','直接输出','[m,b,Δb,lead/6]','c',E2C,'B0仍进入条件；输出初始为0，故比较同时改变输出参数化及优化起点。'),
 ('NEUTRAL_LAST','中性条件+末值锚点','[0,0,0,lead/6]','o+c',E2C,'有效前向不使用B0数值，但仍保留共同的冻结参考模块/生成流程。'),
 ('E0','原始天气线性编码','[m,b,Δb,lead/6]','b+c','E0天气：标准化原始14通道→Linear(14,d)。','与E2C同时改变坐标、地形输入、非线性、深度和参数量，不能归因于单一因素。'),
 ('CLIMATOLOGY','训练期均值天气对照','[m,b,Δb,lead/6]','b+c',E2C,'输入天气换为本站训练期逐lead天气均值；不是无天气通路。均值是模型输入定义，不是跨seed结果平均。'),
]
for family,name,cond,out,weather,note in arms:
    add(family,name,'本站24步功率、24×14天气、静态站点、高程、phase；同站同seed冻结B0',
        COMMON+weather,cond,out,note+' P0I为空间固定双线性采样、native前7节点时间插值；单token注意力权重恒1，不能宣称有效query/key检索。',
        'code/models.py:CachedCorrection/correction; code/column_original.py:WeatherProjection; code/vendor/wind_forcing/models/p0i_transfer.py')
add('TRANSFORMER_FC','Transformer未来协变量适配版','历史功率+相对时刻、未来14天气+4维lead/phase',
    '2层历史encoder+2层未来decoder，4头、FF=2d、GELU、dropout=0；sinusoidal位置编码；[decoder,o]→MLP(d+1,d,1)。',output='o+MLP',source='code/baselines_original.py:TransformerFutureCov')
add('TIDE_FC','TiDE未来协变量适配版','历史功率+相对时刻、未来天气+lead/phase',
    '历史48→d→d；未来432→2d→2d；拼接3d后3个残差MLP块(3d,3d,3d)；decoder(3d,2d,24)；中心化历史Linear(24,24) skip。',output='o+history_skip+decoder',caveat='紧凑TiDE风格，无官方完整temporal decoder。',source='code/baselines_original.py:TiDEFutureCov')
add('NBEATSX_FC','N-BEATSx未来协变量适配版','中心化历史24、天气336、phase1',
    '4个backcast/forecast块；各块361→d→d→d→48，ReLU；前24维从剩余历史扣除，后24维累加；每块重新接收外生天气。',output='o+四块forecast和',caveat='generic块，无interpretable basis stack。')
add('TIMEXER_FC','TimeXer未来协变量适配版','24历史、24×14未来天气；不显式用phase',
    '历史实例归一化→6个长度4不重叠patch Linear(4,d)+位置编码+1个global token；14个外生变量token各Linear(24,d)；'
    '2轮内生self-attention(4头,FF2d,GELU,dropout0)+global token对外生token的4头cross-attention/LayerNorm；展平Linear(7d,24)。',output='预测反归一化')
add('TFT_FC','TFT未来协变量适配版','历史功率+相对时刻、14天气；不显式用phase',
    '14个天气变量向量嵌入，14→d→14/ELU/softmax变量选择；历史Linear(2,d)→2层LSTM；历史state初始化2层未来LSTM；'
    '4头因果attention读取48个历史/未来token；两组GLU门控残差+LayerNorm，含d→2d→d/ELU；Linear(d,1)头。',output='o+head',caveat='无完整官方静态GRN/变量选择；点MAE训练。')
add('TCN_FC','TCN未来协变量适配版','全部24步历史+时刻、天气、lead、phase',
    '历史Conv(2,d,k3,dil1)→GELU→Conv(d,d,k3,dil2)→GELU→展平24d→d；'
    '天气14+历史摘要d+lead+phase经1×1Conv→d；4个残差块Conv(k3,dil1/2/4/8)→GELU→Conv(k1)；1×1头。',output='o+head',caveat='已发布未来天气允许双向时序卷积。')
add('DLINEAR_FC','DLinear未来协变量适配版','24历史、全未来336天气、phase',
    '历史5点复制边界滑动均值分解trend/season；两条Linear(24,24)；天气+phase Linear(337,24)；三项求和。',output='trend+season+weather',caveat='纯线性，无额外末值锚点；config.width字段不参与结构。')
add('GRU_H','纯历史GRU','历史功率+相对时刻；无天气/phase',
    'Linear(2,d)→2层GRU；末态拼o→MLP(d+1,d,24)。',output='o+MLP',source='code/baselines_original.py:HistoryLSTM')
add('PATCHTST_H','纯历史PatchTST适配版','24历史；无天气/phase',
    '历史实例归一化→6个长度4不重叠patch Linear(4,d)+位置编码；3层Transformer encoder(4头,FF2d,GELU,dropout0)；展平Linear(6d,24)。',output='预测反归一化')
add('CNN_W','纯天气CNN','24×14未来天气；无功率历史/phase',
    'Conv(14,d,k3)后直接接Conv(d,d,k5)→GELU；Conv(d,d,k3,dil2)→GELU；Conv(d,d,k3,dil4)→GELU；Conv(d,1,k1)。',output='head',source='code/baselines_original.py:CNNWeather')
add('PERSISTENCE','末值持续性','末次功率o','未来24步均重复o；没有训练。',output='o',caveat='确定性结果，无随机种子；不能复制十遍计作十次训练。',source='code/analyze.py')
add('SELECTED_BASELINE','E10：验证期选定基线','由该站选定family决定','不是新网络；沿用原2023验证协议冻结的每站最优family，并取它每个seed的原始结果。',
    output='实际family输出',caveat='旧选择协议使用2023跨seed验证均值；本次不改选、不呈现该均值、不计作新增训练。actual_family和source_job_id可追溯。',source='analysis/validation_selected_baselines.csv')

BASELINES = ['TRANSFORMER_FC','TIDE_FC','NBEATSX_FC','TIMEXER_FC','TFT_FC','TCN_FC','DLINEAR_FC','GRU_H','PATCHTST_H','CNN_W']
ORDER = ['B0','E2C_CBH','CB','CH','CNEUTRAL','DIRECT','NEUTRAL_LAST','E0','CLIMATOLOGY']+BASELINES+['SELECTED_BASELINE','PERSISTENCE']
DESIGNS = [
 ('D01','逐次天气替代训练均值天气','E2C_CBH','CLIMATOLOGY'),
 ('D02','在无摘要条件下加入轨迹条件','CB','CNEUTRAL'),
 ('D03','在有摘要条件下加入轨迹条件','E2C_CBH','CH'),
 ('D04','在无轨迹条件下加入历史摘要','CH','CNEUTRAL'),
 ('D05','在有轨迹条件下加入历史摘要','E2C_CBH','CB'),
 ('D06','去掉条件通路的历史摘要','CB','E2C_CBH'),
 ('D07','由残差改为直接输出','DIRECT','E2C_CBH'),
 ('D08','中性条件下改用末值锚点','NEUTRAL_LAST','CNEUTRAL'),
 ('D09','原始线性编码改为E2C编码','E2C_CBH','E0'),
 ('D10','E2C编码改回原始线性编码','E0','E2C_CBH'),
]
COMPARISONS = DESIGNS + [(f'B{i:02d}','完整模型对比'+f,'E2C_CBH',f) for i,f in enumerate(['B0']+BASELINES+['SELECTED_BASELINE','PERSISTENCE'],1)]
