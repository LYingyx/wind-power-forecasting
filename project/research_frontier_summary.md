# 省域风电集群超短期预测研究前沿综述

## 研究场景概述

**研究对象：** 省域场站集群总功率/单场站超短期预测（未来6h，15min分辨率）

**数据源：**
- 15min分辨率历史功率数据
- 再分析数据插值得到的场站气象
- 气象大模型预测的1h分辨率区域格点气象（0.25°分辨率）

**核心挑战：**
1. **时间分辨率不对齐**：历史功率15min vs NWP预测1h
2. **空间分辨率不对齐**：0.25°（约28km）格点气象 vs 场站点位功率

---

## 当前研究边界（2024-2025）

### 1. 分辨率处理能力

#### 时间分辨率
- **主流水平**：1h → 10-15min的时间降尺度
- **技术路线**：
  - 时间超分辨率网络（Temporal Super-Resolution）
  - seq2seq架构 + temporal refinement module
- **代表工作**：Nature Scientific Reports (2025) - PatchTST-GRU模型实现了NWP时间尺度细化

#### 空间分辨率
- **主流水平**：9km → 1km的空间降尺度
- **你的挑战**：28km（0.25°）→ 点位场站（更大尺度跨越）
- **技术路线**：
  - 深度学习空间超分辨率（CNN-based, GAN-based, Diffusion-based）
  - 可学习的几何插值网络
  - 多格点融合 + 空间注意力机制

### 2. 预测精度前沿

#### 超短期预测（0-6h）
- **15min分辨率NRMSE**：10-22%（标准化）
  - 基于NWP偏差校正：22% (arXiv 2024)
  - 省域集群时空建模：相比baseline改善38-44% (MDPI 2025)
- **关键技术**：
  - NWP + SCADA历史数据融合
  - 持续学习机制校正系统性偏差
  - 时空联合建模

#### 省域/区域集群预测
- **空间范围**：134个风机 → 省域多场站
- **建模方法**：
  - Graph Neural Networks (GNN)捕捉场站间空间依赖
  - 多点位NWP信息融合
  - 时空对比学习（Spatiotemporal Contrastive Learning）

### 3. 方法论前沿

#### 趋势1：Foundation Model（基础模型）适配
**代表工作：**
- **WindFM** (arXiv 2025)：风电专用基础模型
  - 在150B时间步、126K站点上预训练
  - 零样本泛化到新场站
  - 离散化-生成框架（discretize-and-generate）
- **Weather Foundation Model for Power Grid** (arXiv 2025)
  - 气象基础模型针对电网应用定制
  - 温度MAE降低15%，降水MAE降低35%，风速MAE降低15%

**技术特点：**
- 时间序列tokenization
- Decoder-only Transformer架构
- 大规模预训练 + 下游任务fine-tuning

#### 趋势2：异构数据融合（Heterogeneous Data Fusion）
**核心挑战：** 点数据（SCADA）+ 格点数据（NWP）+ 再分析数据

**技术路线：**
- **早期融合**：特征层面融合多源数据
- **注意力加权**：自适应学习不同数据源在不同预测时刻的可信度
  - 前1-2h：历史功率持续性更可靠
  - 3-6h：NWP预测优势显现
- **多模态编码器**：为每种数据源设计专门的编码器

**代表工作：**
- Fourier Geometric Wind Power Forecasting (arXiv 2024)
- Artificial Intelligent Power Forecasting Based on Multi-Source Data Fusion (MDPI 2023)

#### 趋势3：Physics-Informed + Data-Driven 混合方法
**物理约束：**
- 风电功率曲线约束
- 功率变化率约束（避免不合理跳变）
- 大气边界层物理先验
- 能量守恒约束

**实现方式：**
- 损失函数中加入物理约束项
- 网络架构嵌入物理知识（如功率曲线层）
- 物理模型后处理（统计-动力结合）

### 4. 时空对齐技术路线

#### 方法1：NWP Refinement Module（推荐）
**来源：** Nature Scientific Reports (2025)

**架构：**
```
低分辨率NWP → Spatiotemporal Scaling Network → 高分辨率气象
                ↓
            [空间细化] + [时间细化]
                ↓
        场站级 + 15min分辨率
```

**优势：**
- 端到端可学习
- 同时处理时空两个维度
- 可以为decoder提供未来气象趋势指导

#### 方法2：多格点融合 + 时间超分辨率（两阶段）
**空间阶段：**
```
提取场站周边N×N格点 → GNN/Spatial Attention → 场站气象
```

**时间阶段：**
```
1h NWP序列 → Temporal Super-Resolution Network → 15min序列
```

**代表技术：**
- Graph Neural Networks
- Spatial Attention Mechanism
- GAN/Diffusion-based Super-Resolution

#### 方法3：几何插值 + 物理约束
**空间插值：**
- 输入：{格点坐标、格点值、场站坐标}
- 输出：场站位置的气象值
- 技术：可学习的几何插值器（非简单双线性插值）

**时间细化：**
- 物理约束超分辨率
- 保证生成序列的功率变化率合理性

---

## 核心文献推荐

### 顶刊文献

#### 1. A PatchTST-GRU based heterogeneous seq2seq model with NWP refinement for multi-step wind power forecasting
- **期刊：** Nature Scientific Reports (2025)
- **DOI/链接：** https://www.nature.com/articles/s41598-025-00741-9
- **核心贡献：** NWP refinement模块实现时空尺度变换
- **相关度：** ★★★★★（直接解决时空双重不对齐）

#### 2. Fourier Geometric Wind Power Forecasting with Numerical Weather Prediction
- **平台：** arXiv (2024)
- **链接：** https://arxiv.org/html/2607.17095v1
- **核心贡献：** 异构数据（点+格点）融合框架
- **相关度：** ★★★★★（空间对齐专门处理）

#### 3. WindFM: An Open-Source Foundation Model for Zero-Shot Wind Power Forecasting
- **平台：** arXiv (2025)
- **链接：** https://arxiv.org/pdf/2509.06311
- **核心贡献：** 风电基础模型，150B时间步预训练
- **相关度：** ★★★★★（气象大模型适配）

#### 4. Wind Power Assessment based on Super-Resolution and Downscaling
- **平台：** arXiv (2024)
- **链接：** https://arxiv.org/html/2407.08259v1
- **核心贡献：** 深度学习空间降尺度方法系统对比
- **相关度：** ★★★★☆（空间超分辨率）

#### 5. Bias correction of wind power forecasts with SCADA data and continuous learning
- **平台：** arXiv (2024)
- **链接：** https://arxiv.org/html/2402.13916v1
- **核心贡献：** 用历史SCADA数据校正NWP偏差
- **相关度：** ★★★★☆（模型偏差校正）

### 补充参考文献

- **Probabilistic Forecasting of Provincial Regional Wind Power Considering Spatio-Temporal Features**
  - MDPI Energies (2025): https://www.mdpi.com/1996-1073/18/3/652
  - 省域风电概率预测，时空特征建模

- **Ultra-Short-Term Wind Farm Power Forecasting with Spatial Features**
  - MDPI Sustainability (2025): https://www.mdpi.com/2071-1050/18/14/7387
  - 空间相关性显式建模，改善38-44%

- **A Weather Foundation Model for the Power Grid**
  - arXiv (2025): https://arxiv.org/html/2509.25268v1
  - 电网专用气象基础模型

---

## 你的研究创新空间

### 创新点1：气象大模型 + 场站SCADA端到端融合
**当前空白：**
- 现有工作多用传统NWP（WRF、ECMWF）
- 气象基础模型（如PanguWeather）在风电预测中的应用刚刚起步

**你的优势：**
- 直接使用气象大模型的格点预测
- 结合场站历史数据校正大模型的系统性偏差
- 设计专门的后处理网络适配风电任务

### 创新点2：省域多场站时空联合建模
**当前局限：**
- 单场站预测为主
- 区域预测多采用简单加总

**你的方向：**
- 场站间空间相关性显式建模（GNN）
- 共享格点气象信息
- 集群总功率的整体优化

### 创新点3：大尺度空间降尺度（0.25° → 点位）
**技术挑战：**
- 现有工作多为9km→1km（约10倍）
- 你的场景是28km→点位（更大跨度）

**可能路径：**
- 多尺度级联降尺度（粗→中→细）
- 物理-数据混合驱动
- 利用再分析数据作为中间桥梁

### 创新点4：1h → 15min物理约束超分辨率
**技术要点：**
- 时间超分辨率 + 功率变化率约束
- 保证生成序列的物理合理性
- 多时间尺度特征融合

---

## 建议的技术路线

### 整体架构（端到端）

```
┌─────────────────┐
│ 数据输入        │
├─────────────────┤
│ • 15min历史功率  │
│ • 再分析气象     │──┐
│ • 1h NWP (0.25°)│  │
└─────────────────┘  │
         │           │
         ↓           │
┌─────────────────┐  │
│ 多源编码器      │  │
├─────────────────┤  │
│ Temporal Enc ←──┘  │
│ Reanalysis Enc     │
│ NWP Enc            │
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ 时空对齐模块    │
├─────────────────┤
│ Spatiotemporal  │
│ Refinement      │
│ • 0.25°→点位    │
│ • 1h→15min      │
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ 融合解码器      │
├─────────────────┤
│ Multi-head      │
│ Attention       │
│ + GRU Decoder   │
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ 输出：未来6h    │
│ 24个15min步长   │
│ 集群总功率      │
└─────────────────┘
```

### 关键模块设计

#### 模块1：NWP Spatiotemporal Refinement
```python
# 伪代码示例
class NWPRefinement(nn.Module):
    def __init__(self):
        self.spatial_refine = SpatialInterpolationNet()  # 0.25° → point
        self.temporal_refine = TemporalSuperResolution()  # 1h → 15min
        
    def forward(self, nwp_grid, station_coords):
        # 空间细化：提取周边格点 + 注意力融合
        station_nwp = self.spatial_refine(nwp_grid, station_coords)
        # 时间细化：生成15min序列
        high_res_nwp = self.temporal_refine(station_nwp)
        return high_res_nwp
```

#### 模块2：Multi-Source Attention Fusion
```python
class MultiSourceFusion(nn.Module):
    def __init__(self):
        self.attention = MultiHeadAttention()
        
    def forward(self, hist_power, reanalysis, refined_nwp):
        # 自适应学习不同数据源权重
        # 前1-2h：hist_power权重高
        # 3-6h：refined_nwp权重高
        fused = self.attention([hist_power, reanalysis, refined_nwp])
        return fused
```

#### 模块3：Physics-Constrained Decoder
```python
class PhysicsDecoder(nn.Module):
    def __init__(self):
        self.decoder = GRU()
        self.power_curve = PowerCurveLayer()
        
    def forward(self, fused_features):
        output = self.decoder(fused_features)
        # 功率变化率约束
        output = self.apply_ramp_rate_constraint(output)
        return output
```

---

## 实验设计建议

### Baseline对比
1. **持续性模型**（Persistence）
2. **ARIMA/统计模型**
3. **单一数据源模型**：
   - 仅用历史功率的LSTM
   - 仅用NWP的MLP
4. **简单融合**：特征拼接 + LSTM
5. **现有SOTA**：
   - PatchTST (无NWP)
   - Transformer + 双线性插值NWP

### 消融实验
1. 移除NWP refinement模块（用简单插值替代）
2. 移除时间超分辨率（直接用1h预测）
3. 移除空间对齐（用最近邻格点）
4. 移除多源融合（单一数据源）
5. 移除物理约束

### 评估指标
- **点预测**：RMSE, MAE, MAPE, NRMSE
- **趋势预测**：方向准确率（DA）
- **峰值预测**：峰值误差、峰值时刻偏移
- **不同预测时长**：0-1h, 1-3h, 3-6h分段评估

---

## 参考文献完整列表

1. Nature Scientific Reports (2025): PatchTST-GRU with NWP refinement
2. arXiv:2607.17095 (2024): Fourier Geometric Wind Power Forecasting
3. arXiv:2509.06311 (2025): WindFM Foundation Model
4. arXiv:2407.08259 (2024): Wind Power Super-Resolution
5. arXiv:2402.13916 (2024): Bias Correction with SCADA
6. MDPI Energies (2025): Provincial Wind Power Probabilistic Forecasting
7. MDPI Sustainability (2025): Ultra-Short-Term with Spatial Features
8. arXiv:2509.25268 (2025): Weather Foundation Model for Power Grid
9. arXiv:2507.23147 (2025): Foundation Models for Clean Energy Forecasting

---

**文档生成时间：** 2026-08-18  
**研究场景：** 省域风电集群超短期预测（15min分辨率，未来6h）  
**核心挑战：** 时空分辨率双重不对齐 + 多源异构数据融合
