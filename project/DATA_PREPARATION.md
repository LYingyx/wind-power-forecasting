# 吉林 56 场站数据准备

## 固定目标

- 候选编号：`1043002` 至 `1043061`；
- 排除编号：`1043041`、`1043043`、`1043044`、`1043060`；
- 保留场站数：56；
- 固定总装机容量：6827.01 MW；
- 目标：56 个保留场站的 15 分钟有功功率严格求和；
- 时间切分：2022 年训练、2023 年验证、2024 年测试。

原始场站 CSV、原始 Pangu 归档和旧 58 场站 HDF5 保持不变。场站排除只在新配置中实现。

## Pangu 空间范围

原始吉林归档的空间范围为：

```text
shape:     [1,14,34,51]
latitude:  47.75, 47.50, ..., 39.50
longitude: 120.25, 120.50, ..., 132.75
resolution: 0.25 degree
```

56 场站包络请求向四周外扩 8 个格点。现有归档在北侧和西侧各缺少请求范围的一个格点，因此使用可获得的最大范围：

```text
relative slice: latitude[0:29], longitude[0:30]
latitude:       47.75 to 40.75
longitude:      120.25 to 127.50
shape:          [1,14,29,30]
effective padding (N/S/W/E): 7/8/7/8 grid points
```

该裁剪直接选取原生格点，不执行空间插值。请求外扩量和四个方向的实际外扩量均写入 HDF5 与 JSON 元数据。

## 时间与样本

- 历史功率：6 小时，共 24 个 15 分钟点；
- 未来功率：6 小时，共 24 个 15 分钟点；
- Pangu：6 个历史小时节点和 12 个未来小时节点；
- 起点间隔：15 分钟；
- Pangu 可用性：起报后延迟 15 分钟，按当时最新可用 cycle 组织；
- 功率缺测窗口直接删除，不插值；
- NWP 只保存被样本引用的原生小时帧；
- NWP 按训练集引用帧拟合逐通道均值与标准差。

dry-run 的样本数为：

| 切分 | 15 分钟预测起点 | 独立 Pangu cycle |
|---|---:|---:|
| train | 31190 | 7809 |
| validation | 32550 | 8146 |
| test | 28782 | 7218 |

## 文件与复现

配置：`configs/jilin_56farm_pad8.yaml`

目标数据：`data/jilin_56farm_pad8.h5`

元数据：`data/jilin_56farm_pad8.metadata.json`

构建命令：

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
sh scripts/build_jilin_56farm_pad8.sh
```

构建过程先执行 dry-run，再通过临时文件完成 HDF5 写入，最后原子改名。已有目标文件不会被默认覆盖。

## 模型默认数据视图

模型默认使用场站包络向四周外扩 5 个原生格点，而不是直接读取完整 pad8 范围：

```text
latitude:  47.25 to 41.50
longitude: 120.75 to 126.75
shape:     [8,14,24,25] per weather cycle
```

其中 8 为未来原生小时节点数。模型缓存位于
`data/model_cache_pad5_f8`，包含 train、val 和 test 三个切分的：

- 唯一气象 cycle；
- 56 场站 24 点历史功率；
- 56 场站 24 点未来功率；
- 气象节点对应的场站小时功率监督；
- 起报时刻、气象有效时间差、容量告警和 cycle 映射。

缓存不执行空间插值或时间插值。功率按单场站容量归一化，容量和原始 MW
换算关系保存在静态元数据中。构建命令为：

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
sh scripts/prepare_model_data.sh
```
