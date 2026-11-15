# 风电项目：SSD2 新位置

主项目：[`project/README_CN.md`](project/README_CN.md)。

```bash
cd /shared_data_ssd2/liying/wind_cluster_project_20261018/project
./project status
./project slast-predict --seed 42 --split val --samples 2
```

- `project/`：合并后的代码、现有结果、必要运行缓存和在新路径重建的独立环境。SLAST、transfer 和 few-shot 的指定检查已通过。
- `archives/slast_essential/`：此前整理的轻量归档。
- `archives/slast_full_incomplete/`：旧大归档任务的现存文件快照；旧任务尚未完成，现存文件的此次迁移已校验。
- `migration/`：路径映射、逐目录迁移校验和 SSD1 清理记录。

`column_weather/E2L` 仍缺配套源码、配置、DEM、归一化统计和部分种子结果，因此尚不能声称整个项目已完整可复现。主项目 README 给出了具体缺项与运行边界。

SSD1 上仅删除本次整理产生的三个项目副本。既有 `.conda`、`conda` 环境、用户自行提供的 `new_try_v2`、HDD 源备份，以及另一个正在接收的 SSD2 `new_try` 均保留。
