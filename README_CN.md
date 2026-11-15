# 风电项目 SSD2 统一归档

**接手项目先读：[框架、已有结果、代码地图与贡献计划总入口](project/PROJECT_HANDOFF_CN.md)。** 此文汇总当前主线、全部实验导航、关键实现及下一步工作。

当前唯一工作项目为 `project/`，入口和使用说明见 [project/README_CN.md](project/README_CN.md)。已合并原 /storage 整理副本、new_try_v2 和本次迁回的 /shared_data_ssd2/liying/new_try 项目。

- `project/`：代码、最终模型、数据、统一结果索引和已验证运行入口。
- `archives/slast_essential/`：早期轻量归档。
- `archives/slast_full_incomplete/`：以前中断的大归档，不是完整实验副本，也不是运行入口。
- `migration/`：SSD1 → SSD2 转移、清理和校验记录。

SSD1 的本次整理副本已在 SSD2 校验后清理。源 new_try/new_try_v2、已有环境和 HDD 原备份继续保留。最新合并状态见 `project/integration/MERGE_STATUS.json`。
