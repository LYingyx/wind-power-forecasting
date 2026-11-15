# Cross-Farm / few-shot 迁移包

这个包是当前队列的可断点续跑快照，不包含已经不再需要的 2×2 原始大缓存和旧的 `extra30_2023_2024.h5`。包含：

- Cross-Farm transfer-dev 的已完成结果、队列数据库、代码和目标站缓存；
- 30 场站 few-shot baseline 的已完成结果、队列数据库和代码；
- source CBH checkpoint、source 模型源码和 source static metadata；
- 可重定位的 `chronos` 环境包；
- 路径重写、SQLite 运行项重置、tmux 后台恢复的一键脚本。

在目标服务器解压后执行：

```bash
bash migration/install_and_resume.sh /storage/liying/new_try/wind_cluster_forced_evolution
```

脚本会使用物理卡 `0,3,4,5,6,7`，每卡 8 个 worker，共用每卡锁，并要求至少 4096 MiB 空闲显存。它会把快照中 `running/failed` 行恢复为 `pending`，不会覆盖已完成结果。

验证：

```bash
tmux ls
cat MIGRATION_INSTALLED.json
${PWD}/.migration_env/chronos/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

确认目标端正常运行后，源服务器再执行：

```bash
bash migration/STOP_SOURCE_AFTER_SWITCH.sh
```

