# 当前论文入口

当前已更新的单风场中文手稿位于 `tex/manuscript_cn.pdf`，源码入口为 `tex/main.tex`。

本轮将贡献重心调整为历史轨迹条件化天气修正、训练共享与在线输入的分离证据，以及未见风场适配的收益和边界。CBH-SL 是本站部署主结果；E2C 和当前迁移结果作为区域历史扩展。正文和附录使用审查后的全站结果、条件消融、天气对照、典型风场、分场景和迁移图表。

- 阅读：`tex/manuscript_cn.pdf`
- 编译源码及表图：`tex/manuscript_source.zip`
- 新框架图与 imagegen / gpt-image-2.5 prompt：`../output/imagegen/single_farm_framework_v19/`
- 作者与实验边界：`tex/AUTHOR_NOTES_CN.md`
- 证据映射：`tex/RESULTS_PROVENANCE_CN.md`
- 编译及数据检查：`tex/BUILD_CHECK.json`
- 旧 SLAST 手稿备份：`archive_before_single_farm_rewrite_20261018/`
- 完整实验审查快照：`../runs/single_farm_review_20261018/`

审查快照及其原压缩包保持原样；快照中“旧论文尚未重写”的状态描述属于当时阶段，现已由本次实际手稿更新接续。未完成的 local-only E2C 等实验仍然未完成，不因文字更新而改变状态。
