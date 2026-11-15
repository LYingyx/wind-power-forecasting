# 单风场框架图 v19 绘图记录

使用 imagegen 技能的 bundled CLI fallback，三次请求均设置 `--model gpt-image-2.5 --size 1536x1024 --quality high --no-augment`。模型参数未替换；凭据来自既有本地配置，没有写入交付文件。CLI 未保留服务端底层模型身份元数据，因此 manifest 记录的是实际提交的模型参数。

1. `framework_prompt.txt`：从新的单风场主线重新设计完整科学内容、直角连线和本站/区域版本说明。generate 输出 `framework_v19_generated.png`。
2. `refinement_prompt.txt`：edit，提高印刷文字可读性，去掉大标题色块。输出 `framework_v19_refined.png`。检查发现模型额外画了一条原始历史旁路，因此此版未采用。
3. `connection_fix_prompt.txt`：edit，删除额外旁路，保留历史编码器输出的完整记忆路径。输出最终 `framework_v19_final.png`，已人工查看并核对语义。

最终图由模型直接输出，没有用程序绘制、覆盖或改写连线。信息顺序、mean(M)、b/Δb、双路历史、直接天气、独立上下文和最终两输入求和均已核对。示意曲线不是实测结果。

既有 v18 的近一年会议图形调研用于延续细线、清楚层级和规则连线的原则；本轮依据新科学内容重写 prompt，没有将旧方法图直接改名。v18 中 SLAST/末位摘要及默认区域输入的内容不再保留。
