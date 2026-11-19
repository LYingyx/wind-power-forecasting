# 英文论文总框架图 v27

- [最终 PNG](framework_final.png)
- [论文插图 PDF](framework_final.pdf)
- [英文图注及科学命名](CAPTION_AND_MAPPING.md)
- [完整编辑提示词](edit_prompt.txt)
- [身份支路连线修订提示词](fix_identity_prompt.txt)

配色直接参照用户指定的 v23 `framework_refined.png`：粉红、浅蓝、青绿色及黑色连线，不另加琥珀色类别。

左下采用 **Conditional low-rank refinement**，用收窄的 **Condition projection**、**Low-dimensional features** 和展开的 **Shared linear projection** 表达实际条件化特征修正。不使用系数、基底矩阵或 LoRA 权重更新的说法，不列张量形状及层宽。

右侧的功率解码器、输出头和功率变化输出保持在子框右上部横向连接。下方多源上下文横跨子框，通过区域读取、特征投影、身份向量和共享站点／时间投影利用右下区域。固定站点属性与时间直接进入共享投影，身份查表只读取 Farm ID。

使用当前配置接口所列最新 GPT Image 系列中的 `gpt-image-2.6-sunburst`，通过 **codexzh-image** 技能的图像编辑接口生成与修订。模型列表留存在 `provider_image_models.json`；该接口名称不作为独立确认 OpenAI 官方发布版本的依据。没有使用程序绘制、补字、覆盖连线或拼接图形。最终 PNG 保留模型输出；PDF 仅无损封装原图，不是可编辑矢量图。
