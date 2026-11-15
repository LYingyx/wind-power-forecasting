# 私有 GitHub 仓库与 Gemini 网页版访问

更新：2026-11-15。已通过 GitHub API 核验仓库为 **Private**，默认分支 `main`。

## 仓库地址与阅读入口

| 项目 | 地址 / 值 |
| --- | --- |
| 仓库 | <https://github.com/LYingyx/wind-power-forecasting> |
| 所有者 | `LYingyx` |
| 可见性 | `Private` |
| 分支 | `main` |
| 首页 | [README.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/README.md) |
| Gemini 阅读顺序与提示词 | [GEMINI_READING_GUIDE_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/docs/GEMINI_READING_GUIDE_CN.md) |
| 当前结果 | [DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md) |

下面是可复制给 Gemini 的仓库信息，不是 Gemini 专有配置文件，也不会自动授予访问权限：

```yaml
repository: https://github.com/LYingyx/wind-power-forecasting
visibility: private
branch: main
entry: README.md
reading_guide: docs/GEMINI_READING_GUIDE_CN.md
current_results: project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md
latest_plan: project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md
```

## GitHub 配置

- Visibility：**Private**。
- Default branch：`main`。
- 不需要 GitHub Pages、公开 raw 链接或额外 API 服务。
- 本地 GitHub 登录用于创建/推送；Gemini 读取私有仓库需要它自己的官方 GitHub 授权，两者互不代替。
- 若 Gemini 提供 GitHub App 授权，选择正确的 GitHub 账户，并在可选时只授予本仓库读取权限。组织账户还可能需要组织管理员放行。
- 不把 GitHub token、SSH 私钥或登录密码发到 Gemini 对话里，也不写进仓库。

## 在 Gemini 网页版导入

1. 打开 <https://gemini.google.com/>，新建对话，检查附件/添加文件菜单是否提供“导入代码 / Import code”或 GitHub 入口。
2. **如果入口支持私有仓库并显示 GitHub 授权流程**：完成官方授权，选择 `LYingyx/wind-power-forecasting` 和 `main` 分支；若要求 URL，填写 `https://github.com/LYingyx/wind-power-forecasting`。等待导入完成，再发送[阅读提示词](GEMINI_READING_GUIDE_CN.md)。
3. **如果入口仅接受公开 URL、没有私有仓库授权，或你的账户没有该入口**：使用本地代码 ZIP，通过界面提供的代码/文件上传功能导入。没有相应 ZIP 入口时可解压为代码文件夹，或先上传 README、阅读入口、结果报告和关键源码。
4. 私有仓库 URL 本身不足以让 Gemini 读取；不要为绕过访问限制把仓库改成公开。
5. 导入通常针对当次代码快照。仓库更新后重新导入，并让 Gemini 确认读取到的版本。让它列出实际读到的文件，避免仅根据链接或文件名回答。

Gemini 功能会随账户、套餐、地区及组织策略变化。本次服务器无法连接 Google 帮助站，因此未实时核实你的账户是否支持私有仓库连接，也不保证具体菜单名称、容量限制或自动同步行为。以你账户实际界面及[Google 官方文件上传帮助](https://support.google.com/gemini/answer/14903178?hl=zh-Hans)为准；GitHub 授权入口不可用时使用代码 ZIP。

## 原服务器认证与上传

当前服务器已通过 GitHub CLI 认证为 `LYingyx`。换机器或凭据失效后，在自己的终端运行官方登录流程：

```bash
~/.local/bin/gh auth login --hostname github.com --git-protocol https --web
```

按终端提示在 <https://github.com/login/device> 输入一次性设备码并授权，并等待终端显示 `Logged in as LYingyx`。仅登录 GitHub 网页不等于终端完成认证。不要把访问令牌发到聊天里。

已获授权的其他机器可以克隆：

```bash
gh repo clone LYingyx/wind-power-forecasting
```

原服务器的 ZIP 位于 `github_export/wind-power-forecasting-gemini.zip`，该目录被 Git 忽略。也可在已登录且有权限的 GitHub 网页选择 **Code → Download ZIP**；下载后的 ZIP 可用作 Gemini 的代码导入快照。解压后的完整内容约 35 MiB；以当前 Gemini 界面显示的文件类型及容量限制为准。

## 仓库内容范围

包含源码、配置、手稿文本、当前 2571 项实验的汇总与逐风场/逐种子指标，以及后续计划。排除原始/处理后气象与功率数据、模型权重、逐样本预测、运行环境、日志和迁移归档；旧轮的大结果表也不完整包含。历史文档中指向这些内容的链接只能在原服务器使用。

仓库是代码审阅入口，不是开箱即用的数据/权重分发包。`.gitignore` 采用白名单；新增类型文件时需要检查是否应该纳入版本控制。
