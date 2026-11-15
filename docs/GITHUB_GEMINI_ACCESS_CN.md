# 公开 GitHub 仓库与 Gemini 网页版访问

更新：2026-11-15。已按用户要求将仓库设为 **Public**，并通过 GitHub API 核验，默认分支 `main`。下面说明已同步为公开仓库的导入步骤。

## 仓库地址与阅读入口

| 项目 | 地址 / 值 |
| --- | --- |
| 仓库 | <https://github.com/LYingyx/wind-power-forecasting> |
| 所有者 | `LYingyx` |
| 可见性 | `Public` |
| 分支 | `main` |
| 首页 | [README.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/README.md) |
| Gemini 阅读顺序与提示词 | [GEMINI_READING_GUIDE_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/docs/GEMINI_READING_GUIDE_CN.md) |
| 当前结果 | [DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md](https://github.com/LYingyx/wind-power-forecasting/blob/main/project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md) |

下面是可复制给 Gemini 的仓库信息，不是 Gemini 专有配置文件，也不会自动授予访问权限：

```yaml
repository: https://github.com/LYingyx/wind-power-forecasting
visibility: public
branch: main
entry: README.md
reading_guide: docs/GEMINI_READING_GUIDE_CN.md
current_results: project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md
latest_plan: project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md
```

## GitHub 配置

- Visibility：**Public**。
- Default branch：`main`。
- 不需要 GitHub Pages、公开 raw 链接或额外 API 服务。
- 本地 GitHub 登录用于推送；本仓库的公开读取和 Gemini 导入无需私有仓库访问授权。
- 不把 GitHub token、SSH 私钥或登录密码发到 Gemini 对话里，也不写进仓库。

## 在 Gemini 网页版导入

Google 官方帮助确认：未关联 GitHub 账户也可导入公开仓库。本仓库现已公开，可直接在“导入代码”中提供仓库或分支 URL。

1. 在电脑上打开 <https://gemini.google.com/> 并登录。按官方要求开启 **Keep Activity（保留活动记录）**；工作/学校账户还需符合 Workspace 版本要求，并由管理员允许 Gemini 应用连接。
2. 新建对话，在输入框点击 **添加文件 / Add file → 更多上传 / More Uploads → 导入代码 / Import code**。
3. 输入 `https://github.com/LYingyx/wind-power-forecasting/tree/main`，点击 **Import**。也可使用不带 `/tree/main` 的仓库 URL。
4. 等待导入完成，再发送[阅读提示词](GEMINI_READING_GUIDE_CN.md)。可以先让 Gemini 列出实际读到的文件，并概括 README 和当前结果报告，确认导入成功。

已关联过 GitHub 的账户，可在 **Settings → Connected Apps → GitHub** 检查连接；部分界面需先进入 **Personal Intelligence → Connected Apps**。

官方限制：每个对话只能导入一个仓库/代码文件夹，最多 **5000 个文件、100 MB**。仓库更新后由 `project/scripts/export_gemini_reading.py --check-only --revision HEAD` 检查文件数、体积和文本可读性。仓库仍包含历史源码和文档，阅读时应优先使用本仓库的当前入口。

导入后的内容是快照，**不会自动同步 GitHub 更新**。Gemini 的 GitHub 应用不能仅凭聊天中粘贴的 URL 读取仓库，也不能读取提交历史、PR 等元数据或写回仓库。仓库更新后需要重新导入。

如果账户没有可用的 GitHub 导入入口，可下载 ZIP、解压，再通过 **Import code → Upload folder** 上传代码文件夹。普通“上传文件”的 ZIP 模式另有最多 10 个内部文件的限制，不能用该模式直接上传完整项目 ZIP。若代码文件夹入口也不可用，使用[四份未压缩 TXT](gemini_plain_text/README.md)，其中包含数据、模型、训练、选型和评估源码，而不只是结果摘要。

如果已有会话仍反复引用旧 ZIP 附件，建议新建会话，重新通过“导入代码”导入仓库。是否成功应以实际文件正文和代码引用为准；不能仅凭模型自述“已读完整仓库”判断。详见[结果到代码的证据链与读取验证](RESULTS_TO_CODE_CN.md)。

官方依据（2026-11-15 核对）：[导入 GitHub 仓库](https://support.google.com/gemini/answer/16176929?hl=en)、[上传文件与代码文件夹限制](https://support.google.com/gemini/answer/14903178?hl=en)。账户能否使用相应功能仍以其实际界面和组织策略为准。

## 原服务器认证与上传

当前服务器已通过 GitHub CLI 认证为 `LYingyx`。换机器或凭据失效后，在自己的终端运行官方登录流程：

```bash
~/.local/bin/gh auth login --hostname github.com --git-protocol https --web
```

按终端提示在 <https://github.com/login/device> 输入一次性设备码并授权，并等待终端显示 `Logged in as LYingyx`。仅登录 GitHub 网页不等于终端完成认证。不要把访问令牌发到聊天里。

其他机器可直接克隆公开仓库：

```bash
git clone https://github.com/LYingyx/wind-power-forecasting.git
```

原服务器的 ZIP 位于 `github_export/wind-power-forecasting-gemini.zip`，该目录被 Git 忽略。也可在仓库网页选择 **Code → Download ZIP**；请先解压，再通过 Gemini 的代码文件夹入口导入。

当前服务器按用户提供的代理完成上传，本地仓库已配置：

```bash
git config --local http.proxy http://192.168.100.99:1080
```

该地址的 HTTP 与 SOCKS5 模式均已连通核验，当前采用 HTTP 模式。配置只保存在本机 `.git/config`，不会推送到 GitHub。命令行 GitHub CLI 访问 API 时使用 `HTTPS_PROXY=http://192.168.100.99:1080`；其他机器需使用其自身可达的网络配置。

## 仓库内容范围

包含源码、配置、手稿文本、当前 2571 项实验的汇总与逐风场/逐种子指标、逐任务配置与训练曲线及选轮记录，以及后续计划。排除原始/处理后气象与功率数据、模型权重、逐样本预测、运行环境、完整运行日志和迁移归档；旧轮的大结果表也不完整包含。历史文档中指向这些内容的链接只能在原服务器使用。

仓库是代码审阅入口，不是开箱即用的数据/权重分发包。`.gitignore` 采用白名单；新增类型文件时需要检查是否应该纳入版本控制。
