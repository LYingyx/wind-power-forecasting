"""Publish a navigable review snapshot without training or changing source evidence.

Use --check-only to verify generated text, strict JSON, links and SHA-256 hashes.
Original diagnostic JSON is preserved byte-for-byte, including one NaN value;
the derived strict JSON records that missing value explicitly as null.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2]
BRANCH = 'paper-evidence-20261119'
WEB = f'https://github.com/LYingyx/wind-power-forecasting/blob/{BRANCH}/'
TREE = f'https://github.com/LYingyx/wind-power-forecasting/tree/{BRANCH}/'
RAW = f'https://raw.githubusercontent.com/LYingyx/wind-power-forecasting/{BRANCH}/'
D = 'project/runs/alignment_error_diagnostic_20261120'
P = 'project/runs/daily_paper_evidence_20261118_v10'
RESEARCH = 'project/SINGLE_FARM_CROSS_RESOLUTION_RESEARCH_20261120_CN.md'
REPORT = f'{D}/REPORT_CN.md'
STRICT = 'docs/ALIGNMENT_DIAGNOSTIC_RESULTS_20261121.json'
INDEX = 'docs/GEMINI_READING_INDEX_20261121_CN.md'
OVERVIEW = 'docs/LATEST_CONCLUSIONS_20261121_CN.md'
TEXT10 = 'docs/gemini_plain_text/10_latest_conclusions_20261121.txt'
TEXT11 = 'docs/gemini_plain_text/11_alignment_diagnostic_code_and_results_20261121.txt'
CATALOG = 'docs/GEMINI_FILE_CATALOG_20261121.json'
MANIFEST = 'docs/GEMINI_PUBLICATION_MANIFEST_20261121.json'

SECTIONS = [
    ('最新结论与研究方向', [
        ('本次更新概览与口径说明', OVERVIEW, '区分已完成诊断、原模型结果与研究建议。'),
        ('最新诊断报告', REPORT, '2022–2023、56 站；包括 Pangu 误差、平流插值、历史功率、ERA5/CLDAS 与小时内信息。'),
        ('跨时空分辨率研究路线', RESEARCH, '11-20 的文献与方案讨论；不是新模型收益报告。后续可行性判断须同时读最新诊断。'),
        ('最新结论全文 TXT', TEXT10, '自包含：口径说明 + 最新诊断全文 + 研究路线全文。'),
        ('新诊断源码与结果 TXT', TEXT11, '14 个诊断脚本及 16 份原始汇总 JSON，逐文件带 SHA-256。'),
        ('新诊断严格 JSON 汇总', STRICT, '供程序读取；一处 NaN 转为 null，并保留原路径及说明。'),
    ]),
    ('已完成主模型：科学问题、基线与消融', [
        ('11-19 自包含论文证据包', 'docs/GEMINI_REVIEW_20261119_CN.md', '当前已完成神经网络实验的主要结论；151 项补充任务全部完成。'),
        ('科学问题与贡献表述', 'docs/SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS_20261119_CN.md', '与当时实际实现相对应；未被最新路线讨论自动替换。'),
        ('输入、模块与表示说明', 'docs/INPUTS_AND_REPRESENTATIONS_20261119_CN.md', '历史功率、周边观测、运行统计、气象派生量与表示空间。'),
        ('两年基线与消融总表', f'{P}/analysis/BASELINE_ABLATION_SUMMARY_CN.md', '共同五种子表、十种子复核和未通过项。'),
        ('基线 CSV', f'{P}/analysis/baseline_common5.csv', '相同五种子口径。'),
        ('消融 CSV', f'{P}/analysis/ablation_common5.csv', '相同五种子口径。'),
        ('补充对照 CSV', f'{P}/analysis/extra_common5.csv', '额外对照模型。'),
        ('带标准差的汇总表', f'{P}/analysis/READABLE_TABLES.json', '54 条表格记录。'),
        ('逐种子、逐风场与分时效指标', f'{P}/analysis/METRIC_RECORDS.json', '322 条模型×种子×年份记录；较大文件，建议下载读取。'),
        ('实际配置、完成标志与训练曲线', f'{P}/analysis/TASK_EVIDENCE.json', '151 项任务的配置、训练记录和来源校验。'),
        ('配对比较与门槛', f'{P}/analysis/COMPARISONS.json', '54 项配对比较，保留未通过原因。'),
        ('分组结果', f'{P}/analysis/GROUPS.json', '各实验组聚合指标。'),
        ('两因素交互', f'{P}/analysis/INTERACTIONS.json', '检查两项设计是否独立、是否存在交互。'),
        ('起报日分块区间估计', f'{P}/analysis/ISSUE_BLOCK_UNCERTAINTY.json', '按气象起报日处理样本相关性。'),
        ('完成审计', f'{P}/COMPLETE.json', '151/151；121 次训练（含 16 次调参）和 30 次已有权重复评。'),
        ('冻结协议', f'{P}/PROTOCOL.json', '训练和评价规则。'),
        ('科学假设与判定冻结', f'{P}/SCIENCE_FREEZE.json', '预先约定的比较与门槛。'),
        ('任务清单', f'{P}/manifest.json', '任务标识与实际配置入口。'),
    ]),
    ('当前模型代码、训练与数据处理', [
        ('当前源码 TXT', 'docs/gemini_plain_text/09_current_model_code_20261119.txt', '完整当前 code/、数据准备与结果计算源码；不含模型权重。'),
        ('模型入口和补充基线', f'{P}/code/model.py', 'PaperModel、输入消融和同输入线性模型。'),
        ('完整历史条件表示修正', f'{P}/code/history_model.py', '轨迹卷积、状态/天气窗口条件与低维加性修正。'),
        ('状态调制与加性对照', f'{P}/code/realization_model.py', '兑现模块的历史实现和相关对照。'),
        ('共享主体与解码', f'{P}/code/parent_model.py', '天气、历史、上下文融合及输出路径。'),
        ('表示与流匹配实现', f'{P}/code/representation.py', '包括历史生成式组件；存在代码不代表当前主模型采用它。'),
        ('训练、选轮与评估', f'{P}/code/train.py', '实际损失与选模逻辑。'),
        ('样本和特征构造', f'{P}/code/data.py', '当前训练数据接口。'),
        ('气象派生特征', f'{P}/code/physics.py', '相关物理量的计算。'),
        ('早期日协议数据准备', 'project/runs/daily_protocol_restart_20261114_v1/code/prepare.py', '近邻、三环与相关预处理来源。'),
        ('当前结果汇总', f'{P}/operations/collect.py', '读取完成任务并计算比较。'),
        ('当前易读表生成', f'{P}/operations/readable_report.py', '基线、消融表的生成逻辑。'),
        ('不确定性分析代码', f'{P}/operations/uncertainty.py', '起报日分块计算。'),
        ('日协议天气准备', 'project/scripts/prepare_operational_pangu.py', '业务可用性与天气缓存构造。'),
        ('日协议天气审计', 'project/scripts/audit_operational_pangu.py', '检查数据和时效。'),
        ('数据、选模、评估历史 TXT', 'docs/gemini_plain_text/04_data_selection_evaluation.txt', '历史管线补充，当前模型以 09 为准。'),
    ]),
    ('数据来源、风场范围与论文图', [
        ('服务器数据来源位置', 'project/data/SOURCE_LOCATIONS.json', '功率 CSV/HDF5、Pangu、ERA5 与缓存路径；这是路径记录，不是原始数据。'),
        ('功率数据元信息', 'project/data/jilin_56farm_pad8.metadata.json', '原始 56 站数据元信息。'),
        ('每日起报业务配置', 'project/configs/operational_pangu_d1_08_lag15h.json', '每日 00 UTC 起报、15 h 后可用。'),
        ('主模型固定风场集合', f'{P}/assets/COHORT.json', '55 个风场；不要与新诊断的 56 站混同。'),
        ('主模型质量审计', f'{P}/assets/QC_AUDIT.json', '固定筛选规则及其审计。'),
        ('逐风场质量表', f'{P}/assets/qc_by_farm.csv', '逐场质量信息。'),
        ('归一化参数', f'{P}/assets/normalization.json', '当前共享模型的预处理参数。'),
        ('业务协议说明', 'project/MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md', '已完成 0–6 h 实验的协议。'),
        ('英文论文框架图', 'project/figures/framework_paper_v27_20261118/framework_final.png', '当前已实现结构；不是 11-20 新提案的实现图。'),
        ('论文图 PDF', 'project/figures/framework_paper_v27_20261118/framework_final.pdf', '可下载版本。'),
        ('论文图说明与来源', 'project/figures/framework_paper_v27_20261118/README_CN.md', '图与实际实现的对应关系。'),
    ]),
    ('历史结果与路线演变', [
        ('11-17 已完成实验（2023）', 'docs/EXPERIMENT_RESULTS_20261117_CN.md', 'v4–v7 共 105 次训练；含状态调制、加性修正和隔离教师梯度的负结果。'),
        ('11-17 逐任务证据', 'docs/EXPERIMENT_RESULTS_20261117_EVIDENCE.json', '该批次的配对数据。'),
        ('采样审计', 'docs/SAMPLING_AUDIT_20261117_EVIDENCE.json', '固定检查点采样检验。'),
        ('11-17 跨年复评（2024）', 'docs/TEST2024_RESULTS_20261117_CN.md', '120 个冻结检查点；五种子报告，不代替 11-19 十种子。'),
        ('11-17 跨年逐任务证据', 'docs/TEST2024_RESULTS_20261117_EVIDENCE.json', '2024 逐风场、逐种子结果。'),
        ('11-17 跨年总表', 'docs/TEST2024_SUMMARY_20261117.csv', '历史五种子结果。'),
        ('11-17 跨年配对表', 'docs/TEST2024_COMPARISONS_20261117.csv', '历史配对比较。'),
        ('11-15 日协议结果分析', 'project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md', '原主干、编码和相关消融。'),
        ('11-15 逐风场／种子／时效', 'project/analysis/daily_protocol_review_20261115/FARM_AND_SEED_DETAILS_CN.md', '逐场与稳定性证据。'),
        ('天气编码角色澄清', 'project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md', '编码选择偏差和贡献边界。'),
        ('跨分辨率／兑现早期提案', 'docs/STATE_REALIZATION_AND_CROSS_RESOLUTION_PROPOSAL_CN.md', '方案历史；实际结果以上述完成报告为准。'),
        ('11-16 v3–v5 完整计划存档', 'docs/STATE_REPRESENTATION_PLAN_V3_V5_ARCHIVE_20261121_CN.md', '本次补齐此前本地修订；原文中的“在跑”是当时状态。'),
        ('原 v2 计划', 'project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md', '保留远程原快照，避免改写旧阅读包来源。'),
        ('v2 结果快照', 'project/runs/daily_state_representation_20261116_v2/analysis/CURRENT_RESULTS_CN.md', '历史局部结果；不能据此判断当前完成状态。'),
        ('旧实验结果到代码导航', 'docs/RESULTS_TO_CODE_CN.md', '2571 项早期任务的配置、训练记录与评估来源。'),
        ('旧协议运行状态与天气机制', 'project/TWO_MECHANISM_EVIDENCE_UPDATE_20261027_CN.md', '旧协议机制讨论；不能与每日新协议直接拼接。'),
        ('输出头 2×2 旧实验计划', 'project/runs/readout_factorial_20261028_v1/PLAN_CN.md', '输出头当前功率输入的历史实验入口。'),
        ('共享与逐场适配历史', 'project/runs/shared_site_finetune_20261104_v1/analysis/FINAL_REVIEW_CN.md', '旧适配结果；不代表当前共享模型做了逐场微调。'),
        ('完整历史交接文档', 'project/PROJECT_HANDOFF_CN.md', '仅按需追溯；日期较早的“当前”叙述可能已过时。'),
    ]),
]

TEXTS = [
    ('01_context_and_results.txt', '早期背景、日协议结果与计划（历史）'),
    ('02_models_and_training.txt', '早期模型和训练源码（历史，当前用 09）'),
    ('03_farm_seed_horizon_metrics.txt', '早期逐风场、种子和时效指标（历史）'),
    ('04_data_selection_evaluation.txt', '数据、选型、评估历史源码'),
    ('05_current_experiment_results.txt', '11-16 v2 快照（历史；文件名中的 current 不表示现在）'),
    ('06_completed_experiments_20261117.txt', '11-17 完成的 2023 实验与负结果'),
    ('07_test2024_results_20261117.txt', '11-17 的 2024 五种子复评'),
    ('08_paper_review_20261119.txt', '当前已完成主模型的论文证据、基线与消融'),
    ('09_current_model_code_20261119.txt', '当前主模型、训练、数据和汇总源码'),
    ('10_latest_conclusions_20261121.txt', '最新诊断全文与新研究路线全文'),
    ('11_alignment_diagnostic_code_and_results_20261121.txt', '最新诊断的全部源码和汇总结果'),
]

DIAGNOSTICS = [
    ('气象误差、平滑、位移和时间偏差', 'D1_D4_errors.json', 'diag_errors.py'),
    ('ERA5 平流／线性重建', 'G0_era5.json', 'g0_interp.py'),
    ('Pangu 平流／线性重建', 'G0_pangu.json', 'g0_interp.py'),
    ('气象变量的功率信息', 'Q1_variables.json', 'power_tests.py'),
    ('小时内功率信息', 'Q2_intrahour.json', 'power_tests.py'),
    ('本站与邻站历史信息', 'Q3_history_neighbors.json', 'power_tests.py'),
    ('空间邻域比较', 'Q4_spatial_ceiling.json', 'spatial_ceiling.py'),
    ('平流路线分层检验', 'ROUTE1_stratified.json', 'route1_stratified.py'),
    ('气象订正与功率转换解耦', 'Q5_decoupled.json', 'decoupled_check.py'),
    ('不同时间尺度的波动信息', 'Q6_fluctuation.json', 'fluctuation_info.py'),
    ('24 小时任务', 'Q7_horizon24.json', 'horizon24.py'),
    ('CLDAS 信息、空间均值与时效', 'Q8_cldas.json', 'cldas_check.py'),
    ('CLDAS 追加对照', 'Q8b_cldas_followup.json', 'cldas_followup.py'),
    ('CLDAS 产品逐月核查', 'Q8c_cldas_product_by_month.json', None),
    ('CLDAS 小时内细节检验', 'Q9_cldas_intrahour.json', 'cldas_intrahour.py'),
    ('ERA5 100 m 提取完整性', 'ERA5_100M_READY.json', 'extract_era5_100m.py'),
]


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def link(path, title=None, raw=False):
    return f'[{title or Path(path).name}]({(RAW if raw else WEB)+path})'


def directory(path, title):
    return f'[{title}]({TREE+path}/)'


def strict_results():
    records = []
    for p in sorted((ROOT/D/'output').glob('*.json')):
        missing = []

        def clean(x, pointer=''):
            if isinstance(x, dict):
                return {k: clean(v, pointer+'/'+k.replace('~', '~0').replace('/', '~1')) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v, pointer+'/'+str(i)) for i, v in enumerate(x)]
            if isinstance(x, float) and not math.isfinite(x):
                missing.append({'json_pointer': pointer, 'original_token': str(x), 'meaning': '未得到有限数值；不是零，也不是有效成绩'})
                return None
            return x

        data = clean(json.loads(p.read_text()))
        path = p.relative_to(ROOT).as_posix()
        records.append({'source': path, 'sha256': digest(path), 'nonfinite_values': missing, 'data': data})
    return {'date': '2026-11-21', 'note': '原始 JSON 不改动；此派生版本仅将非有限数值替换为 null，并记录位置。', 'records': records}


def overview():
    return f'''# 最新结论与阅读口径（2026-11-21）

本次发布新增的研究方向文档、截至 11-21 的诊断报告、14 个诊断脚本和 16 份汇总 JSON，并补齐 11-16 的 v3–v5 计划存档。没有在本次发布中重新训练模型或改变已有实验成绩。

完整阅读目录：{link(INDEX)}。远程分支为 `{BRANCH}`；默认 `main` 上的旧报告不能代替该分支的新材料。

## 三套材料分别回答什么

| 材料 | 范围与状态 | 如何使用 |
|---|---|---|
| {link('docs/GEMINI_REVIEW_20261119_CN.md', '已完成主模型实验')} | 55 站共享模型，0–6 h / 15 min，2022 训练、2023 选轮、2024 跨年复评；补充批次 151/151 完成 | 当前基线、消融、十种子和方法有效性证据 |
| {link(RESEARCH, '11-20 研究方向讨论')} | 文中改为单风场日前问题，提出平流插值与生成式后验等方案；文档本身没有新实验结果 | 研究建议、文献线索；不表示新路线已经实现或通过 |
| {link(REPORT, '截至 11-21 的诊断')} | 56 站，2022–2023，不读 2024；功率评价通常剔除 2023 年 11 月，含小时内、1/3/6 h 和 24 h 子任务 | 检查数据中有什么信息，以及新路线在已检验设置下的可行性；不能直接与 55 站主模型成绩比 |

报告所称“零训练诊断”是未重训主神经网络，其中仍包含在 2022 年拟合的梯度提升树和单调回归。原文“等温曲线”对应代码 `IsotonicRegression`，应理解为单调回归曲线。

## 最新诊断值得先看的结论

- 平流插值的总体与分层结果没有支持预期收益。CLDAS 的细尺度结构在所检验的尺度上并不按风速平移；这限制了该具体“空间换时间”方案的依据，不能外推为所有时空方法都无效。
- 在使用同期 ERA5/CLDAS、甚至已知未来整点功率的诊断设置下，小时内细节仍只有很小的可恢复增益。它们是额外信息／神谕对照，不能作为正常部署的输入成绩。
- 气象侧订正再转换到功率，在 Q5 的有限模型比较中出现正反馈，尤其在功率训练数据减少时。它仍需要与同协议强基线比较，不能据此宣称新主模型已胜出。
- CLDAS 周边区域均值和 1–6 h 信息有价值，但未来分析值不是业务预报；报告发现部分产品约 48 h 延迟，起点可用性需要单独满足。
- 11-20 路线文档中的推荐必须与这些后续负结果一起阅读；小时内功率的时间戳究竟代表瞬时值还是区间平均，最新报告仍标为未核实。文献调查记录与代理投票不替代独立的文献或实验验证。

## 已完成模型的结论保持独立

主模型相对同骨干线性插值，在 2023/2024 十种子比较中通过原门槛；相对强 Transformer、部分内部机制必要性仍未全部通过。当前实现为确定性低秩加性特征修正，不能称为已验证的生成式天气超分辨率或局地气象重建。新诊断没有替代或重新计算该组实验。2024 历史上曾被查看，仍称跨年复评，不称盲测。

## 数据和可复核范围

功率、Pangu 与 ERA5 服务器位置见 {link('project/data/SOURCE_LOCATIONS.json')}；CLDAS 来源为 `/shared_data_hdd1/All_Datasets/CLDAS_2.0`，读取逻辑见 {link(D+'/code/extract_cldas.py')}。原始数据、裁剪数组、训练权重与逐样本预测保留在服务器，本仓库提供来源位置、文本源码和汇总指标。

新诊断原始 `Q9_cldas_intrahour.json` 中 `T2.catmull_rom_from_power.R2` 是 `NaN`，不是有效成绩。原文件按字节保留；{link(STRICT, '严格 JSON 副本')}将这一项记为 `null` 并标注来源。它不影响正文报告的其他有效对照。`Q8c_cldas_product_by_month.json` 的单独生成命令未随本批脚本保存，作为结果存档发布，不宣称其执行链已完整复现。

本次检查验证发布内容、哈希、JSON 转换和原有结果表计算；不等于重新运行这些诊断或独立核验研究路线中的全部文献。
'''


def text_bundle(paths, header):
    parts = [header]
    for path in paths:
        parts.extend([f'\n===== FILE: {path} | SHA256: {digest(path)} =====\n',
                      (ROOT/path).read_text(), '\n===== END FILE =====\n'])
    return ''.join(parts)


def index():
    out = [f'''# Gemini 完整阅读目录（截至 2026-11-21）

指定分支：`{BRANCH}`。本目录把最新诊断、已完成主模型、原始结果证据、模型/数据源码和历史路线分开列出。每个文件均提供可浏览地址；纯文本另列 Raw 直链。不要仅导入默认 `main`。

## 最少先读这四份

1. {link(TEXT10, '10：最新诊断和研究路线全文', raw=True)}。
2. {link('docs/gemini_plain_text/08_paper_review_20261119.txt', '08：已完成主模型的科学问题、基线与消融全文', raw=True)}。
3. {link('docs/gemini_plain_text/09_current_model_code_20261119.txt', '09：当前模型与训练／数据／评价源码', raw=True)}。
4. {link(TEXT11, '11：最新诊断源码和汇总结果', raw=True)}。

如果 Gemini 无法直接读取链接，请下载相应 UTF-8 TXT 后导入。先要求它确认实际读取的文件，不要假定链接会被自动遍历。需要重算主模型配对指标，再追加 `METRIC_RECORDS.json` 和 `COMPARISONS.json`。

**口径优先级：**最新诊断读 10/11，已训练主模型读 08/09；01–07 是历史快照。55 站主模型与 56 站诊断、0–6 h 与日前任务、五种子与十种子、实际可用预报与未来再分析对照均须分开。方案不能替代结果；图不能证明每条子路径有效。
''']
    for title, rows in SECTIONS:
        out.append(f'\n## {title}\n\n| 内容 | 文件 | 用途／边界 |\n|---|---|---|\n')
        for label, path, note in rows:
            out.append(f'| {label} | {link(path)} | {note} |\n')
    out.append('\n## 全部 11 份纯文本附件\n\n| 附件 | 内容 | 下载 |\n|---|---|---|\n')
    for name, note in TEXTS:
        path = 'docs/gemini_plain_text/'+name
        out.append(f'| {link(path)} | {note} | {link(path, "Raw", raw=True)} |\n')
    out.append('\n## 新诊断逐项证据与脚本\n\n| 检验 | 原始结果 JSON | 代码 |\n|---|---|---|\n')
    for label, result, script in DIAGNOSTICS:
        out.append(f'| {label} | {link(D+"/output/"+result)} | '+(link(D+'/code/'+script) if script else '生成命令未随本批存档；勿假定已完整复现')+' |\n')
    out.append(f'''
补充：{link(D+'/code/common.py', '路径和共享计算')}、{link(D+'/code/extract_cldas.py', 'CLDAS 提取')}。所有原始结果保留，NaN 的处理说明见{link(OVERVIEW, '本次概览')}。

## 完整代码目录与来源审计

- {directory(P+'/code', '当前主模型全部源码（含实际导入的 legacy 依赖）')}。
- {directory(P+'/operations', '当前结果收集、评估加载修复与审计')}；{directory(P+'/assets/selected', '实际选择的基线配置')}。
- {directory(D+'/code', '最新诊断全部源码')}；{directory(D+'/output', '最新诊断全部汇总 JSON')}。
- {directory('project/analysis/daily_protocol_review_20261115/provenance', '早期 2571 项任务的三组训练来源记录')}。
- {directory('project/runs/daily_protocol_restart_20261114_v1/code', '早期日协议基线源码')}、{directory('project/runs/daily_ablation_restart_20261114_v1/code', '早期编码与输入消融源码')}、{directory('project/runs/daily_site_finetune_20261114_v1/code', '早期逐场微调源码')}。
- {link(CATALOG, '机器可读完整文件目录')}：本目录涉及的全部文件，加当前模型／诊断源码、配置与来源目录；逐文件提供字节数、SHA-256、GitHub 地址和 Raw 地址。
- {link(MANIFEST, '本次发布校验清单')}；{link('docs/PAPER_REVIEW_MANIFEST_20261119.json', '11-19 冻结证据哈希')}；{link('docs/gemini_plain_text/MANIFEST.json', '01–07 历史附件哈希')}。

服务器原始数据路径不会变成 GitHub 下载地址。证据 JSON 内的 `source` 可能指仅在服务器存在的 `jobs/` 原件；远程可读的指标和训练记录分别使用 `METRIC_RECORDS.json`、`TASK_EVIDENCE.json`。

## 可直接复制给 Gemini 的审阅要求

> 请先读本目录的 10、08、09、11 四份 TXT，再按需读结果 JSON。先列出实际读取的文件。区分已有模型净收益、内部机制证据、新诊断与待实现方案；解释最新的小时内／平流负结果如何限制旧研究路线，同时保留气象订正与小时级信息的正反馈。不要混用 55/56 站、0–6 h/24 h、五/十种子或预报/未来分析数据；不要将文献或代理投票当成实验验证，不把历史待办当成当前未完成。逐条引用文件与数字，不根据模型胜负筛选风场。

本次发布检查：`python3 project/scripts/export_gemini_review_20261121.py --check-only`。
已完成模型的独立表格复算：`python3 project/scripts/export_paper_review_20261119.py --check-only`。
''')
    return ''.join(out)


def generated_content():
    generated = {OVERVIEW: overview(), INDEX: index(), STRICT: json_text(strict_results())}
    # Read overview from generated text, keeping check mode read-only.
    generated[TEXT10] = generated[OVERVIEW] + text_bundle(
        [REPORT, RESEARCH], '\n\n以下为两份源文档全文，保留原文；解释边界以上面的发布说明为准。\n')
    diag = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/D/'code').glob('*.py'))
    diag += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/D/'output').glob('*.json'))
    generated[TEXT11] = text_bundle(diag,
        '# 最新诊断源码与汇总结果（2026-11-21）\n\n14 个脚本、16 份原始 JSON。原始 JSON 的一处 NaN 是缺失值；严格 JSON 副本见 docs/ALIGNMENT_DIAGNOSTIC_RESULTS_20261121.json。服务器原始数组未上传。\n')
    return generated


def catalog(generated):
    roles = {}

    def add(path, role):
        roles.setdefault(path, set()).add(role)

    for title, rows in SECTIONS:
        for _, path, _ in rows:
            add(path, title)
    for name, _ in TEXTS:
        add('docs/gemini_plain_text/'+name, '纯文本附件')
    for _, result, script in DIAGNOSTICS:
        add(D+'/output/'+result, '最新诊断原始结果')
        if script:
            add(D+'/code/'+script, '最新诊断源码')
    for base in [P+'/code', P+'/analysis', P+'/operations', P+'/assets', D+'/code',
                 'project/analysis/daily_protocol_review_20261115/provenance']:
        for p in sorted((ROOT/base).rglob('*')):
            if p.is_file() and p.suffix in {'.py', '.json', '.csv', '.md'} and '__pycache__' not in p.parts:
                add(p.relative_to(ROOT).as_posix(), '补充源码／配置／证据')
    for path in [INDEX, 'docs/PAPER_REVIEW_MANIFEST_20261119.json',
                 'docs/gemini_plain_text/MANIFEST.json',
                 'project/scripts/export_gemini_review_20261121.py',
                 'project/scripts/export_paper_review_20261119.py']:
        add(path, '导航／校验')
    records = []
    for path, role in sorted(roles.items()):
        data = generated[path].encode() if path in generated else (ROOT/path).read_bytes()
        records.append({'path': path, 'roles': sorted(role), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(),
                        'github_url': WEB+path, 'raw_url': RAW+path})
    return {'date': '2026-11-21', 'branch': BRANCH, 'files': records}


def check_links(text):
    paths = re.findall(r'https://(?:raw\.githubusercontent\.com/LYingyx/wind-power-forecasting/|github\.com/LYingyx/wind-power-forecasting/(?:blob|tree)/)'+BRANCH+r'/([^\s)]+)', text)
    for path in paths:
        assert (ROOT/path.rstrip('/')).exists(), path
    return len(paths)


def main(check_only):
    generated = generated_content()
    cat = catalog(generated)
    generated[CATALOG] = json_text(cat)
    if not check_only:
        for name, content in generated.items():
            (ROOT/name).write_text(content)
    for name, content in generated.items():
        assert (ROOT/name).read_text() == content, name
    hashes = {r['path']: r['sha256'] for r in cat['files']}
    for path in [CATALOG, 'README.md', 'docs/GEMINI_READING_GUIDE_CN.md', 'docs/gemini_plain_text/README.md', '.gitignore']:
        hashes[path] = digest(path)
    manifest = {'date': '2026-11-21', 'branch': BRANCH,
                'source_policy': '新增诊断原件不改写；派生文件单独命名；不上传原始数组、权重或逐样本预测。',
                'sha256': dict(sorted(hashes.items()))}
    if not check_only:
        (ROOT/MANIFEST).write_text(json_text(manifest))
    assert json.loads((ROOT/MANIFEST).read_text()) == manifest
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    linked = check_links(generated[INDEX]+generated[OVERVIEW])
    results = strict_results()['records']
    assert len(results) == 16
    assert len(list((ROOT/D/'code').glob('*.py'))) == 14
    print(json_text({'passed': True, 'catalog_files': len(cat['files']),
                     'verified_links': linked, 'diagnostic_outputs': len(results),
                     'nonfinite_values_documented': sum(len(r['nonfinite_values']) for r in results)}).strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
