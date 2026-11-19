"""Build and verify the self-contained 2026-11-19 paper review package."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'project/runs/daily_paper_evidence_20261118_v10'
PARTS = [
    ROOT/'docs/SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS_20261119_CN.md',
    RUN/'analysis/BASELINE_ABLATION_SUMMARY_CN.md',
    ROOT/'docs/INPUTS_AND_REPRESENTATIONS_20261119_CN.md',
]
MAIN = ROOT/'docs/GEMINI_REVIEW_20261119_CN.md'
TEXT = ROOT/'docs/gemini_plain_text/08_paper_review_20261119.txt'
CODE_TEXT = ROOT/'docs/gemini_plain_text/09_current_model_code_20261119.txt'
MANIFEST = ROOT/'docs/PAPER_REVIEW_MANIFEST_20261119.json'

HEADER = '''# Gemini 阅读入口：科学问题、最终架构与完整实验结果

更新：2026-11-19。本文件自包含最新科学问题、贡献措辞、两年基线与消融表、十种子结果和实际输入编码说明；无需自动打开所有链接才能理解主要结论。请优先读取本文件，不沿用旧文档中的“仍在运行”或“十种子未完成”。

## 当前完成状态与关键结论

- 最新补充批次已完成 **151/151 个任务**：121 次训练（含 16 次线性基线调参）及 30 次已有权重复评。五次线性基线评估加载错误已修复，训练权重保持不变。该计数是本轮任务数，不是 151 个独立模型，也不替代此前批次计数。
- 完整模型相对同骨干线性插值，在 2023/2024 十种子比较中均通过完整门槛；2024 平均绝对误差降低 0.3098%，均方根误差降低 0.3404%，9/10 个种子同时改善。
- 当前主模型仍包含左侧状态摘要；去掉它在 2024 五种子上平均绝对误差/均方根误差降低 0.0556%/0.0647%，但 2023 上升 0.0917%/0.0489%。这条输入路径的独立价值未确立，不能隐去该负结果。
- 完整模型并未全面稳定胜过所有 Transformer 对照；高功率和跨种子稳定性仍有未通过项。当前方法是确定性低秩特征修正，不能将其收益写成随机生成式超分辨率成功。
- 去掉区域观测、显式运行统计和气象派生特征均有负向结果，但这不直接证明局地气象偏差已被校正，或两个机制具有独立、超加性的因果作用。

## 阅读材料

- [英文论文总框架图](../project/figures/framework_paper_v27_20261118/framework_final.png)；图是当前实现，不是每条路径都已有效的证明。
- [逐种子、逐风场、预测时段及诊断指标](../project/runs/daily_paper_evidence_20261118_v10/analysis/METRIC_RECORDS.json)：322 条“模型×种子×年份”记录，保留来源与校验值。
- [151 项任务的配置、完成标志与训练记录](../project/runs/daily_paper_evidence_20261118_v10/analysis/TASK_EVIDENCE.json)。
- [完整基线和消融表及标准差](../project/runs/daily_paper_evidence_20261118_v10/analysis/READABLE_TABLES.json)；[配对门槛](../project/runs/daily_paper_evidence_20261118_v10/analysis/COMPARISONS.json)；[两因素交互](../project/runs/daily_paper_evidence_20261118_v10/analysis/INTERACTIONS.json)；[按起报日分块的区间估计](../project/runs/daily_paper_evidence_20261118_v10/analysis/ISSUE_BLOCK_UNCERTAINTY.json)。
- [当前源码](../project/runs/daily_paper_evidence_20261118_v10/code/)；[源码纯文本](gemini_plain_text/09_current_model_code_20261119.txt)；[本入口纯文本](gemini_plain_text/08_paper_review_20261119.txt)。
- [此前生成式等路线的 2024 负结果](TEST2024_RESULTS_20261117_CN.md)：修正教师梯度后的流匹配仍未通过净收益门槛；该旧报告中的五种子结果不代替本文件十种子复核。

原始功率和天气数据、训练权重、逐样本预测保留在研究服务器。公开证据中的 source 路径指服务器原件；逐任务指标应读取 METRIC_RECORDS.json 与 TASK_EVIDENCE.json，不能假定仓库包含 jobs 目录。代码用于审阅，缺少研究数据不能直接恢复训练。

## 建议交给 Gemini 的审阅任务

> 请以本文件和实际可读源码、结果为依据，分别评价：①科学问题是否来自真实业务和物理限制；②两项设计是否解决不同问题；③哪些结论得到净收益、内部消融和跨年复评支持，哪些仅是解释假设；④论文应怎样收缩贡献措辞。请明确解释左侧去状态略好、历史反转未变差、完整窗口与局部窗口接近，以及两因素收益重叠的含义。不要混合五种子和十种子均值，不把 2024 称为盲测，不根据模型胜负建议删除风场，不把低秩特征修正叫作 LoRA 或局地气象重建。所有建议引用具体数据或代码；没有读到的材料明确标注。

下面依次附上三份正文。

'''


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_paths():
    return sorted((RUN/'code').rglob('*.py')) + [
        ROOT/'project/runs/daily_protocol_restart_20261114_v1/code/prepare.py',
        RUN/'operations/collect.py', RUN/'operations/readable_report.py', RUN/'operations/uncertainty.py',
    ]


def content():
    main = HEADER + '\n\n---\n\n'.join(p.read_text() for p in PARTS)
    # Embedded implementation links already resolve from docs; the source result
    # table contains no relative Markdown links requiring relocation.
    pieces = ['# 当前模型与结果计算源码（2026-11-19）\n\n仅含文本源码，不含权重和原始数据。每个文件标注仓库路径及 SHA-256。\n']
    for p in code_paths():
        pieces += [f'\n===== FILE: {p.relative_to(ROOT)} | SHA256: {digest(p)} =====\n',p.read_text(), '\n===== END FILE =====\n']
    return main, ''.join(pieces)


def verify_results():
    records = json.loads((RUN/'analysis/METRIC_RECORDS.json').read_text())['records']
    by = {(r['group'],r['seed'],r['year']):r['metrics'] for r in records}
    assert len(by) == len(records) == 322
    rows = json.loads((RUN/'analysis/READABLE_TABLES.json').read_text())['rows']
    mean = statistics.mean
    keys = ['mae','rmse','high_power_rmse']
    for r in rows:
        for k in keys:
            v = mean(by[r['group'],s,r['year']][k] for s in r['seeds'])
            assert abs(v-r[k]) < 1e-12, (r['group'],k)
            full = mean(by['H-FULL',s,r['year']][k] for s in range(42,47))
            assert abs(100*(v/full-1)-r[k+'_change_vs_full_percent']) < 1e-9
    pairs=json.loads((RUN/'analysis/COMPARISONS.json').read_text())
    for x in pairs:
        aa=[by[x['candidate'],s,x['year']] for s in x['seeds']]
        bb=[by[x['reference'],s,x['year']] for s in x['seeds']]
        for k in keys:
            delta=100*(mean(v[k] for v in aa)/mean(v[k] for v in bb)-1)
            assert abs(delta-x['delta_percent'][k])<1e-9
        wins=sum(all(a[k]<b[k] for k in keys[:2]) for a,b in zip(aa,bb))
        assert wins==x['dual_seed_wins']
        for k in keys[:2]:
            farm_wins=sum(mean(v['per_farm_'+k][j] for v in aa)<mean(v['per_farm_'+k][j] for v in bb) for j in range(55))
            assert farm_wins==x['farm_wins'][k]
        gate=(len(aa)>=x['planned_pairs'] and all(x['delta_percent'][k]<=-.1 and x['farm_wins'][k]>=28 for k in keys[:2]) and wins>=math.ceil(.8*len(aa)) and x['delta_percent']['high_power_rmse']<=.1)
        assert gate==x['gate']
    tasks=json.loads((RUN/'analysis/TASK_EVIDENCE.json').read_text())['tasks']
    assert len(tasks)==151 and all(t['done']['passed'] for t in tasks)
    assert sum(t['requested_spec'].get('task')!='evaluate' for t in tasks)==121
    complete=json.loads((RUN/'COMPLETE.json').read_text())
    assert complete['completed']==complete['total']==151 and not complete['failures'] and not complete['errors']
    return dict(metric_records=len(records),table_rows=len(rows),paired_comparisons=len(pairs),completed_tasks=len(tasks))


def main(check_only=False):
    main_text,code_text=content()
    if check_only:
        assert MAIN.read_text()==TEXT.read_text()==main_text
        assert CODE_TEXT.read_text()==code_text
        for name,expected in json.loads(MANIFEST.read_text())['sha256'].items():
            assert digest(ROOT/name)==expected,name
    else:
        MAIN.write_text(main_text);TEXT.write_text(main_text);CODE_TEXT.write_text(code_text)
        files=set(PARTS+code_paths()+[MAIN,TEXT,CODE_TEXT,Path(__file__).resolve()])
        files.update(p for p in (RUN/'analysis').iterdir() if p.is_file())
        files.update(RUN/n for n in ['PROTOCOL.json','manifest.json','SCIENCE_FREEZE.json','COMPLETE.json'])
        files.update(p for p in (ROOT/'project/figures/framework_paper_v27_20261118').iterdir() if p.is_file())
        MANIFEST.write_text(json.dumps({'date':'2026-11-19','branch':'paper-evidence-20261119','sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,**verify_results())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');main(p.parse_args().check_only)
