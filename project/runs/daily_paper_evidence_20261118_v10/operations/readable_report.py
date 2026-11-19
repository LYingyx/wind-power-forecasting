"""Reproducible descriptive tables on a common five-seed cohort."""
from pathlib import Path
import csv
import json
import sys
import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'code'))
from settings import now, atomic_json

BASELINES = [
    ('PERSISTENCE', '持续性预测', '未来各步出力均等于当前实测出力；无需训练'),
    ('LINEAR', '全输入线性回归', '将本站历史、周边功率、未来天气等直接线性映射为未来出力变化'),
    ('TF_R3', 'Transformer', '注意力时序模型；读取未来天气、本站历史及周边功率'),
    ('TF_E2L', 'Transformer：统一气象编码', '将气象编码替换为与本文相同的无激活线性映射'),
    ('TF-WINDOW', 'Transformer：统一编码与完整天气窗口', '再显式加入全部八个小时天气节点及其时间偏移'),
    ('TFT_R3', '时间融合 Transformer', '结合变量选择、循环历史编码和注意力的仓库适配实现'),
    ('TFT-WINDOW', '时间融合 Transformer：完整天气窗口', '上述模型再显式加入全部八个小时天气节点'),
    ('TIDE_R3', '多层感知机时序编码—解码模型（TiDE）', '以全连接网络编码历史及未来协变量；仓库适配实现'),
    ('TIDE-WINDOW', '多层感知机时序编码—解码：完整天气窗口', '上述模型再显式加入全部八个小时天气节点'),
    ('R0', '本文骨干：线性插值版本', '保留多源解码结构，小时天气采用线性插值，不加联合查询修正'),
    ('H-FULL', '本文完整模型', '历史轨迹、状态与天气联合修正查询，再通过多源解码预测出力变化'),
]
ABLATIONS = [
    ('R0', '去掉整个联合查询修正模块', '退回同骨干的小时天气线性插值'),
    ('H-SUMMARY', '修正模块不读完整历史轨迹', '保留状态摘要和天气窗口；主干历史输入不变'),
    ('H-noStateOnly', '修正模块不读状态摘要', '保留完整历史轨迹和天气窗口'),
    ('H-wrongStateOnly', '仅错配修正模块的状态摘要', '换成训练年同站异日状态；真实历史保持不变'),
    ('H-noS', '修正模块不读历史轨迹和状态摘要', '主干真实历史保持不变，仅屏蔽修正模块两路条件'),
    ('H-wrongS', '同时错配修正模块的轨迹和状态', '两者均使用训练年同站异日供体；主干和标签不变'),
    ('H-noWindow', '修正模块不读天气变化窗口', '底层天气线性插值仍保留'),
    ('H-localWindow', '修正模块只读目标所在小时段', '用该小时段两个节点替代完整八节点变化窗口'),
    ('H-reverseHistory', '反转修正模块的历史顺序', '反转前二十三点；当前点和状态摘要不变'),
    ('H-noDecoderState', '去掉解码端运行与区域信息', '联合查询修正模块保留；解码端气象派生特征仍保留'),
    ('H-noRegions', '两处都去掉周边风场信息', '同时移除修正模块与解码端的近邻及区域功率信息'),
    ('H-noOperating', '两处都去掉显式运行状态摘要', '原始功率历史与输出端当前功率仍保留'),
    ('H-noAtmosphere', '去掉气象派生物理特征', '移除解码端十五维派生特征；原始气象仍保留'),
]
EXTRA = [
    ('TF-H-FULL', 'Transformer 加本文联合查询修正', '用于检验修正模块在其他骨干上的迁移效果'),
    ('R0-noDecoderState', '线性插值骨干再去掉解码状态信息', '与完整模型及两个单独消融构成二乘二实验'),
    ('B-additive', '较早的状态与分段天气加性修正', '不读取完整历史轨迹；不是最终模型'),
]
METRICS = ['mae', 'rmse', 'high_power_rmse']
SEEDS = list(range(42, 47))


def load_records():
    records = {}
    for p in sorted((RUN / 'reference_metrics').glob('*.json')):
        x = json.loads(p.read_text())
        g = x.get('group', x.get('spec', {}).get('group'))
        s = x.get('seed', x.get('spec', {}).get('seed'))
        if g and s is not None:
            records[g, s, x.get('evaluation_year', 2023)] = x['metrics']
    for s in json.loads((RUN / 'manifest.json').read_text()):
        if s.get('stage') == 'tune':
            continue
        for split, year in [('val', 2023), ('test', 2024)]:
            p = RUN / 'jobs' / s['id'] / f'metrics_{split}.json'
            if p.exists():
                x = json.loads(p.read_text())
                assert not x.get('smoke_only')
                records[s['group'], s['seed'], year] = x['metrics']
    return records


def main():
    records = load_records()
    specs = json.loads((RUN / 'manifest.json').read_text())
    done = sum((RUN / 'jobs' / s['id'] / 'DONE.json').exists() for s in specs)
    rows = []
    for year in [2023, 2024]:
        full = {k: np.mean([records['H-FULL', s, year][k] for s in SEEDS]) for k in METRICS}
        for kind, groups in [('baseline', BASELINES), ('ablation', ABLATIONS), ('extra', EXTRA)]:
            for group, name, description in groups:
                seeds = SEEDS if group != 'PERSISTENCE' else sorted(s for g, s, y in records if (g, y) == (group, year))[:1]
                assert seeds and all((group, s, year) in records for s in seeds), (group, year, 'incomplete')
                values = [records[group, s, year] for s in seeds]
                row = dict(year=year, kind=kind, group=group, name=name, description=description, seeds=seeds)
                for k in METRICS:
                    row[k] = float(np.mean([v[k] for v in values]))
                    row[k + '_seed_sd'] = float(np.std([v[k] for v in values], ddof=1)) if len(values) > 1 else 0.
                    row[k + '_change_vs_full_percent'] = float(100 * (row[k] / full[k] - 1))
                rows.append(row)
    atomic_json(RUN / 'analysis/READABLE_TABLES.json', dict(timestamp=now(),task_completion=[done,len(specs)],rows=rows))
    for kind in ['baseline', 'ablation', 'extra']:
        with (RUN / f'analysis/{kind}_common5.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(r for r in rows if r['kind'] == kind)
    lines = ['# 已完成实验：基线与消融汇总', '', f'更新：{now()}；本轮任务完成 {done}/{len(specs)}。', '',
        '本轮包括 121 次训练（含 16 次线性基线调参）和 30 次旧权重复评。下列正式模型均使用相同五个随机种子 42–46；持续性预测无需训练。十种子结果另列，不与五种子均值混用。', '',
        '固定 55 个风场共同训练；2022 训练，2023 按逐风场平均绝对误差选检查点，2024 冻结权重跨年复评。未按模型胜负删除风场。功率按容量归一化，指标先逐风场计算再等权平均；数值越小越好。高功率指预测起点容量因子至少 0.75。', '',
        '正式训练基线都使用未来气象、本站历史功率、同一份周边功率缓存及站点／时间信息。周边缓存是近邻汇总历史与三环统计，不是八条独立邻站轨迹。完整窗口版本额外显式读取八个小时节点。', '',
        '平均绝对误差（MAE）；均方根误差（RMSE）。所有相对变化按“该行模型 / 完整模型 − 1”计算；正值表示去除或替换后更差。', '']
    for year in [2024, 2023]:
        lines += [f'## {year} 年基线比较（共同五种子）', '', '| 模型 | 简要说明 | 平均绝对误差 | 均方根误差 | 高功率均方根误差 |', '|---|---|---:|---:|---:|']
        for r in rows:
            if (r['kind'], r['year']) == ('baseline', year):
                lines.append(f"| {r['name']} | {r['description']} | {r['mae']:.8f} | {r['rmse']:.8f} | {r['high_power_rmse']:.8f} |")
        lines += ['', f'## {year} 年消融比较（共同五种子）', '', '| 改动 | 保留或移除的路径 | 平均绝对误差 | 均方根误差 | 平均绝对误差变化 | 均方根误差变化 | 高功率均方根误差变化 |', '|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if (r['kind'], r['year']) == ('ablation', year):
                lines.append(f"| {r['name']} | {r['description']} | {r['mae']:.8f} | {r['rmse']:.8f} | {r['mae_change_vs_full_percent']:+.4f}% | {r['rmse_change_vs_full_percent']:+.4f}% | {r['high_power_rmse_change_vs_full_percent']:+.4f}% |")
        lines += ['', f'### {year} 年迁移及二乘二补充对照', '', '| 模型 | 说明 | 平均绝对误差 | 均方根误差 | 高功率均方根误差 |', '|---|---|---:|---:|---:|']
        for r in rows:
            if (r['kind'], r['year']) == ('extra', year):
                lines.append(f"| {r['name']} | {r['description']} | {r['mae']:.8f} | {r['rmse']:.8f} | {r['high_power_rmse']:.8f} |")
    lines += ['', '## 十种子稳定性复核', '', '以下用十个相同随机种子 42–51 配对。相对变化为“完整模型 / 对照 − 1”；双指标同时改善的种子数及完整门槛不能由均值代替。', '',
        '| 年份 | 对照模型 | 平均绝对误差变化 | 均方根误差变化 | 高功率均方根误差变化 | 两指标同时改善的种子 | 完整门槛 |', '|---|---|---:|---:|---:|---:|---|']
    names = {g:n for g,n,_ in ABLATIONS + EXTRA + BASELINES}
    for x in json.loads((RUN/'analysis/COMPARISONS.json').read_text()):
        if x['candidate'] != 'H-FULL' or len(x['seeds']) != 10:
            continue
        d=x['delta_percent']
        lines.append(f"| {x['year']} | {names[x['reference']]} | {d['mae']:+.4f}% | {d['rmse']:+.4f}% | {d['high_power_rmse']:+.4f}% | {x['dual_seed_wins']}/10 | {'通过' if x['gate'] else '未通过'} |")
    lines += ['', '## 高出力后的下行段与预测时段', '',
        '2024 年共同五种子。下行段指起点容量因子至少 0.75 且未来真实出力低于当前；此列按样本单元汇总误差后计算，区别于主表逐风场等权指标。三个预测时段仍按逐风场指标平均。', '',
        '| 模型 | 高出力后下行段均方根误差 | 0–1 小时均方根误差 | 1–3 小时均方根误差 | 3–6 小时均方根误差 |', '|---|---:|---:|---:|---:|']
    for g in ['R0', 'H-FULL', 'TF_R3', 'TF_E2L', 'TF-WINDOW']:
        values = [records[g,s,2024] for s in SEEDS]
        down = np.mean([np.sqrt(v['diagnostics']['bins']['high_down']['mse']) for v in values])
        horizons = [np.mean([v['horizons'][h]['rmse'] for v in values]) for h in ['0_1h','1_3h','3_6h']]
        lines.append(f"| {names[g]} | {down:.8f} | {horizons[0]:.8f} | {horizons[1]:.8f} | {horizons[2]:.8f} |")
    lines += ['', '## 结论边界', '',
        '- 完整模型相对同骨干线性插值，两个年份十种子均通过预定门槛；整体预测收益有支持。',
        '- 去掉周边功率、显式运行状态、气象派生特征或解码端运行／区域路径均使 2024 平均误差变差，支持这些信息与路径的预测价值。',
        '- 去掉修正模块的状态摘要或反转其历史顺序并未使 2024 误差变差；完整天气窗口与仅局部小时段也接近。因此不能宣称已分别证明这些细节必要。反转保留了历史值且可逆，不等于历史信息无用。',
        '- 相对 Transformer 的整体误差均值更低，但部分比较未通过高功率误差及跨种子稳定性门槛，不能写成全指标稳定胜出。',
        '- 两设计二乘二实验的误差交互项为正，显示收益存在重叠，不支持超加性的协同增益。数据见 INTERACTIONS.json。',
        '- 当前完整模型是确定性低秩特征修正，并非随机生成式气象超分辨率，也不是站点独立的低秩权重适配。此前修正教师梯度的流匹配实验已完成，但不能借用此处收益宣称流匹配成功。',
        '- 气象线性编码曾从七种编码中按 2023 选择；2024 也曾被查看，因此称跨年复评，不称盲测。增加随机种子不增加独立天气年份。', '',
        '逐种子／逐风场原始结果见 jobs 与 reference_metrics；标准差、完整精度数值和内部编号映射见 READABLE_TABLES.json 与三个 CSV。评估加载修复审计见 ../operations/evaluation_loader_repair_20261118/。']
    (RUN/'analysis/BASELINE_ABLATION_SUMMARY_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(tasks_completed=done,tasks_total=len(specs),table_rows=len(rows))))


if __name__ == '__main__':
    main()
