"""Audit the complete neural subset without declaring the unfinished ARX round complete."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math

RUN = Path(__file__).resolve().parents[1]
ANALYSIS = RUN / 'analysis'
TF = 'TF_z1_lr0.0002_dp0.05'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.review.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = read(RUN / 'manifest.json')
    records = {}
    expected_hashes = {}
    counts = collections.Counter()

    def register(path, expected):
        path = str(Path(path).resolve())
        assert path not in expected_hashes or expected_hashes[path] == expected
        expected_hashes[path] = expected

    for spec in manifest:
        if spec['task'] == 'arx':
            continue
        out = RUN / 'jobs' / spec['id']
        assert (out / 'SUCCESS.json').exists(), spec['id']
        assert read(out / 'spec.json') == spec
        result = read(out / 'metrics.json')
        assert not result['smoke_only']
        diagnostic = read(out / 'diagnostics.json')
        high = [r for r in diagnostic['rows'] if r['group'] == 'current_power'
                and r['bin'] == 'high' and r['horizon'] == 'all']
        assert len(high) == 1
        key = (spec['configuration'], spec['variant'], spec['site'], spec['seed'])
        assert key not in records
        records[key] = (result['selections']['mae']['metrics'], high[0])
        if spec['task'] == 'train':
            assert not (out / 'latest.pt').exists()
            assert len(list(out.glob('epoch_*.pt'))) <= 2
            for selection in result['selections'].values():
                register(out / selection['checkpoint'], selection['sha256'])
        else:
            assert not list(out.glob('*.pt'))
            register(result['source_checkpoint'], result['source_sha256'])
        counts[spec['task']] += 1
    assert counts['train'] == 2352 and sum(counts.values()) == 2912
    for index, (path, expected) in enumerate(expected_hashes.items(), 1):
        assert digest(path) == expected, path
        if index % 500 == 0:
            print('verified weights', index, '/', len(expected_hashes), flush=True)
    frozen = read(RUN / 'SOURCE_FREEZE.json')['files']
    for path, expected in frozen.items():
        assert digest(path) == expected, path

    def compare(cc, cv, rc, rv, metric, horizon='all', high=False):
        candidates, references = [], []
        wins_seed = collections.Counter()
        for site in range(56):
            ca, re = [], []
            for seed in (42, 43):
                a = records[(cc, cv, site, seed)][1 if high else 0]
                b = records[(rc, rv, site, seed)][1 if high else 0]
                if horizon != 'all':
                    a, b = a['horizons'][horizon], b['horizons'][horizon]
                assert a['n'] == b['n'] and a['n'] > 0
                assert math.isfinite(a[metric]) and math.isfinite(b[metric])
                ca.append(a[metric]); re.append(b[metric])
                wins_seed[seed] += a[metric] < b[metric]
            candidates.append(sum(ca) / 2); references.append(sum(re) / 2)
        return dict(candidate_configuration=cc, candidate_variant=cv,
                    reference_configuration=rc, reference_variant=rv,
                    metric=metric, horizon=horizon, group='current_power_high' if high else 'all',
                    candidate=sum(candidates) / 56, reference=sum(references) / 56,
                    effect_percent=100 * (sum(candidates) / sum(references) - 1),
                    improved_farms=sum(a < b for a, b in zip(candidates, references)),
                    improved_seed42=wins_seed[42], improved_seed43=wins_seed[43], farms=56)

    # Recompute from source JSON, independent of the pandas aggregate path.
    exported = list(csv.DictReader((ANALYSIS / 'all_actual_per_farm_seed_effects.csv').open()))
    checked = 0
    for row in exported:
        if row['selection'] != 'mae':
            continue
        c = compare(row['candidate_configuration'], row['candidate_variant'],
                    row['reference_configuration'], row['reference_variant'],
                    row['metric'], row['horizon'])
        assert abs(c['effect_percent'] - float(row['effect_percent'])) < 1e-9
        assert c['improved_farms'] == int(row['improved_farms'])
        checked += 1

    effects, gates = [], {}
    for variant in ['R1', 'R2', 'R3', 'R4']:
        comparisons = {}
        for ref in ['N0', 'N1', 'N3', 'C1_' + variant, 'C2_' + variant]:
            comparisons[ref] = {}
            for metric in ['mae', 'rmse']:
                c = compare('W0', variant, 'W0', ref, metric)
                effects.append(c); comparisons[ref][metric] = c
        h = compare('W0', variant, 'W0', 'N3', 'rmse', high=True)
        effects.append(h)
        dual = all(comparisons[r][m]['effect_percent'] <= -.1
                   and comparisons[r][m]['improved_farms'] >= 29
                   for r in ['N1', 'N3'] for m in ['mae', 'rmse'])
        controls = all(comparisons[r][m]['effect_percent'] < 0
                       for r in ['C1_' + variant, 'C2_' + variant] for m in ['mae', 'rmse'])
        gates[variant] = dict(comparisons=comparisons, dual_metric_gate=dual,
                              beats_matched_controls=controls,
                              high_power_rmse_vs_N3_percent=h['effect_percent'],
                              high_power_gate=h['effect_percent'] <= .1,
                              passes=dual and controls and h['effect_percent'] <= .1)
    edges = [('W0', 'R1', 'W0', 'R1_RAWMAP'), ('W0', 'R1', 'W0', 'R1_SELF'),
             ('W0', 'R2', 'W0', 'R2_PAIR'), ('W0', 'R3', 'W0', 'R3_RAW'),
             ('W0', 'R3_RAW', 'W0', 'N3'), ('W0', 'R3_RAW', 'W0', 'N1'),
             ('W0', 'R4', 'W0', 'R3')]
    edges += [(TF, v, TF, r) for v in ['R1', 'R2', 'R3', 'R4'] for r in ['N1', 'N3']]
    edges += [('W0', v, TF, v) for v in ['N0', 'N1', 'N3', 'R1', 'R2', 'R3', 'R4']]
    for edge in edges:
        for metric in ['mae', 'rmse']:
            effects.append(compare(*edge, metric))
    for cc, cv in [('W0', 'R3_RAW')] + [(TF, v) for v in ['R1', 'R2', 'R3', 'R4']]:
        effects.append(compare(cc, cv, cc, 'N3', 'rmse', high=True))
    for cc, cv in [('W0', 'R1'), ('W0', 'R3'), ('W0', 'R3_RAW'), (TF, 'R3')]:
        for horizon in ['0_1h', '1_3h', '3_6h']:
            for metric in ['mae', 'rmse']:
                effects.append(compare(cc, cv, cc, 'N3', metric, horizon))
    with (ANALYSIS / 'NEURAL_REVIEW_EFFECTS_20261026.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    # Task status is a live snapshot; never promote the whole round on subset completion.
    status = read(RUN / 'STATUS.json')
    failures = {j: read(RUN / 'jobs' / j / 'FAILURE.json') for j in status['failed_jobs']}
    audit = dict(timestamp=datetime.datetime.now().astimezone().isoformat(),
                 scope='2352 neural fits and 560 reference replays only', passed=True,
                 neural_complete=True, round_complete=False,
                 training_verified=counts['train'], reference_verified=sum(counts.values()) - counts['train'],
                 unique_selected_weight_hashes_verified=len(expected_hashes),
                 scientific_freeze_files_verified=len(frozen), independent_effect_rows_verified=checked,
                 status_snapshot={k: status[k] for k in ['completed', 'planned', 'failed', 'arx_completed']},
                 unresolved_arx_failures=failures, gates=gates, opens_2024=False,
                 interpretation='R1 primary fails. R3 predeclared secondary passes neural gates; R3_RAW is stronger, so residual necessity is not supported. ARX remains unfinished.')
    write(ANALYSIS / 'NEURAL_SUBSET_AUDIT_20261026.json', audit)

    def lookup(cc, cv, rc, rv, metric, high=False, horizon='all'):
        return compare(cc, cv, rc, rv, metric, horizon, high)

    def change(cc, cv, rc, rv, metric, high=False, horizon='all'):
        return f"{lookup(cc, cv, rc, rv, metric, high, horizon)['effect_percent']:+.4f}%"

    lines = ['# 区域残差筛选：神经网络部分复核（整轮尚未完成）', '',
             f"核验时间：{audit['timestamp']}。2,352 项神经训练和 560 项参考权重重放全部完成；ARX 仅 {status['arx_completed']}/56 项成功，{status['failed']} 项未收敛失败，其余仍运行。不能写成 2,968 项全部成功。", '',
             '所有数值来自 2023 选择年，同一 MAE-selected 权重同时计算 MAE/RMSE；两个 seed 先在站内平均，再按 56 站等权。负值为改善。新 2024 复评仍关闭，本文不是最终选型冻结。', '',
             '## 预先声明的 W0 判定', '',
             '|变体|对 N1 MAE / RMSE|对 N3 MAE / RMSE|对 N3 改善站数 MAE / RMSE|高功率 RMSE 对 N3|判定|',
             '|---|---:|---:|---:|---:|---|']
    for v, g in gates.items():
        q = g['comparisons']; n1 = q['N1']; n3 = q['N3']
        lines.append(f"|{v}|{n1['mae']['effect_percent']:+.4f}% / {n1['rmse']['effect_percent']:+.4f}%|{n3['mae']['effect_percent']:+.4f}% / {n3['rmse']['effect_percent']:+.4f}%|{n3['mae']['improved_farms']}/56 / {n3['rmse']['improved_farms']}/56|{g['high_power_rmse_vs_N3_percent']:+.4f}%|{'通过神经部分全部门槛' if g['passes'] else '未通过'}|")
    lines += ['', 'R1 是预设主机制，不能用次要方案 R3 的成功替代主假设检验。R1 的 RMSE 改善未达到 0.1%，高功率 RMSE 还退化 0.1671%。R3 是预先声明的次要候选：相对 N1/N3 的双指标、改善站数、匹配 C1/C2 与高功率保护均通过；其最终升阶仍待整轮审查及单独冻结。', '',
              '## 对照与机制解释', '', '|对比|MAE|RMSE|', '|---|---:|---:|']
    for cv, rv in [('R1','C1_R1'),('R1','C2_R1'),('R3','C1_R3'),('R3','C2_R3'),
                   ('R1','R1_RAWMAP'),('R1','R1_SELF'),('R2','R2_PAIR'),
                   ('R3','R3_RAW'),('R3_RAW','N3'),('R4','R3')]:
        lines.append(f"|W0 {cv} / {rv}|{change('W0',cv,'W0',rv,'mae')}|{change('W0',cv,'W0',rv,'rmse')}|")
    lines += ['', 'R3_RAW 与 R3 有相同空间范围和投影结构，但仅保留原功率统计。R3 在其基础上加入映射/残差信息后反而变差，因此当前证据不支持“显式邻站残差优于原始区域实况”或“已识别纯 NWP 误差”的主张。R3_RAW 是消融组中的发现，不将它事后改称预设主机制；TF* 未运行该组，不能声称它已经跨骨干验证。', '',
              'R1_RAWMAP 的统计量并非严格信息等价：残差标准差还含功率与映射功率的协方差信息。R2 与 R2_PAIR 才是逐站同信息对照，而其增量接近零。C2 使用另一日的 2022 档案，存在年际分布变化，不能把它单独解释为纯参数量控制。', '',
              'R4 对 R3 退化，继续把方向项作为负结果；不写成已经证实的平流传播机制。区域实况有用与纯天气预报误差被识别，是不同强度的结论。', '',
              '## TF* 与时效', '', '|TF* 变体（相对 TF* N3）|MAE|RMSE|高功率 RMSE|', '|---|---:|---:|---:|']
    for v in ['R1','R2','R3','R4']:
        lines.append(f"|{v}|{change(TF,v,TF,'N3','mae')}|{change(TF,v,TF,'N3','rmse')}|{change(TF,v,TF,'N3','rmse',True)}|")
    lines += ['', 'TF* 的 R3 改善方向一致；TF* 没有本轮匹配 C1/C2 与 R3_RAW，因此不能把 W0 的完整机制对照结论直接外推。', '',
              '|候选相对同骨干 N3|时效|MAE|RMSE|', '|---|---|---:|---:|']
    for cc, cv in [('W0','R1'),('W0','R3'),('W0','R3_RAW'),(TF,'R3')]:
        for h in ['0_1h','1_3h','3_6h']:
            lines.append(f"|{'TF*' if cc == TF else cc} {cv}|{h}|{change(cc,cv,cc,'N3','mae',horizon=h)}|{change(cc,cv,cc,'N3','rmse',horizon=h)}|")
    lines += ['', '## ARX 与完成边界', '',
              '7 个失败均为 alpha=0.0003 的收敛上限问题，不是新的 OOM：s09、s37、s45、s50、s52、s53 在 LOCAL 阶段失败，s33 在 REG_RES 阶段失败。现有成功样本不是随机站点子集，16 站上的 ARX 比较不能代表完整 56 站。未删去失败站、未放宽容差、未更改冻结的迭代上限，也未按原样盲目重启必然重现的确定性失败。', '',
              '运行中的 ARX 保留。原自动最终审计要求 2,968 项全部 SUCCESS，因此当前不会生成整轮完成标志。若修复数值求解，应另存求解器修订和验证记录，保持目标函数、特征、alpha 网格、训练年份及选择规则，并核对收敛与已成功站的一致性；不能覆盖本轮冻结证据或直接接受未收敛系数。', '',
              '## 审计与后续判断', '',
              f"已核对 {len(expected_hashes)} 份不同选定权重的 SHA256、154 项冻结文件、全部神经任务 spec 与诊断，以及 {checked} 行从原始 JSON 独立重算的效应量。审计范围仅为神经网络部分；没有写整轮 COMPLETION_AUDIT，也没有打开 2024。", '',
              '当前可保留的结论是：区域实况补充有效，三距离环汇总值得作为次要候选复评；显式残差与方向项没有展示必要性。R3_RAW 值得在下一份冻结清单中作为简单候选，但尚不能凭筛选年优势改写已预设的主假设。', '',
              '文件：NEURAL_SUBSET_AUDIT_20261026.json、NEURAL_REVIEW_EFFECTS_20261026.csv；完整原值和场景比较见本目录 all_actual_per_farm_seed.csv、diagnostics_per_farm_seed_effects.csv。']
    (ANALYSIS / 'NEURAL_REVIEW_20261026_CN.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['gates','unresolved_arx_failures']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
