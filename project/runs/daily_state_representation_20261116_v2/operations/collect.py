"""Predeclared gates. Missing jobs never count as evidence for a passing stage."""
from pathlib import Path
import sys,json,csv,collections
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
import numpy as np
from settings import JOBS,ASSETS,atomic_json,now,sha


def records():
    groups=collections.defaultdict(dict)
    for group in ['R0','TF_R3']:
        for seed in range(42,47):
            groups[group][seed]=json.loads((RUN/'reference_metrics'/group/str(seed)/'metrics.json').read_text())
    for spec in json.loads((RUN/'manifest.json').read_text()):
        out=JOBS/spec['id']
        if not (out/'SUCCESS.json').exists():continue
        value=json.loads((out/'metrics.json').read_text())
        assert value['requested_spec']==spec and not value['smoke_only']
        groups[spec['group']][spec['seed']]=value
    return dict(groups)


def compare(groups,a,b):
    assert set(groups[a])==set(groups[b])==set(range(42,47))
    out=dict(candidate=a,reference=b)
    seed_improvements=[]
    for metric in ['mae','rmse','high_power_rmse']:
        key={'mae':'per_farm_mae','rmse':'per_farm_rmse','high_power_rmse':'high_power_per_farm_rmse'}[metric]
        x=np.array([groups[a][seed]['metrics'][key] for seed in range(42,47)])
        y=np.array([groups[b][seed]['metrics'][key] for seed in range(42,47)])
        assert x.shape==y.shape==(5,55) and np.isfinite(x).all() and np.isfinite(y).all()
        out[metric+'_change_percent']=float(100*(x.mean()/y.mean()-1))
        out[metric+'_farm_wins']=int((x.mean(0)<y.mean(0)).sum())
        out[metric+'_per_seed_change_percent']=(100*(x.mean(1)/y.mean(1)-1)).tolist()
        if metric!='high_power_rmse':seed_improvements.append(x.mean(1)<y.mean(1))
    out['both_metrics_seed_wins']=int(np.logical_and(*seed_improvements).sum())
    out['dual_improvement']=out['mae_change_percent']<0 and out['rmse_change_percent']<0
    out['main_pass']=bool(out['mae_change_percent']<=-.1 and out['rmse_change_percent']<=-.1 and
        out['mae_farm_wins']>=28 and out['rmse_farm_wins']>=28 and out['both_metrics_seed_wins']>=4 and
        out['high_power_rmse_change_percent']<=.1)
    return out


def enable(name,payload):
    path=RUN/'operations'/name
    if not path.exists():atomic_json(path,dict(timestamp=now(),**payload))


def collect():
    groups=records();complete={g for g,s in groups.items() if set(s)==set(range(42,47))}
    state=dict(timestamp=now(),complete_groups=sorted(complete),stage1_complete=False,
               stage2_enabled=False,cross_backbone_enabled=False,loss_enabled=False,finished=False)
    rows=[]
    for group in sorted(complete):
        for reference in ['R0','TF_R3','TF_E2L']:
            if reference in complete and group!=reference:rows.append(compare(groups,group,reference))
    def dual(a,b):
        row=compare(groups,a,b);rows.append(row);return row['dual_improvement']
    def main_pass(group):
        return all(compare(groups,group,ref)['main_pass'] for ref in ['R0','TF_R3','TF_E2L'] if group!=ref)
    first={'R0+H','R1','R2','R3','TF_E2L'}
    if first<=complete:
        state['stage1_complete']=True
        state['r3_mechanism']=all([dual('R3','R2'),dual('R3','R0+H')])
        state['r3_main_pass']=main_pass('R3')
        state['stage2_enabled']=compare(groups,'R3','R2')['dual_improvement']
        if state['stage2_enabled']:enable('ENABLE_STAGE2.json',dict(reason='R3 improves both metrics versus R2'))
        second={'R4','R4-noS','R4-wrongS'}
        stage2_done=not state['stage2_enabled'] or second<=complete
        state['stage2_complete_or_skipped']=stage2_done
        if state['stage2_enabled'] and second<=complete:
            state['r4_mechanism']=all([dual('R4',g) for g in ['R3','R4-noS','R4-wrongS']])
            state['r4_main_pass']=main_pass('R4')
            state['cross_backbone_enabled']=state['r4_mechanism'] and state['r4_main_pass']
            if state['cross_backbone_enabled']:enable('ENABLE_TF_R4.json',dict(reason='R4 passes mechanism and main gates'))
        if stage2_done:
            candidate='R4' if state.get('r4_mechanism') else 'R3' if state['r3_mechanism'] else 'R1' if main_pass('R1') else 'R0'
            path=ASSETS/'selected_candidate.json'
            value=dict(representation=candidate,rule=json.loads((RUN/'PROTOCOL.json').read_text())['loss_candidate_rule'])
            if path.exists():assert json.loads(path.read_text())==value,'Candidate selection is immutable'
            else:atomic_json(path,value)
            enable('ENABLE_LOSS.json',dict(candidate=candidate,selection_sha256=sha(path)))
            state.update(loss_enabled=True,loss_candidate=candidate)
            if {'LOSS_CANDIDATE','LOSS_TF'}<=complete:
                versus_tf=compare(groups,'LOSS_CANDIDATE','LOSS_TF');rows.append(versus_tf)
                versus_self=compare(groups,'LOSS_CANDIDATE',candidate);rows.append(versus_self)
                state['adopt_mixed_loss']=bool(versus_tf['dual_improvement'] and versus_self['mae_change_percent']<=0 and versus_self['rmse_change_percent']<=0)
                state['finished']=not state['cross_backbone_enabled'] or 'TF_R4' in complete
                state['method_qualified']=main_pass(candidate) if candidate!='R0' else False
    out=RUN/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'GATES.json',state);atomic_json(out/'COMPARISONS.json',rows)
    summaries=[]
    for group in sorted(complete):
        values=[groups[group][s] for s in range(42,47)]
        summaries.append(dict(group=group,seeds=5,**{k:float(np.mean([v['metrics'][k] for v in values])) for k in ['mae','rmse','high_power_rmse']},
                              selected_epochs=[v['checkpoint_epoch'] for v in values]))
    atomic_json(out/'GROUPS.json',summaries)
    flat=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
    if flat:
        with (out/'comparisons.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    text=['# V2 实验进度与判定','',f'更新时间：{state["timestamp"]}。完整组：'+', '.join(sorted(complete))+'。',
          '', '仅完整的五种子组参与比较；2023 同时用于选择，结论不是独立测试结果。',
          '', '| 组 | MAE | RMSE | 高功率 RMSE |','|---|---:|---:|---:|']
    text += [f'| {r["group"]} | {r["mae"]:.8f} | {r["rmse"]:.8f} | {r["high_power_rmse"]:.8f} |' for r in summaries]
    text += ['', '详细阈值与阶段状态见 [GATES.json](GATES.json)，配对结果见 [comparisons.csv](comparisons.csv)。',
             '每项任务的 metrics.json 另含整点/非整点、大变化跟上比例、高功率上行/下行及生成式离散度。']
    (out/'README_CN.md').write_text('\n'.join(text)+'\n')
    return state

if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
