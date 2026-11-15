from settings import *
from collect_physics import collect
from paired_comparisons import comparison

def run():
    freeze=require_test_freeze();state=collect();data={}
    for s in json.loads((RUN/'manifest.json').read_text()):
        p=JOBS/s['id']
        if not (p/'SUCCESS.json').exists():continue
        test=json.loads((p/'test.json').read_text())
        assert test['year']==2024 and test['selection_year']==2023
        assert test['source_sha256']==freeze['parent_checkpoints'][s['source_job']]['sha256']
        row=test['metrics'].copy()
        row['high_rmse']=next((x['rmse'] for x in test['diagnostics'] if x['group']=='current_power' and x['bin']=='high' and x['horizon']=='all'),None)
        data[s['configuration'],s['site'],s['seed']]=row
    selected=freeze['selected'];pairs=[comparison(data,b,selected) for b in ['W0_H32','TF_R3_RAW']]
    pairs.append(comparison(data,'TF_R3_RAW','W0_H32'))
    result=dict(timestamp=now(),completed=state['completed'],planned=1680,complete=state['completed']==1680 and not state['failed'],selected_on2023=selected,comparisons=pairs,interpretation='Exploratory2024 reevaluation after prior test inspection; no test selection or replacement of original main table')
    atomic_json(RUN/'analysis/EXPLORATORY_2024_RESULTS.json',result)
    lines=['# 输出头实验：冻结后的探索性2024复评',f'完成{state["completed"]}/1680。候选模型{selected}仅由2023判定选择；2024剔除11月。','', '|参照|变体|完整配对风场|MAE变化|RMSE变化|','|---|---|---:|---:|---:|']
    for x in pairs:lines.append(f'|{x["base"]}|{x["arm"]}|{x["paired_farms"]}|'+(f'{x["mae"]["relative_change"]*100:+.3f}%|{x["rmse"]["relative_change"]*100:+.3f}%|' if 'mae' in x else '待齐|待齐|'))
    lines+=['','本轮设计前已查看过2024结果，因此属于探索性复评。未据本次2024指标继续选点或换模型；原冻结主表保持不变。逐站、时效、场景及区间详见CSV和JSON。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':run()
