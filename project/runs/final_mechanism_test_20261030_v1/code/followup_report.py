from settings import *
from collect_physics import collect
from interaction_stats import interaction

def run():
    freeze=require_test_freeze();state=collect();data={}
    for s in json.loads((RUN/'manifest.json').read_text()):
        p=JOBS/s['id']
        if not (p/'SUCCESS.json').exists():continue
        t=json.loads((p/'test.json').read_text())
        assert t['year']==2024 and t['selection_year']==2023
        assert t['source_sha256']==freeze['parent_checkpoints'][s['source_job']]['sha256']
        cell=s['variant'][3:] if s['variant'].startswith('R3_H') else 'H1W1'
        data[s['backbone'],cell,s['site'],s['seed']]=t['metrics']
    rows=[interaction(data,b,SEEDS) for b in ['W0','TF']]
    complete=state['completed']==4480 and not state['failed']
    result=dict(timestamp=now(),completed=state['completed'],planned=4480,complete=complete,analyses=rows,evaluation_year=2024,interpretation='Exploratory2024: all8 predeclared cells; no2024 model selection')
    atomic_json(RUN/'analysis/MECHANISM_2024_RESULTS.json',result)
    lines=['# 最终模型历史×天气交互：2024探索性复评',f'完成{state["completed"]}/4480。每格56站×10种子；2024剔除11月。所有格保留真实当前功率输出锚点和区域实况。','', '|骨干|完整配对风场|MAE交互/E11|RMSE交互/E11|MAE/RMSE正交互风场|通过|','|---|---:|---:|---:|---:|---|']
    for x in rows:
        if x['complete']:
            a=x['measures']['all']['mae'];b=x['measures']['all']['rmse']
            lines.append(f'|{x["backbone"]}|56|{100*a["interaction_relative_to_E11"]:.3f}%|{100*b["interaction_relative_to_E11"]:.3f}%|{a["positive_farms"]}/{b["positive_farms"]}|{x["passed"]}|')
        else:lines.append(f'|{x["backbone"]}|{x["paired_farms"]}|待齐|待齐|待齐|待齐|')
    lines+=['','本轮事前冻结8格并全部报告，包括负结果。因为设计前已查看过2024，不能称作从未见过的独立测试。结论仅为给定真实当前功率和区域实况后，本站历史上下文是否增强天气的预测价值。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':run()
