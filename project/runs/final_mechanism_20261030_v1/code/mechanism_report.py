from settings import *
from collect_physics import collect,csv_write
import numpy as np

def read_data():
    data={}
    for s in json.loads((RUN/'manifest.json').read_text()):
        p=JOBS/s['id']
        if not (p/'SUCCESS.json').exists():continue
        m=json.loads((p/'metrics.json').read_text())
        cell=s['variant'][3:] if s['variant'].startswith('R3_H') else 'H1W1'
        data[s['backbone'],cell,s['site'],s['seed']]=m['selections']['mae']['metrics']
    return data

def interaction(data,backbone,seeds):
    cells=['H1W1','H1W0','H0W1','H0W0']
    farms=[i for i in range(56) if all((backbone,c,i,s) in data for c in cells for s in seeds)]
    result=dict(backbone=backbone,seeds=seeds,paired_farms=len(farms),complete=len(farms)==56,passed=False)
    if len(farms)!=56:return result
    rng=np.random.default_rng(301030);indices=rng.integers(0,56,(4000,56));measures={}
    for horizon in ['all','0_1h','1_3h','3_6h']:
        measures[horizon]={}
        for metric in ['mae','rmse']:
            def value(c,i,s):
                m=data[backbone,c,i,s]
                return (m if horizon=='all' else m['horizons'][horizon])[metric]
            bycell={c:np.array([[value(c,i,s) for s in seeds] for i in farms]) for c in cells}
            gain_real=bycell['H1W0']-bycell['H1W1'];gain_fixed=bycell['H0W0']-bycell['H0W1'];effect=gain_real-gain_fixed
            byfarm=effect.mean(1);den=float(bycell['H1W1'].mean());inter=float(effect.mean())
            measures[horizon][metric]=dict(cells={c:float(x.mean()) for c,x in bycell.items()},weather_gain_real_history=float(gain_real.mean()),weather_gain_fixed_history=float(gain_fixed.mean()),gain_ratio=float(gain_real.mean()/gain_fixed.mean()) if abs(gain_fixed.mean())>1e-12 else None,interaction=inter,interaction_relative_to_E11=inter/den,positive_farms=int((byfarm>0).sum()),positive_seed_macros=int((effect.mean(0)>0).sum()),per_farm_interaction=byfarm.tolist(),paired_farm_bootstrap95=np.quantile(byfarm[indices].mean(1),[.025,.975]).tolist())
    result['measures']=measures
    result['passed']=all(measures['all'][m]['interaction_relative_to_E11']>=.001 and measures['all'][m]['positive_farms']>=29 and measures['all'][m]['weather_gain_real_history']>0 for m in ['mae','rmse'])
    return result

def run():
    state=collect();data=read_data();rows=[interaction(data,b,SEEDS[:n]) for n in [2,10] for b in ['W0','TF']]
    complete=state['completed']==4480 and not state['failed']
    result=dict(timestamp=now(),completed=state['completed'],planned=4480,reused=1120,new_training_completed=state['completed']-1120,complete=complete,analyses=rows,test_year_evaluated=False,interpretation=require_protocol()['interpretation'])
    atomic_json(RUN/'analysis/MECHANISM_RESULTS.json',result)
    lines=['# 最终模型的历史×天气交互',f'更新：{now()}。总计{state["completed"]}/4480，其中1120为复用参照；新增训练{state["completed"]-1120}/3360。','2022训练；2023去11月验证；所有格保留真实当前功率输出锚点和区域实况。交互I=(E10−E11)−(E00−E01)，正值表示真实历史增强天气的预测价值。','', '|骨干|种子数|配对风场|MAE交互/E11|RMSE交互/E11|MAE/RMSE正交互风场|通过|','|---|---:|---:|---:|---:|---:|---|']
    for x in rows:
        if x['complete']:
            a=x['measures']['all']['mae'];b=x['measures']['all']['rmse']
            lines.append(f'|{x["backbone"]}|{len(x["seeds"])}|56|{100*a["interaction_relative_to_E11"]:.3f}%|{100*b["interaction_relative_to_E11"]:.3f}%|{a["positive_farms"]}/{b["positive_farms"]}|{x["passed"]}|')
        else:lines.append(f'|{x["backbone"]}|{len(x["seeds"])}|{x["paired_farms"]}|待齐|待齐|待齐|待齐|')
    lines+=['','二种子结果仅作阶段性检查，十种子全部按事前计划运行，包括负结果。不给固定天气格泄露区域注意力query里的真实u/v；固定历史格同步重算STATE。机制结论限定为预测交互，不能解释为状态的因果作用，也不能从正交互直接定位到某个内部响应层。','不引入新门控。所有格完成后冻结，再按同一检查点进行2024探索性复评；不根据2024结果重新选择模型。']
    (RUN/'analysis/REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':run()
