from pathlib import Path
import sys,json,math,collections
import numpy as np
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now

def collect():
 specs=json.loads((RUN/'manifest.json').read_text());records={}
 for p in sorted((RUN/'reference_metrics').glob('*.json')):
  x=json.loads(p.read_text());g=x.get('group',x.get('spec',{}).get('group'));seed=x.get('seed',x.get('spec',{}).get('seed'));year=x.get('evaluation_year',2023)
  if g and seed is not None:records[g,seed,year]=x['metrics']
 for s in specs:
  if s.get('stage')=='tune':continue
  out=RUN/'jobs'/s['id']
  for split,year in [('val',2023),('test',2024)]:
   p=out/f'metrics_{split}.json'
   if p.exists():
    x=json.loads(p.read_text())
    if not x.get('smoke_only'):records[s['group'],s['seed'],year]=x['metrics']
 summaries=[];comparisons=[]
 groups=sorted({k[0] for k in records})
 for year in [2023,2024]:
  for group in groups:
   seeds=sorted(s for g,s,y in records if (g,y)==(group,year))
   if not seeds:continue
   summaries.append(dict(year=year,group=group,seeds=seeds,**{k:float(np.mean([records[group,s,year][k] for s in seeds])) for k in ['mae','rmse','high_power_rmse']}))
  pairs=[('H-FULL',g) for g in groups if g!='H-FULL']+[('TF-H-FULL','TF_E2L'),('H-noDecoderState','R0-noDecoderState'),('R0','R0-noDecoderState')]
  for a,b in pairs:
   seeds=sorted(s for g,s,y in records if (g,y)==(a,year) and (b,s,year) in records)
   if len(seeds)<5:continue
   av=[records[a,s,year] for s in seeds];bv=[records[b,s,year] for s in seeds]
   delta={k:100*(np.mean([m[k] for m in av])/np.mean([m[k] for m in bv])-1) for k in ['mae','rmse','high_power_rmse']}
   farms={k:int(np.sum(np.mean([m['per_farm_'+k] for m in av],0)<np.mean([m['per_farm_'+k] for m in bv],0))) for k in ['mae','rmse']}
   wins=sum(all(x[k]<y[k] for k in ['mae','rmse']) for x,y in zip(av,bv))
   planned=10 if a in ['H-FULL','TF-H-FULL','R0'] and b in ['R0','TF_E2L','TF_R3','H-noS','H-SUMMARY'] else 5
   gate=len(seeds)>=planned and all(delta[k]<=-.1 and farms[k]>=28 for k in ['mae','rmse']) and wins>=math.ceil(.8*len(seeds)) and delta['high_power_rmse']<=.1
   comparisons.append(dict(year=year,candidate=a,reference=b,seeds=seeds,delta_percent=delta,farm_wins=farms,dual_seed_wins=wins,planned_pairs=planned,gate=bool(gate),complete=len(seeds)>=planned))
 interactions=[]
 for year in [2023,2024]:
  gs=['H-FULL','H-noDecoderState','R0','R0-noDecoderState'];ss=[s for s in range(42,47) if all((g,s,year) in records for g in gs)]
  if len(ss)==5:
   interactions.append(dict(year=year,seeds=ss,formula='H_FULL - H_noDecoderState - R0 + R0_noDecoderState; negative means beneficial interaction',
     **{k:float(np.mean([records[gs[0],s,year][k]-records[gs[1],s,year][k]-records[gs[2],s,year][k]+records[gs[3],s,year][k] for s in ss])) for k in ['mae','rmse']}))
 atomic_json(RUN/'analysis/GROUPS.json',summaries);atomic_json(RUN/'analysis/COMPARISONS.json',comparisons);atomic_json(RUN/'analysis/INTERACTIONS.json',interactions)
 lines=['# 论文证据补充：自动结果（运行中未齐的比较不下结论）','',f'更新时间：{now()}','',
 '2024 是历史上已查看过的跨年复评；2023 为开发／选模集。旧结果和新增结果按相同种子配对。',
 '扩展到 10 种子的比较必须齐满 10 个种子才判完整门槛；不能按部分结果提前筛选模型。','',
 '| 年份 | 模型 | seeds | MAE | RMSE | 高功率 RMSE |','|---|---|---:|---:|---:|---:|']
 for x in summaries:lines.append(f"| {x['year']} | {x['group']} | {len(x['seeds'])} | {x['mae']:.8f} | {x['rmse']:.8f} | {x['high_power_rmse']:.8f} |")
 lines+=['','| 年份 | 配对 | ΔMAE | ΔRMSE | Δ高功率 RMSE | 双胜 | 完整门槛 |','|---|---|---:|---:|---:|---:|---|']
 for x in comparisons:
  d=x['delta_percent'];status='通过' if x['gate'] else ('未通过' if x['complete'] else '待种子补齐')
  lines.append(f"| {x['year']} | {x['candidate']} / {x['reference']} | {d['mae']:+.4f}% | {d['rmse']:+.4f}% | {d['high_power_rmse']:+.4f}% | {x['dual_seed_wins']}/{len(x['seeds'])} | {status} |")
 lines+=['','高功率后的下行段、0–1/1–3/3–6 h 分层在每个任务的 metrics_val.json、metrics_test.json 中保留。','两设计的 2×2 交互结果见 INTERACTIONS.json；风场独立性的因果结论不由消融自动推出。']
 (RUN/'CURRENT_RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
 return dict(records=len(records),comparisons=len(comparisons))
if __name__=='__main__':collect()
