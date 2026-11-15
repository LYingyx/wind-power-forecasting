from settings import *
import numpy as np,collections

def ledger():
 out={}
 for s in json.loads((RUN/'manifest.json').read_text()):
  p=JOBS/s['id']
  if not (p/'SUCCESS.json').exists():continue
  m=json.loads((p/'metrics.json').read_text());key=(s['configuration'],s['site'],s['seed'])
  d=json.loads((p/'diagnostics.json').read_text())['rows']
  high=next((x['rmse'] for x in d if x['group']=='current_power' and x['bin']=='high' and x['horizon']=='all'),None)
  out[key]=dict(**m['selections']['mae']['metrics'],high_rmse=high)
 return out

def comparison(data,base,arm,seeds):
 pairs=[]
 for site in range(56):
  if not all((c,site,s) in data for c in [base,arm] for s in seeds):continue
  a=[data[(base,site,s)] for s in seeds];b=[data[(arm,site,s)] for s in seeds]
  pairs.append(dict(site=site,base={k:float(np.mean([r[k] for r in a])) for k in ['mae','rmse']},arm={k:float(np.mean([r[k] for r in b])) for k in ['mae','rmse']},high_base=None if any(r['high_rmse'] is None for r in a) else float(np.mean([r['high_rmse'] for r in a])),high_arm=None if any(r['high_rmse'] is None for r in b) else float(np.mean([r['high_rmse'] for r in b]))))
 if not pairs:return dict(base=base,arm=arm,n=0,passed=False)
 effects={};wins={};ci={};rng=np.random.default_rng(20261027);ix=rng.integers(0,len(pairs),(2000,len(pairs)))
 for k in ['mae','rmse']:
  a=np.array([r['base'][k] for r in pairs]);b=np.array([r['arm'][k] for r in pairs]);effects[k]=float(b.mean()/a.mean()-1);wins[k]=int(sum(b<a));ci[k]=np.quantile(b[ix].mean(1)/a[ix].mean(1)-1,[.025,.975]).tolist()
 hs=[r for r in pairs if r['high_base'] is not None and r['high_arm'] is not None]
 high=float(np.mean([r['high_arm'] for r in hs])/np.mean([r['high_base'] for r in hs])-1) if hs else None
 passed=len(pairs)==56 and all(effects[k]<=-.001 and wins[k]>28 for k in effects) and high is not None and high<=.005
 return dict(base=base,arm=arm,seeds=seeds,n=len(pairs),relative_change=effects,farm_wins=wins,farm_bootstrap95=ci,high_power_rmse_change=high,high_power_farms=len(hs),passed=passed,farms=pairs)

def interaction(data,seeds):
 rows=[];configs=['W0_SEL','W0_H1W0','W0_H0W1','W0_H0W0']
 for site in range(56):
  if not all((c,site,s) in data for c in configs for s in seeds):continue
  row=dict(site=site)
  for k in ['mae','rmse']:
   z=[np.mean([data[(c,site,s)][k] for s in seeds]) for c in configs];row[k]=float((z[1]-z[0])-(z[3]-z[2]))
  rows.append(row)
 return dict(n=len(rows),seeds=seeds,rows=rows,passed=len(rows)==56 and all(np.mean([r[k] for r in rows])>0 and sum(r[k]>0 for r in rows)>28 for k in ['mae','rmse']),interpretation='interaction conditional on actual current-power anchor; not causal proof')

def run():
 data=ledger();pairs=[]
 for b in ['W0','TF']:
  pairs += [(b+'_SEL',b+'_'+v) for v in ['N3','R3_RAW','AGE']]+[(b+'_N3',b+'_R3_RAW'),(b+'_C2_R3_RAW',b+'_R3_RAW')]
 pairs += [('TF_SEL','W0_SEL'),('TF_R3_RAW','W0_R3_RAW'),('W0_B0','W0_SEL')]
 pairs += [(a,b) for a,b in [('W0_AGE','W0_LOCAL_ERR'),('W0_C2_LOCAL_ERR','W0_LOCAL_ERR'),('W0_LOCAL_ERR','W0_REGION_ERR'),('W0_R3_RAW','W0_REGION_ERR'),('W0_C2_REGION_ERR','W0_REGION_ERR')]]
 results=[comparison(data,a,b,[42,43]) for a,b in pairs];ix=interaction(data,[42,43]);chosen=set()
 for b in ['W0','TF']:
  lookup={(r['base'],r['arm']):r for r in results}
  for v in ['N3','AGE','R3_RAW']:
   ok=lookup[b+'_SEL',b+'_'+v]['passed']
   if v=='R3_RAW':ok=ok and lookup[b+'_C2_R3_RAW',b+'_R3_RAW']['passed']
   if ok:chosen.update([b+'_SEL',b+'_'+v]);chosen.update([b+'_C2_R3_RAW',b+'_N3']) if v=='R3_RAW' else None
 if ix['passed']:chosen.update(['W0_SEL','W0_H1W0','W0_H0W1','W0_H0W0'])
 look={(r['base'],r['arm']):r for r in results}
 if look['W0_AGE','W0_LOCAL_ERR']['passed'] and look['W0_C2_LOCAL_ERR','W0_LOCAL_ERR']['passed']:chosen.update(['W0_AGE','W0_LOCAL_ERR','W0_C2_LOCAL_ERR'])
 if all(look[a,'W0_REGION_ERR']['passed'] for a in ['W0_LOCAL_ERR','W0_R3_RAW','W0_C2_REGION_ERR']):chosen.update(['W0_LOCAL_ERR','W0_R3_RAW','W0_REGION_ERR','W0_C2_REGION_ERR'])
 if look['TF_SEL','W0_SEL']['passed'] or look['W0_B0','W0_SEL']['passed']:chosen.update(['W0_B0','W0_SEL','TF_SEL'])
 if chosen:chosen.update(['W0_B0','W0_SEL','TF_SEL'])
 leaderboard=[]
 for config in sorted({k[0] for k in data}):
  farms=[i for i in range(56) if all((config,i,s) in data for s in [42,43])]
  if farms:leaderboard.append(dict(configuration=config,farms=len(farms),metrics={hz:{k:float(np.mean([np.mean([data[config,i,s][k] if hz=='all' else data[config,i,s]['horizons'][hz][k] for s in [42,43]]) for i in farms])) for k in ['mae','rmse']} for hz in ['all','0_1h','1_3h','3_6h']}))
 report=dict(leaderboard=leaderboard,timestamp=now(),selection_year=2023,comparisons=results,interaction=ix,replicate_configurations=sorted(chosen),interpretation='seed-average within farm, then equal-farm means; farm bootstrap is descriptive with spatial dependence; selection-year optimism',ten_seed_comparisons=[comparison(data,a,b,list(range(42,52))) for a,b in pairs],ten_seed_interaction=interaction(data,list(range(42,52))))
 atomic_json(RUN/'analysis/REPORT.json',report)
 lines=['# 新业务协议实验进度与结果',f"更新时间：{now()}。2022训练，2023选择，未启动2024模型评价。",'效应量：先站内平均种子，再对风场等权平均；负值为改善。','|对照|变体|完成风场|MAE变化|RMSE变化|高功率RMSE变化|','|---|---|---:|---:|---:|---:|']
 for r in results:
  if r['n']:lines.append(f"|{r['base']}|{r['arm']}|{r['n']}/56|{r['relative_change']['mae']:.3%}|{r['relative_change']['rmse']:.3%}|{str(None) if r['high_power_rmse_change'] is None else format(r['high_power_rmse_change'],'.3%')}|")
 lines += ['',f"历史×天气交互已完成 {ix['n']}/56 风场；判定通过：{ix['passed']}。",'本轮只识别给定当前功率后的历史上下文与天气交互，不将统计交互等同于因果机制。','高功率组阈值为起报功率>=0.7；筛选容许RMSE相对退化上限0.5%。']
 (RUN/'analysis/REPORT_CN.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
 return report
if __name__=='__main__':run()
