from settings import *
from report import comparison,interaction
import numpy as np,collections

def parent_ledger():
 data={};checkpoints={}
 for s in json.loads((PARENT/'manifest.json').read_text()):
  p=PARENT/'jobs'/s['id'];assert (p/'SUCCESS.json').exists()
  m=json.loads((p/'metrics.json').read_text());assert not m['smoke_only'] and m['data_fingerprint']==FINGERPRINT
  assert json.loads((p/'spec.json').read_text())==s
  sel=m['selections']['mae'];diag=json.loads((p/'diagnostics.json').read_text())['rows']
  high=next((x['rmse'] for x in diag if x['group']=='current_power' and x['bin']=='high' and x['horizon']=='all'),None)
  data[s['configuration'],s['site'],s['seed']]=dict(**sel['metrics'],high_rmse=high)
  checkpoints[s['id']]=dict(checkpoint=sel['checkpoint'],sha256=sel['sha256'],selected_epoch=sel['epoch'],spec_sha256=sha(p/'spec.json'),normalization_sha256=sha(p/'normalization.json'))
 return data,checkpoints

def choose_parent(data):
 choices={};evidence=[]
 for b in ['W0','TF']:
  candidates=[b+'_SEL']
  for v,bases in [('N3',[b+'_SEL']),('R3_RAW',[b+'_SEL',b+'_N3',b+'_C2_R3_RAW'])]:
   tests=[comparison(data,base,b+'_'+v,SEEDS) for base in bases];evidence+=tests
   if all(x['passed'] for x in tests):candidates.append(b+'_'+v)
  choices[b]=min(candidates,key=lambda c:np.mean([data[c,i,s]['mae'] for i in range(56) for s in SEEDS]))
 return dict(choices=choices,evidence=evidence,selection_year=2023,criterion='both metrics improve >=0.1%, each >28 winning farms, high-power RMSE increase <=0.5%; seed-average then farm-average')

def run():
 freeze=require_test_freeze();data={};count=0
 for s in json.loads((RUN/'manifest.json').read_text()):
  p=JOBS/s['id']
  if not (p/'SUCCESS.json').exists():continue
  t=json.loads((p/'test.json').read_text());assert t['year']==2024
  high=next((x['rmse'] for x in t['diagnostics'] if x['group']=='current_power' and x['bin']=='high' and x['horizon']=='all'),None)
  data[s['configuration'],s['site'],s['seed']]=dict(**t['metrics'],high_rmse=high);count+=1
 persist=RUN/'analysis/PERSISTENCE.json'
 if persist.exists():
  for row in json.loads(persist.read_text())['rows']:
   if row['year']==2024:
    for seed in SEEDS:data['PERSISTENCE',row['site'],seed]=dict(**row['metrics'],high_rmse=row['high_power_rmse'])
 main=freeze['selection']['choices']['W0'];families=json.loads((RUN/'baseline_configs.json').read_text())
 baselines=['PERSISTENCE','W0_B0','W0_BASE','TF_BASE','TF_SEL','TF_N3','TF_R3_RAW']+['BASELINE_'+x for x in families]
 pairs=[(c,main) for c in baselines if c!=main]
 for b in ['W0','TF']:
  pairs += [(b+'_BASE',b+'_SEL'),(b+'_SEL',b+'_N3'),(b+'_SEL',b+'_R3_RAW'),(b+'_N3',b+'_R3_RAW'),(b+'_C2_R3_RAW',b+'_R3_RAW')]
 pairs=list(dict.fromkeys(pairs));results=[comparison(data,a,b,SEEDS) for a,b in pairs]
 ix=interaction(data,SEEDS)
 if ix['rows']:
  rng=np.random.default_rng(20261028);sample=np.array([[x[k] for k in ['mae','rmse']] for x in ix['rows']]);idx=rng.integers(0,len(sample),(2000,len(sample)))
  ix['mean']={k:float(sample[:,i].mean()) for i,k in enumerate(['mae','rmse'])};ix['farm_bootstrap95']={k:np.quantile(sample[idx,i].mean(1),[.025,.975]).tolist() for i,k in enumerate(['mae','rmse'])}
 parent_comparisons=freeze['selection']['evidence'];lookup={(x['base'],x['arm']):x for x in parent_comparisons}
 for row in results:
  previous=lookup.get((row['base'],row['arm']))
  if previous and row['n']==56:row['same_direction_as_2023']={k:bool(np.sign(row['relative_change'][k])==np.sign(previous['relative_change'][k])) for k in ['mae','rmse']}
 output=dict(timestamp=now(),completed_test_jobs=count,frozen_main=main,comparisons=results,interaction=ix,year=2024,interpretation='Matched ten seeds, seed mean within farm and equal farm weight. Farm bootstrap is descriptive because spatial dependence remains. Selected on2023; historically2024 was seen under older protocol, so this is operational reevaluation, not pristine holdout.')
 atomic_json(RUN/'analysis/COMPARISONS_2024.json',output)
 lines=['# 2024 固定方案复评',f'冻结主配置：{main}。2024 剔除11月；所有指标来自2023 MAE所选同一检查点。','|对照|模型|完成风场|MAE变化|RMSE变化|MAE/RMSE胜场|高功率RMSE变化|','|---|---|---:|---:|---:|---:|---:|']
 for x in results:
  if x['n']:lines.append(f"|{x['base']}|{x['arm']}|{x['n']}/56|{x['relative_change']['mae']:.3%}|{x['relative_change']['rmse']:.3%}|{x['farm_wins']['mae']}/{x['farm_wins']['rmse']}|{x['high_power_rmse_change']}|")
 lines+=['',f"历史×天气交互：{ix['n']}/56 场完整；通过：{ix['passed']}。",'正交互只表明给定当前功率后的历史上下文与天气存在预测交互，不等同于因果证明。部分站点结果不作最终结论。','完整逐站效应与置信区间见 COMPARISONS_2024.json；误差原值见 per_farm_seed_year.csv。']
 (RUN/'analysis/COMPARISONS_2024_CN.md').write_text('\n'.join(lines)+'\n');return output
if __name__=='__main__':run()
