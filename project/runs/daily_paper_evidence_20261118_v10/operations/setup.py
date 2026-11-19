from pathlib import Path
import sys,json,shutil,itertools
RUN=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,sha,now
ROOT=RUN.parent; V7=ROOT/'daily_history_refinement_20261116_v7'; V9=ROOT/'daily_test2024_20261117_v9'; OLD=ROOT/'daily_protocol_restart_20261114_v1'
for d in ['jobs','schedulers','analysis','data','checkpoints','reference_metrics']:(RUN/d).mkdir(exist_ok=True)
for split,path in [('train',OLD/'data/train'),('val',OLD/'data/val'),('test',V9/'data/test')]:
 p=RUN/'data'/split
 if not p.exists():p.symlink_to(path.resolve(),target_is_directory=True)
shutil.copy2(V9/'data/READY.json',RUN/'data/READY.json')
old_specs=json.loads((V7/'manifest.json').read_text()); templates={s['group']:s for s in old_specs if s['seed']==42}
proto=json.loads((V9/'manifest.json').read_text()); v9map={(s['group'],s['seed']):s for s in proto}
base=dict(templates['H-FULL']); specs=[]
def add(group,seed,template=None,**kw):
 s=dict(template or base); s.update(kw); s.update(id=group.replace('-','_')+'_r'+str(seed),group=group,seed=seed)
 s['stage']=kw.get('stage','paper_evidence');specs.append(s);return s
# Critical precise interventions first, then seed extension interleaved with component tests.
ablations=[('H-noStateOnly',dict(state_only_off=True)),('H-wrongStateOnly',dict(wrong_state=True,wrong_state_only=True)),
 ('H-noWindow',dict(window_mode='none')),('H-localWindow',dict(window_mode='local')),
 ('H-reverseHistory',dict(history_order='reverse_past')),('H-noDecoderState',dict(decoder_state_off=True)),
 ('R0-noDecoderState',dict(refinement=False,representation='R0',decoder_state_off=True)),
 ('H-noRegions',dict(input_ablation='region')),('H-noOperating',dict(input_ablation='operating')),
 ('H-noAtmosphere',dict(input_ablation='atmosphere'))]
extend=['H-FULL','R0','TF_E2L','TF_R3','H-noS','H-SUMMARY','TF-H-FULL']
for i,seed in enumerate(range(42,47)):
 for group,kw in ablations:add(group,seed,**kw)
 for group in extend:
  if group in templates:t=templates[group]
  else:
   t=dict(v9map[(group,42)]['spec']); t.pop('config',None);t.pop('selection_sha256',None)
   t.update(representation=group if group.startswith('TF') else 'R0',realization='none',refinement=False)
  add(group,47+i,t)
 for family in ['TF','TFT','TIDE']:
  add(family+'-WINDOW',seed,refinement=False,family=family,representation='TF_E2L' if family=='TF' else 'R0',
      config_from=family+'_R3',dependencies=['assets/selected/'+family+'_R3.json'],full_window_baseline=True)
# New simple baseline gets the same 8 configurations x 2 tuning seeds as historical families.
configs=[dict(width=64,lr=lr,dropout=0.,weight_decay=wd) for lr,wd in itertools.product([2e-4,1e-3],[0.,1e-4,1e-3,1e-2])]
tune=[]
for ci,cfg in enumerate(configs):
 for seed in [40,41]:
  s=dict(base,id=f'TUNE_LINEAR_c{ci}_r{seed}',group='LINEAR-TUNE',family='LINEAR',refinement=False,
         config=cfg,candidate=ci,seed=seed,epochs=30,patience=6,dependencies=[],stage='tune')
  s.pop('config_from',None);tune.append(s)
# Interleave small tuning jobs to avoid blocking selection near the queue tail.
for i,s in enumerate(tune):specs.insert(min(len(specs),i*6+2),s)
for seed in range(42,47):add('LINEAR',seed,family='LINEAR',refinement=False,config_from='LINEAR',dependencies=['assets/selected/LINEAR.json'])
# Frozen same-input TFT/TiDE weights need only missing 2024 inference.
eval_jobs=[]
for family in ['TFT','TIDE']:
 for seed in range(42,47):
  src=OLD/'jobs'/f'BASE_{family}_R3_r{seed}';name=f'{family}_R3_r{seed}'
  shutil.copy2(src/'best.pt',RUN/'checkpoints'/f'{name}.pt')
  m=json.loads((src/'metrics.json').read_text());atomic_json(RUN/'reference_metrics'/f'{name}_2023.json',m)
  eval_jobs.append(dict(id='EVAL_'+name,group=family+'_R3',seed=seed,task='evaluate',backbone=True,
     checkpoint=f'checkpoints/{name}.pt',checkpoint_sha256=sha(src/'best.pt'),splits=['test'],dependencies=[]))
# Re-inference of core old weights adds issue-day error sums for clustered uncertainty.
for group in ['H-FULL','R0','TF_E2L','TF_R3']:
 for seed in range(42,47):
  old=v9map[(group,seed)];name=f'{group}_r{seed}';src=V9/old['checkpoint']
  shutil.copy2(src,RUN/'checkpoints'/f'{name}.pt')
  eval_jobs.append(dict(id='AUDIT_'+name.replace('-','_'),group=group,seed=seed,task='evaluate',
       backbone=old['backbone'],checkpoint=f'checkpoints/{name}.pt',checkpoint_sha256=sha(src),splits=['val','test'],dependencies=[]))
# Copy already established numerical results without counting them as new jobs.
for s in proto:
 if s['group'] not in extend+['H-wrongS','B-additive']:continue
 name=f"{s['group']}_r{s['seed']}"
 atomic_json(RUN/'reference_metrics'/f'{name}_2023.json',dict(group=s['group'],seed=s['seed'],evaluation_year=2023,metrics=s['validation_metrics']))
 m=json.loads((V9/'jobs'/s['id']/'metrics.json').read_text());atomic_json(RUN/'reference_metrics'/f'{name}_2024.json',m)
specs=eval_jobs[:10]+specs+eval_jobs[10:]
atomic_json(RUN/'manifest.json',specs);atomic_json(RUN/'diagnostic_manifest.json',[])
# Main priority to 4090, overflow distributed; assignments can migrate before launch only.
pattern=['local4090','v1006001','v1006002','v1006001','local4090','v1006002','local4090']
assign={s['id']:pattern[i%len(pattern)] for i,s in enumerate(specs)}
atomic_json(RUN/'operations/ASSIGNMENTS.json',assign)
atomic_json(RUN/'operations/RESOURCES.json',dict(local4090=dict(gpus=[0,2],max_workers_per_gpu=4,max_total_project_workers=4),
 v1006001=dict(gpus=[6,7],max_workers_per_gpu=3,max_total_project_workers=3),
 v1006002=dict(gpus=[7,6,5],max_workers_per_gpu=3,max_total_project_workers=3),minimum_free_mib=3072,memory_reserve_fraction=.15))
atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),train_year=2022,selection_year=2023,reevaluation_year=2024,
 heldout_status='2024 historically viewed; exploratory cross-year reevaluation',farms=55,seeds=list(range(42,47)),
 extension_seeds=list(range(47,52)),loss='MAE',batch=512,max_epochs=80,patience=10,checkpoint='2023 macro-farm MAE once per epoch',
 precision='FP32; TF32 disabled',optimizer='AdamW; unchanged constant LR; clip norm1',
 cohort='unchanged fixed 55 farms and exclusions; no outcome-driven filtering',
 primary_threshold=dict(mae_percent=-.1,rmse_percent=-.1,farm_wins=28,seed_wins_5=4,seed_wins_10=8,high_rmse_max_percent=.1),
 tuning=dict(linear_configs=configs,seeds=[40,41],epochs=30,patience=6,rule='mean 2023 macroMAE; candidate index tie-break'),
 existing_family_configs='historically selected configurations frozen, no new tuning after inspecting 2024',
 inference='frozen best checkpoint, no test-driven selection',source_training_code_sha256=sha(V7/'code/train.py'),
 source_train_loop_unchanged=sha(V7/'code/train.py')==sha(RUN/'code/train.py'),
 uncertainty='paired NWP issue-day error sums; 7-consecutive-issue-day block bootstrap conditional on seed set; not iid 15min samples',
 counts=dict(train=sum(s.get('task')!='evaluate' for s in specs),evaluation_only=len(eval_jobs),total=len(specs)),
 limitations=['E2L was selected among seven encoders using 2023; additional seeds do not erase selection bias.',
 'All regional inputs use cached pooled neighbor history and ring summaries, not independent raw histories for eight neighbors.',
 'TFT/TiDE are repository adaptations, not certified official implementations.',
 'Component ablation supports conditional predictive contribution, not identified meteorological causality.']))
print(json.dumps(dict(total=len(specs),training=sum(s.get('task')!='evaluate' for s in specs),evaluation_only=len(eval_jobs))))
