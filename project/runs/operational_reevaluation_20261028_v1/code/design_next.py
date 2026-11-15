from settings import *

def make_manifest(full=True):
 parent=json.loads((PARENT/'manifest.json').read_text());configs=json.loads((RUN/'baseline_configs.json').read_text())
 families=list(configs) if full else ['TRANSFORMER_FC']
 out=[]
 # Prioritize the inherited main model and baselines, then remaining mechanisms.
 rank={'W0_B0':0,'W0_SEL':1,'TF_SEL':2,'W0_N3':3,'TF_N3':4,'W0_R3_RAW':5,'TF_R3_RAW':6}
 for s in sorted(parent,key=lambda s:(s['seed'],s['site'],rank.get(s['configuration'],10),s['id'])):
  out.append(dict(**{k:v for k,v in s.items() if k not in ['id','task','stage']},id='eval_'+s['id'],task='evaluate',stage='test_2024',source_job=s['id']))
 for seed in range(42,52):
  for site in range(56):
   for family in families:
    config=configs[family]['config'];name='BASELINE_'+family
    out.append(dict(id=f'{name}_s{site:02d}_r{seed}',configuration=name,task='train',stage='baseline_2022_2023_and_test2024',backbone='NAMED',family=family,variant='BASE',site=site,seed=seed,loss='MAE',model_config=config,data_fingerprint=FINGERPRINT))
   for b in ['W0','TF']:
    name=b+'_BASE';out.append(dict(id=f'{name}_s{site:02d}_r{seed}',configuration=name,task='train',stage='baseline_2022_2023_and_test2024',backbone=b,variant='BASE',site=site,seed=seed,loss='MAE',data_fingerprint=FINGERPRINT))
 return out

if __name__=='__main__':
 setup();manifest=make_manifest();atomic_json(RUN/'manifest.json',manifest);atomic_json(RUN/'candidate_manifest.json',manifest);atomic_json(RUN/'screen_manifest.json',manifest)
 atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),parent_run=PARENT.name,user_authorized='2026-10-28: after current experiments complete, evaluate2024 and train/evaluate baselines per farm',trigger='parent COMPLETE_AUDIT passed plus all parent SUCCESS',train_year=2022,selection_year=2023,test_year=2024,excluded_test_month='2024-11 including affected boundaries',normalization='new operational2022 train statistics only',checkpoint_selection='2023 MAE; identical checkpoint for test MAE/RMSE',baseline_hyperparameters='inherited previous2023 selected config; fixed now, not retuned under new protocol, not selected by2024',baseline_families=list(json.loads((RUN/'baseline_configs.json').read_text())),seeds=list(range(42,52)),default_and_tuned_transformer=True,reused_parent_models=len([s for s in manifest if s['task']=='evaluate']),new_training=len([s for s in manifest if s['task']=='train']),persistence='56 farms, deterministic, both2023/2024',test_input='same available Pangu issue at origin; BJT08/20 +12h latency; leads through30h',no_test_model_selection=True,no_full_prediction_dumps=True,output='per farm and seed MAE/RMSE/bias, CF/MW,3 horizons,24 leads,scenarios,daily sufficient statistics; station summaries and seed/farm equal-weight summary',main_choice_rule='on full2023 10seed results: SEL fallback; N3 if paired acceptance; R3 only if acceptance vs SEL,N3,and C2; choose eligible lowest MAE; report all frozen comparisons regardless winner'))
 print('total',len(manifest),'newtraining',sum(s['task']=='train' for s in manifest),'replay',sum(s['task']=='evaluate' for s in manifest))
