from pathlib import Path
import hashlib,json,os,random,datetime
import numpy as np
import torch

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]
OLD=ROOT/'runs/independent_single_farm_20261021_v1'
CACHE=ROOT/'data/model_cache_pad5_f8'
REPS=[6,8,14,16,43,46]
FRONTS=['S1','S2','S3','S4','T1_7','T2_7','T1_8','T2_8','ST00','ST10','ST01','ST11','ST_NO_H']
HISTORY_ARMS=['H1','H2','CLIM_LAST','H2_CLIM']
TRANSFERS=[f'{b}__{f}' for b in ['TRANSFORMER_FC','TIDE_FC'] for f in ['S1','ST00','ST11']]
FORMAL=FRONTS+HISTORY_ARMS+TRANSFERS
PROTOCOL=dict(version='20261022_v1',epochs=80,patience=10,batch_size=256,eval_batch_size=256,precision='float32_no_tf32',
  optimizer='AdamW',schedule='cosine_80',weight_decay=1e-4,gradient_clip=1.,loss='MAE_CF',selection_metric='2023_equal_farm_seed_MAE',
  front_width=32,front_lr_choices=[2e-4,1e-3],k_choices=[3,5,7],tune_seeds=[40,41],formal_seeds=list(range(42,47)),extended_seeds=list(range(47,52)),
  representatives=REPS,train_year=2022,validation_year=2023,reevaluation_year=2024,output_clipping=False,
  train_jobs=8016,tune_jobs=456,formal_jobs=6440,extension_jobs=1120,minimum_improvement=0.,improvement_tolerance=0.,
  intervention_seeds=list(range(1001,1006)),gpu_reserve_mib=4096,max_workers_per_gpu=8,
  evaluation_policy='all_selection_frozen_before_new_2024_predictions',backbone='NEUTRAL_LAST',weather_values='raw_standardized_14')
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def configure():
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def decode_arm(arm):return arm.split('__') if '__' in arm else ('NEUTRAL_LAST',arm)
def backbone_config(backbone):
 return dict(width=128 if backbone=='TIDE_FC' else 64,lr=1e-3 if backbone=='TRANSFORMER_FC' else 2e-4,weight_decay=1e-4)
def group_for(arm):
 b,f=decode_arm(arm)
 if b!='NEUTRAL_LAST':return b
 if f.startswith('ST'):return 'ST'
 if f in ['S2','S3','S4']:return 'SPACE'
 if f.startswith('T'):return 'TIME'
 if f=='S1':return 'S1'
 return 'FIXED'
def spec(arm,site,seed,stage,k=3,front_lr=2e-4,group=None,config_index=None):
 b,f=decode_arm(arm)
 jid=f'{stage}_{arm}_s{site:02d}_r{seed}'
 if stage=='tune':jid+=f'_k{k}_l{0 if front_lr==2e-4 else 1}'
 return dict(id=jid,arm=arm,family=arm,backbone=b,front=f,site=site,seed=seed,stage=stage,k=k,front_lr=front_lr,
             backbone_config=backbone_config(b),group=group or group_for(arm),config_index=config_index)
