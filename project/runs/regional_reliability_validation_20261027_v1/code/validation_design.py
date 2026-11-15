"""Predeclared validation families; only origin-available observations enter models."""
TF='TF_z1_lr0.0002_dp0.05'
REMOVE=['EXCLUDE20_KEEP','EXCLUDE20_REFILL']
MATCH=['MATCH_INNER','MATCH_MIDDLE','MATCH_OUTER','MATCH_RANDOM']
FULL=['FULL_INNER','FULL_MIDDLE','FULL_OUTER','FULL_ALL']
GATES=['G_FREE','G_LEAD','G_STATE','G_STATE_LEAD']
SIMPLIFY=['R3_MEAN_DELTA','C2_R3_MEAN_DELTA']
W0_NEW=REMOVE+MATCH+FULL+GATES+['C2_'+v for v in GATES]+['C2_R3_MEAN_DELTA']
TF_NEW=GATES+['C2_'+v for v in GATES]+SIMPLIFY
NEW=set(W0_NEW+TF_NEW)
REFERENCES=['N0','N3','R3_RAW','R3_ALLPOOL']

def clean(v):return v[3:] if v.startswith('C2_') else v

def spec(config,variant,site,seed):
    x=dict(id=f'{config}_{variant}_s{site:02d}_r{seed}',stage=variant,task='train',configuration=config,
           backbone='W0' if config=='W0' else 'TF',variant=variant,site=site,seed=seed,
           loss='MAE',delay_minutes=15,capacity_control=False,
           protocol='regional_reliability_2023_screen_and_conditional_replication')
    if config==TF:x.update(tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
    return x
