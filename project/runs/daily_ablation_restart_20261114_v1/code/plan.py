"""Prespecified retraining matrix; every seed is fresh, all farms are retained."""
from settings import *
import itertools

def create():
    fp=json.loads((DATA/'READY.json').read_text())['preprocess_fingerprint']
    seeds=SEEDS;specs=[]
    def add(group,family,seed,stage='ablation',inp='R3',**options):
        config_from=family+'_'+inp
        dep=[f'assets/selected/{config_from}.json']
        cond=options.pop('condition','neutral')
        if cond=='cbh':dep.append(f'data/reference_r{seed}/READY.json')
        s=dict(id=f'{group}_r{seed}',stage=stage,group=group,family=family,input=inp,condition=cond,seed=seed,config_from=config_from,epochs=80,patience=10,dependencies=dep,preprocess_fingerprint=fp)
        s.update(options);specs.append(s)
    # W0 local-only requires its own equal-budget tune; TF_BASE already has one.
    configs=[dict(width=w,lr=lr,dropout=d,weight_decay=1e-4) for w,lr,d in itertools.product([64,128],[2e-4,1e-3],[0.,.05])]
    for ci,cfg in enumerate(configs):
        for seed in [40,41]:
            specs.append(dict(id=f'TUNE_W0_BASE_c{ci}_r{seed}',stage='tune',group='W0_BASE',family='W0',input='BASE',condition='neutral',seed=seed,config=cfg,epochs=30,patience=6,dependencies=[],preprocess_fingerprint=fp))
    for seed in seeds:add('SHARED_W0_BASE','W0',seed,stage='shared_local',inp='BASE')
    for condition in ['neutral','cbh']:
        for variant in ['E0','E1','E2L','E2N','E2','E3']:
            for seed in seeds:add(f'ENC_{condition}_{variant}','W0',seed,stage='weather_encoder',condition=condition,weather_variant=variant)
    controls={
        'LOCAL_SEL':dict(no_near=True,no_rings=True),
        'NO_SEL':dict(no_sel=True),
        'NO_NEAR':dict(no_near=True),
        'NO_RINGS':dict(no_rings=True),
        'NO_DENSITY':dict(mask_sel=[[0,3]]),
        'NO_DYNAMICS':dict(mask_sel=[[3,10]]),
        'NO_STABILITY':dict(mask_sel=[[10,15]]),
        'NO_STATE':dict(mask_sel=[[15,22]])}
    for family in ['W0','TF']:
        for label,opts in controls.items():
            for seed in seeds:add(f'INPUT_{family}_{label}',family,seed,stage='input_ablation',**opts)
        for label,opts in {
            'H0W1':dict(history_control='constant_current'),
            'H1W0':dict(weather_control='training_mean'),
            'H0W0':dict(history_control='constant_current',weather_control='training_mean')}.items():
            for seed in seeds:add(f'FACTOR_{family}_{label}',family,seed,stage='history_weather',**opts)
    cohort=json.loads((ASSETS/'COHORT.json').read_text())
    for station,station_id in enumerate(cohort['station_ids']):
        for seed in seeds:
            for family,inp in itertools.product(['W0','TF'],['BASE','R3']):
                group=f'INDEPENDENT_{family}_{inp}'
                add(group,family,seed,stage='independent',inp=inp,station=station,station_id=station_id)
                specs[-1]['id']=f'{group}_s{station:02d}_r{seed}'
    assert len(specs)==1291 and len({s['id'] for s in specs})==len(specs)
    atomic_json(RUN/'manifest.json',specs)
    from collections import Counter
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),preprocess_fingerprint=fp,weather_protocol='daily_08_BJT_latency15h',regional_delay_minutes=0,train_year=2022,selection_year=2023,test2024_read=False,seeds=seeds,farms=cohort['count'],budget=dict(Counter(s['stage'] for s in specs)),total_jobs=len(specs),old_protocol_weights_or_metrics_reused=False,initialization='fresh for every job; no pretrained weights',independent='Each optimizer sees only its target farm labels; regional input is the matched factor, using all observed farms through origin. Same train-fitted feature scaling as shared arms; not an unseen-farm transfer experiment.',factorial='H0 repeats current local power across history and recomputes STATE; current-power anchor is retained. W0 replaces weather with training mean, zeroes ALL normalized weather-derived SEL fields; time, site and regional observed power remain. H1W1 is the new shared R3 baseline. Measures local-history x weather given current power and regional observations.',encoders='E0 raw14 affine; E1 raw14+DEM affine; E2L raw14 deep affine; E2N raw14 deep GELU; E2 raw14+DEM GELU; E2C column15 GELU; E3 column groups (quarter,3/8,3/8 width). Neutral and CBH contexts both rerun; E2C controls are the newly trained shared W0_R3 and M2_CBH.',tuning='W0_BASE:8configs x2seeds, same budget as existing families. Other ablations use their own family/input selected config to hold backbone budget fixed. Independent arms reuse same selected config and max80/patience10; same per-farm sample exposure, optimizer update count differs from shared.',dependencies=['daily_protocol_restart_20261114_v1/COMPLETE.json','daily_site_finetune_20261114_v1/COMPLETE.json']))
    print(json.dumps(dict(total=len(specs),stages=dict(Counter(s['stage'] for s in specs)))))
if __name__=='__main__':create()
