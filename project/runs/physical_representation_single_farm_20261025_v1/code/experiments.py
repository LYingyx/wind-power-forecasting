"""Predeclared 2023-only attribution grid; all configuration factors explicit."""
from settings import *
from physical_data import A1,FEATURES
import itertools

def build():
    setup();jobs=[]
    sites=REPRESENTATIVES+[s for s in range(56) if s not in REPRESENTATIVES]
    combos=['SEL','SEL_NO_DENS','SEL_NO_DYN','SEL_NO_STAB','SEL_NO_STATE','SEL_REF','SEL_REF_DELTA']+['SEL_'+v for v in A1]
    def add(stage,variant,backbone,zero=False,lr=None,drop=None):
        for site in sites:
            for seed in SEEDS:
                config='W0' if backbone=='W0' else f'TF_z{int(zero)}_lr{lr:g}_dp{drop:g}'
                s=dict(stage=stage,task='train',variant=variant,site=site,seed=seed,backbone=backbone,loss='MAE',
                    delay_minutes=15,capacity_control=False,protocol='single_farm_only',configuration=config)
                if backbone=='TF':s.update(tf_zero_init=zero,tf_lr=lr,tf_dropout=drop)
                s['id']=f'{stage}_{config}_{variant}_s{site:02d}_r{seed}';jobs.append(s)
    for variant in A1:
        for bb in ['W0','TF']:add('A1',variant,bb,lr=1e-3,drop=0.)
    for variant in combos:
        for bb in ['W0','TF']:add('A2',variant,bb,lr=1e-3,drop=0.)
    for zero,lr,drop in itertools.product([False,True],[1e-3,2e-4],[0.,.05]):
        if not zero and lr==1e-3 and drop==0.:continue # existing BASE/DENS exactly reused
        for variant in ['BASE','DENS']:add('A3',variant,'TF',zero,lr,drop)
    for variant in ['BASE','DENS','ALL']:
        for bb in ['W0','TF']:
            for site in sites:
                for seed in SEEDS:
                    jid=f'input_{bb}_{variant}_MAE_s{site:02d}_r{seed}_d15'
                    jobs.append(dict(stage='A4_REFERENCE',task='diagnose_reference',local_only=True,variant=variant,site=site,seed=seed,backbone=bb,configuration='W0' if bb=='W0' else 'TF_z0_lr0.001_dp0',loss='MAE',delay_minutes=15,capacity_control=False,protocol='single_farm_only',source_job=jid,id='A4_'+jid))
    # Site/seed/feature coverage early; cannot depend on a validation win.
    jobs.sort(key=lambda s:(s['site'] not in REPRESENTATIVES, {'A1':0,'A3':0,'A2':1,'A4_REFERENCE':2}[s['stage']],s['site'],s['seed'],s['stage'],s['variant'],s['configuration']))
    assert len(jobs)==6944 and len({s['id'] for s in jobs})==len(jobs)
    atomic_json(RUN/'manifest.json',jobs)
    atomic_json(RUN/'PROTOCOL.json',dict(version=RUN.name,created=now(),train_year=2022,validation_year=2023,test_year_accessed=False,
        planned_tasks=len(jobs),planned_training=6272,planned_reference_diagnostics=672,stages=dict(A1=1568,A2=3136,A3=1568,A4_REFERENCE=672),
        seeds=SEEDS,farms=list(range(56)),a1=A1,a2=combos,tf_factorial=dict(zero_init=[False,True],learning_rate=[1e-3,2e-4],dropout=[0,.05],inputs=['BASE','DENS']),
        reused_parent_variants=['BASE','DENS','ALL'],parent_run=str(PARENT_RUN),feature_groups=FEATURES,
        epochs=80,patience=10,stopping_metric='validation_MAE',checkpoint_selection=['mae','rmse'],loss='MAE',precision='float32_no_tf32',batch=256,
        optimizer='AdamW',weight_decay=1e-4,clip_norm=1,extra_lr=2e-4,w0_lr=2e-4,
        a1_controls='Three nominal slots; scalar variants zero-pad; DENS_UNCOUPLED matches dimension and inputs while removing density-speed multiplication.',
        a2_controls='SEL has 22 slots; leave-one-group-out uses zero slots, preserving nominal projection parameters; seven substitutions avoid adaptive extra training.',
        diagnostics='Train-only quartiles and ramp thresholds; wind bins are descriptive, not physical cut-in/rated thresholds; no prediction files.',
        selection='Primary: W0 validation MAE, equal farm then seed weights. No CI-as-equivalence rule, no improvement cutoff; exact score tie uses fewer active inputs. Retain both DENS and best compact variant if Pareto tradeoff.',
        tf_selection='Choose factor configuration using DENS validation MAE, retain default TF in final comparisons; transfer chosen configuration to common main input without 2024 tuning.',
        guard='No 2024 data paths in this stage. A completion is not model selection freeze or B completion. New explicit FROZEN_SELECTION required for B.',
        reporting='Per-farm/per-seed scores mandatory. Equal-farm aggregate and paired site bootstrap are supplementary, exploratory and not multiplicity/spatial-correlation corrected.',
        storage='Reuse prepared inputs and parent checkpoints; at most two unique selected weight files per new train, latest checkpoint only until success.'))
    print(json.dumps(dict(tasks=len(jobs),training=6272,reference_diagnostics=672)))

if __name__=='__main__':build()
