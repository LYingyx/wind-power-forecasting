from settings import *
import itertools

def create():
    families={'TF_BASE':('TF','BASE'),'TF_R3':('TF','R3'),'TIDE_R3':('TIDE','R3'),'TFT_R3':('TFT','R3'),'W0_R3':('W0','R3')}
    configs=[dict(width=w,lr=lr,dropout=drop,weight_decay=1e-4) for w,lr,drop in itertools.product([64,128],[2e-4,1e-3],[0.,.05])]
    specs=[]
    # Interleave families/configs to occupy all machines fairly from the start.
    for ci,cfg in enumerate(configs):
        for seed in [40,41]:
            for group,(family,inp) in families.items():specs.append(dict(id=f'TUNE_{group}_c{ci}_r{seed}',stage='tune',group=group,family=family,input=inp,condition='neutral',seed=seed,config=cfg,epochs=30,patience=6,dependencies=[]))
    for seed in SEEDS:
        for fold in [-1,0,1,2,3]:
            specs.append(dict(id=f'B0_r{seed}_f{fold}',stage='reference',group='B0',family='B0',input='H',condition='neutral',seed=seed,fold=fold,config=dict(width=64,lr=2e-4,dropout=.05,weight_decay=1e-4),epochs=80,patience=10,dependencies=[]))
    for group,(family,inp) in families.items():
        for seed in SEEDS:specs.append(dict(id=f'BASE_{group}_r{seed}',stage='baseline',group=group,family=family,input=inp,condition='neutral',seed=seed,config_from=group,epochs=80,patience=10,dependencies=[f'assets/selected/{group}.json']))
    mechanisms={'M1_CB':('W0','cb'),'M2_CBH':('W0','cbh'),'M3_ANCHOR':('W0','anchor'),'M4_WRONG_CB':('W0','wrong_cb'),'M4H_WRONG_CBH':('W0','wrong_cbh'),'M5_TF_CB':('TF','cb'),'M5W_TF_WRONG_CB':('TF','wrong_cb'),'M5N_TF_NEUTRAL':('TF','neutral_adapter')}
    for seed in SEEDS:
        for group,(family,cond) in mechanisms.items():
            base='W0_R3' if family=='W0' else 'TF_R3';deps=[f'assets/selected/{base}.json']
            if cond!='neutral_adapter':deps.append(f'data/reference_r{seed}/READY.json')
            specs.append(dict(id=f'{group}_r{seed}',stage='mechanism',group=group,family=family,input='R3',condition=cond,seed=seed,config_from=base,epochs=80,patience=10,dependencies=deps))
    protocol=dict(timestamp=now(),train_year=2022,selection_year=2023,test_status='2024 exploratory, not read in stages1-2',seeds=SEEDS,tune_seeds=[40,41],configs=configs,
        data='operational age0 paired train2022/val2023 cache; val November and boundary exclusions inherited; 15 minute delayed regions recomputed after QC',
        checkpoint='minimum equal-farm macro MAE, one checkpoint for all metrics',loss='unclipped capacity-factor MAE',optimizer='AdamW; constant learning rate; grad norm clip1',
        batch_size=512,validation_batch_size=1024,precision='float32, TF32 disabled, no AMP',
        input_fairness='All models share local history, site lat/lon/logcapacity/elevation and learned embedding8. All R3 models receive identical SEL22, N3 pooled history, 3 ring statistics. TF_BASE omits SEL/region.',
        b0='Shared local-history TCN with target attention; no cluster/global power. Architecture adaptation, not the old global-history ForcedEvolution checkpoint.',
        oof='4 quarter blocks of2022, 12h purge; each OOF model early-stops on a separate21-day block from remaining2022 only. Final B0 uses full2022, selected on2023. No in-sample target-fitted trajectory is supplied to2022 mechanism training.',
        trajectory_controls='same month/Beijing clock/different date >=2days;2023 wrong references drawn from2022 OOF archive. CBH also mismatches history summary. TF neutral adapter added for parameter-matched control.',
        budget=dict(tuning=80,shared_baselines=25,reference_oof_and_full=25,mechanism=40,total=170),
        pass_rule=dict(minimum_relative_improvement_percent=.1,minimum_farm_win_fraction_strict=.5,metrics=['mae','rmse'],high_power='current CF>=.75: report and require relative RMSE degradation <=.1% before scale-up'),
        scale_up_gate='M1 or M2 passes against W0_R3 and matched wrong-date control; selected candidate must also pass against best matched-input baseline selected by2023MAE among TF/TiDE/TFT. Stage1 rawW0-vsTF may fail if mechanism rescues it. No personalization or2024 jobs generated before gate report.',
        inference='one forecast per farm; shared parameters before personalization; per-farm equal-weight reporting',
        hardware='Both V100 hosts and idle local4090 cards; no foreign process termination. Multi-process governed by memory reserve, external compute and measured throughput.',
        caveats=['Old independent metrics are descriptive only after QC changes neighbors; same-input independent reruns deferred to stage4.','P3 physical interpretation remains a hypothesis.','2024 cannot provide an untouched confirmation set.'])
    atomic_json(RUN/'PROTOCOL.json',protocol);atomic_json(RUN/'manifest.json',specs)
    print('manifest jobs',len(specs))
if __name__=='__main__':create()
