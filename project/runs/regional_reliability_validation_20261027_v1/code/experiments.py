"""Stage D only. All test-year work remains dependent on a separate freeze."""
from settings import *
from regional_inputs import W0_VARIANTS, TF_VARIANTS, latency
TFSTAR='TF_z1_lr0.0002_dp0.05'

def main():
    setup(); specs=[]
    for site in REPRESENTATIVES+[s for s in range(56) if s not in REPRESENTATIVES]:
        for seed in SEEDS:
            for backbone, variant in [('W0','R3_RAW'),('W0','N0'),('W0','N3'),('TF','N0'),('TF','N3')]:
                config='W0' if backbone=='W0' else TFSTAR
                source=RESIDUAL_RUN if variant=='R3_RAW' or (backbone,variant)==('TF','N3') else NEIGHBOR_RUN
                source_job=f'{config}_{variant}_s{site:02d}_r{seed}'
                if (backbone,variant)==('W0','N0'):
                    source=ATTRIBUTION_RUN; source_job=f'A2_W0_SEL_s{site:02d}_r{seed}'
                specs.append(dict(id=f'{config}_{variant}_s{site:02d}_r{seed}',stage='REFERENCE',
                    task='diagnose_reference',configuration=config,backbone=backbone,variant=variant,
                    site=site,seed=seed,source_run=source.name,source_job=source_job))
            for backbone, variants in [('W0',W0_VARIANTS),('TF',TF_VARIANTS)]:
                config='W0' if backbone=='W0' else TFSTAR
                for variant in variants:
                    spec=dict(id=f'{config}_{variant}_s{site:02d}_r{seed}',stage=variant if backbone=='W0' else 'T_'+variant,
                        task='train',configuration=config,backbone=backbone,variant=variant,site=site,seed=seed,
                        loss='MAE',delay_minutes=latency(variant),capacity_control=False,
                        protocol='independent_target_regional_raw_completion_SEL_2023')
                    if backbone=='TF':spec.update(tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
                    specs.append(spec)
    assert len(specs)==2800 and len({s['id'] for s in specs})==2800
    atomic_json(RUN/'manifest.json',specs)
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),fit_year=2022,selection_year=2023,open_2024=False,
        farms=56,seeds=SEEDS,planned_training=2240,planned_reference_replays=560,
        primary='R3_RAW; previously selected on 2023, now confirm controls, no unseen-year claim',
        simplified_candidate='R3_ALLPOOL (one pool with mean/std/change, retained N3)',
        W0_variants=W0_VARIANTS,TF_variants=TF_VARIANTS,base='SEL, no CAL',
        neighbors=dict(radius_km=150,rings_km=[0,20,60,150],delay_minutes=15,
            default_points=22,target_excluded=True,capacity_weighted=True,N3_nearest=8),
        controls=dict(C2='joint 2022 archive donor, same local month/clock, >=2 days apart; not a pure capacity control',
            SELF='extra summary takes delayed own history; real N3 retained; matched 36 input slots',
            ALLPOOL='same raw mean/std/change as each ring; no ring labels; unused slots zero',
            ALLPOOL_MEAN='mean only, compare to ring MEAN; diagnostic arm, not automatically selectable'),
        projection='same 36 slots and parameter count; masks/unused statistics zero before train-only standardization',
        training=dict(epochs=80,patience=10,batch=256,loss='MAE',precision='FP32_no_TF32',
            lr=.0002,TF_zero_init=True,TF_dropout=.05,selection='MAE checkpoint for both primary metrics'),
        gate=dict(comparators=['N0','N3'],metrics=['mae','rmse'],minimum_improvement_percent=.1,
            each_metric_improved_farms_minimum=29,matched_control='C2_{candidate}',
            high_power='current CF>=0.7',high_power_rmse_max_degradation_vs_N3_percent=.1,
            all_rules_required=True),
        simplification=dict(candidate='R3_ALLPOOL',reference='R3_RAW',
            noninferiority_upper_one_sided_95_percent_max=.1,paired_farm_bootstrap=10000,bootstrap_seed=20261026,
            both_metrics_required=True,also_pass_primary_gate=True,
            note='Noninferiority under a predeclared tolerance, not equality; spatial dependence limits farm bootstrap interpretation'),
        interventions='selected weights; ring edits hold N3 fixed; global mismatch/drop30 affect both branches; sensitivity not causality',
        reporting='two seeds within farm then equal farms; no prediction exports; all negative effects retained',
        stages_E='deferred until D complete, controls reviewed, final models and interaction inputs frozen',
        storage='<=2 selected weights per train job; remove successful resume and verified remote duplicates'))
    print('2240 new fits + 560 selected-weight replays = 2800 tasks; no ARX duplication, no 2024 launch')

if __name__=='__main__':main()
