"""Neighbor-on-SEL screen, before any 2024 evaluation."""
from settings import *
from neighbor_inputs import VARIANTS
TFSTAR='TF_z1_lr0.0002_dp0.05'

def main():
    setup();specs=[]
    sites=REPRESENTATIVES+[s for s in range(56) if s not in REPRESENTATIVES]
    for site in sites:
        for seed in SEEDS:
            for backbone,variants in [('W0',VARIANTS),('TF',['N0','N1'])]:
                for variant in variants:
                    config='W0' if backbone=='W0' else TFSTAR
                    stage=variant if backbone=='W0' else 'N6_'+variant
                    spec=dict(id=f'{config}_{variant}_s{site:02d}_r{seed}',stage=stage,task='train',variant=variant,site=site,seed=seed,configuration=config,backbone=backbone,loss='MAE',delay_minutes=0 if variant=='N7_D0' else 30 if variant=='N7_D30' else 15,capacity_control=False,protocol='single_farm_only' if variant in ['N0','N2_SELF'] else 'independent_target_neighbor_observations')
                    if backbone=='TF':spec.update(tf_zero_init=True,tf_lr=2e-4,tf_dropout=.05)
                    if backbone=='W0' and variant=='N0':spec.update(task='diagnose_reference',local_only=True,source_job=f'A2_W0_SEL_s{site:02d}_r{seed}')
                    specs.append(spec)
    assert len(specs)==1344 and sum(s['task']=='train' for s in specs)==1232
    atomic_json(RUN/'manifest.json',specs)
    atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),train_year=2022,validation_year=2023,test_year_accessed=False,planned_training=1232,planned_reference_diagnostics=112,planned_tasks=1344,farms=56,seeds=SEEDS,backbones=['W0',TFSTAR],base_input='SEL = DENS3 + DYN7 + STAB5 + STATE7; no CAL',variants=VARIANTS,neighbor_defaults=dict(k=8,radius_km=150,delay_minutes=15,history_points=22,target_excluded=True,invalid_mask_and_null_token=True),topology_selection='N5 ranks 2022 contemporaneous power correlation within same 150km candidate pool; all other topology fixed by geography',aggregate='N3 uses same 8 neighbors and capacity-normalized MW sum, repeated to same token slots and same attention parameters',capacity_control='N2_SELF uses only delayed target history through identical neighbor parameters and geometry; N2 alone does not isolate regularization',mismatch='N2 jointly draws all neighbor histories from a different day of 2022 with same calendar month and local quarter-hour. Training excludes +/-1 day, validation uses only 2022 donors. Year-distribution shift is a limitation of this negative control.',latency='All delays use 22 observations; N7 retrains at 0/30 minutes, in addition to fixed-weight delay interventions',interventions=['neighbor_zero','neighbor_identity_valid_only','neighbor_reverse','delay0','delay15','delay30'],intervention_interpretation='Fixed-weight sensitivity, possibly out-of-distribution; not standalone proof of incremental value',diagnostics=['horizons','current_power','history_ramp','forecast_wind_change','outcome_power_change_1h','neighbor_within_50km','season','stability','wind_speed'],outcome_strata='Future realized ramp labels used only to group prediction errors; thresholds fitted on 2022',training=dict(epochs=80,patience=10,loss='MAE',optimizer='AdamW',weight_decay=1e-4,batch=256,clip_norm=1,precision='FP32_no_TF32',stopping='validation_MAE',selected_checkpoints=['mae','rmse']),gate=dict(primary='W0 N1 vs W0 N0, 2023 all-horizon MAE, equal seed within farm then equal farm',minimum_relative_improvement_percent=.1,also_strictly_better_than=['N2','N3'],capacity_control_report='N2_SELF comparison mandatory; if it does not improve, do not claim proven real-time neighbor value',fixed_primary='N1; N4/N5/N7 sensitivity results cannot replace failed primary gate after the fact'),reporting='Actual farm/seed results mandatory; report every improvement and loss, including below 0.1%. The 0.1% threshold is the user-specified next-stage gate only.',next='After complete review, explicitly freeze revised joint 2024 protocol. Preserve previous B1 freeze; no 2024 execution in this screen.',storage='Reuse cache, no prediction files or attention-array copies, retain at most two selected weights, delete latest on success'))
    print('1232 training + 112 reference diagnostics = 1344 tasks')

if __name__=='__main__':main()
