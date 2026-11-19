"""Predeclare full-history refinement and additive-state attribution controls."""
from settings import *
import argparse


def main(freeze=False):
    if freeze:
        for name in ['CHECKS.json','TRAIN_SMOKE.json']:
            assert json.loads((RUN/'operations'/name).read_text())['passed']
        files=list((RUN/'code').rglob('*.py'))
        files += [RUN/n for n in ['manifest.json','PROTOCOL.json','DATA_FREEZE.json']]
        files += [RUN/'operations'/n for n in ['checks.py','collect.py','report_diagnostics.py']]
        files += [p for p in ASSETS.rglob('*') if p.is_file()]
        files += list((RUN/'reference_metrics').rglob('*.json'))
        values={str(p.relative_to(RUN)):sha(p) for p in sorted(files)}
        target=RUN/'SCIENCE_FREEZE.json'
        if target.exists():assert json.loads(target.read_text())['files']==values
        else:atomic_json(target,dict(timestamp=now(),files=values))
        return
    assert not (RUN/'SCIENCE_FREEZE.json').exists()
    fingerprint=json.loads((DATA/'READY.json').read_text())['preprocess_fingerprint']
    groups=[('H-FULL','W0',True,'full',True,False),
            ('H-SUMMARY','W0',True,'summary',True,False),
            ('H-noS','W0',True,'no_state',False,False),
            ('H-wrongS','W0',True,'full',True,True),
            ('TF-H-FULL','TF',True,'full',True,False),
            ('ADD-noS','W0',False,'none',False,False),
            ('ADD-wrongS','W0',False,'none',True,True),
            ('TF-ADD','TF',False,'none',True,False)]
    specs=[]
    for seed in SEEDS:
        for group,family,refine,mode,state,wrong in groups:
            specs.append(dict(id=f'{group.replace("-","_")}_r{seed}',group=group,
                stage='history_refinement',representation=group,family=family,
                refinement=refine,history_mode=mode,realization='none' if refine else 'additive',
                use_state=state,wrong_state=wrong,input='R3',condition='neutral',weather_variant='E2L',
                seed=seed,config_from=family+'_R3',epochs=80,patience=10,mse_weight=0.,
                dependencies=[f'assets/selected/{family}_R3.json'],preprocess_fingerprint=fingerprint))
    atomic_json(RUN/'manifest.json',specs)
    atomic_json(RUN/'PROTOCOL.json',dict(version=7,created_at=now(),training_jobs=40,seeds=SEEDS,
        question='Does directly conditioning 15-minute query residuals on the complete historical power trajectory improve power prediction beyond summary state and linear interpolation?',
        data_fingerprint=fingerprint,farms=55,train_year=2022,selection_year=2023,
        task='24 quarter-hour power targets, 0-6h; no reconstructed-weather target',
        effective_batch=512,optimization='unchanged per-family frozen config',epochs_max=80,patience=10,
        loss='MAE',selection='strict minimum farm-macro validation MAE once each epoch',
        source_training_code_sha256=sha(PROJECT/'runs/daily_realization_20261116_v4/code/train.py'),
        architecture=dict(formula='u_k = E2L(interpolated weather)_k + U g(full history, c, whole low-dimensional wind window, t_k)',
            history='24 past power points; two 16-channel convolution layers; flattened 16-dimensional encoding',
            state='same 17 state features as v4; no future power',
            wind='eight hourly 10m/1000hPa speeds relative to origin, seven adjacent differences, node offsets',
            generator='73-to-32 GELU-to-2, .5*tanh; direct 24 queries, learned 2-to-d basis',
            zero_initialization='coefficient output zero; exact corresponding baseline at initialization',
            summary='same network and parameter count; only full-history input zeroed',
            no_state='same network; full-history input and all 17 state inputs zeroed',
            wrong_state='module history and c use same-site same-month/clock training donor >=2 days away; backbone unchanged',
            uncertainty='deterministic; no new generative model in this batch'),
        additive_followups='exact v4 additive implementation/initialization: remove or mismatch c, transfer to TF',
        references=['R0','TF_R3','TF_E2L','B-additive'],
        main_gate=dict(mae_change_percent_max=-.1,rmse_change_percent_max=-.1,
            farm_wins_min=28,both_metrics_seed_wins_min=4,high_power_rmse_change_percent_max=.1),
        history_gate='H-FULL main_pass vs R0 and B-additive; dual >=0.1% vs H-SUMMARY/H-noS/H-wrongS',
        history_main='H-FULL main_pass vs R0, B-additive, TF_R3 and TF_E2L',
        additive_state='B-additive main_pass vs R0 and dual >=0.1% vs ADD-noS/ADD-wrongS',
        transfer='TF-H-FULL main_pass vs TF_E2L and TF-ADD; TF-ADD main_pass vs TF_E2L reported separately',
        required_diagnostics=['high-power up/down','true large changes','0-1/1-3/3-6h','clock phase'],
        limitations=['2023 is repeatedly used development data, not blind confirmation',
            'full-history representation is not recovery of physical weather',
            'not a pure within-hour timing module separated from amplitude',
            'H-SUMMARY controls extra capacity; B-additive differs in window and expansion as well as history',
            'same parameter count does not imply same effective capacity after zeroing inputs',
            'fixed 55-farm cohort; no performance-dependent exclusion',
            'no claim of improvement until completed paired comparisons']))
    assignment={}
    for s in specs:
        g=s['group']
        if g in ['H-FULL','H-SUMMARY','TF-H-FULL','H-wrongS']:
            host='local4090'
        elif g=='ADD-noS' or (g=='TF-ADD' and s['seed']==42):host='v1006001'
        else:host='v1006002'
        assignment[s['id']]=host
    atomic_json(RUN/'operations/ASSIGNMENTS.json',assignment)
    atomic_json(RUN/'operations/RESOURCES.json',dict(
        local4090=dict(gpus=[7,4,5,6],max_workers_per_gpu=3,max_total_project_workers=4),
        v1006001=dict(gpus=[7,0,1],max_workers_per_gpu=3,max_total_project_workers=2),
        v1006002=dict(gpus=[4,5,7],max_workers_per_gpu=3,max_total_project_workers=3),
        minimum_free_mib=3072,memory_reserve_fraction=.15,
        note='4090 preferred; free V100 cards for overflow; do not stop existing queues. Pair workers then calibrate extra workers.'))
    print(json.dumps(dict(jobs=len(specs),groups=[g[0] for g in groups])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
