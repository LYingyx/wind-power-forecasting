"""Predeclare all conditional jobs; future activation never changes the science specs."""
from settings import *
import argparse, shutil

def main(freeze=False):
    if freeze:
        assert json.loads((RUN/'operations/CHECKS.json').read_text())['passed']
        names=[p for p in (RUN/'code').rglob('*.py')]
        names += [RUN/'manifest.json',RUN/'diagnostic_manifest.json',RUN/'PROTOCOL.json']
        names += [p for p in ASSETS.rglob('*') if p.is_file() and p.name!='selected_candidate.json']
        names += list((RUN/'reference_metrics').rglob('*.json'))
        values={str(p.relative_to(RUN)):sha(p) for p in sorted(names)}
        path=RUN/'SCIENCE_FREEZE.json'
        if path.exists():assert json.loads(path.read_text())['files']==values,'Refuse changing a frozen experiment'
        else:atomic_json(path,dict(timestamp=now(),files=values))
        return
    assert not (RUN/'SCIENCE_FREEZE.json').exists()
    fingerprint=json.loads((DATA/'READY.json').read_text())['preprocess_fingerprint']
    specs=[]
    stages=[('stage1',['R0+H','R1','R2','R3','TF_E2L'],None),
            ('stage2',['R4','R4-noS','R4-wrongS'],'operations/ENABLE_STAGE2.json'),
            ('cross_backbone',['TF_R4'],'operations/ENABLE_TF_R4.json'),
            ('loss_review',['LOSS_CANDIDATE','LOSS_TF'],'operations/ENABLE_LOSS.json')]
    for stage,groups,gate in stages:
        for group in groups:
            for seed in SEEDS:
                family='TF' if group in ['TF_E2L','TF_R4','LOSS_TF'] else 'W0'
                rep={'TF_R4':'R4','LOSS_CANDIDATE':'SELECTED','LOSS_TF':'TF_R3'}.get(group,group)
                specs.append(dict(id=f'{group.replace("+","plus").replace("-","_")}_r{seed}',
                    stage=stage,group=group,representation=rep,family=family,input='R3',condition='neutral',
                    weather_variant='E2L',seed=seed,config_from=family+'_R3',epochs=80,patience=10,
                    mse_weight=.2 if stage=='loss_review' else 0.,
                    dependencies=[f'assets/selected/{family}_R3.json']+([gate] if gate else []),
                    preprocess_fingerprint=fingerprint))
    atomic_json(RUN/'manifest.json',specs)
    diagnostic=[]
    for group,source_run,source_group in [('R0','daily_ablation_restart_20261114_v1','ENC_neutral_E2L'),
                                          ('TF_R3','daily_protocol_restart_20261114_v1','TF_R3')]:
        for source in json.loads((RUN.parent/source_run/'manifest.json').read_text()):
            if source['group']!=source_group or source['seed'] not in SEEDS or source['stage']=='tune':continue
            rel=f'assets/reference_weights/{group}_r{source["seed"]}.pt'
            target=RUN/rel;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(RUN.parent/source_run/'jobs'/source['id']/'best.pt',target)
            metric=json.loads((RUN/'reference_metrics'/group/str(source['seed'])/'metrics.json').read_text())
            assert sha(target)==metric['checkpoint_sha256']
            spec=dict(source,id=f'REFERENCE_{group}_r{source["seed"]}',group='REFERENCE_'+group,
                      representation=group,stage='diagnostics',task='reference_evaluation',
                      checkpoint=rel,checkpoint_sha256=sha(target),dependencies=[],mse_weight=0.)
            diagnostic.append(spec)
    atomic_json(RUN/'diagnostic_manifest.json',diagnostic)
    # Eight jobs on four local GPUs; seventeen on eight V100s (one queued).
    assignments={}
    for stage,_,_ in stages:
        jobs=[s for s in specs if s['stage']==stage]
        for i,s in enumerate(jobs):assignments[s['id']]='local4090' if i%3==1 else 'v1006001'
    for i,s in enumerate(diagnostic):assignments[s['id']]='local4090' if i%3==1 else 'v1006001'
    atomic_json(RUN/'operations/ASSIGNMENTS.json',assignments)
    atomic_json(RUN/'operations/RESOURCES.json',dict(local4090=dict(gpus=[0,1,2,3],target_per_gpu=2),
                v1006001=dict(gpus=list(range(8)),target_per_gpu=2),minimum_free_mib=3072,
                memory_reserve_fraction=.15,max_workers_per_gpu=3,
                note='At least two processes on each used GPU when enough jobs and memory; third only with measured aggregate throughput gain.'))
    atomic_json(RUN/'PROTOCOL.json',dict(version=2,date='2026-11-16',data_fingerprint=fingerprint,
        task='single-farm 0-6h every 15min; shared training over 55 farms',train_year=2022,selection_year=2023,
        pangu_issue_utc=0,latency_hours=15,crop_leads=[9,45],regional_delay_minutes=0,
        effective_batch=512,optimizer='AdamW',lr=.0002,weight_decay=.0001,dropout=.05,
        checkpoint_selection='strict minimum 2023 farm-macro MAE; once each epoch; no RMSE tie break',
        generator=dict(width=32,layers=1,heads=2,dropout=0.,hourly_nodes=8,segments=7,parameters_per_segment=4,
                       kernel='normalized Gaussian centers at hourly-segment midpoints, sigma=0.75h',
                       state_dimension=17,history_span_hours=5.75),
        flow=dict(euler_steps=8,training_samples=4,evaluation_samples=8,posterior_prior_std=.1,
                  loss='teacher MAE + student-mean MAE + 0.1 flow MSE + 0.001 posterior KL',
                  posterior_target_detached_for_flow=True,evaluation_noise='stateless per origin/site/seed/sample',
                  inference_teacher=False,dispersion='reported; mean-of-4 MAE does not guarantee noncollapse'),
        stage1_runs=25,stage2_conditional_runs=15,cross_backbone_conditional_runs=5,loss_review_runs=10,
        main_gate=dict(mae_change_percent_max=-.1,rmse_change_percent_max=-.1,farm_wins_min=28,
                       both_metrics_seed_wins_min=4,high_power_rmse_change_percent_max=.1,
                       references=['R0','TF_R3','TF_E2L']),
        stage2_gate='R3 strictly improves macro MAE and RMSE versus R2',
        mechanisms=dict(R3=['R2','R0+H'],R4=['R3','R4-noS','R4-wrongS']),
        loss_candidate_rule='R4 if mechanism passes; else R3 if mechanism passes; else R1 if main gate passes; else R0',
        loss_adoption='candidate beats loss-matched TF in MAE/RMSE and itself not worse than its MAE-trained version',
        limitations=['R0 selected among seven encodings on 2023','2023 is not an independent test set',
                     '2024 previously inspected; later review is exploratory','R1 and R0 are affine-equivalent'],
        baseline_reuse='five existing R0 and five existing TF_R3; no changes to frozen original runs'))
    shutil.copyfile(RUN.parent/'daily_protocol_restart_20261114_v1/DATA_FREEZE.json',RUN/'DATA_FREEZE.json')
    print(json.dumps(dict(stage1=25,conditional=30,all_specs=len(specs))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
