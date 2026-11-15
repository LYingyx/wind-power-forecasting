"""Meaningful activation, resource-yield, new-seed data/model and resume checks."""
from pathlib import Path
import sys,tempfile,json,datetime,copy
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from extend_seeds import TF,select_arms,build_manifest,spec_for
from efficiency import EfficiencyGovernor
from gpu_policy import decide,Policy

def guard_checks():
    times=[1000.]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);governor=EfficiencyGovernor(root,clock=lambda:times[0])
        def row(epoch,seconds):return dict(epoch=epoch,seconds=seconds,timestamp=datetime.datetime.fromtimestamp(times[0]).astimezone().isoformat())
        def history(jid,rows):
            (root/jid).mkdir(exist_ok=True);(root/jid/'history.json').write_text(json.dumps(rows))
        def base():return dict(launch=True,yield_count=0,reason='spare_compute_and_memory')
        assert not governor.control('0',[('a',{})],base())['launch']
        times[0]+=100;history('a',[row(e,10.) for e in [2,3,4]])
        assert governor.control('0',[('a',{})],base())['launch']
        governor.control('0',[('a',{}),('b',{})],base())
        times[0]+=100
        history('a',[row(e,20.) for e in [5,6,7]]);history('b',[row(e,20.) for e in [2,3,4]])
        assert governor.control('0',[('a',{}),('b',{})],base())['yield_count']==0
        times[0]+=20;history('a',[row(e,20.) for e in [6,7,8]])
        d=governor.control('0',[('a',{}),('b',{})],base())
        assert d['yield_count']==1 and d['reason']=='matched_job_epoch_slowdown'
        d=governor.control('0',[('a',{})],base());assert not d['launch']
        foreign=dict(launch=False,yield_count=1,reason='other_workload_became_active')
        assert governor.control('0',[('a',{})],foreign)['reason']=='other_workload_became_active'
    snapshot=dict(total_mib=32768,free_mib=20000,other_pids=[99],process_utilization_known=True,external_sm=90,utilization=95)
    result=decide([snapshot]*3,2,0,1000,2048,Policy())
    assert not result['launch'] and result['yield_count']==2
    snapshot.update(other_pids=[],external_sm=0,utilization=0,free_mib=3500)
    result=decide([snapshot]*3,1,0,1000,2048,Policy());assert not result['launch'] and result['yield_count']==1

def main():
    guard_checks()
    rec=dict(complete=True,open_2024=False,gates={})
    for config in ['W0',TF]:
        for v in ['R3_RAW','R3_ALLPOOL']:
            rec['gates'][config+'/'+v]=dict(passes=False,dual=True,beats_control=True,high_power_rmse_percent=0.)
    assert len(build_manifest(select_arms(rec)))==1792
    rec['gates']['W0/R3_RAW']['passes']=True
    assert len(build_manifest(select_arms(rec)))==2688
    assert ('W0','C2_R3_RAW') in select_arms(rec) and ('W0','R3_ALLPOOL') not in select_arms(rec)
    for g in rec['gates'].values():g['passes']=True
    full=build_manifest(select_arms(rec));assert len(full)==len({s['id'] for s in full})==5376
    rec['complete']=False
    try:select_arms(rec)
    except AssertionError:pass
    else:raise AssertionError('Partial screening must not activate')
    import torch
    from physical_models import PhysicalModel,frozen
    import train_physics as training
    frozen.configure()
    # D's actual learned fixtures must remain numerically unchanged in the extension.
    for item in torch.load(RUN/'operations/compatibility_fixture.pt',weights_only=False):
        model=PhysicalModel(item['spec'],item['feature_dim'],item['prior']);model.load_state_dict(item['weights']);model.eval()
        with torch.no_grad():pred=model(item['batch'])
        assert torch.allclose(pred,item['expected'],rtol=0,atol=2e-7)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='regional_extension_check_') as tmp:
        training.RUN=Path(tmp)
        for i,(config,variant) in enumerate([(c,v) for c in ['W0',TF] for v in ['N0','N3','R3_RAW','R3_ALLPOOL','C2_R3_RAW','C2_R3_ALLPOOL']]):
            spec=spec_for(config,variant,6,44 if i%2==0 else 51)
            training.run(spec,'cpu',smoke=True)
            out=Path(tmp)/'smoke_jobs'/spec['id'];m=json.loads((out/'metrics.json').read_text())
            assert m['epochs']==2 and m['smoke_only'] and not (out/'latest.pt').exists()
            cases.append(dict(configuration=config,variant=variant,seed=spec['seed'],mae=m['selections']['mae']['metrics']['mae']))
        # Exact CPU resume from an optimizer/RNG checkpoint, using a new seed and C2 donor.
        spec=spec_for('W0','C2_R3_RAW',8,44)
        training.run(spec,'cpu',smoke=True,stop_after_epoch=1)
        training.run(spec,'cpu',smoke=True)
        out=Path(tmp)/'smoke_jobs'/spec['id'];resumed=json.loads((out/'metrics.json').read_text())
        training.RUN=Path(tmp)/'fresh';training.RUN.mkdir()
        training.run(spec,'cpu',smoke=True)
        fresh=json.loads((training.RUN/'smoke_jobs'/spec['id']/'metrics.json').read_text())
        for criterion in ['mae','rmse']:
            assert resumed['selections'][criterion]['metrics']==fresh['selections'][criterion]['metrics']
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,activation_checks=True,
        efficiency_slowdown_and_foreign_yield_checks=True,learned_D_fixture_parity=True,
        new_seed_smokes=cases,resume_exact=True,open_2024=False))
    print('Passed activation guards, efficiency/yield, five learned fixtures, twelve new-seed training cases, exact resume')

if __name__=='__main__':main()
