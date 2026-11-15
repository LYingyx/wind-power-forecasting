"""Admission edge cases and checkpoint-release/resume equivalence."""
from pathlib import Path
import sys,json,shutil
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import atomic_json,now
from gpu_policy import Policy,decide
from scheduled_worker import training,ResourceYield,install_yield_hook
import torch


def check():
    result={}
    def sample(util=0,other=False,sm=0,known=True,free=20000):
        return dict(total_mib=24564,free_mib=free,utilization=util,other_pids=[999] if other else [],external_sm=sm,process_utilization_known=known)
    def decision(s,own=0,starting=0,age=90):return decide([s]*3,own,starting,age,2048)
    assert decision(sample())['launch'];result['empty_gpu']='pass'
    assert decision(sample(other=True))['launch'];result['idle_allocated_gpu']='pass'
    assert not decision(sample(util=80,other=True,sm=75))['launch'];result['busy_other_process']='pass'
    assert not decision(sample(other=True,known=False))['launch'];result['unknown_other_usage']='pass'
    assert not decision(sample(free=6000))['launch'];result['memory_margin']='pass'
    assert not decision(sample(),starting=1)['launch'];result['startup_reservation']='pass'
    assert not decision(sample(),age=10)['launch'];result['warmup']='pass'
    assert not decision(sample(util=98),own=4)['launch'];result['no_useless_oversubscription']='pass'
    assert decision(sample(util=30),own=4)['launch'];result['grow_beyond_four_when_idle']='pass'
    assert decision(sample(util=90,other=True,sm=70),own=3)['yield_count']==3;result['release_our_jobs_for_others']='pass'
    assert decision(sample(free=3000),own=3)['yield_count']>=1;result['release_memory_reserve']='pass'
    assert not decision(sample(),own=12)['launch'];result['safety_cap']='pass'
    base=dict(id='resource_reference',site=6,seed=43,backbone='W0',variant='CAL',loss='MAE',delay_minutes=15,capacity_control=False)
    resume=dict(base,id='resource_yield_resume')
    training.run(base,'cpu',smoke=True)
    folder=RUN/'smoke_jobs'/resume['id'];folder.mkdir(exist_ok=True)
    request=folder/'YIELD_REQUEST.json';atomic_json(request,dict(test=True))
    original=install_yield_hook(request)
    try:
        try:training.run(resume,'cpu',smoke=True)
        except ResourceYield:pass
        else:raise AssertionError('Did not release after checkpoint')
    finally:training.save_torch=original
    assert (folder/'latest.pt').exists() and not (folder/'SUCCESS.json').exists()
    request.unlink();training.run(resume,'cpu',smoke=True)
    a=json.loads((RUN/'smoke_jobs'/base['id']/'metrics.json').read_text());b=json.loads((folder/'metrics.json').read_text())
    for key,sa in a['selections'].items():
        sb=b['selections'][key];assert sa['epoch']==sb['epoch'] and sa['metrics']==sb['metrics']
        ma=torch.load(RUN/'smoke_jobs'/base['id']/sa['checkpoint'],weights_only=False)['model']
        mb=torch.load(folder/sb['checkpoint'],weights_only=False)['model'];assert all(torch.equal(ma[k],mb[k]) for k in ma)
    for spec in [base,resume]:shutil.rmtree(RUN/'smoke_jobs'/spec['id'])
    result['cooperative_release_resume_exact']='pass'
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=result,science_code_changed=False))
    print(json.dumps(result,indent=2))
if __name__=='__main__':check()
