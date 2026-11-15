"""Tiny real-data optimization and selected-checkpoint replay on CPU."""
from settings import *
from physical_data import FarmData
from physical_models import PhysicalModel
from train_physics import run,predict,metrics
import torch,shutil

def main():
    results=[]
    for backbone,variant,loss,capacity in [('W0','BASE','MAE',False),('TF','BASE','MSE',False),('W0','CURVE_BIAS','MIX',False),('TF','NEIGH_ADVECT','MAE',False),('W0','SHIFT_SAMPLE','MAE',False),('TF','ALL','MAE',True)]:
        spec=dict(id=f'check_{backbone}_{variant}_{loss}',site=6,seed=42,backbone=backbone,variant=variant,loss=loss,delay_minutes=15,capacity_control=capacity)
        folder=RUN/'smoke_jobs'/spec['id']
        run(spec,'cpu',smoke=True)
        meta=json.loads((folder/'metrics.json').read_text());assert meta['smoke_only'] and meta['n_train']==64
        normalization=json.loads((folder/'normalization.json').read_text())
        data=FarmData('val',6,variant,stats=normalization,limit=64,capacity_control=capacity)
        for key,selected in meta['selections'].items():
            state=torch.load(folder/selected['checkpoint'],map_location='cpu',weights_only=False)
            # Prior is part of the saved state. Initial placeholder never controls replay.
            model=PhysicalModel(spec,data.x.shape[-1],torch.zeros(114));model.load_state_dict(state['model'])
            calculated=metrics(predict(model,data),data.y.numpy())
            assert abs(calculated['mae']-selected['metrics']['mae'])<1e-9
            assert abs(calculated['rmse']-selected['metrics']['rmse'])<1e-9
        results.append(dict(spec=spec,selections=meta['selections'],epochs=meta['epochs'],checkpoint_replay=True))
        # Only smoke scratch owned by this script; retained proof is small JSON.
        shutil.rmtree(folder)
    # Compare uninterrupted optimization with a genuine saved-state restart.
    original=dict(id='resume_reference',site=6,seed=43,backbone='W0',variant='CAL',loss='MAE',delay_minutes=15,capacity_control=False)
    restarted=dict(original,id='resume_restarted')
    run(original,'cpu',smoke=True)
    run(restarted,'cpu',smoke=True,stop_after_epoch=1)
    assert (RUN/'smoke_jobs'/restarted['id']/'latest.pt').exists()
    run(restarted,'cpu',smoke=True)
    left=json.loads((RUN/'smoke_jobs'/original['id']/'metrics.json').read_text())
    right=json.loads((RUN/'smoke_jobs'/restarted['id']/'metrics.json').read_text())
    for key in left['selections']:
        a=left['selections'][key];b=right['selections'][key]
        assert a['epoch']==b['epoch'] and a['metrics']==b['metrics']
        sa=torch.load(RUN/'smoke_jobs'/original['id']/a['checkpoint'],weights_only=False)['model']
        sb=torch.load(RUN/'smoke_jobs'/restarted['id']/b['checkpoint'],weights_only=False)['model']
        assert all(torch.equal(sa[k],sb[k]) for k in sa)
    for s in [original,restarted]:shutil.rmtree(RUN/'smoke_jobs'/s['id'])
    results.append(dict(interruption_resume_exact=True))
    atomic_json(RUN/'SMOKE_CHECK.json',dict(timestamp=now(),passed=True,cases=results,not_formal_results=True))
    print('SMOKE CHECK PASSED',len(results))
if __name__=='__main__':main()
