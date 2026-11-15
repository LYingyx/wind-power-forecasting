"""Cross-build/device parity on the same real inputs and learned fixture weights."""
from pathlib import Path
import sys,json,argparse
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from relocate import activate
activate(RUN)
from settings import atomic_json,now
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
from train_physics import loss_value
import torch,numpy as np

def make():
    frozen.configure();fixtures=[]
    for backbone,variant in [('W0','N1'),('TF','N1'),('W0','N3'),('TF','N4_K16')]:
        data=FarmData('train',6,variant,limit=32);spec=dict(site=6,seed=42,backbone=backbone,variant=variant)
        if backbone=='TF':spec.update(tf_zero_init=True,tf_dropout=.05,tf_lr=2e-4)
        prior=data.historical_input.mean(0);model=PhysicalModel(spec,data.x.shape[-1],prior)
        opt=torch.optim.AdamW(model.groups(),weight_decay=1e-4)
        batch=data.batch(slice(None))
        for i in range(2):
            model.train();opt.zero_grad();loss=loss_value(model(batch),data.y,'MAE');loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():expected=model(batch)
        fixtures.append(dict(spec=spec,prior=prior,feature_dim=data.x.shape[-1],batch=batch,target=data.y,weights=model.state_dict(),expected=expected))
    torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')
    print('Prepared shared real-data compatibility fixture')

def add_cuda_reference():
    frozen.configure();fixtures=torch.load(RUN/'operations/compatibility_fixture.pt',weights_only=False)
    for item in fixtures:
        model=PhysicalModel(item['spec'],item['feature_dim'],item['prior']);model.load_state_dict(item['weights']);model=model.cuda();model.eval()
        with torch.no_grad():item['expected_cuda']=model({k:v.cuda() for k,v in item['batch'].items()}).cpu()
        item['cuda_reference_runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        print(json.dumps(dict(spec=item['spec'],cuda_vs_cpu_max=float((item['expected_cuda']-item['expected']).abs().max()))),flush=True)
    torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')

def verify():
    frozen.configure();assert torch.cuda.is_available()
    architecture=torch.cuda.get_device_capability(0);flags=torch.cuda.get_arch_list()
    assert 'sm_'+str(architecture[0])+str(architecture[1]) in flags,(architecture,flags)
    results=[]
    for item in torch.load(RUN/'operations/compatibility_fixture.pt',weights_only=False):
        m=PhysicalModel(item['spec'],item['feature_dim'],item['prior']);m.load_state_dict(item['weights']);m=m.cuda();m.eval()
        batch={k:v.cuda() for k,v in item['batch'].items()}
        with torch.no_grad():pred=m(batch).cpu()
        # CPU fixture parity first; optional CUDA reference when available.
        # Avoid occupying a busy 4090 merely to create a fixture.
        reference=item.get('expected_cuda',item['expected']);tolerance=2e-5 if 'expected_cuda' in item else 1e-4
        maximum=float((pred-reference).abs().max());assert maximum<tolerance,(item['spec'],maximum)
        cpu_difference=float((pred-item['expected']).abs().max());assert cpu_difference<1e-4,(item['spec'],cpu_difference)
        m.train();opt=torch.optim.AdamW(m.groups(),weight_decay=1e-4,fused=True)
        loss=loss_value(m(batch),item['target'].cuda(),'MSE');loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in m.parameters())
        results.append(dict(spec=item['spec'],max_abs_prediction_difference=maximum,cpu_max_abs_difference=cpu_difference,reference_kind='CUDA' if 'expected_cuda' in item else 'CPU',numerical_tolerance=tolerance,reference_runtime=item.get('cuda_reference_runtime',dict(device='CPU')),fused_optimizer_finite=True))
    atomic_json(RUN/'operations/GPU_COMPATIBILITY.json',dict(timestamp=now(),passed=True,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,gpu=torch.cuda.get_device_name(),architecture=architecture,cases=results,numerical_tolerance=max(c['numerical_tolerance'] for c in results),cuda_numerical_tolerance=2e-5,cpu_numerical_tolerance=1e-4))
    print(json.dumps(results))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--make',action='store_true');ap.add_argument('--cuda-reference',action='store_true');a=ap.parse_args()
    if a.make:make()
    elif a.cuda_reference:add_cuda_reference()
    else:verify()
