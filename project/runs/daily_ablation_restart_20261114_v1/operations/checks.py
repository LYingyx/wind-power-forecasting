"""Check unchanged controls, station isolation and joint input interventions."""
from pathlib import Path
import sys,json,copy,importlib.util,argparse,shutil,time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from data import SharedData
from model import SharedModel,configure
from train import run
import torch,numpy as np

def main(device='cpu'):
    configure();specs=json.loads((RUN/'manifest.json').read_text());cfg=dict(width=64,lr=2e-4,dropout=.05,weight_decay=1e-4)
    base=dict(specs[0],family='W0',input='R3',config=cfg,condition='neutral');base.pop('config_from',None)
    d=SharedData('train',base);ix=np.array([0,1,54,55,100,101,900,901]);b,y,si,ti=d.batch(ix,device)
    origin=json.loads((PROJECT/'runs/daily_protocol_restart_20261114_v1/SCIENCE_FREEZE.json').read_text())
    src=PROJECT/'runs/daily_protocol_restart_20261114_v1/code/model.py';assert sha(src)==origin['files']['code/model.py']
    imp=importlib.util.spec_from_file_location('daily_original_model_for_audit',src);old=importlib.util.module_from_spec(imp);imp.loader.exec_module(old)
    for fam in ['W0','TF']:
        sp=dict(base,family=fam);a=old.SharedModel(sp).to(device).eval();c=SharedModel(sp).to(device).eval()
        assert a.state_dict().keys()==c.state_dict().keys()
        for k in a.state_dict():assert torch.equal(a.state_dict()[k],c.state_dict()[k]),k
        with torch.no_grad():
            for m in [a,c]:
                head=[x for x in (m.head if fam=='W0' else m.original.head).modules() if isinstance(x,torch.nn.Linear)][-1];head.weight.fill_(.01);head.bias.fill_(.02)
            torch.testing.assert_close(a(b),c(b),atol=0,rtol=0)
    # Independent labels/inputs are exactly the global-loader target slice.
    station=17;sd=SharedData('val',dict(base,station=station));full=SharedData('val',base);indices=np.array([0,20,100,1000,sd.n-1]);sb,sy,ss,st=sd.batch(indices,device);fb,fy,fs,ft=full.batch(indices*55+station,device)
    assert sd.farms==1 and sd.n==29647 and (ss==0).all() and (sb['site']==station).all()
    for k in sb:torch.testing.assert_close(sb[k],fb[k],atol=0,rtol=0)
    torch.testing.assert_close(sy,fy,atol=0,rtol=0)
    class Perturb:
        def __init__(self,a,history=False):self.a=a;self.history=history
        def __getitem__(self,k):
            v=np.array(self.a[k]);v[...,:-1]+=.2 if self.history else 0
            if not self.history:v+=.25
            return v
    for h,w in [(0,1),(1,0),(0,0)]:
        sp=dict(base,history_control='constant_current' if h==0 else 'real',weather_control='training_mean' if w==0 else 'real')
        dd=SharedData('train',sp);before,yy,_,_=dd.batch(ix,device)
        if h==0:
            assert torch.equal(before['h'],b['h'][:,-1:].expand_as(b['h']))
            dd.a['h']=Perturb(dd.a['h'],True)
        if w==0:
            assert torch.count_nonzero(before['weather'])==0 and torch.count_nonzero(before['x'][...,:15])==0
            dd.a['nodes']=Perturb(dd.a['nodes']);dd.a['dynamics']=Perturb(dd.a['dynamics'])
        after,yafter,_,_=dd.batch(ix,device)
        for k in before:torch.testing.assert_close(before[k],after[k],atol=0,rtol=0)
        torch.testing.assert_close(yy,yafter,atol=0,rtol=0)
    cases={}
    for s in specs:
        if s['stage'] in ['tune','independent']:continue
        fields={k:v for k,v in s.items() if k in ['family','input','condition','weather_variant','no_sel','no_near','no_rings','mask_sel','history_control','weather_control']}
        cases[json.dumps(fields,sort_keys=True)]=fields
    results=[];counts={}
    for fields in cases.values():
        sp=dict(base,**fields);x={k:v.clone() for k,v in b.items()}
        x.update(base=x['h'][:,-1:].expand(-1,24).clone(),base_delta=torch.zeros_like(y),condition_h=x['h'].clone())
        m=SharedModel(sp).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
        for _ in range(2):
            out=m(x);loss=(out-y).abs().mean();loss.backward()
            assert torch.isfinite(out).all() and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            opt.step();opt.zero_grad(set_to_none=True)
        m.eval();perm=torch.arange(len(y)-1,-1,-1,device=device)
        with torch.no_grad():torch.testing.assert_close(m(x)[perm],m({k:v[perm] for k,v in x.items()}),atol=2e-5,rtol=2e-5)
        n=sum(p.numel() for p in m.parameters());results.append(dict(**fields,parameters=n));counts[(sp['condition'],sp.get('weather_variant'))]=n
    for c in ['neutral','cbh']:assert counts[c,'E2L']==counts[c,'E2N']
    # Resume must give the same checkpoint as uninterrupted training.
    sp=dict(base,id='CHECK_RESUME',epochs=1,patience=2)
    for name in ['CHECK_RESUME','CHECK_DIRECT','CHECK_STATION']:shutil.rmtree(RUN/'smoke_jobs'/name,ignore_errors=True)
    assert run(sp,smoke=True,device=device,max_batches=1)==75
    assert run(sp,smoke=True,device=device)==0
    assert run(dict(sp,id='CHECK_DIRECT'),smoke=True,device=device)==0
    aa=torch.load(RUN/'smoke_jobs/CHECK_RESUME/best.pt',map_location='cpu',weights_only=False);bb=torch.load(RUN/'smoke_jobs/CHECK_DIRECT/best.pt',map_location='cpu',weights_only=False)
    for k in aa['model']:torch.testing.assert_close(aa['model'][k],bb['model'][k],atol=0,rtol=0)
    assert run(dict(sp,id='CHECK_STATION',station=station),smoke=True,device=device)==0
    mm=json.loads((RUN/'smoke_jobs/CHECK_STATION/metrics.json').read_text());assert len(mm['metrics']['per_farm_mae'])==1 and mm['metrics']['counts']==[16]
    # Include learned/nonzero head in cross-device fixture.
    m=SharedModel(base).eval()
    with torch.no_grad():m.head[-1].weight.fill_(.01);m.head[-1].bias.fill_(.02);out=m({k:v.cpu() for k,v in b.items()})
    torch.save(dict(spec=base,b={k:v.cpu() for k,v in b.items()},weights=m.state_dict(),output=out),RUN/'operations/fixture.pt')
    atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,cases=results,checks=['unchanged_W0_TF_exact_weights_and_nonzero_outputs','independent_station_labels_and_site_index','constant_history_no_hidden_STATE_path','masked_weather_all_derived_paths','finite_gradients_and_sample_independence','E2L_E2N_exact_parameter_match','checkpoint_resume_exact_match','one_farm_metric_counts']))
    print('passed',len(results),'model cases',flush=True)

def fixture():
    configure();f=torch.load(RUN/'operations/fixture.pt',map_location='cpu',weights_only=False);m=SharedModel(f['spec']).cuda().eval();m.load_state_dict(f['weights'])
    with torch.no_grad():out=m({k:v.cuda() for k,v in f['b'].items()}).cpu()
    torch.testing.assert_close(out,f['output'],atol=3e-5,rtol=3e-5)
    atomic_json(RUN/'operations/GPU_CHECKS.json',dict(timestamp=now(),passed=True,max_absolute_error=float((out-f['output']).abs().max()),gpu=torch.cuda.get_device_name()))
    print('GPU fixture passed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');p.add_argument('--device',default='cpu');a=p.parse_args();fixture() if a.fixture else main(a.device)
