"""Numerical/gradient equivalence, input controls, isolation, and throughput."""
from common import *
from models import *
from train import Data,forward
import time,warnings

def main():
    configure_torch();warnings.filterwarnings('ignore',message='std\\(\\): degrees of freedom')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');site=6;seed=42
    b0=baseline(site,seed);state={k:v.detach().clone() for k,v in b0.state_dict().items()}
    fast=correction(site,seed,64,'E2C_CBH',state).to(device)
    with torch.no_grad():
        # The source initializes these maps to zero: randomize to avoid a trivial test.
        for mod in [fast.readout[-1],fast.align_out,fast.transfer_adapter[-1]]:
            mod.weight.normal_(0,.03);mod.bias.normal_(0,.01)
    slow=correction(site,seed,64,'E2C_CBH',state,fast=False).to(device);slow.load_state_dict(fast.state_dict())
    fast.eval();slow.eval()
    d=Data('train',site,device,True,limit=512);ix=torch.tensor([0,1,2,3,101,203],device=device)
    raw=np.load(LOCAL/'train.npy',mmap_mode='r');ci=np.load(CACHE/'train/weather_cycle_index.npy')[ix.cpu().numpy()]
    nodes=torch.tensor(np.array(raw[ci,site]),device=device)
    times=torch.tensor(np.load(CACHE/'train/weather_time_offset_hours.npy')[ix.cpu().numpy()],device=device)
    with torch.no_grad():
        base=fast.baseline_model(station_history=d.h[ix,None],cluster_history=d.h[ix],origin_phase=d.phase[ix])['station_factor'][:,0]
    ff=fast(d.h[ix],d.w[ix],d.phase[ix],base)
    ss=slow(station_history=d.h[ix,None],cluster_history=d.h[ix],origin_phase=d.phase[ix],weather=nodes[...,None,None],
            weather_batch_index=torch.arange(len(ix),device=device),weather_time_offset_hours=times)['station_factor'][:,0]
    ferr=float((ff-ss).abs().max());assert torch.allclose(ff,ss,atol=2e-6,rtol=2e-5),ferr
    ff.square().mean().backward();ss.square().mean().backward()
    grad_error=0.;checked=0
    for (n,p),(sn,sp) in zip(fast.named_parameters(),slow.named_parameters()):
        assert n==sn
        if p.grad is None:
            if sp.grad is not None:assert float(sp.grad.abs().max())<1e-7,(n,sp.grad.abs().max())
        else:
            assert sp.grad is not None,n
            err=float((p.grad-sp.grad).abs().max());grad_error=max(err,grad_error)
            assert torch.allclose(p.grad,sp.grad,atol=3e-6,rtol=5e-4),(n,err);checked+=1
    report=dict(forward_max_abs_error=ferr,gradient_max_abs_error=grad_error,gradient_tensors_checked=checked,tests=[])
    # Independent constructors must not share tensor storage.
    second=baseline(site+1,seed)
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(b0.parameters(),second.parameters()))
    report['tests'].append('separate_farm_parameter_storage')
    # Shared seed means all four condition arms start with identical parameters.
    for arm in ['CB','CH','CNEUTRAL','DIRECT','NEUTRAL_LAST']:
        a=correction(site,seed,64,arm,state).to(device).eval()
        reference=correction(site,seed,64,'E2C_CBH',state).to(device).eval()
        assert all(torch.equal(v,reference.state_dict()[n]) for n,v in a.state_dict().items())
        with torch.no_grad():
            pred=a(d.h[ix],d.w[ix],d.phase[ix],base)
            expected=torch.zeros_like(base) if arm=='DIRECT' else d.h[ix,-1:] if arm=='NEUTRAL_LAST' else base
            assert torch.allclose(pred,expected),arm
        del a,reference
    report['tests'].append('matched_initialization_and_output_anchors')
    # Check checkpoint optimizer+RNG continuation with dropout and real CBH model.
    import io
    a=fast; a.train();opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=2e-4)
    def step():
        opt.zero_grad();o=a(d.h[ix],d.w[ix],d.phase[ix],base);o.square().mean().backward();opt.step();return o.detach().clone()
    step();buf=io.BytesIO();torch.save(dict(m=a.state_dict(),o=opt.state_dict(),rng=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if device.type=='cuda' else None),buf)
    expected=step();expected_state={k:v.detach().clone() for k,v in a.state_dict().items()}
    buf.seek(0);ck=torch.load(buf,weights_only=False);a.load_state_dict(ck['m']);opt.load_state_dict(ck['o']);torch.set_rng_state(ck['rng'])
    if device.type=='cuda':torch.cuda.set_rng_state(ck['cuda'])
    actual=step();assert torch.equal(expected,actual)
    assert all(torch.equal(v,a.state_dict()[n]) for n,v in expected_state.items())
    report['tests'].append('optimizer_rng_checkpoint_exact_continuation')
    del fast,slow,a,opt,second,b0
    if device.type=='cuda':torch.cuda.empty_cache()
    benchmark={}
    for family in ['B0','E2C_CBH']+FAMILIES:
        seed_all(seed)
        model=baseline(site,seed) if family=='B0' else correction(site,seed,128,family,state) if family=='E2C_CBH' else named(family,128)
        model=model.to(device);model.train();h=d.h[:256];w=d.w[:256];p=d.phase[:256]
        with torch.no_grad():base_batch=torch.zeros_like(h)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
        def call(hh,ww):
            if family=='B0':return model(station_history=hh[:,None],cluster_history=hh,origin_phase=p)['station_factor'][:,0]
            if family=='E2C_CBH':return model(hh,ww,p,base_batch)
            return model(hh,ww,p)
        for i in range(12):
            if i==2:
                if device.type=='cuda':torch.cuda.synchronize()
                t=time.perf_counter()
            optim.zero_grad();pred=call(h,w);loss=(pred-d.y[:256]).abs().mean();loss.backward();optim.step()
            assert torch.isfinite(loss)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=(time.perf_counter()-t)/10
        model.eval()
        with torch.no_grad():
            original=call(h,w);hh=h.clone();hh[:,0]+=1;changed=call(hh,w)
            if family=='CNN_W':assert torch.equal(original,changed)
            else:assert float((original-changed).abs().max())>1e-8,(family,'ignores early history')
            if family.endswith('_H'):assert torch.equal(original,call(h,w+1))
            elif family!='B0':assert float((original-call(h,w+1)).abs().max())>1e-8,(family,'ignores weather')
        benchmark[family]=dict(train_step_seconds=seconds,batch_size=256,width=128 if family!='B0' else 64,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0)
        print(family,benchmark[family],flush=True)
        del model,optim
        if device.type=='cuda':torch.cuda.empty_cache()
    report['benchmark']=benchmark;report['tests'].append('all_family_forward_backward_and_input_sensitivity')
    report['device']=str(device);report['passed']=True
    atomic_json(RUN/'PREFLIGHT.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
