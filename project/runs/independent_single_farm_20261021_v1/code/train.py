"""Resumable, one process / one farm / one seed training. No shared parameters."""
from common import *
import argparse, datetime, time, traceback, signal
from models import baseline, correction, named

def now(): return datetime.datetime.now().astimezone().isoformat()

def save_torch(path,obj):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(obj,temp);os.replace(temp,path)

def save_array(path,array):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with open(temp,'wb') as f:np.save(f,array)
    os.replace(temp,path)

class Data:
    def __init__(self,split,site,device,use_weather,reference=None,climatology=False,limit=None):
        p=RUN/'prepared'/split/f'{site:02d}'
        def tensor(path,dtype=torch.float32):
            a=np.load(path,mmap_mode='r');a=a[:limit] if limit else a
            return torch.tensor(np.array(a),dtype=dtype,device=device)
        self.h=tensor(p/'history.npy');self.y=tensor(p/'target.npy')
        self.phase=tensor(p.parent/'origin_phase.npy',torch.long)
        self.w=tensor(p/'weather.npy') if use_weather else None
        if climatology:
            self.w=torch.tensor(np.load(RUN/'prepared/train'/f'{site:02d}'/'weather_climatology.npy'),device=device)[None].expand(len(self.h),-1,-1)
        self.b=tensor(reference/f'{split}_prediction.npy') if reference else None
        self.n=len(self.h)

def forward(model,data,indices,family):
    h=data.h[indices];p=data.phase[indices]
    if family=='B0':
        return model(station_history=h[:,None],cluster_history=h,origin_phase=p)['station_factor'][:,0]
    w=data.w[indices] if data.w is not None else None
    if family=='E2C_CBH' or family in ARMS:return model(h,w,p,data.b[indices])
    return model(h,w,p)

@torch.no_grad()
def predict(model,data,family,batch):
    model.eval();pred=[]
    for st in range(0,data.n,batch):pred.append(forward(model,data,slice(st,st+batch),family).float())
    out=torch.cat(pred)
    if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite prediction')
    return out

def metrics(prediction,truth,history,site,split):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(truth,dtype=np.float64);e=p-y
    n=len(p);pre=RUN/'prepared'/split;origin=np.load(pre/'origin_unix.npy')[:n]
    dates=(origin+8*3600).astype('datetime64[s]');months=dates.astype('datetime64[M]').astype(int)%12+1
    hours=((origin+8*3600)//3600)%24
    desc=json.loads((RUN/'prepared/train'/f'{site:02d}'/'train_descriptors.json').read_text())
    ramp=np.abs(history[:,-1]-history[:,-5]);current=history[:,-1]
    masks={}
    for label,ms in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]: masks['season_'+label]=np.isin(months,ms)
    for k in range(4):masks[f'local_hour_{k*6:02d}_{(k+1)*6:02d}']=(hours>=k*6)&(hours<(k+1)*6)
    for label,lo,hi in [('low',-np.inf,.1),('medium_low',.1,.3),('medium_high',.3,.7),('high',.7,np.inf)]:masks['current_cf_'+label]=(current>=lo)&(current<hi)
    masks['observed_recent_ramp']=ramp>desc['ramp_q90'];masks['observed_nonramp']=~masks['observed_recent_ramp']
    one_hour_change=np.abs(y-np.concatenate((history[:,-4:],y[:,:-4]),axis=1))
    masks['future_ramp_diagnostic_only']=(one_hour_change>=.2).any(1)
    def score(mask):
        a=e[mask];return dict(n_origins=int(len(a)),mae=float(np.abs(a).mean()),rmse=float(np.sqrt((a*a).mean())),bias=float(a.mean())) if len(a) else dict(n_origins=0,mae=None,rmse=None,bias=None)
    result=score(np.ones(n,dtype=bool));result.update(lead_mae=np.abs(e).mean(0).tolist(),lead_rmse=np.sqrt((e*e).mean(0)).tolist(),
                scenarios={k:score(v) for k,v in masks.items()},outside_0_1_fraction=float(np.mean((p<0)|(p>1))))
    daily={}
    for day in np.unique(origin//86400):
        mask=origin//86400==day;a=e[mask];daily[str(day)]=dict(n=int(len(a)),abs_sum=float(np.abs(a).sum()),sq_sum=float((a*a).sum()))
    result['daily_utc']=daily
    return result

def run(spec,smoke=False):
    configure_torch();seed_all(spec['seed']);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
    family=spec['family'];site=spec['site'];seed=spec['seed'];cfg=spec['config']
    out=RUN/('smoke_jobs' if smoke else 'jobs')/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    if (out/'spec.json').exists() and json.loads((out/'spec.json').read_text())!=spec:raise ValueError('Existing job specification mismatch')
    atomic_json(out/'spec.json',spec)
    reference=RUN/'jobs'/ref_id(site,seed) if family=='E2C_CBH' or family in ARMS else None
    refhash=None
    if reference:
        if smoke and spec.get('smoke_reference'):reference=RUN/'smoke_jobs'/spec['smoke_reference']
        assert (reference/'SUCCESS.json').exists(),reference
        state=torch.load(reference/'best.pt',map_location='cpu',weights_only=False)['model'];refhash=sha(reference/'best.pt')
        model=correction(site,seed,cfg['width'],family,state)
    elif family=='B0':model=baseline(site,seed)
    else:model=named(family,cfg['width'])
    model.to(device);pars=[p for p in model.parameters() if p.requires_grad]
    use_weather=family!='B0' and not family.endswith('_H')
    limit=512 if smoke else None
    train=Data('train',site,device,use_weather,reference,family=='CLIMATOLOGY',limit)
    val=Data('val',site,device,use_weather,reference,family=='CLIMATOLOGY',limit)
    optimizer=torch.optim.AdamW(pars,lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=device.type=='cuda')
    epochs=2 if smoke else PROTOCOL['epochs'];patience=PROTOCOL['patience']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    best=float('inf');best_epoch=0;stale=0;start_epoch=0;elapsed_prior=0.;history=[]
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert state['reference_sha256']==refhash
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];start_epoch=state['epoch']+1;elapsed_prior=state['elapsed']
        history=state['history'];torch.set_rng_state(state['rng_cpu'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['rng_cuda'])
    started=time.perf_counter();batch=PROTOCOL['batch_size'];eval_batch=PROTOCOL['eval_batch_size']
    for epoch in range(start_epoch,epochs):
        if stale>=patience:break
        t=time.perf_counter();model.train()
        order=np.random.default_rng(seed*1000003+epoch).permutation(train.n)
        order=torch.tensor(order,device=device);total=torch.zeros((),device=device,dtype=torch.float64)
        for st in range(0,train.n,batch):
            ix=order[st:st+batch];optimizer.zero_grad(set_to_none=True)
            pred=forward(model,train,ix,family);loss=(pred-train.y[ix]).abs().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(pars,PROTOCOL['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            total+=loss.detach().double()*len(ix)
        pv=predict(model,val,family,eval_batch);vm=float((pv-val.y).abs().double().mean());scheduler.step()
        if vm<best:
            best=vm;best_epoch=epoch+1;stale=0
            save_torch(out/'best.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},spec=spec,epoch=best_epoch,val_mae=best,reference_sha256=refhash))
        else:stale+=1
        row=dict(epoch=epoch+1,train_mae=float(total/train.n),val_mae=vm,best_val_mae=best,best_epoch=best_epoch,stale=stale,
                seconds=time.perf_counter()-t,lr=optimizer.param_groups[0]['lr'],timestamp=now())
        history.append(row);print(json.dumps(row),flush=True);atomic_json(out/'history.json',history)
        elapsed=elapsed_prior+time.perf_counter()-started
        save_torch(out/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,elapsed=elapsed,reference_sha256=refhash,
                   rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state() if device.type=='cuda' else None))
        atomic_json(out/'progress.json',dict(**row,pid=os.getpid(),device=str(device),elapsed_seconds=elapsed))
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    result=dict(id=spec['id'],family=family,site=site,station_id=str(static()['station_id'][site]),seed=seed,stage=spec['stage'],
                best_epoch=best_epoch,epochs_run=len(history),validation_selection_mae=best,config=cfg,reference_sha256=refhash,
                trainable_parameters=sum(p.numel() for p in pars),total_parameters=sum(p.numel() for p in model.parameters()),
                active_gradient_parameters=sum(p.numel() for p in pars if p.grad is not None),
                seconds_training=elapsed_prior+time.perf_counter()-started,precision=PROTOCOL['precision'],completed_at=now(),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0,
                gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',smoke_only=smoke)
    # Tuning never loads or evaluates 2024. References also export train B0 cache.
    splits=['val'] if spec['stage']=='tune' else ['val','test']
    if family=='B0':splits=['train','val'] if seed in TUNE_SEEDS else ['train','val','test']
    for split in splits:
        data=train if split=='train' else val if split=='val' else Data(split,site,device,use_weather,reference,family=='CLIMATOLOGY',limit)
        pred=predict(model,data,family,eval_batch).cpu().numpy();save_array(out/f'{split}_prediction.npy',pred)
        if data.b is not None:
            save_array(out/f'{split}_correction.npy',pred-data.b.cpu().numpy())
        if split!='train':result[split]=metrics(pred,data.y.cpu().numpy(),data.h.cpu().numpy(),site,split)
        if split=='test':del data
    result['checkpoint_sha256']=sha(out/'best.pt')
    atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(id=spec['id'],completed_at=now(),metrics_sha256=sha(out/'metrics.json'),checkpoint_sha256=result['checkpoint_sha256']))
    (out/'latest.pt').unlink(missing_ok=True)
    print('COMPLETE '+spec['id'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    spec=json.loads(Path(a.spec).read_text())
    try:run(spec,a.smoke)
    except Exception as e:
        out=RUN/('smoke_jobs' if a.smoke else 'jobs')/spec['id']
        atomic_json(out/'FAILURE.json',dict(timestamp=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
