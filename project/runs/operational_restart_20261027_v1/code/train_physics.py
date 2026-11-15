"""Fit 2022, select on 2023. Save two distinct selected epochs at most, no predictions."""
from settings import *
from physical_data import FarmData
from physical_models import PhysicalModel,frozen

import torch,numpy as np,time,argparse,traceback


def save_torch(path,obj):
    tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

@torch.no_grad()
def predict(model,data,batch=256,intervention=None):
    model.eval();out=[]
    for st in range(0,data.n,batch):out.append(model(data.batch(slice(st,st+batch)),intervention=intervention).cpu())
    out=torch.cat(out).numpy();assert np.isfinite(out).all();return out

def metrics(p,y):
    e=np.asarray(p,dtype='f8')-np.asarray(y,dtype='f8')
    def calc(a):return dict(n=int(a.size),mae=float(abs(a).mean()),rmse=float(np.sqrt(np.square(a).mean())),bias=float(a.mean()),sq_sum=float(np.square(a).sum()))
    return dict(**calc(e),horizons={label:calc(e[:,a:b]) for label,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]},lead_mae=abs(e).mean(0).tolist(),lead_rmse=np.sqrt(np.square(e).mean(0)).tolist())

def loss_value(p,y,kind):
    e=p-y
    if kind=='MAE':return e.abs().mean()
    if kind=='MSE':return e.square().mean()
    if kind=='MIX':return .5*e.abs().mean()+.5*e.square().mean()/.1
    raise ValueError(kind)

def run(spec,device='cuda:0',smoke=False,stop_after_epoch=None):
    assert spec.get('data_fingerprint',FINGERPRINT)==FINGERPRINT
    setup();frozen.configure();frozen.seed_all(spec['seed'])
    out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    if (out/'spec.json').exists():assert json.loads((out/'spec.json').read_text())==spec
    atomic_json(out/'spec.json',spec)
    limit=64 if smoke else None
    train=FarmData('train',spec['site'],spec['variant'],device,limit=limit,delay_minutes=spec.get('delay_minutes',15),capacity_control=spec.get('capacity_control',False),seed=spec['seed'])
    val=FarmData('val',spec['site'],spec['variant'],device,stats=train.stats,limit=limit,delay_minutes=spec.get('delay_minutes',15),capacity_control=spec.get('capacity_control',False),seed=spec['seed'])
    atomic_json(out/'normalization.json',train.stats)
    model=PhysicalModel(spec,train.x.shape[-1],train.historical_input.mean(0).cpu()).to(device)
    opt=torch.optim.AdamW(model.groups(),weight_decay=1e-4,fused=device.startswith('cuda'))
    epochs=2 if smoke else 80;scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    selections={};best={'mae':float('inf'),'rmse':float('inf')};history=[];stale=0;elapsed=0.;start=0
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert saved['spec']==spec
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        selections=saved['selections'];best=saved['best'];history=saved['history'];stale=saved['stale'];elapsed=saved['elapsed'];start=saved['epoch']
        torch.set_rng_state(saved['rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    began=time.monotonic()
    for epoch in range(start,epochs):
        if stale>=10:break
        model.train();total=0.;t=time.monotonic()
        order=torch.as_tensor(np.random.default_rng(spec['seed']*1000003+epoch).permutation(train.n),device=device)
        for begin in range(0,train.n,256):
            ix=order[begin:begin+256];opt.zero_grad(set_to_none=True)
            loss=loss_value(model(train.batch(ix)),train.y[ix],spec['loss'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
        p=predict(model,val);m=metrics(p,val.y.cpu().numpy());scheduler.step()
        improved=[key for key in best if m[key]<best[key]]
        # Patience is tied to a predeclared primary criterion; selection factors do not alter trajectories.
        primary='mae' # identical stopping rule across all loss arms
        stale=0 if primary in improved else stale+1
        if improved:
            name=f'epoch_{epoch+1:03d}.pt'
            save_torch(out/name,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},spec=spec,epoch=epoch+1))
            for key in improved:best[key]=m[key];selections[key]=dict(epoch=epoch+1,checkpoint=name,value=m[key])
        row=dict(epoch=epoch+1,train_loss=total/train.n,val_mae=m['mae'],val_rmse=m['rmse'],stale=stale,seconds=time.monotonic()-t,timestamp=now())
        history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'selection.json',selections)
        save_torch(out/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),spec=spec,
            selections=selections,best=best,history=history,stale=stale,elapsed=elapsed+time.monotonic()-began,epoch=epoch+1,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.startswith('cuda') else None))
        for path in out.glob('epoch_*.pt'):
            if path.name not in {s['checkpoint'] for s in selections.values()}:path.unlink()
        if stop_after_epoch is not None and epoch+1>=stop_after_epoch:
            assert smoke, 'Controlled interruption only for smoke checks'
            return
        atomic_json(out/'progress.json',dict(**row,pid=os.getpid(),device=device));print(json.dumps(row),flush=True)
    results={};computed={}
    from diagnostics import stratified, daily_sums
    for criterion,selection in selections.items():
        cp=selection['checkpoint']
        if cp not in computed:
            model.load_state_dict(torch.load(out/cp,map_location=device,weights_only=False)['model'])
            p=predict(model,val);computed[cp]=metrics(p,val.y.cpu().numpy())
            if criterion=='mae':
                atomic_json(out/'diagnostics.json',dict(selection='mae',year=2023,rows=stratified(p,val)))
                atomic_json(out/'daily.json',daily_sums(p,val))
        results[criterion]=dict(**selection,metrics=computed[cp],sha256=sha(out/cp))
    interventions={}
    result=dict(**spec,finished=now(),seconds=elapsed+time.monotonic()-began,epochs=len(history),smoke_only=smoke,
        precision='float32_no_tf32',n_train=train.n,n_val=val.n,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        selections=results,interventions=interventions,outputs='selected weights and metrics only; no prediction copies',device=device)
    atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=smoke))
    (out/'latest.pt').unlink(missing_ok=True);(out/'FAILURE.json').unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec');ap.add_argument('--job-id');ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    spec=json.loads(Path(a.spec).read_text()) if a.spec else next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==a.job_id)
    try:run(spec,a.device,a.smoke)
    except BaseException:
        out=(RUN/'smoke_jobs' if a.smoke else JOBS)/spec['id'];atomic_json(out/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
