"""Independent farm training, deployable forward inputs, checkpointed resource yield."""
from settings import *
from data import FarmData
from model import CorrectedModel,interp
from physical_models import frozen
import numpy as np,torch,time,argparse,socket,traceback

def save_torch(path,value):
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def metrics(p,y):
    e=np.asarray(p,'f8')-np.asarray(y,'f8')
    calc=lambda x:dict(mae=float(abs(x).mean()),rmse=float(np.sqrt((x*x).mean())),n=int(x.size))
    return dict(**calc(e),horizons={name:calc(e[:,a:b]) for name,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]},lead_mae=abs(e).mean(0).tolist(),lead_rmse=np.sqrt((e*e).mean(0)).tolist())
@torch.no_grad()
def predict(model,data):
    model.eval();out=[];winds=[]
    for i in range(0,data.n,256):
        p,w=model(data.batch(slice(i,i+256)));out.append(p.cpu());winds.append(w.cpu())
    return torch.cat(out).numpy(),torch.cat(winds).numpy()
def daily_and_scenes(out,p,uv,data,model):
    from operational import interpolate_nodes
    e=p.astype('f8')-data.y_numpy;days=(data.origin+28800)//86400;unique=np.unique(days)
    sums=np.array([np.stack([np.full(24,(days==d).sum()),abs(e[days==d]).sum(0),(e[days==d]**2).sum(0)],-1) for d in unique])
    sd=model.nwp_sd.cpu().numpy()[1:3];mu=model.nwp_mu.cpu().numpy()[1:3]
    off=data.inputs['offset'].cpu().numpy();raw=data.inputs['nodes'].cpu().numpy()[...,1:3]
    truth=data.era.cpu().numpy();lead=np.arange(1,7,dtype='f4')
    speed=lambda x:np.linalg.norm(interpolate_nodes(x*sd+mu,off,lead),axis=-1)
    obs=speed(truth);before=(speed(raw)-obs)**2;after=(speed(uv)-obs)**2
    weather=np.array([np.stack([np.full(6,(days==d).sum()),before[days==d].sum(0),after[days==d].sum(0)],-1) for d in unique])
    np.savez_compressed(out/'daily.npz',day=unique,power=sums,weather=weather)
    h=data.h_numpy[:,-1];scenes={}
    for name,mask in [('low',h<.1),('midlow',(h>=.1)&(h<.3)),('midhigh',(h>=.3)&(h<.7)),('high',h>=.7)]:
        if mask.any():scenes[name]=metrics(p[mask],data.y_numpy[mask])
    return dict(weather=dict(raw_wind_mse=float(before.mean()),corrected_wind_mse=float(after.mean()),
        raw_wind_mse_4_6=float(before[:,3:].mean()),corrected_wind_mse_4_6=float(after[:,3:].mean()),
        lead_raw_wind_mse=before.mean(0).tolist(),lead_corrected_wind_mse=after.mean(0).tolist()),current_power=scenes)

def run(spec,smoke=False,limit=None,epochs_override=None,stop_after_epoch=None,device='cuda:0'):
    frozen.configure();frozen.seed_all(spec['seed'])
    out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return 0
    atomic_json(out/'spec.json',spec)
    atomic_json(out/'runtime.json',dict(host=socket.gethostname(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=now(),
        gpu_name=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu',torch_version=torch.__version__,cuda_version=torch.version.cuda,numpy_version=np.__version__))
    limit=limit or (96 if smoke else None)
    train=FarmData('train',spec,device,limit=limit);val=FarmData('val',spec,device,stats=train.stats,limit=limit)
    atomic_json(out/'normalization.json',train.stats)
    model=CorrectedModel(spec,train.stats).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=device.startswith('cuda'))
    epochs=epochs_override or (2 if smoke else 80);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=float('inf');stale=0;history=[];start=0;elapsed=0.;best_epoch=0
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert saved['spec']==spec
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['opt']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history'];elapsed=saved['elapsed'];best_epoch=saved['best_epoch']
        torch.set_rng_state(saved['rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    began=time.monotonic()
    for epoch in range(start,epochs):
        if stale>=10:break
        model.train();t=time.monotonic();power_sum=0.;aux_sum=0.
        order=torch.as_tensor(np.random.default_rng(spec['seed']*1000003+epoch).permutation(train.n),device=device)
        for begin in range(0,train.n,256):
            ix=order[begin:begin+256];opt.zero_grad(set_to_none=True)
            pred,uv=model(train.batch(ix));pl=(pred-train.y[ix]).abs().mean()
            auxiliary=(uv-train.era[ix]).square().mean() if spec['aux_weight'] else pl.new_zeros(())
            loss=pl+spec['aux_weight']*auxiliary
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            power_sum+=float(pl.detach())*len(ix);aux_sum+=float(auxiliary.detach())*len(ix)
        pred,_=predict(model,val);m=metrics(pred,val.y_numpy);scheduler.step()
        improved=m['mae']<best;stale=0 if improved else stale+1
        if improved:
            best=m['mae'];best_epoch=epoch+1
            save_torch(out/'best.pt',dict(spec=spec,epoch=best_epoch,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
        row=dict(epoch=epoch+1,train_power_mae=power_sum/train.n,train_aux_mse=aux_sum/train.n,val_mae=m['mae'],val_rmse=m['rmse'],
            seconds=time.monotonic()-t,timestamp=now(),stale=stale)
        history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',dict(**row,pid=os.getpid()))
        save_torch(out/'latest.pt',dict(spec=spec,epoch=epoch+1,model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict(),
            best=best,stale=stale,history=history,elapsed=elapsed+time.monotonic()-began,best_epoch=best_epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.startswith('cuda') else None))
        print(json.dumps(row),flush=True)
        if smoke and stop_after_epoch is not None and epoch+1>=stop_after_epoch:return 75
        if not smoke and (out/'YIELD_REQUEST.json').exists():
            atomic_json(out/'YIELDED.json',dict(timestamp=now(),checkpoint_epoch=epoch+1));return 75
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    pred,uv=predict(model,val);m=metrics(pred,val.y_numpy);diag=daily_and_scenes(out,pred,uv,val,model)
    result=dict(spec=spec,finished=now(),epochs=len(history),best_epoch=best_epoch,seconds=elapsed+time.monotonic()-began,
        metrics=m,diagnostics=diag,n_train=train.n,n_val=val.n,evaluation_year=2023,smoke_only=smoke,
        trainable_parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(out/'best.pt'),
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,precision='float32_no_tf32',labels_in_forward=False)
    atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=smoke,passed=True))
    (out/'latest.pt').unlink(missing_ok=True);(out/'FAILURE.json').unlink(missing_ok=True);return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job-id',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--epochs',type=int);ap.add_argument('--stop-after-epoch',type=int);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==a.job_id)
    try:rc=run(spec,a.smoke,a.limit,a.epochs,a.stop_after_epoch,a.device)
    except torch.cuda.OutOfMemoryError:
        out=(RUN/'smoke_jobs' if a.smoke else JOBS)/spec['id']
        atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='cuda_oom',checkpoint_exists=(out/'latest.pt').exists()))
        rc=75
    except BaseException:
        out=(RUN/'smoke_jobs' if a.smoke else JOBS)/spec['id'];atomic_json(out/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
    raise SystemExit(rc)
