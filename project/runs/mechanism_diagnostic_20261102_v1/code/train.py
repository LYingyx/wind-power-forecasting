"""Independent jobs with durable resume and two separated diagnostic routes."""
from settings import *
from experiment import *
import time,argparse,socket,traceback

def save_torch(path,value):
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)

@torch.no_grad()
def predict(model,data,correction):
    model.eval();out=[]
    for i in range(0,data.n,256):
        ix=slice(i,i+256)
        y=model(data.x[ix],data.raw[ix]) if correction else model(data.batch(ix))
        out.append(y.cpu().numpy())
    return np.concatenate(out)

def power_metrics(p,y):
    e=np.asarray(p,'f8')-np.asarray(y,'f8')
    calc=lambda x:dict(mae=float(abs(x).mean()),rmse=float(np.sqrt(np.mean(x*x))),n=int(x.size))
    return dict(**calc(e),horizons={name:calc(e[:,a:b]) for name,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]},lead_mae=abs(e).mean(0).tolist(),lead_rmse=np.sqrt(np.mean(e*e,0)).tolist())

def linear_baselines(train,val):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    results={}
    features=[('MOS_UV',train.mos_features(),val.mos_features()),
              ('RIDGE_UV_CONTEXT',train.weather_context(),val.weather_context()),
              ('RIDGE_UV_OBS',train.context,val.context)]
    delta=(train.truth-train.nodes[...,1:3]).reshape(train.n,-1)
    for name,x,v in features:
        model=make_pipeline(StandardScaler(),Ridge(alpha=10,solver='cholesky')).fit(x,delta)
        z=model.predict(v).reshape(val.n,8,2)+val.nodes[...,1:3]
        results[name]=val.metrics(z)
    trspeed=np.linalg.norm(train.hourly_uv(train.truth),axis=-1)
    trraw=np.linalg.norm(train.hourly_uv(train.nodes[...,1:3]),axis=-1)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10,solver='cholesky')).fit(train.mos_features(),trspeed-trraw)
    valraw=np.linalg.norm(val.hourly_uv(val.nodes[...,1:3]),axis=-1)
    valspeed=np.linalg.norm(val.hourly_uv(val.truth),axis=-1)
    err=valraw+model.predict(val.mos_features())-valspeed
    results['MOS_SPEED']=dict(speed_mse=float(np.mean(err**2)),speed_rmse=float(np.sqrt(np.mean(err**2))),speed_mse_1_3=float(np.mean(err[:,:3]**2)),speed_mse_4_6=float(np.mean(err[:,3:]**2)),lead_speed_mse=np.mean(err**2,axis=0).tolist())
    return results

def run(spec,smoke=False,device='cuda:0',stop_after_epoch=None):
    frozen.configure();frozen.seed_all(spec['seed']);corr=spec['role']=='correction'
    out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return 0
    atomic_json(out/'spec.json',spec)
    atomic_json(out/'runtime.json',dict(host=socket.gethostname(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=now()))
    cls=CorrectionData if corr else InteractionData;limit=96 if smoke else None
    train=cls('train',spec,device,limit=limit);val=cls('val',spec,device,stats=train.stats,limit=limit)
    atomic_json(out/'normalization.json',train.stats)
    model=(PureCorrector(spec['seed']) if corr else InteractionModel(spec)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=device.startswith('cuda'))
    epochs=2 if smoke else 80;sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=float('inf');best_epoch=0;stale=0;history=[];start=0;elapsed=0.;best_speed=float('inf')
    if (out/'latest.pt').exists():
        s=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert s['spec']==spec
        model.load_state_dict(s['model']);opt.load_state_dict(s['opt']);sched.load_state_dict(s['scheduler'])
        best=s['best'];best_epoch=s['best_epoch'];stale=s['stale'];history=s['history'];start=s['epoch'];elapsed=s['elapsed'];best_speed=s['best_speed']
        torch.set_rng_state(s['rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(s['cuda_rng'].cpu())
    begun=time.monotonic()
    for epoch in range(start,epochs):
        if stale>=10:break
        tick=time.monotonic();model.train();loss_sum=0.
        order=torch.as_tensor(np.random.default_rng(spec['seed']*1000003+epoch).permutation(train.n),device=device)
        for begin in range(0,train.n,256):
            ix=order[begin:begin+256];opt.zero_grad(set_to_none=True)
            prediction=model(train.x[ix],train.raw[ix]) if corr else model(train.batch(ix))
            loss=(prediction-train.y[ix]).square().mean() if corr else (prediction-train.y[ix]).abs().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        pred=predict(model,val,corr);m=val.metrics(pred) if corr else power_metrics(pred,val.y_numpy)
        score=m['node_normalized_uv_mse'] if corr else m['mae'];sched.step()
        if score<best:
            best=score;best_epoch=epoch+1;stale=0
            save_torch(out/'best.pt',dict(spec=spec,epoch=best_epoch,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
        else:stale+=1
        if corr and m['speed_mse']<best_speed:
            best_speed=m['speed_mse'];save_torch(out/'best_speed.pt',dict(spec=spec,epoch=epoch+1,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
        row=dict(epoch=epoch+1,train_loss=loss_sum/train.n,val_selection=score,val_metrics=m,seconds=time.monotonic()-tick,timestamp=now(),stale=stale)
        history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',dict(**row,pid=os.getpid()))
        save_torch(out/'latest.pt',dict(spec=spec,epoch=epoch+1,model=model.state_dict(),opt=opt.state_dict(),scheduler=sched.state_dict(),best=best,best_epoch=best_epoch,stale=stale,history=history,elapsed=elapsed+time.monotonic()-begun,best_speed=best_speed,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.startswith('cuda') else None))
        print(json.dumps({k:v for k,v in row.items() if k!='val_metrics'}),flush=True)
        if stop_after_epoch and epoch+1>=stop_after_epoch:return 75
        if not smoke and (out/'YIELD_REQUEST.json').exists():
            atomic_json(out/'YIELDED.json',dict(timestamp=now(),checkpoint_epoch=epoch+1));return 75
    last=predict(model,val,corr) if corr else None
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    pred=predict(model,val,corr);diagnostics={}
    if corr:
        metrics=val.metrics(pred);trpred=predict(model,train,True)
        diagnostics=dict(raw_val=val.metrics(val.nodes[...,1:3]),raw_train=train.metrics(train.nodes[...,1:3]),
            pure_train=train.metrics(trpred),last_val=val.metrics(last),linear=linear_baselines(train,val))
        model.load_state_dict(torch.load(out/'best_speed.pt',map_location=device,weights_only=False)['model'])
        diagnostics['speed_selected_val']=val.metrics(predict(model,val,True))
        for variant in ['E1','E3']:
            old=ASSETS/'previous_correctors'/f'{variant}_s{spec["site"]:02d}_r{spec["seed"]}.pt'
            if old.exists():
                model.load_state_dict(torch.load(old,map_location=device,weights_only=False))
                if variant=='E3':
                    vx=val.x.clone();tx=train.x.clone();val.x[:,196:286]=0;train.x[:,196:286]=0
                diagnostics[variant+'_val']=val.metrics(predict(model,val,True));diagnostics[variant+'_train']=train.metrics(predict(model,train,True))
                if variant=='E3':val.x=vx;train.x=tx
        err=val.hourly_uv(pred)-val.hourly_uv(val.truth)
        speederr=np.linalg.norm(val.hourly_uv(pred),axis=-1)-np.linalg.norm(val.hourly_uv(val.truth),axis=-1)
        days=(val.origin+28800)//86400;unique=np.unique(days)
        np.savez_compressed(out/'daily.npz',day=unique,n=np.array([(days==d).sum() for d in unique]),uv_sse=np.array([(err[days==d]**2).sum((0,2)) for d in unique]),speed_sse=np.array([(speederr[days==d]**2).sum(0) for d in unique]))
    else:
        metrics=power_metrics(pred,val.y_numpy);h=val.h_numpy[:,-1];scenes={}
        for name,mask in [('low',h<.1),('midlow',(h>=.1)&(h<.3)),('midhigh',(h>=.3)&(h<.7)),('high',h>=.7)]:
            if mask.any():scenes[name]=power_metrics(pred[mask],val.y_numpy[mask])
        diagnostics['current_power']=scenes
        e=pred.astype('f8')-val.y_numpy;days=(val.origin+28800)//86400;unique=np.unique(days)
        power=np.array([np.stack([np.full(24,(days==d).sum()),abs(e[days==d]).sum(0),(e[days==d]**2).sum(0)],-1) for d in unique])
        np.savez_compressed(out/'daily.npz',day=unique,power=power)
    result=dict(spec=spec,finished=now(),epochs=len(history),best_epoch=best_epoch,seconds=elapsed+time.monotonic()-begun,
        metrics=metrics,diagnostics=diagnostics,n_train=train.n,n_val=val.n,evaluation_year=2023,smoke_only=smoke,
        checkpoint_sha256=sha(out/'best.pt'),trainable_parameters=sum(p.numel() for p in model.parameters()),labels_in_forward=False)
    atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),passed=True))
    (out/'latest.pt').unlink(missing_ok=True);(out/'FAILURE.json').unlink(missing_ok=True);return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job-id',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda:0');ap.add_argument('--stop-after-epoch',type=int);args=ap.parse_args()
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==args.job_id)
    try:rc=run(spec,args.smoke,args.device,args.stop_after_epoch)
    except torch.cuda.OutOfMemoryError:
        out=(RUN/'smoke_jobs' if args.smoke else JOBS)/spec['id'];atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='cuda_oom'));rc=75
    except BaseException:
        out=(RUN/'smoke_jobs' if args.smoke else JOBS)/spec['id'];atomic_json(out/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
    raise SystemExit(rc)
