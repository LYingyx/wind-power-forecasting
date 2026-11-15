from settings import *
from data import SharedData
from model import SharedModel,configure,seed_all
import torch,numpy as np,time,socket,argparse,traceback,signal

class YieldJob(Exception):pass
def save_torch(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def requested(out):
    if (out/'YIELD_REQUEST.json').exists():raise YieldJob()
def resolve(spec):
    s=dict(spec)
    if 'config_from' in s:
        p=ASSETS/'selected'/f'{s["config_from"]}.json';sel=json.loads(p.read_text());s['config']=sel['config'];s['selection_sha256']=sha(p)
    return s

@torch.no_grad()
def evaluate(model,data,batch_size,device,out=None,pred_path=None):
    model.eval();nf=data.farms;ab=np.zeros((nf,24));sq=ab.copy();counts=np.zeros(nf);hi_ab=ab.copy();hi_sq=ab.copy();hi_count=np.zeros(nf)
    pred=None
    if pred_path is not None:pred=np.lib.format.open_memmap(pred_path,mode='w+',dtype='f4',shape=(len(data.rows),nf,24))
    for st in range(0,data.n,batch_size):
        if out is not None:requested(out)
        inds=np.arange(st,min(st+batch_size,data.n));b,y,si,_=data.batch(inds,device);p=model(b);err=(p-y).cpu().numpy().astype('f8')
        if not np.isfinite(err).all():raise FloatingPointError('nonfinite predictions')
        np.add.at(ab,si,abs(err));np.add.at(sq,si,err**2);np.add.at(counts,si,1)
        high=b['h'][:,-1].cpu().numpy()>=.75
        np.add.at(hi_ab,si[high],abs(err[high]));np.add.at(hi_sq,si[high],err[high]**2);np.add.at(hi_count,si[high],1)
        if pred is not None:pred.reshape(-1,24)[inds]=p.cpu().numpy()
    if pred is not None:pred.flush()
    lead_mae=ab/counts[:,None];lead_mse=sq/counts[:,None];mae=lead_mae.mean(1);rmse=np.sqrt(lead_mse.mean(1))
    high_rmse=np.sqrt(hi_sq.sum(1)/np.maximum(1,hi_count*24));valid=hi_count>0
    return dict(mae=float(mae.mean()),rmse=float(rmse.mean()),per_farm_mae=mae.tolist(),per_farm_rmse=rmse.tolist(),lead_mae=lead_mae.tolist(),lead_rmse=np.sqrt(lead_mse).tolist(),
        high_power_rmse=float(high_rmse[valid].mean()) if valid.any() else None,high_power_per_farm_rmse=high_rmse.tolist(),high_power_counts=hi_count.astype(int).tolist(),counts=counts.astype(int).tolist(),
        horizons={label:dict(mae=float(lead_mae[:,a:b].mean()),rmse=float(np.sqrt(lead_mse[:,a:b].mean(1)).mean())) for label,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]})

def run(requested_spec,smoke=False,device='cuda:0',max_batches=None):
    configure();spec=resolve(requested_spec);seed_all(spec['seed']);out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if not smoke and (out/'SUCCESS.json').exists():return
    atomic_json(out/'spec.json',spec)
    if spec['family']=='B0' and spec['fold']>=0:
        f=np.load(ASSETS/f'fold{spec["fold"]}.npz');train=SharedData('train',spec,f['train']);valid=SharedData('train',spec,f['valid'])
    else:train=SharedData('train',spec);valid=SharedData('val',spec)
    if smoke:
        train.rows=train.rows[:32];train.n=len(train.rows)*train.farms;valid.rows=valid.rows[:16];valid.n=len(valid.rows)*valid.farms
    model=SharedModel(spec).to(device);opt=torch.optim.AdamW(model.parameters(),lr=spec['config']['lr'],weight_decay=spec['config']['weight_decay'])
    epoch=0;cursor=0;history=[];best=float('inf');bad=0;loss_sum=0.;seen=0;active_seconds=0.;micro=512
    ap=out/'adaptive_microbatch.json'
    if ap.exists():micro=int(json.loads(ap.read_text())['microbatch'])
    epoch_started=time.monotonic();epoch_elapsed_prior=0.
    def checkpoint():
        return dict(spec=spec,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,cursor=cursor,history=history,best=best,bad=bad,loss_sum=loss_sum,seen=seen,active_seconds=active_seconds,
            epoch_elapsed=epoch_elapsed_prior+time.monotonic()-epoch_started,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert ck['spec']==spec;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        epoch=ck['epoch'];cursor=ck['cursor'];history=ck['history'];best=ck['best'];bad=ck['bad'];loss_sum=ck['loss_sum'];seen=ck['seen'];active_seconds=ck['active_seconds'];epoch_elapsed_prior=ck['epoch_elapsed']
        torch.set_rng_state(ck['rng'].cpu())
        if ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    epochs=1 if smoke else spec['epochs'];start=time.monotonic();chunk_start=time.monotonic();chunk_steps=0;steps_done=0;last_save=time.monotonic()
    if not (out/'latest.pt').exists():save_torch(out/'latest.pt',checkpoint())
    signal.signal(signal.SIGTERM,lambda *_:atomic_json(out/'YIELD_REQUEST.json',dict(reason='signal')))
    try:
        while epoch<epochs and bad<spec['patience']:
            model.train();order=np.random.default_rng(spec['seed']*100000+epoch).permutation(train.n)
            while cursor<train.n:
                requested(out);end=min(cursor+512,train.n);ids=order[cursor:end];opt.zero_grad(set_to_none=True);step_loss=0.
                for j in range(0,len(ids),micro):
                    ib=ids[j:j+micro];b,y,_,_=train.batch(ib,device);p=model(b);loss=(p-y).abs().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    (loss*len(ib)/len(ids)).backward();step_loss+=float(loss.detach())*len(ib)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();loss_sum+=step_loss;seen+=len(ids);cursor=end;steps_done+=1;chunk_steps+=1
                if chunk_steps>=25 or cursor==train.n:
                    elapsed=time.monotonic()-chunk_start;active_seconds+=elapsed
                    atomic_json(out/'progress.json',dict(timestamp=now(),pid=os.getpid(),host=socket.gethostname(),epoch=epoch+1,step=cursor//512,total_steps=(train.n+511)//512,completed_steps=steps_done,chunk_steps=chunk_steps,seconds_per_step=elapsed/chunk_steps,loss=loss_sum/max(seen,1),best_mae=None if best==float('inf') else best,microbatch=micro,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,active_seconds=active_seconds))
                    chunk_start=time.monotonic();chunk_steps=0
                if time.monotonic()-last_save>120:save_torch(out/'latest.pt',checkpoint());last_save=time.monotonic()
                if max_batches and steps_done>=max_batches:raise YieldJob()
            vm=evaluate(model,valid,max(micro,2*micro),device,out)
            improved=vm['mae']<best
            if improved:
                best=vm['mae'];bad=0;save_torch(out/'best.pt',dict(model=model.state_dict(),spec=spec,epoch=epoch+1,validation=vm,normalization_sha256=sha(ASSETS/'normalization.json'),cohort_sha256=sha(ASSETS/'COHORT.json')))
            else:bad+=1
            row=dict(timestamp=now(),epoch=epoch+1,seconds=epoch_elapsed_prior+time.monotonic()-epoch_started,train_mae=loss_sum/max(seen,1),validation_mae=vm['mae'],validation_rmse=vm['rmse'],best_mae=best)
            history.append(row);atomic_json(out/'history.json',history);print(json.dumps(row),flush=True)
            epoch+=1;cursor=0;loss_sum=0.;seen=0;epoch_elapsed_prior=0.;epoch_started=time.monotonic();save_torch(out/'latest.pt',checkpoint());last_save=time.monotonic();chunk_start=time.monotonic();chunk_steps=0
        ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
        evaluation=valid;predict_path=None
        if spec['family']=='B0' and not smoke:
            if spec['fold']>=0:evaluation=SharedData('train',spec,np.load(ASSETS/f'fold{spec["fold"]}.npz')['heldout'])
            predict_path=out/'reference_prediction.npy'
        final=evaluate(model,evaluation,max(512,2*micro),device,out,predict_path)
        result=dict(timestamp=now(),requested_spec=requested_spec,spec=spec,smoke_only=smoke,evaluation_year=2022 if spec['family']=='B0' and spec['fold']>=0 else 2023,validation_selection='2022 internal block' if spec['family']=='B0' and spec['fold']>=0 else '2023 macroMAE',checkpoint_epoch=ck['epoch'],checkpoint_sha256=sha(out/'best.pt'),elapsed_this_session_seconds=time.monotonic()-start,parameter_count=sum(x.numel() for x in model.parameters()),metrics=final)
        if predict_path is not None:result['reference_sha256']=sha(predict_path)
        atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),passed=True,smoke_only=smoke));(out/'latest.pt').unlink(missing_ok=True)
    except (YieldJob,torch.cuda.OutOfMemoryError) as exc:
        opt.zero_grad(set_to_none=True)
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            micro//=2
            if micro<32:raise
            atomic_json(ap,dict(microbatch=micro,effective_batch=512))
        else:save_torch(out/'latest.pt',checkpoint())
        # OOM rolls back to the last stable checkpoint, including optimizer/RNG;
        # never persist a possibly partial Adam update.
        atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='cuda_oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'cooperative',epoch=epoch,cursor=cursor));return 75
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--max-batches',type=int);a=p.parse_args()
    specs={s['id']:s for s in json.loads((RUN/'manifest.json').read_text())};spec=specs[a.job_id]
    try:rc=run(spec,a.smoke,a.device,a.max_batches)
    except Exception:
        out=(RUN/'smoke_jobs' if a.smoke else JOBS)/spec['id'];atomic_json(out/'FAILURE.json',dict(timestamp=now(),error=traceback.format_exc()));raise
    raise SystemExit(rc)
