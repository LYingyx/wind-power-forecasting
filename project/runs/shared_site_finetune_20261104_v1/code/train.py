"""Per-station adaptation of immutable shared checkpoints; no 2024 access."""
from settings import *
from model import SharedModel,configure,seed_all
from site_data import StationData
from train_shared import evaluate,save_torch,YieldJob,requested
import torch,numpy as np,time,socket,argparse,traceback,signal

def load_model(spec,device):
    path=RUN/spec['pretrained'];assert sha(path)==spec['source_sha256']
    source=torch.load(path,map_location='cpu',weights_only=False)
    model=SharedModel(source['spec']).to(device);model.load_state_dict(source['model'])
    if spec['mode']=='P2_HEAD':
        for p in model.parameters():p.requires_grad_(False)
        head=model.head if spec['family']=='W0' else model.original.head
        for p in head.parameters():p.requires_grad_(True)
        model.embedding.weight.requires_grad_(True)
    else:assert spec['mode']=='P1_FULL'
    # Only the target row is personalized, and AdamW must not decay other rows.
    mask=torch.zeros_like(model.embedding.weight);mask[spec['station']]=1
    model.embedding.weight.register_hook(lambda grad:grad*mask)
    params=[p for n,p in model.named_parameters() if p.requires_grad and n!='embedding.weight']
    optimizer=torch.optim.AdamW([dict(params=params,weight_decay=spec['weight_decay']),dict(params=[model.embedding.weight],weight_decay=0.)],lr=spec['finetune_lr'])
    return model,optimizer,source['spec']

def run(spec,smoke=False,device='cuda:0',max_batches=None):
    configure();out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return 0
    atomic_json(out/'spec.json',spec)
    model,opt,shared_spec=load_model(spec,device)
    train=StationData('train',shared_spec,spec['station'],smoke);valid=StationData('val',shared_spec,spec['station'],smoke)
    seed_all(spec['seed']*1000+spec['station'])
    epoch=0;cursor=0;history=[];best=float('inf');bad=0;loss_sum=0.;seen=0;active_seconds=0.;micro=512
    ap=out/'adaptive_microbatch.json'
    if ap.exists():micro=int(json.loads(ap.read_text())['microbatch'])
    baseline=None;epoch_started=time.monotonic();epoch_elapsed_prior=0.
    def checkpoint():
        return dict(spec=spec,shared_spec=shared_spec,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,cursor=cursor,history=history,best=best,bad=bad,loss_sum=loss_sum,seen=seen,active_seconds=active_seconds,baseline=baseline,
            epoch_elapsed=epoch_elapsed_prior+time.monotonic()-epoch_started,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
    def save_best(at,metrics):
        save_torch(out/'best.pt',dict(model=model.state_dict(),spec=spec,shared_spec=shared_spec,epoch=at,validation=metrics,source_sha256=spec['source_sha256'],normalization_sha256=sha(ASSETS/'normalization.json'),cohort_sha256=sha(ASSETS/'COHORT.json')))
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert ck['spec']==spec
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        epoch=ck['epoch'];cursor=ck['cursor'];history=ck['history'];best=ck['best'];bad=ck['bad'];loss_sum=ck['loss_sum'];seen=ck['seen'];active_seconds=ck['active_seconds'];epoch_elapsed_prior=ck['epoch_elapsed'];baseline=ck['baseline']
        torch.set_rng_state(ck['rng'].cpu())
        if ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    else:
        baseline=evaluate(model,valid,1024,device)
        if not smoke:
            source=json.loads((RUN/'pretrained'/f"{spec['family']}_r{spec['seed']}.metrics.json").read_text())['metrics']
            for key in ['mae','rmse']:assert np.isclose(baseline[key],source['per_farm_'+key][spec['station']],rtol=2e-5,atol=2e-7),(spec['id'],key,baseline[key],source['per_farm_'+key][spec['station']])
        best=baseline['mae'];save_best(0,baseline);atomic_json(out/'epoch0.json',baseline);epoch_started=time.monotonic();save_torch(out/'latest.pt',checkpoint())
    epochs=2 if smoke else spec['epochs'];start=time.monotonic();chunk_start=time.monotonic();chunk_steps=0;steps_done=0;last_save=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:atomic_json(out/'YIELD_REQUEST.json',dict(reason='signal')))
    try:
        while epoch<epochs and bad<spec['patience']:
            model.train();order=np.random.default_rng(spec['seed']*100000+spec['station']*100+epoch).permutation(train.n)
            while cursor<train.n:
                requested(out);end=min(cursor+512,train.n);ids=order[cursor:end];opt.zero_grad(set_to_none=True);step_loss=0.
                for j in range(0,len(ids),micro):
                    ib=ids[j:j+micro];b,y,_,_=train.batch(ib,device);pred=model(b);loss=(pred-y).abs().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                    (loss*len(ib)/len(ids)).backward();step_loss+=float(loss.detach())*len(ib)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();loss_sum+=step_loss;seen+=len(ids);cursor=end;steps_done+=1;chunk_steps+=1
                if chunk_steps>=10 or cursor==train.n:
                    elapsed=time.monotonic()-chunk_start;active_seconds+=elapsed
                    atomic_json(out/'progress.json',dict(timestamp=now(),pid=os.getpid(),host=socket.gethostname(),epoch=epoch+1,step=(cursor+511)//512,total_steps=(train.n+511)//512,completed_steps=steps_done,chunk_steps=chunk_steps,seconds_per_step=elapsed/chunk_steps,loss=loss_sum/max(seen,1),best_mae=best,microbatch=micro,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,active_seconds=active_seconds))
                    chunk_start=time.monotonic();chunk_steps=0
                if time.monotonic()-last_save>120:save_torch(out/'latest.pt',checkpoint());last_save=time.monotonic()
                if max_batches and steps_done>=max_batches:raise YieldJob()
            vm=evaluate(model,valid,max(micro,2*micro),device,out)
            if vm['mae']<best:best=vm['mae'];bad=0;save_best(epoch+1,vm)
            else:bad+=1
            row=dict(timestamp=now(),epoch=epoch+1,seconds=epoch_elapsed_prior+time.monotonic()-epoch_started,train_mae=loss_sum/max(seen,1),validation_mae=vm['mae'],validation_rmse=vm['rmse'],best_mae=best)
            history.append(row);atomic_json(out/'history.json',history);print(json.dumps(row),flush=True)
            epoch+=1;cursor=0;loss_sum=0.;seen=0;epoch_elapsed_prior=0.;epoch_started=time.monotonic();save_torch(out/'latest.pt',checkpoint());last_save=time.monotonic();chunk_start=time.monotonic();chunk_steps=0
        ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);final=evaluate(model,valid,max(micro,2*micro),device,out)
        result=dict(timestamp=now(),requested_spec=spec,spec=spec,shared_spec=shared_spec,smoke_only=smoke,evaluation_year=2023,station=spec['station'],station_id=spec['station_id'],validation_selection='per-station 2023 MAE including unmodified epoch0',checkpoint_epoch=ck['epoch'],checkpoint_sha256=sha(out/'best.pt'),source_sha256=spec['source_sha256'],elapsed_this_session_seconds=time.monotonic()-start,parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad)-(model.embedding.weight.numel()-model.embedding.embedding_dim),trained_epochs=epoch,metrics=final,baseline_metrics=baseline)
        atomic_json(out/'metrics.json',result);atomic_json(out/'SUCCESS.json',dict(timestamp=now(),passed=True,smoke_only=smoke));(out/'latest.pt').unlink(missing_ok=True)
    except (YieldJob,torch.cuda.OutOfMemoryError) as exc:
        opt.zero_grad(set_to_none=True)
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            micro//=2
            if micro<32:raise
            atomic_json(ap,dict(microbatch=micro,effective_batch=512))
        else:save_torch(out/'latest.pt',checkpoint())
        atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='cuda_oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'cooperative',epoch=epoch,cursor=cursor));return 75
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--max-batches',type=int);a=p.parse_args();spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==a.job_id)
    try:rc=run(spec,a.smoke,a.device,a.max_batches)
    except Exception:
        out=(RUN/'smoke_jobs' if a.smoke else JOBS)/spec['id'];atomic_json(out/'FAILURE.json',dict(timestamp=now(),error=traceback.format_exc()));raise
    raise SystemExit(rc)
