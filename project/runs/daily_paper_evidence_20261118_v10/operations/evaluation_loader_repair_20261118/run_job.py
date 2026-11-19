"""Durable train -> frozen validation audit -> 2024 reevaluation transaction."""
from settings import *
from train import run, evaluate, YieldJob, requested
from data import SharedData
from model import SharedModel, configure, seed_all
from operational import select_issue
import torch,numpy as np,argparse,time,socket,traceback,signal

def evaluate_split(model,spec,split,out,device,bs,smoke=False):
    dest=out/f'metrics_{split}.json'
    if dest.exists() and not smoke:return
    data=SharedData(split,spec)
    assert data.farms==55
    if smoke:data.rows=data.rows[:4];data.n=len(data.rows)*55
    issues=select_issue(np.asarray(data.a['origin'])[data.rows]);days,di=np.unique(issues,return_inverse=True)
    ab=np.zeros((len(days),55,24));sq=np.zeros_like(ab);counts=np.zeros((len(days),55),dtype='int64')
    original=data.batch;cache={};started=time.monotonic();last=[0.]
    def batch(inds,dev='cpu'):
        result=original(inds,dev);cache['si']=result[2];cache['day']=di[np.asarray(inds)//55];cache['y']=result[1]
        elapsed=time.monotonic()-started
        if elapsed-last[0]>20 or last[0]==0:
            atomic_json(out/'evaluation_progress.json',dict(timestamp=now(),pid=os.getpid(),host=socket.gethostname(),split=split,
                processed=int(inds[0]),total=data.n,seconds=elapsed,examples_per_second=int(inds[0])/max(elapsed,.001)))
            last[0]=elapsed
        return result
    def hook(module,inputs,p):
        err=(p-cache['y']).detach().cpu().numpy().astype('float64');ix=(cache['day'],cache['si'])
        np.add.at(ab,ix,abs(err));np.add.at(sq,ix,err**2);np.add.at(counts,ix,1)
    data.batch=batch;handle=model.register_forward_hook(hook)
    try:metrics=evaluate(model,data,bs,device,out=out,diagnostics=True)
    finally:handle.remove()
    if not smoke:
        np.savez_compressed(out/f'issue_stats_{split}.npz',issue=days,abs_sum=ab,sq_sum=sq,counts=counts)
    atomic_json(dest,dict(timestamp=now(),group=spec['group'],seed=spec['seed'],evaluation_year=2024 if split=='test' else 2023,
        metrics=metrics,smoke_only=smoke,seconds=time.monotonic()-started,checkpoint_sha256=cache.get('checkpoint'),
        issue_clusters=len(days),validation_selection='2023 macro-farm MAE; frozen weights',batch_size=bs))

def main(args):
    spec=next(s for s in json.loads((RUN/'manifest.json').read_text()) if s['id']==args.job_id)
    out=RUN/('smoke_jobs' if args.smoke else 'jobs')/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists() and not args.smoke:return 0
    configure();signal.signal(signal.SIGTERM,lambda *_:atomic_json(out/'YIELD_REQUEST.json',dict(reason='signal')))
    try:
        if spec.get('task')!='evaluate':
            rc=run(spec,smoke=args.smoke,device=args.device,max_batches=2 if args.smoke else None)
            if rc==75:return rc
            if spec['stage']=='tune':
                atomic_json(out/'DONE.json',dict(timestamp=now(),passed=True,stage='tune'));return 0
            checkpoint=out/'best.pt';backbone=False;splits=['val','test']
        else:
            checkpoint=RUN/spec['checkpoint'];assert sha(checkpoint)==spec['checkpoint_sha256'];backbone=spec['backbone'];splits=spec['splits']
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);s=dict(ck['spec'])
        assert ck['normalization_sha256']==sha(ASSETS/'normalization.json')
        assert ck['cohort_sha256']==sha(ASSETS/'COHORT.json')
        if backbone:
            from backbone import SharedModel as Backbone
            model=Backbone(s);model.generative=False
        elif 'wind_mean' not in ck['model']:
            from parent_model import SharedModel as Parent
            model=Parent(s)
        else:model=SharedModel(s)
        model.load_state_dict(ck['model'],strict=True);model.to(args.device).eval()
        for p in model.parameters():p.requires_grad_(False)
        # Group can differ from legacy checkpoint spelling; no effect on data/model.
        s.update(group=spec['group'])
        ep=out/'eval_batch.json';bs=json.loads(ep.read_text())['batch'] if ep.exists() else 512
        for split in splits:
            seed_all(spec['seed']);evaluate_split(model,s,split,out,args.device,bs,args.smoke)
            p=out/f'metrics_{split}.json';m=json.loads(p.read_text());m['checkpoint_sha256']=sha(checkpoint);m['requested_spec']=spec
            atomic_json(p,m)
            if split=='val' and not args.smoke:
                for metric in ['mae','rmse']:
                    assert abs(m['metrics'][metric]-ck['validation'][metric])<2e-6,(spec['id'],metric,m['metrics'][metric],ck['validation'][metric])
        atomic_json(out/'DONE.json',dict(timestamp=now(),passed=True,smoke_only=args.smoke,checkpoint_sha256=sha(checkpoint),splits=splits))
        print(json.dumps(dict(id=spec['id'],done=True)),flush=True);return 0
    except torch.cuda.OutOfMemoryError:
        bs=locals().get('bs',512)//2
        if bs<32:raise
        atomic_json(out/'eval_batch.json',dict(batch=bs));atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='eval_oom'));return 75
    except YieldJob:
        atomic_json(out/'YIELDED.json',dict(timestamp=now(),reason='cooperative_evaluation'));return 75
    except Exception:
        atomic_json(out/'FAILURE.json',dict(timestamp=now(),error=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    raise SystemExit(main(p.parse_args()))
