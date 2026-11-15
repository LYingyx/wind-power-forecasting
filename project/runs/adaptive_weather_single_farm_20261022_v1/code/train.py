"""Train on 2022, select epochs on 2023. No test loading in this worker."""
from config import *
from models import Model
from data import Data,score
import argparse,time,traceback

def save_torch(path,obj):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def save_array(path,array):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with open(tmp,'wb') as f:np.save(f,array)
 os.replace(tmp,path)
@torch.no_grad()
def predict(model,data):
 model.eval();out=[]
 for st in range(0,data.n,PROTOCOL['eval_batch_size']):out.append(model(*data.batch(slice(st,st+PROTOCOL['eval_batch_size']))).float())
 out=torch.cat(out);assert torch.isfinite(out).all();return out

def run(spec,smoke=False,stop_after_epoch=None):
 configure();seed_all(spec['seed']);device='cuda:0' if torch.cuda.is_available() else 'cpu'
 if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 out=RUN/('smoke_jobs' if smoke else 'jobs')/spec['id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'TRAIN_SUCCESS.json').exists():return
 if (out/'spec.json').exists():assert json.loads((out/'spec.json').read_text())==spec
 atomic_json(out/'spec.json',spec);model=Model(spec).to(device)
 train=Data('train',spec['site'],spec['k'],device,limit=512 if smoke else None)
 val=Data('val',spec['site'],spec['k'],device,limit=512 if smoke else None)
 if spec['front']=='H1':
  train.load_base(spec['seed'],device);val.load_base(spec['seed'],device)
 groups=model.groups();pars=[p for g in groups for p in g['params']]
 opt=torch.optim.AdamW(groups,weight_decay=1e-4,fused=device.startswith('cuda'))
 epochs=2 if smoke else 80;scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
 history=[];best=float('inf');best_epoch=0;stale=0;elapsed_prior=0.;start_epoch=0
 if (out/'latest.pt').exists():
  checkpoint=torch.load(out/'latest.pt',map_location=device,weights_only=False);assert checkpoint['spec']==spec
  model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
  history=checkpoint['history'];best=checkpoint['best'];best_epoch=checkpoint['best_epoch'];stale=checkpoint['stale'];elapsed_prior=checkpoint['elapsed'];start_epoch=checkpoint['epoch']+1
  torch.set_rng_state(checkpoint['rng_cpu'].cpu())
  if device.startswith('cuda'):torch.cuda.set_rng_state(checkpoint['rng_cuda'].cpu())
 started=time.perf_counter()
 for epoch in range(start_epoch,epochs):
  if stale>=10:break
  ts=time.perf_counter();model.train();total=torch.zeros((),device=device,dtype=torch.float64)
  order=torch.tensor(np.random.default_rng(spec['seed']*1000003+epoch).permutation(train.n),device=device)
  for st in range(0,train.n,256):
   ix=order[st:st+256];opt.zero_grad(set_to_none=True);loss=(model(*train.batch(ix))-train.y[ix]).abs().mean()
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);opt.step();total+=loss.detach().double()*len(ix)
  pv=predict(model,val);vm=float((pv.double()-val.y.double()).abs().mean());scheduler.step()
  if vm<best:
   best=vm;best_epoch=epoch+1;stale=0
   save_torch(out/'best.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},spec=spec,epoch=best_epoch,val_mae=best))
  else:stale+=1
  row=dict(epoch=epoch+1,train_mae=float(total/train.n),val_mae=vm,best_val_mae=best,best_epoch=best_epoch,stale=stale,seconds=time.perf_counter()-ts,lrs=scheduler.get_last_lr(),timestamp=now())
  history.append(row);atomic_json(out/'history.json',history);print(json.dumps(row),flush=True)
  elapsed=elapsed_prior+time.perf_counter()-started
  save_torch(out/'latest.pt',dict(model=model.state_dict(),spec=spec,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,best_epoch=best_epoch,stale=stale,epoch=epoch,elapsed=elapsed,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None))
  atomic_json(out/'progress.json',dict(**row,pid=os.getpid(),elapsed_seconds=elapsed))
  if stop_after_epoch is not None and epoch+1>=stop_after_epoch:
   assert smoke,'Controlled interruption only permitted for smoke checks'
   return
 model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
 pred=predict(model,val).cpu().numpy();save_array(out/'val_prediction.npy',pred)
 result=dict(**spec,station_id=str(np.load(CACHE/'static/station_id.npy')[spec['site']]),best_epoch=best_epoch,epochs_run=len(history),validation_selection_mae=best,
  total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in pars),active_gradient_parameters=sum(p.numel() for p in pars if p.grad is not None),
  front_parameters=sum(p.numel() for p in model.frontend.parameters()),front_gate_tanh=float(model.frontend.gate.detach().tanh()) if model.frontend.enabled else None,seconds_training=elapsed_prior+time.perf_counter()-started,precision='float32_no_tf32',completed_at=now(),
  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.startswith('cuda') else 0.,gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else 'cpu',smoke_only=smoke,
  val=score(pred,val.y.cpu().numpy(),val.h.cpu().numpy(),spec['site'],'val'),checkpoint_sha256=sha(out/'best.pt'))
 atomic_json(out/'metrics.json',result);atomic_json(out/'TRAIN_SUCCESS.json',dict(timestamp=now(),checkpoint_sha256=result['checkpoint_sha256'],validation_only=True))
 (out/'latest.pt').unlink(missing_ok=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--stop-after-epoch',type=int);a=ap.parse_args();s=json.loads(Path(a.spec).read_text())
 try:run(s,a.smoke,a.stop_after_epoch)
 except BaseException:
  p=RUN/('smoke_jobs' if a.smoke else 'jobs')/s['id'];atomic_json(p/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
