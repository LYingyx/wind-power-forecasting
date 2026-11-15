from settings import *
import torch,numpy as np,time
import fit_model as fit
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
from diagnostics import stratified,daily_sums
save_torch=fit.save_torch
evaluation_yield_check=None

def evaluate(spec,source,split='test',limit=None):
 freeze=require_test_freeze() if split=='test' else None
 saved=json.loads((source/'metrics.json').read_text());original=json.loads((source/'spec.json').read_text());stats=json.loads((source/'normalization.json').read_text());sel=saved['selections']['mae'];checkpoint=source/sel['checkpoint']
 assert saved['data_fingerprint']==FINGERPRINT and stats['data_fingerprint']==FINGERPRINT
 assert sha(checkpoint)==sel['sha256']
 assert original['site']==spec['site'] and original['seed']==spec['seed']
 assert original['configuration']==spec['configuration']
 if freeze and spec['task']=='evaluate':
  assert freeze['parent_checkpoints'][spec['source_job']]['sha256']==sel['sha256']
 d=FarmData(split,spec['site'],original['variant'],device=spec.get('_device','cuda:0'),stats=stats,seed=spec['seed'],limit=limit)
 m=PhysicalModel(original,d.x.shape[-1]).to(d.y.device);m.load_state_dict(torch.load(checkpoint,map_location=d.y.device,weights_only=False)['model']);p=fit.predict(m,d,before_batch=evaluation_yield_check)
 return dict(year=2024 if split=='test' else 2023,n_origins=d.n,metrics=fit.metrics(p,d.y.cpu().numpy()),diagnostics=stratified(p,d),daily=daily_sums(p,d),source_checkpoint=sel['checkpoint'],source_sha256=sel['sha256'],selected_epoch=sel['epoch'],selection_year=2023,selection_metric='mae',data_fingerprint=FINGERPRINT)

def run(spec,device='cuda:0',smoke=False):
 require_test_freeze()
 setup();frozen.configure();frozen.seed_all(spec['seed']);out=JOBS/spec['id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'SUCCESS.json').exists():
  if spec['task']=='evaluate' or (out/'test.json').exists():return
  (out/'SUCCESS.json').unlink()
 started=time.monotonic()
 if spec['task']=='train':
  if not (out/'FIT_COMPLETE.json').exists():
   fit.run(spec,device,smoke)
   (out/'SUCCESS.json').unlink(missing_ok=True)
   atomic_json(out/'FIT_COMPLETE.json',dict(timestamp=now(),selected_weights_verified=True))
  # Fit artifacts remain immutable while evaluation reads them.
  result=evaluate(dict(spec,_device=device),out)
  atomic_json(out/'test.json',result)
 else:
  atomic_json(out/'spec.json',spec)
  source=RUN/'sources'/spec['source_job']
  if not source.exists() and not (RUN/'operations/REMOTE_DEPLOYMENT.json').exists():source=PARENT/'jobs'/spec['source_job']
  result=evaluate(dict(spec,_device=device),source)
  atomic_json(out/'test.json',result)
  atomic_json(out/'metrics.json',dict(**spec,finished=now(),seconds=time.monotonic()-started,smoke_only=False,evaluation_year=2024,source_sha256=result['source_sha256'],selections={'mae':dict(epoch=result['selected_epoch'],checkpoint=result['source_checkpoint'],sha256=result['source_sha256'],metrics=result['metrics'])}))
  atomic_json(out/'diagnostics.json',dict(selection='2023_mae',year=2024,rows=result['diagnostics']))
 atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=False,test_evaluated=True))
 (out/'FAILURE.json').unlink(missing_ok=True)
