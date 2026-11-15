from pathlib import Path
import sys,json,copy,shutil,compileall
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import torch,numpy as np
from physical_models import PhysicalModel,frozen
from physical_data import FarmData,ReadoutFarmData
import fit_model


def run():
 frozen.configure();require_protocol();specs=json.loads((RUN/'manifest.json').read_text())[:8]
 assert compileall.compile_dir(RUN/'code',quiet=1) and compileall.compile_dir(RUN/'operations',quiet=1)
 base=FarmData('train',0,'R3_RAW',limit=64);batch_data={};models={};rngs={};fixtures=[]
 for spec in specs:
  d=FarmData('train',0,spec['variant'],stats=copy.deepcopy(base.stats),limit=64)
  m=PhysicalModel(spec,d.x.shape[-1]);rngs[spec['configuration']]=torch.get_rng_state().clone()
  cell=spec['variant'][3:] if spec['variant'].startswith('R3_H') else 'H1W1'
  # Real power and neighbors retained, only the predeclared factor paths change.
  assert torch.equal(d.anchor,base.anchor) and torch.equal(d.neighbor_raw,base.neighbor_raw) and torch.equal(d.region_features,base.region_features)
  if cell[1]=='0':
   assert torch.equal(d.h[0],d.h[-1]) and not torch.equal(d.h,base.h)
   assert torch.equal(d.x[0,:,-7:],d.x[-1,:,-7:])
  else:assert torch.equal(d.h,base.h) and torch.equal(d.x[...,-7:],base.x[...,-7:])
  if cell[3]=='0':
   assert torch.equal(d.weather[0],d.weather[-1]) and torch.equal(d.physical[0],d.physical[-1])
   assert torch.count_nonzero(d.x[...,:15])==0
  else:assert torch.equal(d.weather,base.weather) and torch.equal(d.physical,base.physical)
  models[spec['configuration']]=m;batch_data[spec['configuration']]=d.batch(slice(0,4));m.eval()
  with torch.no_grad():assert torch.allclose(m(batch_data[spec['configuration']]),d.anchor[:4].expand(-1,24),atol=1e-7,rtol=0)
  fixture=copy.deepcopy(m)
  with torch.no_grad():
   head=fixture.backbone.readout if spec['backbone']=='W0' else fixture.backbone.original.head
   last=[x for x in head.modules() if isinstance(x,torch.nn.Linear)][-1];last.weight.fill_(.01);last.bias.fill_(.02)
   expected=fixture(batch_data[spec['configuration']])
  fixtures.append(dict(spec=spec,feature_dim=d.x.shape[-1],weights=fixture.state_dict(),batch=batch_data[spec['configuration']],expected=expected,target=d.y[:4]))
  m.train();(m(batch_data[spec['configuration']])-d.y[:4]).abs().mean().backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 for b,ref in [('W0','W0_H32'),('TF','TF_R3_RAW')]:
  for cell in ['H1W0','H0W1','H0W0']:
   name=b+'_'+cell;assert torch.equal(rngs[ref],rngs[name])
   assert all(torch.equal(v,models[name].state_dict()[k]) for k,v in models[ref].state_dict().items())
 # Reused H1W1 reference predictions retain exactly the old model/data behavior.
 for b,ref in [('W0','W0_H32'),('TF','TF_R3_RAW')]:
  spec=next(s for s in specs if s['configuration']==ref);p=PARENT/'jobs'/spec['id']
  stats=json.loads((p/'normalization.json').read_text());sel=json.loads((p/'metrics.json').read_text())['selections']['mae']
  d=FarmData('val',0,'R3_RAW',stats=copy.deepcopy(stats),limit=64);old=ReadoutFarmData('val',0,'R3_RAW',stats=copy.deepcopy(stats),limit=64)
  assert all(torch.equal(v,old.batch(slice(None))[k]) for k,v in d.batch(slice(None)).items())
  m=models[ref];m.load_state_dict(torch.load(p/sel['checkpoint'],weights_only=False)['model']);m.eval()
  with torch.no_grad():assert torch.equal(m(d.batch(slice(None))),m(old.batch(slice(None))))
 torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')
 # Exact save/resume on a true ablation with identical training controls.
 a=dict(specs[3],id='SMOKE_FULL');b=dict(specs[3],id='SMOKE_RESUME')
 for spec in [a,b]:
  path=RUN/'smoke_jobs'/spec['id']
  if path.exists():shutil.rmtree(path)
 fit_model.run(a,'cpu',True);fit_model.run(b,'cpu',True,stop_after_epoch=1);fit_model.run(b,'cpu',True)
 for metric in ['mae','rmse']:
  results=[]
  for spec in [a,b]:
   path=RUN/'smoke_jobs'/spec['id'];x=json.loads((path/'metrics.json').read_text())['selections'][metric]
   results.append((x['metrics'],torch.load(path/x['checkpoint'],weights_only=False)['model']))
  assert results[0][0]==results[1][0]
  assert all(torch.equal(v,results[1][1][k]) for k,v in results[0][1].items())
 try:FarmData('test',0,'R3_RAW',stats=base.stats,limit=1)
 except RuntimeError:pass
 else:raise AssertionError('Test accessed before freeze')
 from mechanism_report import interaction
 data={}
 for site in range(56):
  for seed in [42,43]:
   for cell,value in [('H1W1',1.),('H1W0',3.),('H0W1',4.),('H0W0',5.)]:
    data['W0',cell,site,seed]=dict(mae=value,rmse=value,horizons={h:dict(mae=value,rmse=value) for h in ['0_1h','1_3h','3_6h']})
 assert interaction(data,'W0',[42,43])['passed']
 del data['W0','H1W1',0,42]
 assert not interaction(data,'W0',[42,43])['complete']
 checks=['all8 cells preserve same architecture/common initialization/RNG','history-derivedSTATE and attentionquery masked together','physicalweatherquery and normalizedSEL masked together','real regionalinputs and currentpower outputanchor retained','reference checkpoint/data equivalence','finite backward and zero initialization','exact CPU interruption/resume','test2024 blocked before freeze','interaction formula and incomplete-pair guard']
 atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,checks=checks,fixture_cases=8));print(json.dumps(checks),flush=True)

if __name__=='__main__':run()
