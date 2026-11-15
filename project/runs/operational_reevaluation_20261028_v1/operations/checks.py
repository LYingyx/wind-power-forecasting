from pathlib import Path
import sys,json,copy,time
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from physical_models import PhysicalModel,OriginalPhysicalModel,frozen
from physical_data import FarmData
from train_physics import evaluate
import fit_model,torch,numpy as np

def main():
 frozen.configure();setup();results=[];fixtures=[]
 assert not (RUN/'TEST_FREEZE.json').exists()
 try:FarmData('test',0,'SEL')
 except RuntimeError:results.append(dict(check='test_blocked_before_parent_freeze',passed=True))
 else:raise AssertionError('Test loader opened before freeze')
 specs=json.loads((RUN/'manifest.json').read_text());cases={}
 for s in specs:
  if s['site']==0 and s['seed']==42:cases[s['configuration']]=s
 for name,s in cases.items():
  d=FarmData('train',0,s['variant'],'cpu',limit=64,seed=42);batch=d.batch(slice(0,8));m=PhysicalModel(s,d.x.shape[-1]);m.eval()
  with torch.no_grad():out=m(batch)
  assert out.shape==(8,24) and torch.isfinite(out).all(),name
  if s['backbone']=='NAMED' and s['family'] in ['GRU_H','PATCHTST_H','CNN_W']:
   altered=dict(batch);key='h' if s['family']=='CNN_W' else 'weather';altered[key]=torch.randn_like(batch[key])*3
   with torch.no_grad():assert torch.equal(m(altered),out),name
  if s['backbone']!='NAMED':
   original=OriginalPhysicalModel(s,d.x.shape[-1]);original.load_state_dict(m.state_dict());original.eval()
   with torch.no_grad():assert torch.equal(original(batch),out),name
  fixtures.append(dict(spec=s,feature_dim=d.x.shape[-1],weights={k:v.clone() for k,v in m.state_dict().items()},batch=batch,expected=out,target=d.y[:8]))
  m.train();opt=torch.optim.AdamW(m.groups(),weight_decay=1e-4);opt.zero_grad();loss=(m(batch)-d.y[:8]).abs().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  assert all(torch.isfinite(x).all() for x in m.parameters()),name
  results.append(dict(check='forward_backward_and_protocol',configuration=name,passed=True));print('PASS',name,flush=True)
 torch.save(fixtures,RUN/'operations/compatibility_fixture.pt')
 # Replay an entire completed validation fold, checking both metrics against saved selected weights.
 for config in ['W0_SEL','TF_R3_RAW']:
  s=cases[config];source=PARENT/'jobs'/s['source_job'];old=json.loads((source/'metrics.json').read_text())['selections']['mae']['metrics']
  replay=evaluate(dict(s,_device='cpu'),source,split='val')
  delta={k:abs(old[k]-replay['metrics'][k]) for k in ['mae','rmse']};assert max(delta.values())<2e-6,delta
  results.append(dict(check='full_2023_checkpoint_replay',configuration=config,max_abs_error=delta,passed=True));print('REPLAY',config,delta,flush=True)
 # Exercise actual two-epoch training + optimizer/RNG resume without touching formal jobs or test.
 s=dict(cases['BASELINE_TRANSFORMER_FC'],id='check_named_resume')
 fit_model.run(s,'cpu',True,stop_after_epoch=1);out=RUN/'smoke_jobs'/s['id'];assert (out/'latest.pt').exists() and not (out/'FIT_COMPLETE.json').exists()
 fit_model.run(s,'cpu',True);assert (out/'FIT_COMPLETE.json').exists() and not (out/'SUCCESS.json').exists()
 met=json.loads((out/'metrics.json').read_text());assert met['smoke_only'] and met['epochs']==2
 for sel in met['selections'].values():assert sha(out/sel['checkpoint'])==sel['sha256']
 results.append(dict(check='fit_resume_selected_weights_separate_from_test_completion',passed=True))
 from collect_physics import rows_for,collect
 sample=dict(n=24,mae=.1,rmse=.2,bias=.05,horizons={})
 row=rows_for(s,2023,sample)[0];assert abs(row['mae_mw']-.1*row['capacity_mw'])<1e-12
 status=collect();assert status['completed']==0 and not status['test_year_evaluated']
 results.append(dict(check='per_farm_export_CF_MW_and_no_premature_test_completion',passed=True))
 atomic_json(RUN/'operations/CHECKS.json',dict(timestamp=now(),passed=True,test_2024_evaluated=False,cases=results))
 print('ALL CHECKS PASSED',flush=True)
if __name__=='__main__':main()
