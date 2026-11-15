from pathlib import Path
import sys,json,tempfile
from unittest.mock import patch
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'));sys.path.insert(0,str(RUN/'operations'))
from settings import *
import train_physics,fit_model,finalize,remote_controller
from reevaluation_report import choose_parent
from physical_models import PhysicalModel,frozen
from physical_data import FarmData
from scheduled_worker import install_yield_hook,ResourceYield
frozen.configure();checks=[]
assert not finalize.activate(), 'Parent must still be incomplete during preparation'
s=json.loads((RUN/'manifest.json').read_text())[0]
try:train_physics.run(s,'cpu')
except RuntimeError as e:assert 'awaits' in str(e)
else:raise AssertionError('Dispatcher bypassed freeze')
assert not list(JOBS.glob('*/SUCCESS.json'))
checks.append(dict(check='automatic_activation_and_dispatcher_wait_for_parent',passed=True))
data={}
for b in ['W0','TF']:
 for i in range(56):
  for seed in SEEDS:
   for name,value,high in [('SEL',1.,1.),('N3',.99,.99),('R3_RAW',.98,.98),('C2_R3_RAW',1.,1.)]:data[b+'_'+name,i,seed]=dict(mae=value,rmse=value,high_rmse=high)
choice=choose_parent(data);assert all(v.endswith('R3_RAW') for v in choice['choices'].values())
for key,value in data.items():
 if key[0].endswith('R3_RAW') and 'C2' not in key[0]:value['high_rmse']=1.2
assert all(v.endswith('N3') for v in choose_parent(data)['choices'].values())
checks.append(dict(check='selection_requires_both_metrics_controls_and_high_power_guard',passed=True))
with tempfile.TemporaryDirectory() as tmp:
 control=Path(tmp)/'YIELD_REQUEST.json';control.write_text('{}');original=install_yield_hook(control)
 d=FarmData('train',0,'SEL',limit=8);m=PhysicalModel(dict(s,variant='SEL',backbone='W0'),d.x.shape[-1])
 try:fit_model.predict(m,d,before_batch=train_physics.evaluation_yield_check)
 except ResourceYield:pass
 else:raise AssertionError('Evaluation did not release resources')
 fit_model.save_torch=original;train_physics.evaluation_yield_check=None
checks.append(dict(check='evaluation_yields_at_batch_boundary_without_corrupting_fit',passed=True))
# Threaded receives release successful job locks even when another transfer fails.
class Claim:
 def __init__(self):self.closed=False
 def close(self):self.closed=True
claims={'ok':Claim(),'bad':Claim()};original=claims['ok']
def fake_receive(host,jid,expected,success):
 if jid=='bad':raise OSError('simulated interrupted transport')
with patch.object(remote_controller,'receive',fake_receive):
 try:remote_controller.receive_many('test',dict(done=['ok','bad'],failed=[]),claims,dict(ok={},bad={}))
 except RuntimeError:pass
assert original.closed and 'ok' not in claims and 'bad' in claims
checks.append(dict(check='partial_parallel_transfer_preserves_recoverable_reservations',passed=True))
p=RUN/'operations/CHECKS.json';result=json.loads(p.read_text());result['cases']+=checks;result['timestamp']=now();atomic_json(p,result)
print('FOLLOW-UP CHECKS PASSED')
