from pathlib import Path
import sys,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from physical_models import PhysicalModel,frozen
from settings import atomic_json,now
import torch,numpy as np
frozen.configure();assert torch.cuda.is_available();results=[]
for item in torch.load(RUN/'operations/compatibility_fixture.pt',weights_only=False):
 m=PhysicalModel(item['spec'],item['feature_dim']);m.load_state_dict(item['weights']);m=m.cuda();m.eval();batch={k:v.cuda() for k,v in item['batch'].items()}
 with torch.no_grad():p=m(batch).cpu()
 diff=float((p-item['expected']).abs().max());assert diff<1e-4,(item['spec'],diff)
 m.train();opt=torch.optim.AdamW(m.groups(),weight_decay=1e-4,fused=True);opt.zero_grad();loss=(m(batch)-item['target'].cuda()).abs().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
 assert all(torch.isfinite(p).all() for p in m.parameters())
 results.append(dict(variant=item['spec']['variant'],backbone=item['spec']['backbone'],cpu_cuda_max_difference=diff))
atomic_json(RUN/'operations/GPU_COMPATIBILITY.json',dict(timestamp=now(),passed=True,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cases=results))
print('GPU compatibility passed',torch.cuda.get_device_name(),flush=True)
