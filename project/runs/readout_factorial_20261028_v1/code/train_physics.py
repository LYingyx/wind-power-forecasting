"""2022 fitting and November-excluded2023 selection only."""
from settings import *
import torch,numpy as np
import fit_model as fit
save_torch=fit.save_torch
evaluation_yield_check=None

def run(spec,device='cuda:0',smoke=False):
    require_protocol()
    assert spec['task']=='train'
    out=(RUN/'smoke_jobs' if smoke else JOBS)/spec['id']
    if (out/'SUCCESS.json').exists():return
    fit.run(spec,device,smoke)
    result=json.loads((out/'metrics.json').read_text())
    assert result['evaluation_year']==2023
    assert result['validation_mask_sha256']==sha(ASSETS/'val_keep.npy')
    for sel in result['selections'].values():assert sha(out/sel['checkpoint'])==sel['sha256']
    atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=smoke,evaluation_year=2023))
    (out/'FAILURE.json').unlink(missing_ok=True)
