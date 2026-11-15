"""Replay existing selected checkpoints; never retrain or copy their weights."""
from settings import *
from physical_data import FarmData
from physical_models import PhysicalModel,frozen
from train_physics import predict,metrics
from diagnostics import stratified
import torch,time

def run(spec,device='cuda:0'):
    setup();frozen.configure();out=JOBS/spec['id'];out.mkdir(exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    source=ATTRIBUTION_RUN/'jobs'/spec['source_job'];old=json.loads((source/'metrics.json').read_text())
    selection=old['selections']['mae'];cp=source/selection['checkpoint'];assert sha(cp)==selection['sha256']
    atomic_json(out/'spec.json',spec);stats=json.loads((source/'normalization.json').read_text())
    data=FarmData('val',spec['site'],'N0',device,stats=stats)
    model=PhysicalModel(old,data.x.shape[-1],torch.zeros(114)).to(device)
    model.load_state_dict(torch.load(cp,map_location=device,weights_only=False)['model'])
    t=time.monotonic();prediction=predict(model,data);score=metrics(prediction,data.y.cpu().numpy())
    for name in ['mae','rmse']:assert abs(score[name]-selection['metrics'][name])<2e-6,(spec['id'],name)
    atomic_json(out/'diagnostics.json',dict(selection='mae',year=2023,source_checkpoint=str(cp),source_sha256=sha(cp),rows=stratified(prediction,data)))
    atomic_json(out/'metrics.json',dict(**spec,finished=now(),seconds=time.monotonic()-t,smoke_only=False,source_checkpoint=str(cp),source_sha256=sha(cp),selections={'mae':dict(epoch=selection['epoch'],metrics=score)},retrained=False))
    atomic_json(out/'SUCCESS.json',dict(timestamp=now(),smoke_only=False,retrained=False))
