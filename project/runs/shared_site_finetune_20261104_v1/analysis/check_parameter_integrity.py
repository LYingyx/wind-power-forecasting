from pathlib import Path
import sys,json
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'));from settings import atomic_json,now
# Verify actual saved weights, independently of requires_grad flags or runtime smoke checks.
torch.set_num_threads(1);manifest=json.loads((R/'manifest.json').read_text());source={};checked=0;head_checked=0;epoch0_checked=0
for spec in manifest:
 key=spec['family'],spec['seed']
 if key not in source:source[key]=torch.load(R/spec['pretrained'],map_location='cpu',weights_only=False)['model']
 p=R/'jobs'/spec['id'];d=json.loads((p/'metrics.json').read_text());ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False);assert ck['spec']==spec;w=ck['model'];src=source[key];assert set(w)==set(src)
 assert ck['epoch']==d['checkpoint_epoch'];site=spec['station'];mask=torch.ones(55,dtype=torch.bool);mask[site]=False
 assert torch.equal(w['embedding.weight'][mask],src['embedding.weight'][mask]),spec['id']
 if spec['mode']=='P2_HEAD':
  prefix='head.' if spec['family']=='W0' else 'original.head.'
  for name,tensor in w.items():
   if name=='embedding.weight' or name.startswith(prefix):continue
   assert torch.equal(tensor,src[name]),(spec['id'],name)
  head_checked+=1
 if ck['epoch']==0:
  assert all(torch.equal(w[k],src[k]) for k in w),spec['id'];epoch0_checked+=1
 checked+=1
report=dict(timestamp=now(),passed=True,total_models=checked,head_frozen_parameter_checks=head_checked,unchanged_epoch0_models=epoch0_checked,other_station_embeddings_unchanged=checked)
atomic_json(R/'analysis/PARAMETER_INTEGRITY.json',report);print(json.dumps(report))
