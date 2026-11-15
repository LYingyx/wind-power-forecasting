from pathlib import Path
import os,json,hashlib,datetime
RUN=Path(__file__).resolve().parents[1]
PROJECT=RUN.parents[1]
BUNDLE=(PROJECT/'data/operational_bundle_v1').resolve()
CACHE=BUNDLE/'cache';ASSETS=RUN/'assets';PREPARED=BUNDLE/'physical'
JOBS=RUN/'jobs';SEEDS=list(range(42,52))
FINGERPRINT='1052d8456c9b950e91ad2a15813c6ceffc541fe7ddba7450c5abd7cc892eef80'
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_name(path.name+f'.{os.getpid()}.tmp')
 t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def setup():
 for d in [JOBS,ASSETS,RUN/'analysis',RUN/'schedulers']:d.mkdir(parents=True,exist_ok=True)
def eligible(spec):
 gate=spec.get('gate')
 if not gate:return True
 p=RUN/'analysis/P1_GATE.json'
 return p.exists() and json.loads(p.read_text()).get(gate,False)

def require_protocol():
 p=json.loads((RUN/'PROTOCOL.json').read_text())
 assert p['data_fingerprint']==FINGERPRINT and p['train_year']==2022 and p['selection_year']==2023
 assert p['validation_mask_sha256']==sha(ASSETS/'val_keep.npy')
 return p

PARENT=PROJECT/'runs/final_mechanism_20261030_v1'
def require_test_freeze():
 p=RUN/'TEST_FREEZE.json'
 if not p.exists():raise RuntimeError('Await complete2023 selection freeze')
 f=json.loads(p.read_text())
 assert f['data_fingerprint']==FINGERPRINT and f['parent_completion_passed']
 assert f['comparison_manifest_sha256']==sha(RUN/'manifest.json')
 assert f['source_freeze_sha256']==sha(RUN/'SOURCE_FREEZE.json')
 assert len(json.loads((RUN/'manifest.json').read_text()))==4480
 return f
