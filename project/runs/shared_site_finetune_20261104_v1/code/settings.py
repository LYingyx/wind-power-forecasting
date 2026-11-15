from pathlib import Path
import os,json,hashlib,datetime
RUN=Path(__file__).resolve().parents[1]
PROJECT=RUN.parents[1]
DATA=RUN/'data'; ASSETS=RUN/'assets'; JOBS=RUN/'jobs'
SOURCE=PROJECT/'runs/privileged_wind_age_pilot_20261101_v1/data'
SEEDS=list(range(42,47))
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
