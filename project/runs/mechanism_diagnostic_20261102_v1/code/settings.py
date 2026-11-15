from pathlib import Path
import json,os,hashlib,datetime
RUN=Path(__file__).resolve().parents[1]
PROJECT=RUN.parents[1]
DATA=RUN/'data';CACHE=DATA/'cache';ASSETS=RUN/'assets';JOBS=RUN/'jobs'
SEEDS=[42,43]
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,path)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
