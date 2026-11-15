from pathlib import Path
import sys,json
def activate(run):
 sys.path.insert(0,str(run/'code'))
 p=run/'operations/REMOTE_DEPLOYMENT.json'
 return json.loads(p.read_text()) if p.exists() else None
def source_path(run,path):
 p=run/'operations/REMOTE_DEPLOYMENT.json'
 if not p.exists():return Path(path)
 origin=Path(json.loads(p.read_text())['source_project'])
 return run.parents[1]/Path(path).relative_to(origin)
