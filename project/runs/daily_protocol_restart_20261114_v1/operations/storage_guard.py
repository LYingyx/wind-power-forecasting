"""Include user quotas: df free space alone does not ensure writable output."""
from pathlib import Path
import os,shutil,subprocess

def storage_headroom(run):
    rows=[];quota_cache={}
    for label,path,reserve in [('metadata',Path(run),512*2**20),('jobs',Path(run)/'jobs',2*2**30),('data',Path(run)/'data',2*2**30)]:
        target=path.resolve();available=shutil.disk_usage(target).free;quota=None
        mount=next((m for m in ['/shared_data_ssd2','/shared_data_ssd3'] if target.is_relative_to(m)),None)
        if mount and shutil.which('lfs'):
            if mount not in quota_cache:
                out=subprocess.check_output(['lfs','quota','-u',str(os.getuid()),mount],text=True,timeout=15)
                line=next(x.split() for x in out.splitlines() if x.startswith(mount+' '))
                used=int(line[1].rstrip('*'));limit=int(line[3].rstrip('*'))
                quota_cache[mount]=max(0,limit-used)*1024 if limit else None
            quota=quota_cache[mount]
            if quota is not None:available=min(available,quota)
        rows.append(dict(label=label,path=str(target),available_mib=available/2**20,reserve_mib=reserve/2**20,quota_headroom_mib=quota/2**20 if quota is not None else None,ok=available>=reserve))
    return dict(ok=all(x['ok'] for x in rows),paths=rows)
