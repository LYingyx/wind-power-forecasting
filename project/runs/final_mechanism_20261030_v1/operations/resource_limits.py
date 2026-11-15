"""Host and hierarchical cgroup memory headroom with pending-worker reservations."""
from pathlib import Path
import os,time
RUN=Path(__file__).resolve().parents[1]

def memory_status():
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    available=mem['MemAvailable'];total=mem['MemTotal'];groups=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if not line.startswith('0::'):continue
        start=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
        for folder in [start,*start.parents]:
            if not folder.is_relative_to('/sys/fs/cgroup'):break
            limit=folder/'memory.max';current=folder/'memory.current'
            if not limit.exists() or not current.exists():continue
            raw=limit.read_text().strip()
            if raw=='max':continue
            ceiling=int(raw);used=int(current.read_text());statfile=folder/'memory.stat'
            stats=dict(line.split() for line in statfile.read_text().splitlines()) if statfile.exists() else {}
            reclaim=max(0,int(stats.get('inactive_file',0))-int(stats.get('file_dirty',0))-int(stats.get('file_writeback',0)))
            free=max(0,ceiling-used+reclaim);available=min(available,free);total=min(total,ceiling)
            groups.append(dict(path=str(folder),limit_mib=ceiling/2**20,current_mib=used/2**20,inactive_reclaimable_mib=reclaim/2**20,available_mib=free/2**20))
    # A worker not yet resident must not look free to simultaneous CPU/GPU dispatchers.
    uptime=float(Path('/proc/uptime').read_text().split()[0]);hz=os.sysconf('SC_CLK_TCK');reserved=0;pending=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            if folder.stat().st_uid!=os.getuid():continue
            cmd=(folder/'cmdline').read_bytes()
            if str(RUN.parents[1]/'runs').encode() not in cmd or b'--job-id' not in cmd:continue
            stat=(folder/'stat').read_text().rsplit(')',1)[1].split();age=uptime-int(stat[19])/hz
            if age>=60 or stat[0] in ['Z','X']:continue
            row=next((x for x in (folder/'status').read_text().splitlines() if x.startswith('VmRSS:')),None)
            rss=int(row.split()[1])*1024 if row else 0
            amount=max(0,3*2**30-rss);reserved+=amount;pending.append(dict(pid=int(folder.name),age_seconds=age,reserved_mib=amount/2**20))
        except (OSError,ValueError,IndexError,ProcessLookupError):continue
    return dict(available_mib=max(0,available-reserved)/2**20,total_mib=total/2**20,pending_reserved_mib=reserved/2**20,cgroups=groups,pending_workers=pending)

def effective_memory_available():
    d=memory_status();return d['available_mib'],d['total_mib']
