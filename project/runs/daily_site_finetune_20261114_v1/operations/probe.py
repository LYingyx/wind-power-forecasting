import subprocess
def command(args):
    return subprocess.check_output(args,text=True,timeout=15,stderr=subprocess.DEVNULL)


def probe(own_pids=()):
    own=set(own_pids);gpus={}
    for line in command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']).splitlines():
        idx,uuid,name,total,free,util=[s.strip() for s in line.split(',')]
        gpus[idx]=dict(uuid=uuid,name=name,total_mib=int(total),free_mib=int(free),utilization=int(util),other_pids=[],own_pids=[],process_memory_mib={},external_sm=0,process_utilization_known=True)
    uuids={g['uuid']:g for g in gpus.values()}
    for line in command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).splitlines():
        fields=[s.strip() for s in line.split(',')]
        if len(fields)!=3 or fields[0] not in uuids:continue
        uuid,pid,memory=fields;pid=int(pid);g=uuids[uuid]
        g['own_pids' if pid in own else 'other_pids'].append(pid)
        if memory.isdigit():g['process_memory_mib'][str(pid)]=int(memory)
    process_sm={}
    try:
        for line in command(['nvidia-smi','pmon','-c','1','-s','um']).splitlines():
            cols=line.split()
            if not cols or cols[0].startswith('#') or len(cols)<5:continue
            if cols[1].isdigit() and cols[3].isdigit():process_sm[(cols[0],int(cols[1]))]=int(cols[3])
    except (subprocess.SubprocessError,OSError):pass
    for idx,g in gpus.items():
        g['external_sm']=sum(process_sm.get((idx,pid),0) for pid in g['other_pids'])
        g['process_utilization_known']=all((idx,pid) in process_sm for pid in g['other_pids'])
    return gpus


