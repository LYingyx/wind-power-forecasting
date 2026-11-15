from pathlib import Path
import json,tempfile,datetime
from efficiency_v3 import EfficiencyGovernor
from gpu_policy import decide,Policy

def main():
    clock=[1000.]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);g=EfficiencyGovernor(root,clock=lambda:clock[0])
        base=lambda:dict(launch=True,yield_count=0,reason='spare_compute_and_memory')
        def data(jid,epoch,cost):
            path=root/jid;path.mkdir(exist_ok=True)
            (path/'history.json').write_text(json.dumps([dict(epoch=e,seconds=cost,
                timestamp=datetime.datetime.fromtimestamp(clock[0]).astimezone().isoformat()) for e in [epoch-1,epoch]]))
        assert not g.control('0',[('a',{})],base())['launch']
        clock[0]+=25;data('a',3,10)
        assert g.control('0',[('a',{})],base())['launch']
        assert not g.control('0',[('a',{}),('b',{})],base())['launch']
        clock[0]+=25;data('a',5,12);data('b',3,12)
        assert g.control('0',[('a',{}),('b',{})],base())['efficiency']['safe_slots']==2
        # A completion must refill the tested second slot on the very next resource poll.
        assert g.control('0',[('b',{})],base())['reason']=='refill_proven_parallel_slot'
        # The refill rule must never override an external-compute or RAM/VRAM veto.
        veto=dict(launch=False,yield_count=0,reason='other_workload_compute_busy')
        assert not g.control('0',[('b',{})],veto)['launch']
        veto=dict(launch=False,yield_count=1,reason='release_our_memory_for_reserve')
        assert g.control('0',[('b',{})],veto)['yield_count']==1
        # Replacing an ID at the same concurrency cannot reset the expansion timer.
        g.control('0',[('b',{}),('c',{})],base());clock[0]+=25;data('b',5,12);data('c',3,12)
        g.control('0',[('b',{}),('c',{})],base());changed=g.cards['0']['changed']
        g.control('0',[('b',{}),('d',{})],base());assert g.cards['0']['changed']==changed
        # Probe third slot; persistent matched-job slowdown must withdraw only one of ours.
        g.control('0',[('b',{}),('d',{}),('e',{})],base());clock[0]+=25
        for jid in ['b','d','e']:data(jid,7,25)
        assert g.control('0',[('b',{}),('d',{}),('e',{})],base())['yield_count']==0
        clock[0]+=10
        for jid in ['b','d','e']:data(jid,8,25)
        result=g.control('0',[('b',{}),('d',{}),('e',{})],base())
        assert result['yield_count']==1 and result['efficiency']['cap']==2
        assert not g.control('0',[('b',{}),('d',{})],base())['launch']
        assert g.control('0',[('b',{})],base())['reason']=='refill_proven_parallel_slot'
    snapshot=dict(total_mib=16384,free_mib=13000,other_pids=[99],process_utilization_known=True,external_sm=90,utilization=95)
    result=decide([snapshot]*3,2,0,1000,2048,Policy(warmup_seconds=20))
    assert result['yield_count']==2 and not result['launch']
    print('Passed immediate refill, same-count replacement, resource veto, expansion calibration, slowdown backoff and foreign-workload protection')

if __name__=='__main__':main()
