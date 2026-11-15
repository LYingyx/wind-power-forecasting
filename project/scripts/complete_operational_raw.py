"""Complete the existing raw plan with measured I/O concurrency, preserving its identity."""
from pathlib import Path
import argparse
import fcntl
import json
import numpy as np
import prepare_operational_pangu as builder


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();out=a.bundle.resolve()
    with (out/'builder.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (out/'SUCCESS.json').exists()
        identity=json.loads((out/'PROTOCOL.json').read_text())
        for name,value in identity['code_hashes'].items():assert builder.digest(builder.PROJECT/name)==value,name
        builder.OLD=Path(identity['old_cache']);builder.PROTOCOL=builder.operational.PROTOCOLS[identity['protocol']]
        issue=np.load(out/'frame_issue_unix.npy');lead=np.load(out/'frame_lead_hours.npy')
        keys=[(int(i),int(l)) for i,l in zip(issue,lead)]
        workers=max(1,min(a.workers,32))
        builder.atomic_json(out/'RAW_COMPLETER.json',dict(timestamp=builder.now(),workers=workers,preprocess_fingerprint=identity['preprocess_fingerprint'],extractor='unchanged prepare_operational_pangu.extract_raw',driver_sha256=builder.digest(__file__)))
        builder.extract_raw(out,Path(identity['root']),keys,workers)
        builder.atomic_json(out/'STATUS.json',dict(timestamp=builder.now(),state='raw_complete_pending_derived_cache',done=len(keys),total=len(keys)))


if __name__=='__main__':main()
