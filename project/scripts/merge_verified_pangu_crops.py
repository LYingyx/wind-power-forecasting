"""Reuse identical physical crops across protocols, never normalized caches or models."""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wind_forcing.data.operational import source_path
from prepare_operational_pangu import atomic_json, digest, now


def merge(old, out):
    lock = (out / 'builder.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'SUCCESS.json').exists(), 'Only merge into an unfinished raw store'
    protocol = json.loads((out / 'PROTOCOL.json').read_text())
    assert json.loads((old / 'AUDIT.json').read_text())['passed']
    assert (old / 'SUCCESS.json').is_file()
    for name in ['weather_latitude', 'weather_longitude', 'nwp_channel_name']:
        np.testing.assert_array_equal(np.load(old / f'cache/static/{name}.npy'),
                                      np.load(Path(protocol['old_cache']) / f'static/{name}.npy'))
    oi = np.load(old / 'frame_issue_unix.npy')
    ol = np.load(old / 'frame_lead_hours.npy')
    lookup = {(int(t), int(l)): i for i, (t, l) in enumerate(zip(oi, ol))}
    ni = np.load(out / 'frame_issue_unix.npy')
    nl = np.load(out / 'frame_lead_hours.npy')
    raw = np.load(out / 'raw_frames.npy', mmap_mode='r+')
    done = np.load(out / 'raw_done.npy', mmap_mode='r+')
    previous = np.load(old / 'raw_frames.npy', mmap_mode='r')
    assert raw.shape[1:] == previous.shape[1:] == (14, 24, 25)
    with (old / 'source_crop_receipts.csv').open() as f:
        receipts = {int(r['frame']): r for r in csv.DictReader(f)}
    candidates = [int(i) for i in np.flatnonzero(~done) if (int(ni[i]), int(nl[i])) in lookup]
    # Fresh source values independently verify relocated-file identity across years/leads.
    fresh = set(np.asarray(candidates)[np.unique(np.linspace(0, len(candidates)-1, min(18, len(candidates))).astype(int))].tolist()) if candidates else set()
    merged, skipped, verified = [], [], []
    fields = ['frame','issue_unix','lead_hours','path','size','mtime_ns','clean_crop_sha256','q1000_clipped','q925_clipped']
    with (out / 'source_crop_receipts.csv').open('a') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for start in range(0, len(candidates), 256):
            batch = []
            for i in candidates[start:start+256]:
                j = lookup[(int(ni[i]), int(nl[i]))]
                receipt = receipts[j]
                path = source_path(protocol['root'], int(ni[i]), int(nl[i]))
                st = path.stat()
                if (st.st_size, st.st_mtime_ns) != (int(receipt['size']), int(receipt['mtime_ns'])):
                    skipped.append(i)
                    continue
                block = np.asarray(previous[j])
                assert np.isfinite(block).all()
                assert hashlib.sha256(block.tobytes()).hexdigest() == receipt['clean_crop_sha256']
                if i in fresh:
                    global_data = np.load(path, mmap_mode='r')
                    assert global_data.shape == (1, 14, 721, 1440) and global_data.dtype == np.dtype('f4')
                    check = np.array(global_data[0, :, 171:195, 483:508])
                    check[[6, 7]] = np.maximum(check[[6, 7]], 0)
                    np.testing.assert_array_equal(block, check)
                    verified.append(i)
                raw[i] = block
                writer.writerow(dict(receipt, frame=i, path=str(path)))
                batch.append(i)
            raw.flush(); f.flush(); os.fsync(f.fileno())
            done[batch] = True; done.flush(); merged.extend(batch)
    report = dict(timestamp=now(),old_bundle=str(old),new_bundle=str(out),
                  preprocess_fingerprint=protocol['preprocess_fingerprint'],
                  source_receipts_sha256=digest(old/'source_crop_receipts.csv'),
                  reused=len(merged),skipped_source_metadata_mismatch=len(skipped),
                  fresh_source_comparisons=verified,all_reused_crop_hashes_verified=True,
                  normalized_features_or_statistics_reused=False,remaining=int((~done).sum()))
    np.save(out / 'merged_old_frame_indices.npy', np.asarray(merged, dtype='i8'))
    atomic_json(out / 'MERGE.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();merge(a.old,a.output)
