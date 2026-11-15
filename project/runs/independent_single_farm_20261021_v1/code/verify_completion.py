"""Audit all expected result artifacts before a run can be marked complete."""
from common import *
import sqlite3,time

def main():
    db=sqlite3.connect(RUN/'queue.sqlite');db.row_factory=sqlite3.Row
    rows=db.execute('SELECT * FROM jobs').fetchall();assert len(rows)==9500
    assert all(r['status']=='done' for r in rows),'Not all training stages are done'
    frozen=json.loads((RUN/'TRAINING_CODE_FROZEN.json').read_text())
    assert all(sha(RUN/'code'/n)==h for n,h in frozen.items()),'Training code changed'
    for name in ['PREFLIGHT','ANALYSIS_PREFLIGHT','ACTUAL_JOB_AUDIT','FORMAL_RESULT_AUDIT']:
        assert json.loads((RUN/f'{name}.json').read_text())['passed'],name
    references={};counts={'jobs':0,'predictions':0,'corrections':0,'checkpoints':0};started=time.time()
    for row in rows:
        jid=row['id'];p=RUN/'jobs'/jid;spec=json.loads(row['spec']);success=json.loads((p/'SUCCESS.json').read_text())
        m=json.loads((p/'metrics.json').read_text());assert not m['smoke_only']
        assert m['precision']==PROTOCOL['precision'] and m['id']==jid
        assert json.loads((p/'spec.json').read_text())==spec
        assert sha(p/'metrics.json')==success['metrics_sha256']
        h=sha(p/'best.pt');assert h==success['checkpoint_sha256']==m['checkpoint_sha256'];counts['checkpoints']+=1
        family=spec['family'];seed=spec['seed'];site=spec['site']
        splits=['val'] if spec['stage']=='tune' else ['val','test']
        if family=='B0':
            splits=['train','val'] if seed in TUNE_SEEDS else ['train','val','test']
            references[jid]=h
        if spec['stage']=='tune' or seed in TUNE_SEEDS:assert 'test' not in m and not (p/'test_prediction.npy').exists()
        for split in splits:
            truth=np.load(RUN/'prepared'/split/f'{site:02d}'/'target.npy',mmap_mode='r');pred=np.load(p/f'{split}_prediction.npy',mmap_mode='r')
            assert pred.shape==truth.shape and pred.dtype==np.float32 and np.isfinite(pred).all(),(jid,split)
            if split!='train':
                mae=float(np.abs(pred.astype(np.float64)-truth).mean());assert abs(mae-m[split]['mae'])<1e-10,(jid,split,mae)
            if family=='E2C_CBH' or family in ARMS:
                ref=RUN/'jobs'/ref_id(site,seed)
                if ref.name not in references:references[ref.name]=sha(ref/'best.pt')
                assert m['reference_sha256']==references[ref.name]
                delta=np.load(p/f'{split}_correction.npy',mmap_mode='r');base=np.load(ref/f'{split}_prediction.npy',mmap_mode='r')
                assert delta.shape==pred.shape and np.allclose(delta,pred-base,rtol=0,atol=1e-7)
                counts['corrections']+=1
            counts['predictions']+=1
        counts['jobs']+=1
        if counts['jobs']%100==0:
            atomic_json(RUN/'COMPLETION_AUDIT_PROGRESS.json',dict(**counts,seconds=time.time()-started));print(counts,flush=True)
    for family in ['E2C_CBH']+FAMILIES:
        selected=json.loads((RUN/'selected'/f'{family}.json').read_text())
        assert len(selected['trial_metrics_sha256'])==48
        for jid,h in selected['trial_metrics_sha256'].items():assert sha(RUN/'jobs'/jid/'metrics.json')==h
    audit=dict(passed=True,**counts,seconds=time.time()-started,training_source_hashes_verified=True,all_shapes_and_mae_recomputed=True,
               tuning_excludes_2024=True,selected_trial_hashes_verified=True,reference_hashes_and_corrections_verified=True)
    atomic_json(RUN/'COMPLETION_AUDIT.json',audit);print(audit,flush=True)

if __name__=='__main__':main()
