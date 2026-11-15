"""Verify label coverage, source identity, interpolation and diagnostic exports."""
import hashlib,json
import numpy as np
from prepare_era5 import OUT,BUNDLE,PROJECT,RUN,atomic,stamp,load

def main():
    raw=np.load(OUT/'era5_crop.npy',mmap_mode='r');times=np.load(OUT/'valid_unix.npy')
    assert len(raw)==len(times)==16700 and raw.shape[1:]==(14,24,25)
    for name in ['surface','pressure']:assert np.load(OUT/f'{name}_done.npy').all()
    assert all(np.isfinite(raw[s:s+128]).all() for s in range(0,len(raw),128))
    years=times.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    assert set(years)=={2022,2023}
    ids=load('bilinear_patch_index');w=load('bilinear_weight');mu=load('nwp_mean');sd=load('nwp_std')
    checked=[]
    for split in ['train','val']:
        for part in ['future','past']:
            idx=np.load(OUT/f'{split}_{part}_index.npy');a=np.load(OUT/'labels'/split/f'era5_{part}.npy',mmap_mode='r')
            se=np.load(OUT/'labels'/split/f'era5_{part}_weather_sel.npy',mmap_mode='r')
            assert a.shape==(len(idx),56,idx.shape[1],14) and se.shape==(*a.shape[:-1],15)
            for i in np.unique(np.linspace(0,len(a)-1,12).astype(int)):
                x=np.asarray(raw[idx[i]])
                expected=((x.reshape(len(x),14,-1)[...,ids]*w).sum(-1).transpose(2,0,1)-mu)/sd
                np.testing.assert_array_equal(a[i],expected)
            assert all(np.isfinite(a[i:i+128]).all() and np.isfinite(se[i:i+128]).all() for i in range(0,len(a),128))
            checked.append(dict(split=split,part=part,shape=list(a.shape)))
    d=json.loads((OUT/'D2_SUMMARY.json').read_text());assert d['passed'] and not d['test_era5_read']
    rows=sum(1 for _ in (OUT/'D2_per_farm.csv').open())-1;assert rows==56*6*8
    records=[]
    for model,wrong in [('LOCAL_REGION_RAW','WRONGDAY_RAW'),('LOCAL_REGION_EQ','WRONGDAY_EQ')]:
        at=lambda m,h:next(x for x in d['results'] if x['model']==m and x['horizon']==h)
        a,b,c=at(model,'1-6'),at(model,'4-6'),at(wrong,'1-6')
        passed=(a['mean_incremental_MSE_reduction_percent']>=1 and a['positive_farms']>=29 and
                a['ci95_week_bootstrap'][0]>0 and b['mean_incremental_MSE_reduction_percent']>0 and
                a['mean_incremental_MSE_reduction_percent']>c['mean_incremental_MSE_reduction_percent'])
        records.append(dict(model=model,passes_small_pilot_budget_gate=passed))
    atomic(OUT/'D2_PILOT_GATE.json',dict(timestamp=stamp(),rule='>=1% incremental weather MSE reduction vs MOS over1-6h; >=29 positive farms; synchronized week CI lower>0; positive4-6h; beats wrongday',
        interpretation='budget heuristic for a small D3/E pilot, not proof of power gains or nonlinear impossibility',results=records,
        recommendation='small_oracle_and_correction_pilot' if any(r['passes_small_pilot_budget_gate'] for r in records) else 'do_not_launch_large_training',automatic_power_training=False))
    comparison=json.loads((OUT/'PANGU_ZERO_HOUR_COMPARISON.json').read_text()) if (OUT/'PANGU_ZERO_HOUR_COMPARISON.json').exists() else None
    assert comparison and len(comparison['samples'])==3
    assert all(max(x['max_absolute_difference'])==0 for x in comparison['samples'])
    atomic(OUT/'COMPLETENESS_AUDIT.json',dict(timestamp=stamp(),passed=True,unique_hours=len(times),years=[2022,2023],crop_shape=[24,25],
        labels=checked,diagnostic_rows=rows,normalization_source=str(BUNDLE/'cache/static'),test_era5_read=False,pangu_zero_comparison=comparison,
        pressure_versions=json.loads((OUT/'PRESSURE_VERSION_AUDIT.json').read_text()),provisional_label_masks_exported=True))
    from summarize import main as summarize
    summarize()
    active_path=PROJECT/'ACTIVE_EXPERIMENT.json';active=json.loads(active_path.read_text())
    if active.get('run')==str(RUN):
        active.update(timestamp=stamp(),stage='completed_D1_D2',audit=str(OUT/'COMPLETENESS_AUDIT.json'),report=str(OUT/'REPORT_CN.md'))
        atomic(active_path,active)
if __name__=='__main__':main()
