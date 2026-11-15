"""Exercise complete family-selection logic on synthetic results, never real claims."""
from pathlib import Path
import sys,tempfile,json,shutil
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
import analyze_validation as analysis
from settings import atomic_json,now
from validation_design import GATES,REFERENCES,TF,REMOVE,MATCH,FULL,spec

def main():
    with tempfile.TemporaryDirectory(prefix='regional_reporting_') as tmp:
        root=Path(tmp)/RUN.name;root.mkdir();(root/'analysis').mkdir();(root/'jobs').mkdir();(root/'assets').mkdir()
        shutil.copy2(RUN/'assets/matched_subsets.npz',root/'assets/matched_subsets.npz')
        arms=REFERENCES+GATES+['C2_'+v for v in GATES]+['R3_MEAN_DELTA','C2_R3_MEAN_DELTA']+REMOVE+MATCH+FULL
        factors={v:1 for v in arms};factors.update(N0=1.02,N3=1.015,R3_RAW=1.01,R3_ALLPOOL=1.012,
            G_FREE=1.01,G_LEAD=1.008,G_STATE=1.007,G_STATE_LEAD=1.,R3_MEAN_DELTA=1.006,
            MATCH_INNER=1.02,MATCH_MIDDLE=1.015,MATCH_OUTER=1.005,MATCH_RANDOM=1.016)
        for v in arms:
            if v.startswith('C2_'):factors[v]=1.03
        manifest=[]
        for site in range(56):
            for seed in [42,43]:
                for v in arms:
                    s=spec('W0',v,site,seed);manifest.append(s);out=root/'jobs'/s['id'];out.mkdir()
                    factor=factors[v]*(1+site*.0001+seed*.00001)
                    score=dict(n=240,mae=.1*factor,rmse=.2*factor,bias=.01*factor)
                    metrics=dict(score,horizons={h:dict(score) for h in ['0_1h','1_3h','3_6h']},lead_mae=[score['mae']]*24,lead_rmse=[score['rmse']]*24)
                    atomic_json(out/'metrics.json',dict(selections=dict(mae=dict(epoch=2,metrics=metrics))))
                    atomic_json(out/'diagnostics.json',dict(rows=[dict(group='current_power',bin='high',horizon=h,**score) for h in ['all','0_1h','1_3h','3_6h']]))
                    atomic_json(out/'SUCCESS.json',dict(synthetic=True))
        atomic_json(root/'manifest.json',manifest)
        analysis.RUN=root;analysis.JOBS=root/'jobs';analysis.ASSETS=root/'assets'
        result=analysis.main();selected={tuple(x) for x in result['selected_extension_arms']}
        assert result['gates']['W0/G_STATE_LEAD']['passes']
        assert ('W0','C2_G_FREE') in selected and ('W0','C2_R3_MEAN_DELTA') in selected
        assert all(('W0',v) in selected for v in MATCH+FULL+REMOVE)
        assert result['geometry']['MATCH_OUTER/MATCH_INNER']['effect']['farms']==41
        # A high-power-only regression must fail despite good overall performance.
        for s in manifest:
            if s['variant']!='G_STATE_LEAD':continue
            out=root/'jobs'/s['id'];data=json.loads((out/'diagnostics.json').read_text())
            for row in data['rows']:row['mae']*=1.1;row['rmse']*=1.1
            atomic_json(out/'diagnostics.json',data)
        bad=analysis.main()['gates']['W0/G_STATE_LEAD']
        assert bad['base_dual'] and bad['beats_control'] and not bad['high_protected'] and not bad['passes']
    atomic_json(RUN/'operations/REPORTING_CHECKS.json',dict(timestamp=now(),passed=True,
        checks=['complete paired farm/seed aggregation','matched geometry common support','whole gate/control family replication','simplification own C2','deletion and geometry family activation','high-power regression veto despite overall benefit'],synthetic_only=True))
    print('Reporting and conditional family selection passed',flush=True)

if __name__=='__main__':main()
