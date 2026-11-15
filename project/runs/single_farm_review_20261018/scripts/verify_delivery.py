"""Consistency checks for the review, without retraining or broad project tests."""
import ast,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from audit_tables import ROOT,OUT,ART

def main():
    issues=[]
    for f in (OUT/'scripts').glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    for f in OUT.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if '://' in link or link.startswith('#'):continue
            target=(f.parent/link.split('#')[0]).resolve()
            if not target.exists():issues.append(f'Missing link in {f}: {link}')
    audit=json.loads((OUT/'audit_status.json').read_text());assert audit['duplicate_run_split_keys']==0
    pv=json.loads((OUT/'prediction_verification.json').read_text());assert pv['archives']==40 and pv['max_mae_abs_difference']<1e-7
    assert len(list((OUT/'figures').glob('*.pdf')))==8
    for key in ['source56','target20']:
        d=pd.read_csv(OUT/f'tables/{key}_representatives.csv')
        assert d.represented_farms.sum()==(56 if key=='source56' else 20)
    main=pd.read_csv(OUT/'tables/paper_source_main.csv').set_index('model')
    assert abs(main.loc['E2C','test_mae']-.08070298904253345)<1e-10
    assert abs(main.loc['CBH_SL','test_mae']-.08179857879877764)<1e-10
    assert main.loc['E2C','context']=='regional history'
    cond=pd.read_csv(OUT/'tables/condition_actual_inputs.csv').set_index('variant')
    assert cond.loc['CB','increment_abs_max']>0 and cond.loc['CB','summary_abs_max']==0
    target=pd.read_csv(OUT/'tables/target20_paired_contrasts.csv')
    r=target[(target.budget=='30d')&(target.comparison=='E2C_T5-Pretrained_Transformer')].iloc[0]
    assert r.ci_low>0 and r.wins==7
    cases=pd.read_csv(OUT/'tables/forecast_case_selection.csv');assert len(cases)==14
    assert len(pd.read_csv(OUT/'tables/forecast_cases_transformer.csv'))==14*10*24
    source_code=ART/'p0i_attribution_closure_20260915_v1/snapshot/src/wind_forcing/models/p0i_transfer.py'
    out=dict(passed=not issues,issues=issues,checks=['Python syntax','local Markdown links','source counts and key metrics','prediction replication','representative cluster coverage','condition field semantics','target paired comparison direction','case inference shapes','8 exportable PDF figures'],
             condition_model_source=str(source_code.relative_to(ROOT)),condition_model_sha256=hashlib.sha256(source_code.read_bytes()).hexdigest())
    (OUT/'DELIVERY_CHECKS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    assert not issues,issues
    manifest=[]
    for f in sorted(OUT.rglob('*')):
        if not f.is_file() or any(v in f.parts for v in ['cache','__pycache__']) or f.name=='DELIVERY_SHA256.json':continue
        manifest.append(dict(file=str(f.relative_to(OUT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (OUT/'DELIVERY_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2));print('Verified files:',len(manifest))

if __name__=='__main__':main()
