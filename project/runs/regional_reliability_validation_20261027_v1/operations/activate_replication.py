"""Materialize only prespecified passing families, reuse paired screening results."""
from pathlib import Path
import sys,json,shutil,subprocess,time,os
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
from validation_design import REFERENCES,TF
EXT=PROJECT/'runs/regional_reliability_10seed_20261027_v1'

def main():
    assert json.loads((RUN/'COMPLETION_AUDIT.json').read_text())['passed']
    rec=json.loads((RUN/'analysis/SELECTION_RECOMMENDATION.json').read_text())
    arms={tuple(x) for x in rec['selected_extension_arms']}
    if not arms:
        atomic_json(RUN/'operations/REPLICATION_STATUS.json',dict(timestamp=now(),state='no_family_passed_predeclared_replication_rule',open_2024=False));return
    for config in {x[0] for x in arms}:arms.update((config,v) for v in REFERENCES)
    if not (EXT/'ACTIVATION.json').exists():
        for folder in ['code','operations','assets','analysis','schedulers','smoke_jobs']:(EXT/folder).mkdir(parents=True,exist_ok=True)
        for folder in ['code','assets']:
            for f in (RUN/folder).iterdir():
                if f.is_file():shutil.copy2(f,EXT/folder/f.name)
        for f in (RUN/'operations').glob('*.py'):shutil.copy2(f,EXT/'operations'/f.name)
        for f in ['HOST_POLICY.json','CHECKS.json','REPORTING_CHECKS.json','compatibility_fixture.pt','README_CN.md']:
            shutil.copy2(RUN/'operations'/f,EXT/'operations'/f)
        shutil.copy2(RUN/'analysis/train_only_bins.json',EXT/'analysis/train_only_bins.json')
        for name in ['README_CN.md','SELECTION_RULES.json']:shutil.copy2(RUN/name,EXT/name)
        specs=[s for s in json.loads((RUN/'candidate_extension_manifest.json').read_text()) if (s['configuration'],s['variant']) in arms]
        for config in sorted({x[0] for x in arms}):
            for v in REFERENCES:
                for seed in range(44,52):
                    for site in range(56):
                        jid=f'{config}_{v}_s{site:02d}_r{seed}';source=TEN_RUN/'jobs'/jid
                        assert (source/'SUCCESS.json').exists()
                        specs.append(dict(id=jid,stage='REFERENCE',task='diagnose_reference',configuration=config,
                            backbone='W0' if config=='W0' else 'TF',variant=v,site=site,seed=seed,source_run=TEN_RUN.name,source_job=jid))
        # References first per site/seed for balanced reporting, not outcome-based ordering.
        specs.sort(key=lambda s:(s['seed'],s['site'],s['configuration'],s['variant']))
        assert len({s['id'] for s in specs})==len(specs)
        reuse=[]
        for s in json.loads((RUN/'manifest.json').read_text()):
            if (s['configuration'],s['variant']) in arms:
                source=JOBS/s['id'];reuse.append(dict(spec=s,path=str(source),metrics_sha256=sha(source/'metrics.json'),diagnostics_sha256=sha(source/'diagnostics.json')))
        atomic_json(EXT/'manifest.json',specs);atomic_json(EXT/'REUSED_SCREEN.json',reuse)
        atomic_json(EXT/'PROTOCOL.json',dict(timestamp=now(),parent=str(RUN),fit_year=2022,selection_year=2023,open_2024=False,
            seeds=list(range(42,52)),new_seeds=list(range(44,52)),arms=sorted(arms),tasks=len(specs),reused=len(reuse),
            scientific_rule_sha256=sha(RUN/'SELECTION_RULES.json'),report='separate seeds42/43,44..51 and combined ten; no independent-year claim'))
        atomic_json(EXT/'ACTIVATION.json',dict(timestamp=now(),manifest_sha256=sha(EXT/'manifest.json'),
            reuse_sha256=sha(EXT/'REUSED_SCREEN.json'),parent_recommendation_sha256=sha(RUN/'analysis/SELECTION_RECOMMENDATION.json'),arms=sorted(arms)))
    if not (EXT/'SOURCE_FREEZE.json').exists():subprocess.run([sys.executable,str(EXT/'operations/freeze_validation.py')],check=True)
    if not (EXT/'operations/DEPLOYED.json').exists():subprocess.run([sys.executable,str(EXT/'operations/deploy_round.py')],check=True)
    if not (EXT/'operations/LAUNCH_REMOTES.json').exists():subprocess.run([sys.executable,str(EXT/'operations/launch_round.py')],check=True)
    atomic_json(RUN/'operations/REPLICATION_STATUS.json',dict(timestamp=now(),state='activated',run=str(EXT),arms=sorted(arms),open_2024=False))

if __name__=='__main__':main()
