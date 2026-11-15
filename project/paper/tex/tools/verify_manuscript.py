"""Check manuscript claims, dependencies, and final LaTeX log against audited tables."""
from pathlib import Path
import csv, hashlib, json, re, subprocess

P = Path(__file__).resolve().parents[1]
D = P/'data/single_farm'
def read(name): return list(csv.DictReader((D/(name+'.csv')).open()))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,tol=5e-7):
    assert abs(float(a)-b)<tol,(a,b)

checks={}
main={r['model']:r for r in read('paper_source_main')}
for m,v,t in [('CBH_SL',.084140,.081799),('E2C',.082667,.080703),('TRANSFORMER_FC',.087732,.084699)]:
    close(main[m]['val_mae'],v);close(main[m]['test_mae'],t)
checks['primary_metrics']=True
checks['relative_gain_percent']={m:{s:100*(1-float(main[m][s+'_mae'])/float(main['TRANSFORMER_FC'][s+'_mae'])) for s in ['val','test']} for m in ['CBH_SL','E2C']}
assert round(checks['relative_gain_percent']['CBH_SL']['test'],2)==3.42
assert round(checks['relative_gain_percent']['E2C']['test'],2)==4.72
farms=[r for r in read('source_all56_per_farm') if r['split']=='test']
assert len(farms)==56 and len({r['station'] for r in farms})==56
for model,expected in [('CBH_SL',49),('E2C',50)]:
    count=sum(float(r[model])<float(r['TRANSFORMER_FC']) for r in farms)
    assert count==expected
    checks[model+'_wins_vs_transformer']=count
assert sum(float(r['CBH_SL'])<float(r['CBH_IL']) for r in farms)==54
checks['sharing_wins_SL_vs_IL']=54
conds={(r['model'],r['split']):float(r['mean']) for r in read('condition_ablation_fp16')}
close(conds['CB','test']-conds['CNEUTRAL','test'],-.004419)
close(conds['CBH','test']-conds['CB','test'],-.000260)
close(conds['CBH','test']-conds['CH','test'],-.004593)
checks['condition_contrasts']=True
transfer=read('paper_target_learning_curve')
assert [r['budget'] for r in transfer]==['7d','30d','90d','180d','full']
assert [r['budget'] for r in transfer if float(r['E2C_T5'])<float(r['Pretrained_Transformer'])]==['full']
close(float(transfer[1]['E2C_T5'])-float(transfer[1]['Pretrained_Transformer']),.001235)
checks['transfer_boundary_preserved']=True
assert len(read('paper_source_representatives_2024'))==6
cases=read('forecast_case_selection')
assert len(cases)==14 and all(r['fallback_full_year']=='False' for r in cases)
checks['representatives_and_case_count']=True
assert all(main[m]['test_macro_rmse']=='' for m in ['CBH_SL','CBH_IL','CBH_SC','CBH_IC'])
checks['unavailable_metrics_left_missing']=True

active=['main.tex','preamble.tex']+[f'sections/{x}.tex' for x in ['introduction','method','results','discussion','appendix']]
tex='\n'.join((P/p).read_text() for p in active)
labels=re.findall(r'\\label\{([^}]+)\}',tex)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
assert len(labels)==len(set(labels))
assert set(refs)<=set(labels),set(refs)-set(labels)
figs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex)
assert len(figs)==9
assert all((P/'figures'/f).is_file() for f in figs)
assert 'architecture_overview.png' not in tex
assert 'results_single_farm_review' not in (P/'main.tex').read_text()
checks['references_and_9_figures']=True
sources=json.loads((D/'sources.json').read_text())
for name,record in sources.items():
    local=(P/'figures/single_farm'/name) if name.endswith('.pdf') else (D/(name if name.endswith('.json') else name+'.csv'))
    assert sha(local)==record['sha256'],local
checks['data_and_numeric_figure_hashes']=True
im=json.loads((P/'figures/imagegen_provenance/manifest.json').read_text())
assert im['model_parameter']=='gpt-image-2.5'
assert sha(P/'figures/architecture_single_farm_v19.png')==im['files']['framework_v19_final.png']
checks['image_model_parameter_and_final_hash']=True
log=(P/'build/main.log').read_text()
assert 'Output written on build/main.pdf' in log
for bad in ['Overfull','Undefined control sequence','undefined references','LaTeX Error','Missing character']:
    assert bad not in log,bad
assert not re.search(r'Reference .+ undefined',log)
checks['latex_no_overflow_or_unresolved_references']=True
info=subprocess.check_output(['pdfinfo',str(P/'manuscript_cn.pdf')],text=True)
checks['pages']=int(re.search(r'Pages:\s*(\d+)',info)[1])
checks['pdf_sha256']=sha(P/'manuscript_cn.pdf')
assert sha(P/'manuscript_cn.pdf')==sha(P/'manuscript.pdf')==sha(P/'build/main.pdf')
checks['pdf_copies_identical']=True
checks['benign_font_warning']='Fandol fontspec CJK script warning; glyph rendering visually inspected.'
visual=json.loads((P/'VISUAL_REVIEW.json').read_text())
assert visual['status']=='passed'
for n,h in visual['compiled_input_sha256'].items(): assert sha(P/n)==h,n
text_bytes=subprocess.check_output(['pdftotext','-layout',str(P/'manuscript_cn.pdf'),'-'])
assert hashlib.sha256(text_bytes).hexdigest()==visual['pdf_text_sha256']
checks['visual_review']='passed; all pages plus enlarged framework/conclusion'
checks['status']='passed'
(P/'BUILD_CHECK.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
