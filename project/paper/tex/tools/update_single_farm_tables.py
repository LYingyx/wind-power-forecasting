"""Render manuscript tables from the audited review, without estimating missing metrics."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

PAPER = Path(__file__).resolve().parents[1]
REVIEW = PAPER.parents[1] / 'runs/single_farm_review_20261018'
OUT = PAPER / 'data/single_farm'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}

def read(name):
    path = REVIEW / 'tables' / (name + '.csv')
    SOURCES[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    shutil.copy2(path, OUT / path.name)
    return list(csv.DictReader(path.open()))

def fmt(value, digits=6):
    return f'{float(value):.{digits}f}' if value != '' else '--'

def line(values):
    return ' & '.join(map(str, values)) + r' \\' + '\n'

def save(name, body):
    (OUT / (name + '.tex')).write_text('% Generated from audited CSV; do not hand-edit.\n' + body)

NAMES = {'GRU_H':'GRU (H)', 'LSTM_H':'LSTM (H)', 'PATCHTST_H':'PatchTST (H)',
         'ITRANSFORMER_H':'iTransformer (H)', 'MODERNTCN_H':'ModernTCN (H)',
         'CNN_W':'CNN (W)', 'VIT_W':'ViT (W)', 'SWIN_W':'Swin (W)',
         'DLINEAR_FC':'DLinear (FC)', 'NBEATSX_FC':'NBEATSx (FC)',
         'TCN_FC':'TCN (FC)', 'TFT_FC':'TFT (FC)', 'TIMEXER_FC':'TimeXer (FC)',
         'TIDE_FC':'TiDE (FC)', 'TRANSFORMER_FC':'Transformer (FC)',
         'CBH_SL':r'\textbf{CBH-SL}', 'CBH_SC':'CBH-SC', 'E0':'E0-CBH (FP32)',
         'E2C':r'\textbf{E2C-CBH (FP32)}'}

rows = read('paper_source_main')
body = ''
for group, ids in [
    ('本站输入：朴素参照与共享命名基线', [r['model'] for r in rows[:16]]),
    ('本站输入：共享训练的条件修正', ['CBH_SL']),
    ('额外区域历史：扩展系统', ['CBH_SC','E0','E2C'])]:
    body += r'\midrule'+'\n'+r'\multicolumn{5}{l}{\textit{'+group+r'}} \\'+'\n'
    for model in ids:
        r=next(r for r in rows if r['model']==model)
        body += line([NAMES.get(model,model), fmt(r['val_mae']),fmt(r['test_mae']),
                      fmt(r['test_macro_rmse']),fmt(r['test_macro_r2'],4)])
save('source_main',body)

rows=read('sharing_2x2')
body=''
for model, train, inputs in [('IL','独立','本站'),('SL','共享','本站'),('IC','独立','区域'),('SC','共享','区域')]:
    body+=line([model,train,inputs]+[fmt(next(r['mean'] for r in rows if r['model']==model and r['split']==s)) for s in ['val','test']])
save('sharing',body)

rows=read('condition_ablation_fp16')
body=''
for model,h,b in [('CBH','有','有'),('CB','无','有'),('CH','有','无'),('CNEUTRAL','无','无')]:
    body+=line([model,h,b]+[fmt(next(r['mean'] for r in rows if r['model']==model and r['split']==s)) for s in ['val','test']])
save('condition',body)

rows=read('weather_ablation_fp32')
body=''
for model,inputs,enc in [('E0','14 天气变量','线性'),('E1','14 天气变量 + 高程','线性'),
    ('E2L','14 天气变量','三层线性'),('E2N','14 天气变量','三层 MLP'),
    ('E2','14 天气变量 + 高程','三层 MLP'),('E2C','15 维柱坐标','三层 MLP'),('E3','15 维柱坐标','分组 MLP')]:
    body+=line(['E2(raw)' if model=='E2' else model,inputs,enc]+[fmt(next(r['mean'] for r in rows if r['model']==model and r['split']==s)) for s in ['val','test']])
save('weather',body)

rows=read('paper_source_representatives_2024')
save('source_representatives',''.join(line([r['station_id'],r['station_name'],fmt(r['capacity_mw'],1)]+[fmt(r[k]) for k in ['TRANSFORMER_FC','CBH_SL','E2C']]) for r in rows))

rows=read('paper_target_learning_curve')
save('transfer',''.join(line([r['budget'],NAMES.get(r['selected_family'],r['selected_family'])]+[fmt(r[k]) for k in ['Scratch_dev_selected','Pretrained_Transformer','T5_r16','E2C_T5']]) for r in rows))

rows=read('source_all56_per_farm')
save('source_all56',''.join(line([r['station']]+[fmt(r[k]) for k in ['TRANSFORMER_FC','CBH_IL','CBH_SL','E0','E2C']]) for r in rows if r['split']=='test'))

rows=read('independent_baselines_pilot_only')
save('pilot',''.join(line([NAMES.get(r['model'],r['model'])]+[fmt(r[k]) for k in ['mae','rmse_macro','r2_macro']]) for r in rows))

for name in ['matched_output_fp16','matched_output_contrasts','condition_actual_inputs',
             'condition_farm_contrasts','sharing_2x2_contrasts','weather_factorial_contrasts',
             'source_e2c_week_bootstrap','equal_week_sensitivity','scenario_e2c_minus_e0',
             'scenario_summary','valid_origin_coverage','paper_target_representatives',
             'target20_paired_contrasts','local_only_cbh_vs_shared_transformer',
             'forecast_case_selection','source56_descriptors','source56_representatives',
             'target20_descriptors','target20_representatives']:
    read(name)

for name in ['farm_selection.json','prediction_verification.json','case_inference.json']:
    source = REVIEW/name
    shutil.copy2(source, OUT/name)
    SOURCES[name] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

figdir=PAPER/'figures/single_farm'
figdir.mkdir(parents=True,exist_ok=True)
for p in sorted((REVIEW/'figures').glob('*.pdf')):
    shutil.copy2(p,figdir/p.name)
    SOURCES[p.name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/'sources.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2)+'\n')
print(f'Wrote manuscript tables and copied eight numeric figures to {PAPER}')
