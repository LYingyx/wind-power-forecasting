"""Incremental per-farm, per-seed reporting, no result averaging across fits."""
from config import *
import csv,gzip,sqlite3,fcntl
import pandas as pd

CONTRASTS=[('S3','S2'),('S4','S3'),('S1','W0'),('S2','W0'),('T2_7','T1_7'),('T2_8','T1_8'),('T1_8','T1_7'),('T2_8','T2_7'),
 ('ST00','W0'),('ST10','ST00'),('ST01','ST00'),('ST11','W0'),('ST11','S1'),('ST11','ST00'),('ST11','ST10'),('ST11','ST01'),('ST11','ST_NO_H'),('H1','W0'),('W0','H2'),('W0','CLIM_LAST'),('H2','H2_CLIM')]
for b in ['TRANSFORMER_FC','TIDE_FC']:
 CONTRASTS.extend([(b+'__ST11',b+'__ST00'),(b+'__ST11',b+'__S1'),(b+'__ST11',b+'__W0'),
                   (b+'__S1',b+'__W0'),(b+'__ST00',b+'__W0'),('W0',b+'__W0'),
                   ('ST11',b+'__W0'),('ST11',b+'__S1'),('ST11',b+'__ST00')])

def paired_table(frame,extra_keys=()):
 keys=['site','seed','split','year',*extra_keys];pairs=[]
 for a,b in CONTRASTS:
  x=frame[frame.arm==a].merge(frame[frame.arm==b],on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
  for metric in ['mae','rmse']:
   x[metric+'_gain']=x[metric+'_reference']-x[metric+'_candidate']
   # Missing scenario metrics stay missing; they are not counted as ties.
   x[metric+'_direction']=np.select([x[metric+'_gain'].isna(),x[metric+'_gain']>0,x[metric+'_gain']<0],['unavailable','improved','worsened'],default='tie')
  pairs.append(x)
 return pd.concat(pairs,ignore_index=True)

def finite_mean(x):
 x=np.asarray(x);v=x[np.isfinite(x)];return float(v.mean()) if len(v) else None

def correlation(x,y):
 x,y=np.broadcast_arrays(x,y);ok=np.isfinite(x)&np.isfinite(y);x=x[ok].astype(np.float64);y=y[ok].astype(np.float64)
 if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])

def attention_summary(diag,origin,front,gate):
 """Within-fit descriptive associations only; no aggregation across farms/seeds."""
 assert diag.shape==(len(origin),24,6)
 has_space=front.startswith('S');has_time=front.startswith(('ST','T'))
 month=(origin+28800).astype('datetime64[s]').astype('datetime64[M]').astype(int)%12+1
 subsets={'all':np.ones(len(origin),bool),**{'season_'+k:np.isin(month,v) for k,v in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]}}
 result=[]
 for season,mask in subsets.items():
  if not mask.any():continue
  for horizon,a,b in [('all',0,24),('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
   z=np.asarray(diag[mask,a:b],dtype=np.float64);dx,dy,tm,tp,u,v=np.moveaxis(z,-1,0)
   speed=np.hypot(u,v);radius=np.hypot(dx,dy);lead=np.arange(1,25)[None,a:b]/4
   wind_ok=speed>1e-6;direction_ok=wind_ok&(radius>1e-6)
   along=np.divide(-(dx*u+dy*v),speed,out=np.full_like(speed,np.nan),where=wind_ok)
   cosine=np.divide(along,radius,out=np.full_like(radius,np.nan),where=direction_ok)
   row=dict(season=season,horizon=horizon,n_origins=int(mask.sum()),n_origin_leads=int(speed.size),front_gate_tanh=gate,
            attention_is_candidate_not_final_weight=True,gate_negative=gate<0 if gate is not None else None,
            wind_speed_mean_mps=finite_mean(speed),has_space=has_space,has_time=has_time)
   if has_space:
    row.update(centroid_distance_mean_km=finite_mean(radius),upwind_distance_mean_km=finite_mean(along),upwind_cosine_mean=finite_mean(cosine),
               valid_wind_directions=int(wind_ok.sum()),valid_centroid_directions=int(direction_ok.sum()),
               upwind_positive_fraction=finite_mean((along[wind_ok]>0).astype(float)),centroid_speed_pearson=correlation(radius,speed))
   if has_time:
    row.update(expected_node_offset_mean_hours=finite_mean(tm),expected_minus_target_mean_hours=finite_mean(tm-lead),
               peak_minus_target_mean_hours=finite_mean(tp-lead),expected_target_pearson=correlation(tm,lead),
               time_gap_speed_pearson=correlation(tm-lead,speed))
   result.append(row)
 return result

def attention_rows(folder,m,params_by_id,out):
 path=folder/'attention_diagnostics.npy'
 if not path.exists():path=folder/'attention_diagnostics.npz'
 if not path.exists():return []
 split=m.get('evaluation_split','test');orig=np.load(CACHE/split/'origin_unix.npy',mmap_mode='r')
 try:stat=path.stat()
 except FileNotFoundError:
  path=folder/'attention_diagnostics.npz';stat=path.stat()
 stamp=dict(version=1,size=stat.st_size,mtime_ns=stat.st_mtime_ns,checkpoint_sha256=m['checkpoint_sha256'])
 cache=out/'attention_cache'/f'{folder.name}.json'
 if cache.exists():
  old=json.loads(cache.read_text())
  if old['stamp']==stamp:return old['rows']
 if path.suffix=='.npz':
  with np.load(path) as archive:diag=archive['diagnostics']
 else:
  try:diag=np.load(path,mmap_mode='r')
  except FileNotFoundError:
   # A successful compactor publishes the lossless archive before removing the npy.
   path=folder/'attention_diagnostics.npz'
   with np.load(path) as archive:diag=archive['diagnostics']
   stat=path.stat();stamp.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
 assert len(diag)==len(orig),'Only full official evaluations enter reporting'
 gate=params_by_id.get(m['source_job_id'],{}).get('front_gate_tanh')
 ident={k:m[k] for k in ['arm','site','seed','front','source_job_id']};ident.update(split=split,year=2023 if split=='val' else 2024)
 rows=[{**ident,**r} for r in attention_summary(diag,orig,m['front'],gate)]
 atomic_json(cache,dict(stamp=stamp,rows=rows));return rows

def main():
 out=RUN/'analysis';out.mkdir(exist_ok=True);names=np.load(CACHE/'static/station_name.npy');ids=np.load(CACHE/'static/station_id.npy')
 lock=open(out/'analysis.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 rows=[];lead=[];scenario=[];horizon=[];interventions=[];params=[];attention=[];intervention_slices=[]
 def add(m,arm,source,split,data,kind):
  ident=dict(arm=arm,site=m['site'],station_id=str(ids[m['site']]),station_name=str(names[m['site']]),seed=m['seed'],split=split,year=2023 if split=='val' else 2024,source_job_id=source,row_kind=kind)
  rows.append({**ident,**{k:data.get(k) for k in ['mae','rmse','bias','n_origins','outside_0_1_fraction']},'stage':m.get('stage'),'k':m.get('k'),'front_lr':m.get('front_lr')})
  if m.get('stage')=='tune':return
  for i,(a,b) in enumerate(zip(data['lead_mae'],data['lead_rmse']),1):lead.append({**ident,'lead_step':i,'mae':a,'rmse':b})
  for name,x in data['scenarios'].items():scenario.append({**ident,'scenario':name,**x})
  for name,a,b in [('0_1h',0,4),('1_3h',4,12),('3_6h',12,24)]:
   # Aggregate forecast samples/leads within ONE fit only.
   horizon.append({**ident,'horizon':name,'mae':float(np.mean(data['lead_mae'][a:b])),'rmse':float(np.sqrt(np.mean(np.square(data['lead_rmse'][a:b]))))})
 for success in (RUN/'jobs').glob('*/TRAIN_SUCCESS.json') if (RUN/'jobs').exists() else []:
  path=success.parent/'metrics.json';m=json.loads(path.read_text());add(m,m['arm'],m['id'],'val',m['val'],'trained')
  params.append({k:m.get(k) for k in ['id','arm','site','seed','stage','k','front_lr','total_parameters','trainable_parameters','active_gradient_parameters','front_parameters','front_gate_tanh','seconds_training','peak_reserved_mib','best_epoch','epochs_run','checkpoint_sha256']})
 params_by_id={x['id']:x for x in params}
 for success in (RUN/'evaluation').glob('*/SUCCESS.json') if (RUN/'evaluation').exists() else []:
  m=json.loads((success.parent/'metrics.json').read_text());add(m,m['arm'],m['source_job_id'],'test',m['test'],'legacy_reuse' if m.get('legacy_source') else 'trained')
  attention.extend(attention_rows(success.parent,m,params_by_id,out))
  for label,x in m['interventions'].items():
   if label=='matched':continue
   ident=dict(arm=m['arm'],site=m['site'],station_id=str(ids[m['site']]),seed=m['seed'],split='test',year=2024,variant=label,source_job_id=m['source_job_id'])
   interventions.append(dict(**ident,mae=x['mae'],rmse=x['rmse'],matched_mae=m['test']['mae'],matched_rmse=m['test']['rmse'],mae_increase=x['mae']-m['test']['mae'],rmse_increase=x['rmse']-m['test']['rmse']))
   for kind in ['horizons','scenarios']:
    for name,values in x[kind].items():
     ref=m['test'][kind][name];entry=dict(**ident,slice_kind=kind,slice_name=name,n_origins=values['n_origins'])
     for metric in ['mae','rmse']:
      val,matched=values[metric],ref[metric]
      entry.update({metric:val,'matched_'+metric:matched,metric+'_increase':val-matched if val is not None and matched is not None else None})
     intervention_slices.append(entry)
 # Fixed old comparators. W0 seeds47..51 come only from newly trained extensions.
 for site in range(56):
  for seed in range(42,52):
   for arm,prefix in [('W0','ablation_NEUTRAL_LAST'),('TRANSFORMER_FC__W0','formal_TRANSFORMER_FC'),('TIDE_FC__W0','formal_TIDE_FC')]:
    if arm=='W0' and seed>46:continue
    path=OLD/'jobs'/f'{prefix}_s{site:02d}_r{seed}'/'metrics.json';m=json.loads(path.read_text())
    for split in ['val','test']:
     # W0 test is included once; use old audited values until its diagnostic job completes.
     if arm=='W0' and split=='test' and any(x['arm']=='W0' and x['site']==site and x['seed']==seed and x['split']=='test' for x in rows):continue
     add(m,arm,m['id'],split,m[split],'legacy_comparator')
 def save(data,name):
  df=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
  temp=out/(name+'.tmp');df.to_csv(temp,index=False,float_format='%.17g',encoding='utf-8-sig',compression='gzip' if name.endswith('.gz') else None);os.replace(temp,out/name);return df
 df=save(rows,'results_per_farm_seed.csv');save(params,'actual_configs_and_costs.csv');ld=save(lead,'lead_per_farm_seed.csv.gz');sc=save(scenario,'scenario_per_farm_seed.csv.gz');hz=save(horizon,'horizon_per_farm_seed.csv');save(interventions,'intervention_results_per_seed.csv')
 save(attention,'attention_associations_per_seed.csv');save(intervention_slices,'intervention_slices_per_seed.csv.gz')
 main=df[df.stage!='tune'];assert not main.duplicated(['arm','site','seed','split']).any()
 paired=save(paired_table(main),'paired_actual_results.csv');counts=[]
 paired_hz=save(paired_table(hz,['horizon']),'paired_horizons_per_seed.csv')
 save(paired_table(sc,['scenario']),'paired_scenarios_per_seed.csv.gz')
 save(paired_table(ld,['lead_step']),'paired_leads_per_seed.csv.gz')
 failures=save(paired.loc[(paired.mae_gain<0)|(paired.rmse_gain<0)],'regressions_per_farm_seed.csv')
 for key,g in paired.groupby(['arm_candidate','arm_reference','year','seed']):
  for metric in ['mae','rmse']:
   delta=g[metric+'_gain'];counts.append(dict(candidate=key[0],reference=key[1],year=key[2],seed=key[3],metric=metric,n_pairs=len(g),improved=int((delta>0).sum()),worsened=int((delta<0).sum()),ties=int((delta==0).sum())))
 countdf=save(counts,'improvement_counts_by_seed.csv')
 interaction=[]
 for (site,seed,split),g in main.groupby(['site','seed','split']):
  lookup=g.set_index('arm')
  if all(k in lookup.index for k in ['W0','H2','CLIM_LAST','H2_CLIM']):
   for metric in ['mae','rmse']:
    real=lookup.loc['CLIM_LAST',metric]-lookup.loc['W0',metric];fixed=lookup.loc['H2_CLIM',metric]-lookup.loc['H2',metric]
    interaction.append(dict(site=site,seed=seed,split=split,metric=metric,weather_gain_real_history=real,weather_gain_fixed_history=fixed,interaction=real-fixed))
 save(interaction,'history_weather_interaction_per_seed.csv')
 status=json.loads((RUN/'STATUS.json').read_text()) if (RUN/'STATUS.json').exists() else {}
 report='# 自适应天气提取：实时逐风场逐seed结果\n\n更新时间：'+now()+'。全时效MAE为预先指定主指标；不设置改善幅度门槛；不跨seed或风场平均。\n\n'
 report+='[逐次实测值](results_per_farm_seed.csv) · [配对原值](paired_actual_results.csv) · [逐seed方向次数](improvement_counts_by_seed.csv) · [实际配置与成本](actual_configs_and_costs.csv) · [干预](intervention_results_per_seed.csv) · [历史×天气交互](history_weather_interaction_per_seed.csv)\n\n'
 report+='[分时效配对](paired_horizons_per_seed.csv) · [逐lead配对](paired_leads_per_seed.csv.gz) · [分场景配对](paired_scenarios_per_seed.csv.gz) · [逐站逐seed退步清单](regressions_per_farm_seed.csv) · [干预分时效/场景](intervention_slices_per_seed.csv.gz) · [注意力物理关联](attention_associations_per_seed.csv) · [贡献证据台账](CONTRIBUTION_EVIDENCE_CN.md)\n\n'
 report+='当前队列：`'+json.dumps(status.get('by_kind',[]),ensure_ascii=False)+'`。未完成项没有结果，不补值。旧W0和原Transformer/TiDE明确标为复用。2024仍为跨年复评。\n\n'
 report+='ST00/10/01/11分别表示空间/时间的静态或自适应；ST_NO_H仅冻结权重生成中的历史，保留解码器真实历史；H2保留网络层，以训练期历史替代真实历史，末值锚点仍使用当前功率。完整结构与实验协议见[运行入口](../README_CN.md)。\n'
 report+='\n所有gain列为参考误差减候选误差，正值表示改善；干预increase列为干预误差减匹配误差。缺失场景写unavailable，不计为持平。退步清单收录MAE或RMSE任一变差的配对，保留另一指标，避免隐藏指标间取舍。\n'
 report+='\n注意力关联只在同一模型的起报样本/lead内统计，分别列全年和各季节，不跨风场/seed平均。upwind为天气风矢量反方向；时间差为所读取节点时刻减目标时刻。注意力对应candidate分支；最终前端是point+tanh(gate)×(candidate−point)，gate可能为负，因此权重和空间重心不能直接解释为最终天气输入的正向物理贡献。关联不代表因果；需同时检查天气与权重交换、gate和误差变化。\n'
 (out/'REPORT_CN.md').write_text(report)
 evidence='# 贡献点证据台账（随完整配对自动更新）\n\n'
 evidence+='本表每一行只对应一个年份、一个随机种子；n为已有配对风场数，56才是该seed的完整覆盖。改善/退步按精确符号记录，没有最小幅度门槛，也不把计数当作统计独立性证明。尚未完成全矩阵时不作最终贡献判断。调参seed40/41不进入此表。\n\n'
 evidence+='解释顺序：S1/W0和ST00/W0检验邻域/静态汇聚；S3/S2、T2/T1、ST11/ST00检验样本自适应；ST10/ST00与ST01/ST00区分空间和时间的增量；跨Transformer/TiDE对照检验前端收益的可移植性（不等同于跨风场迁移）。H1/W0检验轨迹条件，W0/H2检验历史信息；历史×天气2×2和ST11/ST_NO_H进一步检验作用位置。\n\n'
 evidence+='MAE改善即列为正向证据，同时保留RMSE退步、无收益站点和各seed的不一致。只有邻域有效时只主张邻域提取；自适应配对、干预、不同骨干共同支持时才讨论自适应贡献；若ST11不优于W0，完整报告其限制。所有变化都要结合逐次原值判断，不能用一两个正例宣称普遍有效。\n\n'
 evidence+='| 候选 | 参考 | 年份 | seed | 指标 | 已配对风场 | 改善 | 退步 | 持平 |\n|---|---|---:|---:|---|---:|---:|---:|---:|\n'
 for x in counts:evidence+='| '+' | '.join(str(x[k]) for k in ['candidate','reference','year','seed','metric','n_pairs','improved','worsened','ties'])+' |\n'
 (out/'CONTRIBUTION_EVIDENCE_CN.md').write_text(evidence)
 # Excel generation is modest and includes every completed actual fit.
 excel_tmp=out/f'EXPERIMENT_RESULTS_PER_SEED.{os.getpid()}.xlsx'
 with pd.ExcelWriter(excel_tmp,engine='xlsxwriter',engine_kwargs={'options':{'strings_to_urls':False}}) as writer:
  for name,data in [('逐站逐seed结果',df),('实际配置成本',pd.DataFrame(params)),('配对实际值',paired),('逐seed方向次数',countdf),('分时效实际值',hz),('分时效配对',paired_hz),('退步配对',failures),('历史天气交互',pd.DataFrame(interaction)),('干预逐seed',pd.DataFrame(interventions)),('注意力关联',pd.DataFrame(attention))]:
   data.to_excel(writer,sheet_name=name,index=False);ws=writer.sheets[name];ws.freeze_panes(1,0);ws.autofilter(0,0,len(data),max(0,len(data.columns)-1));ws.set_column(0,max(0,len(data.columns)-1),22)
 os.replace(excel_tmp,out/'EXPERIMENT_RESULTS_PER_SEED.xlsx')
 print(now(),'analysis rows',len(df),'pairs',len(paired),flush=True)

if __name__=='__main__':main()
