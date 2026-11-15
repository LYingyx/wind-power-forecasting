from settings import *

def configurations():
 out=[('W0','B0')]+[(b,v) for b in ['W0','TF'] for v in ['SEL','N3','R3_RAW','C2_R3_RAW','AGE']]+[('W0',v) for v in ['H0W1','H1W0','H0W0']]
 gate=json.loads((RUN/'analysis/P1_GATE.json').read_text())
 if gate['own']:out += [('W0',v) for v in ['LOCAL_ERR','C2_LOCAL_ERR']]
 if gate['region']:out += [('W0',v) for v in ['REGION_ERR','C2_REGION_ERR']]
 return out

def make_spec(b,v,site,seed):
 return dict(id=f'{b}_{v}_s{site:02d}_r{seed}',task='train',backbone=b,variant=v,site=site,seed=seed,loss='MAE',stage='screen' if seed<44 else 'replication',configuration=b+'_'+v,data_fingerprint=FINGERPRINT,tf_zero_init=b=='TF',tf_dropout=.05,tf_lr=.0002)

def build():
 configs=configurations();screen=[make_spec(b,v,site,seed) for seed in [42,43] for site in range(56) for b,v in configs]
 alljobs=[make_spec(b,v,site,seed) for seed in range(42,52) for site in range(56) for b,v in configs]
 atomic_json(RUN/'manifest.json',screen);atomic_json(RUN/'screen_manifest.json',screen);atomic_json(RUN/'candidate_manifest.json',alljobs)
 atomic_json(RUN/'PROTOCOL.json',dict(timestamp=now(),data_fingerprint=FINGERPRINT,train_year=2022,selection_year=2023,test_2024_enabled=False,screen_seeds=[42,43],replication_seeds=list(range(44,52)),farms=56,epochs=80,patience=10,loss='MAE',selection='2023 MAE; report paired MAE and RMSE of same MAE-selected checkpoint',precision='float32_no_tf32',batch_size=256,learning_rate=.0002,weight_decay=.0001,regional_rings_km=[[0,20],[20,60],[60,150]],regional_weight='capacity MW; exclude target',neighbor_delay_minutes=15,neighbor_n3='nearest 8 within150km; same legacy N3 readout',R3_RAW='N3 + three rings mean/std/1h change/presence; removes identically zero slots from original36dim encoding',mismatch='joint donor for all neighbors from2022 same BJT month and clock; at least48h away; seed-specific; matches forecast age',interaction='real current-power persistence anchor retained in all four cells; H0 fixes earlier history and all SEL STATE; W0 fixes weather and weather-derived SEL expectations; estimate history-context by weather interaction conditional on current power',P1='same available issue pastweather; monotone2022 mapping, train monthOOF+24h; observed-minus-mapped residual; ownlast3h throughorigin, neighbors delayed15min; linear conditional gate before GPU',age='age/24 at origin; deterministic cycle/time-of-day covariate, not separate causal age identification',success='MAE and RMSE relative improvement >=0.1%, >28/56 farm wins each, high-power group RMSE increase <=0.5%; trained mismatch controls must be worse',replication='eligible two-seed comparisons and matched controls expand to10 seeds; mechanism full2x2 expands together if positive both metrics with >28 farm signs; no2024 evaluation',initialization='fresh random weights; original W0 neutral-last active graph has no B0 weight dependency; new E2C and SEL normalization',forecast_time='BJT08/20 issue +12h delivery; UTC filenames; same issue for entire6h trajectory; lead up to30h'))
 print('Screen tasks',len(screen),'possible10seed',len(alljobs))
if __name__=='__main__':build()
