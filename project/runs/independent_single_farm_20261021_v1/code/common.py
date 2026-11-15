"""Frozen protocol and paths for independent single-farm experiments."""
from pathlib import Path
import hashlib, json, os, random, sys
import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
CACHE = ROOT / 'data/model_cache_pad5_f8'
LOCAL = ROOT / 'artifacts/shared_all_station_baselines_20260922_v1/data/local_weather'
COLUMN = ROOT / 'artifacts/column_weather_20260928_v1/data'
sys.path.insert(0, str(RUN / 'code/vendor'))
FAMILIES = ['TRANSFORMER_FC', 'TIDE_FC', 'NBEATSX_FC', 'TIMEXER_FC',
            'TFT_FC', 'TCN_FC', 'DLINEAR_FC', 'GRU_H', 'PATCHTST_H', 'CNN_W']
ARMS = ['CB', 'CH', 'CNEUTRAL', 'DIRECT', 'NEUTRAL_LAST', 'E0', 'CLIMATOLOGY']
REPRESENTATIVES = ['1043008', '1043010', '1043016', '1043018', '1043048', '1043051']
SEEDS = list(range(42, 52))
TUNE_SEEDS = [40, 41]
CONFIGS = [dict(width=d, lr=lr, weight_decay=1e-4) for d in (64, 128) for lr in (2e-4, 1e-3)]
PROTOCOL = dict(version='20261021_v1', precision='float32_no_tf32', epochs=80, patience=10,
                batch_size=256, eval_batch_size=512, loss='MAE_CF', gradient_clip=1.0,
                optimizer='AdamW', schedule='cosine_80', train_year=2022, validation_year=2023,
                reevaluation_year=2024, main_seeds=SEEDS, ablation_seeds=SEEDS[:5],
                tuning_seeds=TUNE_SEEDS, tuning_farms=REPRESENTATIVES,
                configurations=CONFIGS, output_clipping=False, independent_parameters=True,
                minimum_improvement=0.0, equivalence_margin_cf=0.001,
                max_gpu_workers=8, gpu_reserve_mib=4096, initial_worker_budget_mib=4096)

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def static():
    return {p.stem:np.load(p, allow_pickle=False) for p in (CACHE/'static').glob('*.npy')}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def configure_torch():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

def config_for(family, index):
    cfg = dict(CONFIGS[index])
    if family == 'DLINEAR_FC':
        cfg.update(width=64, lr=(2e-4, 1e-3)[index % 2], weight_decay=(1e-4, 1e-2)[index//2])
    return cfg

def ref_id(site, seed): return f'reference_B0_s{site:02d}_r{seed}'

def source_hashes():
    return {str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'code').rglob('*.py'))}
