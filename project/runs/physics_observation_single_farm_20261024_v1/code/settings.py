"""Independent, as-of-origin physical/observation experiments; old runs are frozen."""
from pathlib import Path
import datetime
import hashlib
import json
import os

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
PREVIOUS = PROJECT / 'runs/adaptive_weather_single_farm_20261022_v1'
OLD = PROJECT / 'runs/independent_single_farm_20261021_v1'
CACHE = PROJECT / 'data/model_cache_pad5_f8'
H5 = PROJECT / 'data/jilin_56farm_pad8.h5'
LOCAL_WEATHER = PROJECT / 'artifacts/shared_all_station_baselines_20260922_v1/data/local_weather'
COLUMN_DATA = PROJECT / 'artifacts/column_weather_20260928_v1/data'
STORAGE = Path('/shared_data_ssd3/liying/physics_observation_single_farm_20261024_v1')
PREPARED = STORAGE / 'prepared'
JOBS = STORAGE / 'jobs'
SEEDS = [42, 43]
REPRESENTATIVES = [6, 8, 14, 16, 43, 46]
CHANNELS = ['msl','u10','v10','t2m','z1000','z925','q1000','q925','t1000','t925','u1000','u925','v1000','v925']

def now():
    return datetime.datetime.now().astimezone().isoformat()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def setup():
    RUN.mkdir(exist_ok=True); PREPARED.mkdir(parents=True, exist_ok=True); JOBS.mkdir(parents=True, exist_ok=True)
    for name, target in [('prepared', PREPARED), ('jobs', JOBS)]:
        link = RUN / name
        if not link.exists(): link.symlink_to(target, target_is_directory=True)
        assert link.resolve() == target.resolve()
