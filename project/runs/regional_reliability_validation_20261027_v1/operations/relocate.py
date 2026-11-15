"""Explicit deployment path adapter; frozen model/data/training sources stay byte-identical."""
from pathlib import Path
import json,sys

def activate(run):
    sys.path.insert(0,str(run/'code'))
    marker=run/'operations/REMOTE_DEPLOYMENT.json'
    if not marker.exists():return None
    import settings
    settings.STORAGE=run;settings.JOBS=run/'jobs'
    return json.loads(marker.read_text())

def source_path(run,path):
    marker=run/'operations/REMOTE_DEPLOYMENT.json'
    if not marker.exists():return Path(path)
    config=json.loads(marker.read_text());origin=Path(config['source_project'])
    storage=Path(config.get('source_storage','/shared_data_ssd3/liying/'+run.name))
    candidate=Path(path)
    oldstorage=Path('/shared_data_ssd3/liying/physics_observation_single_farm_20261024_v1')
    if candidate.is_relative_to(oldstorage):return run.parents[1]/'runs/physics_observation_single_farm_20261024_v1'/candidate.relative_to(oldstorage)
    neighborstorage=Path('/shared_data_ssd3/liying/neighbor_observation_sel_single_farm_20261025_v1')
    if candidate.is_relative_to(neighborstorage):return run.parents[1]/'runs/neighbor_observation_sel_single_farm_20261025_v1'/candidate.relative_to(neighborstorage)
    if candidate.is_relative_to(storage):return run/candidate.relative_to(storage)
    return run.parents[1]/candidate.relative_to(origin)
