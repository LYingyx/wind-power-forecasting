"""Paths for the merged installation; archived experiment files remain unchanged."""
from pathlib import Path
import os
import sys

ROOT = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SLAST = ROOT / "artifacts/p0i_summary_calibration_20260916_v1"
CROSS = ROOT / "artifacts/cross_farm_shared_specific_20260920_v1"
FEWSHOT = ROOT / "artifacts/selected30_fewshot_baselines_20260920_v1"
OLD_ROOTS = (
    "/storage/liying/new_try/wind_cluster_forced_evolution",
    "/data/liying/new_try/wind_cluster_forced_evolution",
)

def rebase(value):
    if not isinstance(value, str):
        return value
    for old in OLD_ROOTS:
        value = value.replace(old, str(ROOT))
    value = value.replace("/data/liying/anaconda3/envs/chronos/bin/python", sys.executable)
    value = value.replace("/storage/liying/new_try/datasets", str(ROOT / "datasets"))
    for name, profile in [("cross_farm_shared_specific_20260920_v1", "transfer"), ("selected30_fewshot_baselines_20260920_v1", "fewshot")]:
        value = value.replace(str(ROOT / "artifacts" / name / "code"), str(ROOT / "runtime/experiments" / profile))
    return value

def source(profile):
    experiment = SLAST if profile == "slast" else ROOT / "artifacts/p0i_attribution_closure_20260915_v1"
    return experiment / "snapshot/src"

def select_source(profile):
    sys.path.insert(0, str(source(profile)))

def read_slast_config(seed):
    import yaml
    original = SLAST / f"configs/SLAST_seed{seed}.yaml"
    config = yaml.safe_load(original.read_text())
    config["data"]["cache_dir"] = str(ROOT / "data/model_cache_pad5_f8")
    config["data"]["source_h5"] = str(ROOT / "data/jilin_56farm_pad8.h5")
    return config

def new_output(value):
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    for protected in (ROOT / "artifacts", ROOT / "provenance", ROOT / "data", ROOT / ".runtime"):
        if path.is_relative_to(protected):
            raise ValueError(f"Use a new output path outside imported records: {path}")
    return path
