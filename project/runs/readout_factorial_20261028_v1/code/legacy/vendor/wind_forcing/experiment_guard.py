"""Fail-closed 2024 access for the registered farm-enhancement experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_freeze(root: Path) -> dict:
    path = root / "selection/validation_freeze.json"
    if not path.is_file():
        raise RuntimeError("2024 access blocked: validation freeze is missing")
    freeze = json.loads(path.read_text())
    if (freeze.get("protocol_version") not in {2, 3}
            or freeze.get("test_results_read") is not False
            or freeze.get("selected_spec", {}).get("context") == "id_film"
            or not freeze.get("registered_runs")
            or not freeze.get("train_q90_ramp_thresholds")):
        raise RuntimeError("Invalid farm-enhancement validation freeze")
    snapshot = json.loads((root / "code_snapshot.json").read_text())
    if freeze.get("code_sha256") != snapshot["sha256"]:
        raise RuntimeError("Freeze/code snapshot mismatch")
    return freeze


def require_test_authorized(config: dict, checkpoint: str | Path) -> None:
    experiment = config.get("experiment", {})
    checkpoint = Path(checkpoint).resolve()
    path_root = next((p for p in checkpoint.parents if p.name == "farm_interaction_enhancement_v1"), None)
    if not (path_root is not None or str(experiment.get("id", "")).startswith("farm_interaction_enhancement_v1")
            or config.get("model", {}).get("response_output_mode") in {"cf_residual", "bounded_cf_mix"}):
        return
    root = path_root if path_root is not None else Path(experiment["protocol_root"])
    freeze = read_freeze(root)
    for record in freeze["registered_runs"]:
        registered = root / record["checkpoint"]
        if registered.resolve() == checkpoint:
            if sha256(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
            if "config_sha256" in record:
                import yaml
                config_path = registered.parent / "config.yaml"
                if sha256(config_path) != record["config_sha256"]:
                    raise RuntimeError("Frozen config changed")
                supplied = {k:v for k,v in config.items() if not k.startswith("_")}
                if supplied != yaml.safe_load(config_path.read_text()):
                    raise RuntimeError("Evaluation config differs from frozen config")
            if "validation_metrics_sha256" in record:
                if sha256(registered.parent / "metrics_val.json") != record["validation_metrics_sha256"]:
                    raise RuntimeError("Frozen validation metrics changed")
            return
    raise RuntimeError("Checkpoint is not registered in the validation freeze")
