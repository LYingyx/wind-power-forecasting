from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    for section in ("data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration misses mapping section: {section}")

    project_root = config_path.parent.parent
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    required_paths = ("source_h5", "cache_dir")
    optional_paths = (
        "response_cache_dir", "response_latent_cache_root",
        "dynamics_time_cache_dir", "dynamics_latent_cache_root", "realization_cache_root",
    )
    for key in (*required_paths, *optional_paths):
        value = config["data"].get(key)
        if value is None and key in required_paths:
            raise ValueError(f"Configuration misses data.{key}")
        if value is None:
            continue
        path_value = Path(str(value))
        if not path_value.is_absolute():
            path_value = (config_path.parent / path_value).resolve()
        config["data"][key] = str(path_value)
    return config
