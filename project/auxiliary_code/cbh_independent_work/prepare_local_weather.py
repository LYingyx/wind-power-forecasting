"""Pre-extract the canonical bilinear P0I weather for all 56 sites."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from independent_models import CACHE, OUT, SOURCE, atomic_json, load_static, sha256_file

sys.path.insert(0, str(SOURCE))
from wind_forcing.models.components import build_station_patch_metadata


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = OUT / "input_cache"
    output.mkdir(parents=True, exist_ok=True)
    static = load_static()
    indices, _, weights = build_station_patch_metadata(
        static["weather_latitude"],
        static["weather_longitude"],
        static["station_latitude"],
        static["station_longitude"],
        "bilinear",
        1,
    )
    if indices.shape != (56, 4) or weights.shape != (56, 4):
        raise RuntimeError((indices.shape, weights.shape))
    rows = {}
    for split in ("train", "val", "test"):
        source_path = CACHE / split / "weather.npy"
        destination = output / f"{split}_p0i_local.npy"
        if destination.exists() and not args.overwrite:
            values = np.load(destination, mmap_mode="r")
            if values.shape[1:] != (56, 8, 14):
                raise RuntimeError(f"Invalid existing cache: {destination} {values.shape}")
        else:
            weather = np.load(source_path, mmap_mode="r")
            temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
            values = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(len(weather), 56, 8, 14),
            )
            for start in range(0, len(weather), args.chunk):
                stop = min(len(weather), start + args.chunk)
                flat = np.asarray(weather[start:stop], dtype=np.float32).reshape(
                    stop - start, 8, 14, -1
                )
                gathered = flat[..., indices.reshape(-1)].reshape(
                    stop - start, 8, 14, 56, 4
                )
                local = (gathered * weights[None, None, None]).sum(-1, dtype=np.float32)
                values[start:stop] = local.transpose(0, 3, 1, 2)
            values.flush()
            del values
            os.replace(temporary, destination)
            values = np.load(destination, mmap_mode="r")
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite local weather: {destination}")
        rows[split] = {
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "output": str(destination),
            "output_sha256": sha256_file(destination),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    audit = {
        "passed": True,
        "method": "canonical bilinear station extraction used by CovariateBaseline.station_weather",
        "indices_shape": list(indices.shape),
        "weights_shape": list(weights.shape),
        "indices_sha256": array_sha256(indices),
        "weights_sha256": array_sha256(weights),
        "maximum_weight_sum_error": float(np.abs(weights.sum(1) - 1).max()),
        "splits": rows,
    }
    atomic_json(OUT / "status/LOCAL_WEATHER_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
