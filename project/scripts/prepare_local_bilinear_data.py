from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from wind_forcing.models.components import build_station_patch_metadata


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache raw 2x2 bilinear Pangu values at all farm coordinates"
    )
    parser.add_argument("--source", default="data/model_cache_pad5_f8")
    parser.add_argument("--output", default="data/local_bilinear_56farm_v1")
    parser.add_argument("--chunk-cycles", type=int, default=64)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    static = {
        name: np.load(source / "static" / f"{name}.npy", allow_pickle=False)
        for name in (
            "weather_latitude",
            "weather_longitude",
            "station_latitude",
            "station_longitude",
        )
    }
    patch, _, weights = build_station_patch_metadata(
        static["weather_latitude"],
        static["weather_longitude"],
        static["station_latitude"],
        static["station_longitude"],
        "bilinear",
        1,
    )
    output.mkdir(parents=True, exist_ok=True)
    for path in (source / "static").glob("*.npy"):
        link_or_copy(path, output / "static" / path.name)
    np.save(output / "static" / "bilinear_patch_index.npy", patch)
    np.save(output / "static" / "bilinear_weight.npy", weights)

    new_arrays = []
    for split in ("train", "val", "test"):
        source_split = source / split
        target_split = output / split
        target_split.mkdir(parents=True, exist_ok=True)
        for path in source_split.glob("*.npy"):
            if path.name != "weather.npy":
                link_or_copy(path, target_split / path.name)
        weather = np.load(source_split / "weather.npy", mmap_mode="r")
        shape = (weather.shape[0], weather.shape[1], len(patch), weather.shape[2])
        target_weather = np.lib.format.open_memmap(
            target_split / "weather.npy", mode="w+", dtype=np.float32, shape=shape
        )
        flat_patch = patch.reshape(-1)
        for start in range(0, len(weather), args.chunk_cycles):
            stop = min(len(weather), start + args.chunk_cycles)
            block = np.asarray(weather[start:stop], dtype=np.float32)
            values = block.reshape(*block.shape[:3], -1)[:, :, :, flat_patch]
            values = values.reshape(
                stop - start, block.shape[1], block.shape[2], len(patch), 4
            ).transpose(0, 1, 3, 4, 2)
            target_weather[start:stop] = (values * weights[None, None, :, :, None]).sum(axis=3)
            target_weather.flush()
            print(f"[{split}] {stop}/{len(weather)}", flush=True)
        del target_weather
        manifest["splits"][split]["weather_shape"] = list(shape[1:])

    identity = {
        "source_fingerprint": manifest["preprocess_fingerprint"],
        "method": "raw_2x2_bilinear_before_point_encoding",
        "patch_sha256": digest(patch),
        "weight_sha256": digest(weights),
    }
    manifest["source_preprocess_fingerprint"] = manifest["preprocess_fingerprint"]
    manifest["preprocess_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest["local_weather_cache"] = identity
    manifest["arrays"] = [
        row for row in manifest["arrays"] if row.get("relative_path") not in {
            f"{split}/weather.npy" for split in ("train", "val", "test")
        }
    ]
    for split in ("train", "val", "test"):
        path = output / split / "weather.npy"
        array = np.load(path, mmap_mode="r")
        manifest["arrays"].append(
            {
                "relative_path": f"{split}/weather.npy",
                "file": "weather.npy",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    (output / "COMPLETE").touch()
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
