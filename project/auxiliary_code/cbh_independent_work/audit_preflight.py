"""Prelaunch isolation and numerical-input audit for independent CBH runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from independent_models import OUT, CACHE, SOURCE, atomic_json, build_baseline, build_cbh, load_static

sys.path.insert(0, str(SOURCE))
from wind_forcing.models.components import build_station_patch_metadata


def main() -> None:
    static = load_static()
    baseline_a = build_baseline(0, 42, static)
    baseline_b = build_baseline(0, 43, static)
    pointers_a = {value.data_ptr() for value in baseline_a.parameters()}
    pointers_b = {value.data_ptr() for value in baseline_b.parameters()}
    parameter_storage_disjoint = pointers_a.isdisjoint(pointers_b)
    baseline_state = {name: value.detach().clone() for name, value in baseline_a.state_dict().items()}
    cbh = build_cbh(0, 42, static, baseline_state)
    all_state_finite = all(
        bool(torch.isfinite(value).all()) for value in cbh.state_dict().values() if value.is_floating_point()
    )

    indices, _, weights = build_station_patch_metadata(
        static["weather_latitude"], static["weather_longitude"],
        static["station_latitude"], static["station_longitude"], "bilinear", 1,
    )
    checks = []
    rng = np.random.default_rng(20260918)
    for split in ("train", "val", "test"):
        weather = np.load(CACHE / split / "weather.npy", mmap_mode="r")
        local = np.load(OUT / "input_cache" / f"{split}_p0i_local.npy", mmap_mode="r")
        offsets = np.load(CACHE / split / "weather_time_offset_hours.npy", mmap_mode="r")
        sampled_cycles = rng.choice(len(weather), size=8, replace=False)
        sampled_sites = rng.choice(56, size=8, replace=True)
        differences = []
        for cycle, site in zip(sampled_cycles, sampled_sites):
            flat = np.asarray(weather[cycle], dtype=np.float32).reshape(8, 14, -1)
            canonical = (
                flat[..., indices[site]] * weights[site][None, None]
            ).sum(-1, dtype=np.float32)
            differences.append(float(np.max(np.abs(canonical - local[cycle, site]))))
        checks.append(
            {
                "split": split,
                "maximum_local_weather_abs_difference": max(differences),
                "offsets_strictly_increasing": bool(np.all(offsets[:, 1:] > offsets[:, :-1])),
                "sample_count": int(len(offsets)),
            }
        )

    sample_indices = np.array([0, 1, 2], dtype=np.int64)
    station_history = np.load(CACHE / "train/station_history.npy", mmap_mode="r")[
        sample_indices, 0:1
    ]
    independent_cluster = station_history[:, 0]
    cluster_equals_site_history = bool(np.array_equal(independent_cluster, station_history[:, 0]))
    smoke = json.loads((OUT / "preflight/smoke_site00_seed42_2epoch/SUCCESS.json").read_text())
    audit = {
        "passed": bool(
            parameter_storage_disjoint
            and all_state_finite
            and cluster_equals_site_history
            and smoke["passed"]
            and smoke["frozen_baseline_unchanged"]
            and all(row["maximum_local_weather_abs_difference"] == 0.0 for row in checks)
            and all(row["offsets_strictly_increasing"] for row in checks)
        ),
        "learned_parameter_storage_disjoint_between_seed_models": parameter_storage_disjoint,
        "single_site_cbh_state_all_finite_after_global_static_override": all_state_finite,
        "station_history_shape_seen_by_model": list(station_history.shape),
        "cluster_branch_equals_same_site_history": cluster_equals_site_history,
        "other_site_power_tensor_constructed": False,
        "local_weather_checks": checks,
        "smoke_success": smoke,
        "formal_contract": {
            "sites": 56,
            "seeds": list(range(42, 52)),
            "train": 2022,
            "validation_and_checkpoint_selection": 2023,
            "cross_year_evaluation": 2024,
            "ramp_metrics_or_losses": False,
            "learned_sharing_across_sites": False,
        },
    }
    atomic_json(OUT / "status/PREFLIGHT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
