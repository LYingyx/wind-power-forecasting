from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wind_forcing.config import load_config


SEEDS = tuple(range(42, 52))
SCREEN_SEEDS = {42, 46, 50}
SPATIAL_SLUG = {
    "S0_raw_bilinear": "s0_raw_bilinear",
    "S1_latent_bilinear": "s1_latent_bilinear",
    "S2_learned_2x2": "s2_learned_2x2",
    "S3_fixed_gaussian_4x4": "s3_fixed_gaussian_4x4",
    "S4_learned_4x4_nogeom": "s4_learned_4x4_nogeom",
}
LINK_FILES = (
    "config.yaml",
    "config.log",
    "best_model.pt",
    "history.json",
    "metrics_val.json",
    "predictions_val.npz",
    "run.log",
)


def compatible(source: Path, temporal: dict, spatial: dict, seed: int) -> bool:
    if not (source / "COMPLETE").is_file():
        return False
    required = ("config.yaml", "best_model.pt", "history.json", "metrics_val.json", "predictions_val.npz")
    if any(not (source / name).is_file() or (source / name).stat().st_size == 0 for name in required):
        return False
    config = load_config(source / "config.yaml")
    model = config["model"]
    deterministic = config["training"]["deterministic"]
    return all(
        (
            int(config["seed"]) == seed,
            model["time_query_mode"] == temporal["selected_time_query_mode"],
            model.get("bracket_time_feature_mode", "relative_distances")
            == temporal["selected_bracket_time_feature_mode"],
            model.get("bracket_mapper_activation", "gelu")
            == temporal["selected_bracket_mapper_activation"],
            model.get("forcing_difference_input", "actual")
            == temporal["selected_forcing_difference_input"],
            model["station_query_mode"] == spatial["selected_station_query_mode"],
            int(model["station_neighborhood_cells"])
            == int(spatial["selected_station_neighborhood_cells"]),
            bool(model["include_wind_geometry"])
            == bool(spatial["selected_include_wind_geometry"]),
            deterministic["anchor_schedule_mode"] == "horizon_taper",
            float(deterministic["distillation_weight"]) == 0.0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse exact frozen-architecture Normal checkpoints")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = project / "artifacts/cross_resolution_temporal_v2"
    temporal = json.loads((root / "temporal_selection.json").read_text(encoding="utf-8"))
    spatial = json.loads((root / "spatial/analysis_formal/decision.json").read_text(encoding="utf-8"))
    if not temporal.get("temporal_structure_frozen") or not spatial.get("spatial_structure_frozen"):
        raise SystemExit("Architecture is not frozen")
    temporal_sources = {int(record["seed"]): Path(record["directory"]) for record in temporal["runs"]}
    selected = spatial["selected_spatial_group"]
    reused = []
    for seed in SEEDS:
        if selected == "S5_learned_4x4_wind":
            source = temporal_sources[seed]
            source_stage = "temporal_selection"
        else:
            slug = SPATIAL_SLUG[selected]
            stage = "screening" if seed in SCREEN_SEEDS else "formal"
            source = root / f"spatial/{stage}/groups/{slug}/seed{seed}"
            source_stage = f"spatial_{stage}"
        destination = root / f"final/groups/normal/seed{seed}"
        if not compatible(source, temporal, spatial, seed):
            continue
        if (destination / "best_model.pt").exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for name in LINK_FILES:
            source_file = source / name
            target = destination / name
            if source_file.is_file() and not target.exists() and not target.is_symlink():
                os.symlink(source_file.resolve(), target)
        (destination / "REUSE_SOURCE.json").write_text(
            json.dumps(
                {"source": str(source.resolve()), "source_stage": source_stage, "exact_config_reuse": True},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (destination / "COMPLETE").touch()
        reused.append(seed)
    print(json.dumps({"selected_spatial_group": selected, "reused_seeds": reused}, ensure_ascii=False))


if __name__ == "__main__":
    main()
