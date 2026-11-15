from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEEDS = tuple(range(42, 52))
SCREEN_SEEDS = {42, 46, 50}


def resolve(project: Path) -> dict[str, Any]:
    root = project / "artifacts/cross_resolution_temporal_v2"
    formal_decision = json.loads((root / "formal/analysis_val/decision.json").read_text(encoding="utf-8"))
    formal_manifest = json.loads((root / "formal/run_manifest.json").read_text(encoding="utf-8"))
    if not formal_manifest.get("all_complete"):
        raise SystemExit("Formal temporal manifest is incomplete")

    selected_group = formal_decision["selected_group"]
    selected_variant = selected_group
    query_mode = formal_decision["selected_time_query_mode"]
    bracket_time_mode = "relative_distances"
    bracket_activation = "gelu"
    difference_input = "actual"
    source = "formal"
    frozen = bool(formal_decision.get("temporal_structure_frozen", False))
    if formal_decision.get("stage_b_authorized", False):
        mechanism_path = root / "mechanism/analysis_formal/decision.json"
        if not mechanism_path.is_file() or not (mechanism_path.parent / "ANALYSIS_COMPLETE").is_file():
            frozen = False
        else:
            mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
            frozen = bool(mechanism.get("temporal_structure_frozen", False))
            selected_variant = mechanism["selected_variant"]
            query_mode = mechanism["selected_time_query_mode"]
            bracket_time_mode = mechanism["selected_bracket_time_feature_mode"]
            bracket_activation = mechanism["selected_bracket_mapper_activation"]
            difference_input = mechanism["selected_forcing_difference_input"]
            source = "mechanism" if selected_variant != "t2_bracket_nonlinear" else "formal"

    runs = []
    if selected_variant == "t2_no_diff":
        for seed in SEEDS:
            stage = "screening" if seed in SCREEN_SEEDS else "formal"
            directory = root / f"mechanism/{stage}/groups/t2_no_diff/seed{seed}"
            runs.append({"seed": seed, "directory": str(directory.resolve()), "source_stage": f"mechanism_{stage}"})
    else:
        matches = {
            int(record["seed"]): record
            for record in formal_manifest["runs"]
            if record["group"] == selected_group
        }
        if set(matches) != set(SEEDS):
            raise SystemExit(f"Selected formal group is incomplete: {selected_group}")
        runs = [
            {
                "seed": seed,
                "directory": matches[seed]["directory"],
                "source_stage": "formal",
            }
            for seed in SEEDS
        ]
    if frozen:
        for record in runs:
            directory = Path(record["directory"])
            for name in ("COMPLETE", "config.yaml", "best_model.pt", "metrics_val.json", "predictions_val.npz"):
                if not (directory / name).is_file():
                    raise SystemExit(f"Frozen selection is missing {directory / name}")
    return {
        "data_split": "2023_validation",
        "test_2024_read": False,
        "temporal_structure_frozen": frozen,
        "selected_group": selected_group,
        "selected_variant": selected_variant,
        "selected_time_query_mode": query_mode,
        "selected_bracket_time_feature_mode": bracket_time_mode,
        "selected_bracket_mapper_activation": bracket_activation,
        "selected_forcing_difference_input": difference_input,
        "selection_source": source,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the frozen cross-resolution temporal architecture")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--field")
    args = parser.parse_args()
    result = resolve(Path(args.project_root).resolve())
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.field:
        value = result[args.field]
        print(value if isinstance(value, str) else json.dumps(value))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
