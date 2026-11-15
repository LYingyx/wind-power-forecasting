from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SEEDS = tuple(range(42, 52))
SCREEN_SEEDS = {42, 46, 50}
LABELS = {
    "t2_phase_off": "T2_PhaseOff",
    "t2_linear": "T2_Linear",
    "t2_no_diff": "T2_NoDiff",
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def ci(values: np.ndarray, probability: float = 0.975) -> tuple[float, float]:
    half = stats.t.ppf(probability, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze expanded 10-seed T2 mechanism controls")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    base = project / "artifacts/cross_resolution_temporal_v2"
    screening = json.loads(
        (base / "mechanism/analysis_screening/decision.json").read_text(encoding="utf-8")
    )
    expanded = tuple(screening.get("formal_expansion_groups", ()))
    formal_manifest = json.loads((base / "formal/run_manifest.json").read_text(encoding="utf-8"))
    if not formal_manifest.get("all_complete"):
        raise SystemExit("Formal temporal manifest is incomplete")
    reference_paths = {
        int(record["seed"]): Path(record["directory"])
        for record in formal_manifest["runs"]
        if record["group"] == "T2_bracket_nonlinear"
    }
    if set(reference_paths) != set(SEEDS):
        raise SystemExit("The ten T2 reference checkpoints are incomplete")

    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    conclusions: dict[str, Any] = {}
    no_diff_noninferior = False
    for group in expanded:
        if group not in LABELS:
            raise ValueError(f"Unknown expanded mechanism group: {group}")
        differences = []
        candidate_maes = []
        reference_maes = []
        for seed in SEEDS:
            stage = "screening" if seed in SCREEN_SEEDS else "formal"
            directory = base / f"mechanism/{stage}/groups/{group}/seed{seed}"
            required = ("COMPLETE", "config.yaml", "best_model.pt", "metrics_val.json", "predictions_val.npz")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise SystemExit(f"Incomplete {group} seed{seed}: {missing}")
            candidate = load(directory / "predictions_val.npz")
            reference = load(reference_paths[seed] / "predictions_val.npz")
            np.testing.assert_array_equal(candidate["origin_unix"], reference["origin_unix"])
            np.testing.assert_array_equal(candidate["cluster_target_mw"], reference["cluster_target_mw"])
            candidate_error = np.abs(candidate["cluster_prediction_mw"] - candidate["cluster_target_mw"])
            reference_error = np.abs(reference["cluster_prediction_mw"] - reference["cluster_target_mw"])
            differences.append(float((candidate_error - reference_error).mean()))
            candidate_maes.append(float(candidate_error.mean()))
            reference_maes.append(float(reference_error.mean()))
            run_manifest.append(
                {
                    "group": group,
                    "seed": seed,
                    "source_stage": stage,
                    "directory": str(directory.resolve()),
                    "status": "complete",
                }
            )
        values = np.asarray(differences, dtype=np.float64)
        low, high = ci(values)
        row: dict[str, Any] = {
            "group": LABELS[group],
            "candidate_mae_mean_mw": float(np.mean(candidate_maes)),
            "reference_t2_mae_mean_mw": float(np.mean(reference_maes)),
            "candidate_minus_t2_mean_mw": float(values.mean()),
            "sd_mw": float(values.std(ddof=1)),
            "ci95_low_mw": low,
            "ci95_high_mw": high,
            "same_direction_count": int(np.sum(values > 0)),
            "seed_differences_mw": json.dumps(differences),
        }
        if group in {"t2_phase_off", "t2_linear"}:
            supported = bool(values.mean() >= 1.0 and low > 0 and np.sum(values > 0) >= 7)
            row["mechanism_supported"] = supported
            conclusions[f"{LABELS[group]}_mechanism_supported"] = supported
        else:
            one_sided_upper = float(
                values.mean()
                + stats.t.ppf(0.95, len(values) - 1)
                * values.std(ddof=1)
                / np.sqrt(len(values))
            )
            no_diff_noninferior = bool(one_sided_upper < 1.0)
            row["noninferiority_margin_mw"] = 1.0
            row["one_sided_ci95_upper_mw"] = one_sided_upper
            row["noninferior"] = no_diff_noninferior
            conclusions["T2_NoDiff_noninferior_1mw"] = no_diff_noninferior
        rows.append(row)

    # A control that did not pass the preregistered three-seed expansion gate
    # cannot support a ten-seed mechanism claim or an architectural change.
    for group, label in LABELS.items():
        if group not in expanded:
            conclusions[f"{label}_not_formally_tested"] = True
    selected_variant = "t2_no_diff" if no_diff_noninferior else "t2_bracket_nonlinear"
    decision = {
        "data_split": "2023_validation",
        "test_2024_read": False,
        "single_checkpoint_statistics": True,
        "expanded_groups": list(expanded),
        **conclusions,
        "selected_variant": selected_variant,
        "selected_time_query_mode": "bracketing_mlp",
        "selected_bracket_time_feature_mode": "relative_distances",
        "selected_bracket_mapper_activation": "gelu",
        "selected_forcing_difference_input": "zero" if no_diff_noninferior else "actual",
        "temporal_structure_frozen": True,
    }
    output = base / "mechanism/analysis_formal"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_formal_comparisons.csv", rows)
    (output / "run_manifest.json").write_text(
        json.dumps({"runs": run_manifest, "all_complete": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "ANALYSIS_COMPLETE").touch()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
