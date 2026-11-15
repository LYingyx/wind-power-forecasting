from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SEEDS = tuple(range(42, 52))
LEADS = np.arange(1, 25, dtype=np.float64) / 4.0
HORIZON_BANDS = {
    "0.25-1h": np.arange(0, 4),
    "1.25-3h": np.arange(4, 12),
    "3.25-6h": np.arange(12, 24),
}
PHASE_NAMES = {0: "anchor", 1: "+15min", 2: "+30min", 3: "+45min"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    half = stats.t.ppf(0.5 + confidence / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal 10-seed temporal validation")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    formal = project / "artifacts/cross_resolution_temporal_v2/formal"
    manifest = json.loads((formal / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest["all_complete"] or not (formal / "FORMAL_COMPLETE").is_file():
        raise SystemExit("Formal manifest has not passed its 40-run audit")
    output = formal / "analysis_val"
    output.mkdir(parents=True, exist_ok=True)

    records = {(record["group"], record["seed"]): record for record in manifest["runs"]}
    archives: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    seed_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    cache = project / "data/model_cache_pad5_f8/val"
    cache_index = np.load(cache / "sample_index.npy").astype(np.int64)
    offsets_all = np.load(cache / "weather_time_offset_hours.npy")
    groups = sorted({record["group"] for record in manifest["runs"]})
    for group in groups:
        archives[group] = {}
        maes = []
        lead_seed = []
        for seed in SEEDS:
            record = records[(group, seed)]
            with np.load(Path(record["directory"]) / "predictions_val.npz") as data:
                run = {key: data[key] for key in data.files}
            archives[group][seed] = run
            error = np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"])
            mae = float(error.mean())
            maes.append(mae)
            per_lead = error.mean(axis=0)
            lead_seed.append(per_lead)
            seed_rows.append({"group": group, "seed": seed, "cluster_mae_mw": mae})
            for band, indices in HORIZON_BANDS.items():
                band_rows.append(
                    {"group": group, "seed": seed, "horizon_band": band, "mae_mw": float(error[:, indices].mean())}
                )
            sample_index = run["sample_index"].astype(np.int64)
            local = np.searchsorted(cache_index, sample_index)
            if np.any(local >= len(cache_index)) or not np.array_equal(cache_index[local], sample_index):
                raise ValueError(f"Cache mapping failed for {group} seed{seed}")
            offsets = offsets_all[local]
            phase = np.mod(np.rint(np.mod(LEADS[None, :] - offsets[:, :1], 1.0) * 4).astype(np.int64), 4)
            for phase_id, phase_name in PHASE_NAMES.items():
                phase_rows.append(
                    {"group": group, "seed": seed, "phase": phase_name, "mae_mw": float(error[phase == phase_id].mean())}
                )
        low, high = ci(np.asarray(maes))
        seed_rows.append(
            {
                "group": group,
                "seed": "mean",
                "cluster_mae_mw": float(np.mean(maes)),
                "sd_mw": float(np.std(maes, ddof=1)),
                "ci95_low_mw": low,
                "ci95_high_mw": high,
            }
        )
        lead_seed_array = np.asarray(lead_seed)
        for index, lead in enumerate(LEADS):
            lead_rows.append(
                {
                    "group": group,
                    "lead_hours": lead,
                    "mae_mean_mw": float(lead_seed_array[:, index].mean()),
                    "mae_sd_mw": float(lead_seed_array[:, index].std(ddof=1)),
                }
            )

    comparisons = (
        ("T2_vs_T0", "T2_bracket_nonlinear", "T0_raw_linear"),
        ("T2_vs_T1", "T2_bracket_nonlinear", "T1_latent_linear"),
        ("T2_vs_T4", "T2_bracket_nonlinear", "T4_native_query"),
        # Descriptive checks used to choose the practical fallback after the
        # preregistered T2 mechanism claim fails.  They are not added to the
        # two-comparison Holm family above.
        ("T1_vs_T0", "T1_latent_linear", "T0_raw_linear"),
        ("T1_vs_T4", "T1_latent_linear", "T4_native_query"),
    )
    comparison_rows = []
    differences: dict[str, np.ndarray] = {}
    p_values = {}
    for label, candidate, reference in comparisons:
        values = []
        for seed in SEEDS:
            cand = archives[candidate][seed]
            ref = archives[reference][seed]
            np.testing.assert_array_equal(cand["origin_unix"], ref["origin_unix"])
            np.testing.assert_array_equal(cand["cluster_target_mw"], ref["cluster_target_mw"])
            cand_error = np.abs(cand["cluster_prediction_mw"] - cand["cluster_target_mw"])
            ref_error = np.abs(ref["cluster_prediction_mw"] - ref["cluster_target_mw"])
            values.append(float((cand_error - ref_error).mean()))
        values_array = np.asarray(values)
        differences[label] = values_array
        low, high = ci(values_array)
        p_value = float(stats.ttest_1samp(values_array, 0.0).pvalue)
        if label in {"T2_vs_T0", "T2_vs_T1"}:
            p_values[label] = p_value
        comparison_rows.append(
            {
                "comparison": label,
                "candidate_minus_reference_mean_mw": float(values_array.mean()),
                "sd_mw": float(values_array.std(ddof=1)),
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "candidate_wins": int(np.sum(values_array < 0)),
                "p_value_two_sided": p_value,
                "seed_differences_mw": json.dumps(values),
            }
        )
    adjusted = holm(p_values)
    for row in comparison_rows:
        row["holm_adjusted_p"] = adjusted.get(row["comparison"], "")

    t2_t0 = next(row for row in comparison_rows if row["comparison"] == "T2_vs_T0")
    t2_t1 = next(row for row in comparison_rows if row["comparison"] == "T2_vs_T1")
    t2_t4_values = differences["T2_vs_T4"]
    noninferior_upper = float(
        t2_t4_values.mean()
        + stats.t.ppf(0.95, len(t2_t4_values) - 1)
        * t2_t4_values.std(ddof=1)
        / np.sqrt(len(t2_t4_values))
    )
    t2_vs_t0_pass = (
        t2_t0["candidate_minus_reference_mean_mw"] <= -3.0
        and t2_t0["ci95_high_mw"] < 0
        and t2_t0["candidate_wins"] >= 7
        and adjusted["T2_vs_T0"] < 0.05
    )
    t2_vs_t1_pass = (
        t2_t1["candidate_minus_reference_mean_mw"] <= -1.0
        and t2_t1["ci95_high_mw"] < 0
        and t2_t1["candidate_wins"] >= 7
        and adjusted["T2_vs_T1"] < 0.05
    )
    t2_vs_t4_noninferior = noninferior_upper < 1.0
    stage_b_authorized = bool(
        t2_vs_t0_pass and t2_vs_t1_pass and t2_vs_t4_noninferior
    )
    if stage_b_authorized:
        selected_group, selected_mode = "T2_bracket_nonlinear", "bracketing_mlp"
    elif t2_vs_t0_pass and t2_vs_t1_pass:
        # T2 clears the two interpolation controls but is more than the
        # preregistered 1 MW non-inferiority margin behind T4.  In that case
        # simplicity is not enough to select T2; retain the stronger T4.
        selected_group, selected_mode = "T4_native_query", "full_attention"
    elif not t2_vs_t1_pass:
        # The preregistration explicitly falls back to T1 when nonlinear
        # bracketing does not improve over latent-linear interpolation.  A
        # failed T2-vs-T0 effect-size gate downgrades the paper claim; it does
        # not make the weaker T0 the deployment choice.
        selected_group, selected_mode = "T1_latent_linear", "latent_linear"
    else:
        selected_group, selected_mode = "T0_raw_linear", "raw_linear"
    decision = {
        "data_split": "2023_validation",
        "test_2024_read": False,
        "single_checkpoint_statistics": True,
        "t2_vs_t0_pass": t2_vs_t0_pass,
        "t2_vs_t1_pass": t2_vs_t1_pass,
        "t2_vs_t4_noninferior_1mw": t2_vs_t4_noninferior,
        "t2_vs_t4_one_sided_ci95_upper_mw": noninferior_upper,
        "stage_b_authorized": stage_b_authorized,
        # A confirmed T2 still requires the preregistered mechanism screen
        # (including the possible removal of the explicit difference input)
        # before the temporal structure is frozen.
        "temporal_structure_frozen": not stage_b_authorized,
        "selected_group": selected_group,
        "selected_time_query_mode": selected_mode,
        "stage_b_stop_reasons": [
            reason
            for reason, passed in (
                ("T2_vs_T0_effect_or_inference_gate_failed", t2_vs_t0_pass),
                ("T2_vs_T1_superiority_gate_failed", t2_vs_t1_pass),
                ("T2_vs_T4_noninferiority_gate_failed", t2_vs_t4_noninferior),
            )
            if not passed
        ],
        "selection_rationale": (
            "T2 passed all formal gates"
            if stage_b_authorized
            else "T1 is the preregistered simpler fallback because T2 did not improve over T1"
            if selected_group == "T1_latent_linear"
            else "T4 retained because T2 failed its noninferiority margin"
            if selected_group == "T4_native_query"
            else "raw-linear fallback because the cross-resolution contribution gates failed"
        ),
        "paper_claim": (
            "task_space_local_nonlinear_bracket_alignment"
            if stage_b_authorized
            else "cross_resolution_as_task_context"
        ),
    }
    write_csv(output / "single_checkpoint_summary.csv", seed_rows)
    write_csv(output / "paired_comparisons.csv", comparison_rows)
    write_csv(output / "per_lead_mae.csv", lead_rows)
    write_csv(output / "per_horizon_band_mae.csv", band_rows)
    write_csv(output / "per_within_hour_phase_mae.csv", phase_rows)
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
