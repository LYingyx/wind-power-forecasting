from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from tqdm import tqdm


SEEDS = (42, 46, 50)
LEAD_HOURS = np.arange(1, 25, dtype=np.float64) / 4.0
GROUPS = {
    "T0_raw_linear": "t0_raw_linear",
    "T1_latent_linear": "t1_latent_linear",
    "T2_bracket_nonlinear": "t2_bracket_nonlinear",
    "T3_nearest": "t3_nearest",
}
PHASE_LABELS = {0: "anchor", 1: "+15min", 2: "+30min", 3: "+45min"}
HORIZON_BANDS = {
    "0.25-1h": np.arange(0, 4),
    "1.25-3h": np.arange(4, 12),
    "3.25-6h": np.arange(12, 24),
}
FOCUSED_PAIRS = (
    ("T1_latent_linear", "T0_raw_linear"),
    ("T2_bracket_nonlinear", "T0_raw_linear"),
    ("T2_bracket_nonlinear", "T1_latent_linear"),
    ("T2_bracket_nonlinear", "T3_nearest"),
    ("T2_bracket_nonlinear", "T4_native_query"),
    ("T4_native_query", "T1_latent_linear"),
)


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    point = float(values.mean())
    if len(values) < 2:
        return point, point, point
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return point, point - half, point + half


def aggregate_stratified(
    rows: list[dict[str, Any]], value_key: str
) -> list[dict[str, Any]]:
    """Aggregate seed-wise paired deltas without averaging model predictions."""
    aggregate: list[dict[str, Any]] = []
    keys = sorted({(row["stratum_type"], row["stratum"]) for row in rows})
    for stratum_type, stratum in keys:
        values = np.asarray(
            [
                row[value_key]
                for row in rows
                if (row["stratum_type"], row["stratum"])
                == (stratum_type, stratum)
            ],
            dtype=np.float64,
        )
        point, low, high = mean_ci(values)
        aggregate.append(
            {
                "stratum_type": stratum_type,
                "stratum": stratum,
                "candidate_minus_reference_mean_mw": point,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "seed_wins": int(np.sum(values < 0)),
            }
        )
    return aggregate


def forcing_variation(
    cache: Path, output: Path, chunk_size: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output / "forcing_variation_cache.npz"
    if cache_path.is_file():
        with np.load(cache_path) as archive:
            return archive["train"], archive["val"], archive["thresholds"]
    scale = np.load(cache / "static/nwp_std.npy").astype(np.float64)
    scale = np.maximum(scale, 1e-6).reshape(1, 1, -1, 1, 1)
    results: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        weather = np.load(cache / split / "weather.npy", mmap_mode="r")
        values = np.empty(len(weather), dtype=np.float64)
        for start in tqdm(
            range(0, len(weather), chunk_size),
            desc=f"{split} forcing variation",
            unit="chunk",
        ):
            block = np.asarray(weather[start : start + chunk_size], dtype=np.float64)
            delta = np.diff(block, axis=1) / scale
            values[start : start + len(block)] = np.sqrt(np.mean(delta**2, axis=(1, 2, 3, 4)))
        results[split] = values
    thresholds = np.quantile(results["train"], (1.0 / 3.0, 2.0 / 3.0))
    np.savez_compressed(
        cache_path,
        train=results["train"],
        val=results["val"],
        thresholds=thresholds,
    )
    return results["train"], results["val"], thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="T0-T4 cross-resolution audit")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--root", default="artifacts/cross_resolution_temporal_v1")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = (project / args.root).resolve()
    output = root / "analysis_val"
    output.mkdir(parents=True, exist_ok=True)
    group_roots = {
        name: root / "groups" / directory for name, directory in GROUPS.items()
    }
    group_roots["T4_native_query"] = (
        project / "artifacts/horizon_specialist_v1/groups/l1_anchor_taper"
    )
    missing = [
        str(path / f"seed{seed}/predictions_val.npz")
        for path in group_roots.values()
        for seed in SEEDS
        if not (path / f"seed{seed}/predictions_val.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing predictions:\n" + "\n".join(missing))

    cache = project / "data/model_cache_pad5_f8"
    offsets_all = np.load(cache / "val/weather_time_offset_hours.npy")
    ages_all = np.load(cache / "val/nwp_age_minutes.npy")
    cycles_all = np.load(cache / "val/weather_cycle_index.npy")
    cache_sample_index = np.load(cache / "val/sample_index.npy").astype(np.int64)
    _, val_variation, thresholds = forcing_variation(cache, output)

    predictions: dict[str, list[dict[str, np.ndarray]]] = {}
    summaries: list[dict[str, Any]] = []
    for name, path in group_roots.items():
        runs = [
            load_prediction(path / f"seed{seed}/predictions_val.npz")
            for seed in SEEDS
        ]
        predictions[name] = runs
        seed_mae = [
            float(np.abs(run["cluster_prediction_mw"] - run["cluster_target_mw"]).mean())
            for run in runs
        ]
        parameter_count = json.loads(
            (path / f"seed{SEEDS[0]}/metrics_val.json").read_text(encoding="utf-8")
        )["parameter_count"]
        summaries.append(
            {
                "group": name,
                "single_checkpoint_mae_mean_mw": float(np.mean(seed_mae)),
                "single_checkpoint_mae_std_mw": float(np.std(seed_mae, ddof=1)),
                "seed42_mae_mw": seed_mae[0],
                "seed46_mae_mw": seed_mae[1],
                "seed50_mae_mw": seed_mae[2],
                "parameter_count": int(parameter_count),
            }
        )
    write_csv(output / "summaries.csv", summaries)

    reference_runs = predictions["T0_raw_linear"]
    comparison_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    for name, runs in predictions.items():
        seed_deltas = []
        for run, reference in zip(runs, reference_runs):
            candidate_error = np.abs(
                run["cluster_prediction_mw"] - run["cluster_target_mw"]
            )
            reference_error = np.abs(
                reference["cluster_prediction_mw"] - reference["cluster_target_mw"]
            )
            seed_deltas.append(float((candidate_error - reference_error).mean()))
        point, low, high = mean_ci(np.asarray(seed_deltas))
        comparison_rows.append(
            {
                "comparison": f"{name}_vs_T0_raw_linear",
                "candidate_minus_t0_mean_mw": point,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "candidate_seed_wins": int(sum(value < 0 for value in seed_deltas)),
                "seed_differences_mw": json.dumps(seed_deltas),
            }
        )

        for seed, run, reference in zip(SEEDS, runs, reference_runs):
            sample_index = run["sample_index"].astype(np.int64)
            if not np.array_equal(sample_index, reference["sample_index"]):
                raise ValueError(f"Sample mismatch for {name} seed {seed}")
            local_index = np.searchsorted(cache_sample_index, sample_index)
            if (
                np.any(local_index >= len(cache_sample_index))
                or not np.array_equal(cache_sample_index[local_index], sample_index)
            ):
                raise ValueError(f"Cache sample mapping failed for {name} seed {seed}")
            delta = np.abs(
                run["cluster_prediction_mw"] - run["cluster_target_mw"]
            ) - np.abs(
                reference["cluster_prediction_mw"] - reference["cluster_target_mw"]
            )
            offsets = offsets_all[local_index]
            phase = np.mod(
                np.rint(
                    np.mod(LEAD_HOURS[None, :] - offsets[:, :1], 1.0) * 4.0
                ).astype(np.int64),
                4,
            )
            for phase_id, phase_label in PHASE_LABELS.items():
                stratified_rows.append(
                    {
                        "group": name,
                        "seed": seed,
                        "stratum_type": "within_hour_phase",
                        "stratum": phase_label,
                        "candidate_minus_t0_mw": float(delta[phase == phase_id].mean()),
                        "points": int(np.sum(phase == phase_id)),
                    }
                )
            for band, indices in HORIZON_BANDS.items():
                stratified_rows.append(
                    {
                        "group": name,
                        "seed": seed,
                        "stratum_type": "horizon",
                        "stratum": band,
                        "candidate_minus_t0_mw": float(delta[:, indices].mean()),
                        "points": int(delta[:, indices].size),
                    }
                )
            sample_variation = val_variation[cycles_all[local_index]]
            variation_id = np.digitize(sample_variation, thresholds)
            for regime_id, regime in enumerate(("low", "medium", "high")):
                selected = variation_id == regime_id
                stratified_rows.append(
                    {
                        "group": name,
                        "seed": seed,
                        "stratum_type": "forcing_variation",
                        "stratum": regime,
                        "candidate_minus_t0_mw": float(delta[selected].mean()),
                        "points": int(delta[selected].size),
                    }
                )
            ages = ages_all[local_index]
            for age in (15, 30, 45, 60):
                selected = ages == age
                stratified_rows.append(
                    {
                        "group": name,
                        "seed": seed,
                        "stratum_type": "nwp_age_minutes",
                        "stratum": str(age),
                        "candidate_minus_t0_mw": float(delta[selected].mean()),
                        "points": int(delta[selected].size),
                    }
                )
    write_csv(output / "comparisons_vs_t0.csv", comparison_rows)
    write_csv(output / "stratified_seedwise.csv", stratified_rows)

    aggregate_rows: list[dict[str, Any]] = []
    keys = sorted(
        {(row["group"], row["stratum_type"], row["stratum"]) for row in stratified_rows}
    )
    for group, stratum_type, stratum in keys:
        values = np.asarray(
            [
                row["candidate_minus_t0_mw"]
                for row in stratified_rows
                if (row["group"], row["stratum_type"], row["stratum"])
                == (group, stratum_type, stratum)
            ]
        )
        point, low, high = mean_ci(values)
        aggregate_rows.append(
            {
                "group": group,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "candidate_minus_t0_mean_mw": point,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "seed_wins": int(np.sum(values < 0)),
            }
        )
    write_csv(output / "stratified_summary.csv", aggregate_rows)

    # Direct paired comparisons among the scientifically relevant alternatives.
    # Deltas are always candidate minus reference; each entry is computed from
    # single-checkpoint predictions with the same seed and samples.
    focused_overall_rows: list[dict[str, Any]] = []
    focused_stratified_rows: list[dict[str, Any]] = []
    stratified_lookup = {
        (row["group"], row["seed"], row["stratum_type"], row["stratum"]): row[
            "candidate_minus_t0_mw"
        ]
        for row in stratified_rows
    }
    stratified_keys = sorted(
        {(row["stratum_type"], row["stratum"]) for row in stratified_rows}
    )
    for candidate, reference in FOCUSED_PAIRS:
        seed_deltas = []
        for candidate_run, reference_run in zip(
            predictions[candidate], predictions[reference]
        ):
            candidate_error = np.abs(
                candidate_run["cluster_prediction_mw"]
                - candidate_run["cluster_target_mw"]
            )
            reference_error = np.abs(
                reference_run["cluster_prediction_mw"]
                - reference_run["cluster_target_mw"]
            )
            seed_deltas.append(float((candidate_error - reference_error).mean()))
        point, low, high = mean_ci(np.asarray(seed_deltas))
        focused_overall_rows.append(
            {
                "comparison": f"{candidate}_vs_{reference}",
                "candidate_minus_reference_mean_mw": point,
                "ci95_low_mw": low,
                "ci95_high_mw": high,
                "candidate_seed_wins": int(sum(value < 0 for value in seed_deltas)),
                "seed_differences_mw": json.dumps(seed_deltas),
            }
        )
        for stratum_type, stratum in stratified_keys:
            values = np.asarray(
                [
                    stratified_lookup[(candidate, seed, stratum_type, stratum)]
                    - stratified_lookup[(reference, seed, stratum_type, stratum)]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            stratum_point, stratum_low, stratum_high = mean_ci(values)
            focused_stratified_rows.append(
                {
                    "comparison": f"{candidate}_vs_{reference}",
                    "stratum_type": stratum_type,
                    "stratum": stratum,
                    "candidate_minus_reference_mean_mw": stratum_point,
                    "ci95_low_mw": stratum_low,
                    "ci95_high_mw": stratum_high,
                    "candidate_seed_wins": int(np.sum(values < 0)),
                }
            )
    write_csv(output / "focused_pairwise.csv", focused_overall_rows)
    write_csv(
        output / "focused_pairwise_stratified_summary.csv",
        focused_stratified_rows,
    )

    # T4 versus T1 is the decisive comparison for the method claim: T0 only
    # establishes that interpolation in raw weather space can be insufficient,
    # whereas T1 tests whether a learned native-path query is needed after the
    # hourly observations have already been encoded into task-relevant features.
    t4_t1_seed_deltas: list[float] = []
    t4_t1_stratified_rows: list[dict[str, Any]] = []
    for seed, t4_run, t1_run in zip(
        SEEDS, predictions["T4_native_query"], predictions["T1_latent_linear"]
    ):
        sample_index = t4_run["sample_index"].astype(np.int64)
        if not np.array_equal(sample_index, t1_run["sample_index"]):
            raise ValueError(f"T4/T1 sample mismatch for seed {seed}")
        local_index = np.searchsorted(cache_sample_index, sample_index)
        if (
            np.any(local_index >= len(cache_sample_index))
            or not np.array_equal(cache_sample_index[local_index], sample_index)
        ):
            raise ValueError(f"T4/T1 cache sample mapping failed for seed {seed}")
        delta = np.abs(
            t4_run["cluster_prediction_mw"] - t4_run["cluster_target_mw"]
        ) - np.abs(
            t1_run["cluster_prediction_mw"] - t1_run["cluster_target_mw"]
        )
        t4_t1_seed_deltas.append(float(delta.mean()))

        offsets = offsets_all[local_index]
        phase = np.mod(
            np.rint(
                np.mod(LEAD_HOURS[None, :] - offsets[:, :1], 1.0) * 4.0
            ).astype(np.int64),
            4,
        )
        for phase_id, phase_label in PHASE_LABELS.items():
            t4_t1_stratified_rows.append(
                {
                    "seed": seed,
                    "stratum_type": "within_hour_phase",
                    "stratum": phase_label,
                    "t4_minus_t1_mw": float(delta[phase == phase_id].mean()),
                    "points": int(np.sum(phase == phase_id)),
                }
            )
        for band, indices in HORIZON_BANDS.items():
            t4_t1_stratified_rows.append(
                {
                    "seed": seed,
                    "stratum_type": "horizon",
                    "stratum": band,
                    "t4_minus_t1_mw": float(delta[:, indices].mean()),
                    "points": int(delta[:, indices].size),
                }
            )
        sample_variation = val_variation[cycles_all[local_index]]
        variation_id = np.digitize(sample_variation, thresholds)
        for regime_id, regime in enumerate(("low", "medium", "high")):
            selected = variation_id == regime_id
            t4_t1_stratified_rows.append(
                {
                    "seed": seed,
                    "stratum_type": "forcing_variation",
                    "stratum": regime,
                    "t4_minus_t1_mw": float(delta[selected].mean()),
                    "points": int(delta[selected].size),
                }
            )
        ages = ages_all[local_index]
        for age in (15, 30, 45, 60):
            selected = ages == age
            t4_t1_stratified_rows.append(
                {
                    "seed": seed,
                    "stratum_type": "nwp_age_minutes",
                    "stratum": str(age),
                    "t4_minus_t1_mw": float(delta[selected].mean()),
                    "points": int(delta[selected].size),
                }
            )
    write_csv(output / "t4_vs_t1_stratified_seedwise.csv", t4_t1_stratified_rows)
    t4_t1_aggregate = aggregate_stratified(
        t4_t1_stratified_rows, "t4_minus_t1_mw"
    )
    write_csv(output / "t4_vs_t1_stratified_summary.csv", t4_t1_aggregate)
    t4_t1_point, t4_t1_low, t4_t1_high = mean_ci(
        np.asarray(t4_t1_seed_deltas)
    )
    t4_t1_phase = {
        row["stratum"]: row
        for row in t4_t1_aggregate
        if row["stratum_type"] == "within_hour_phase"
    }
    t4_t1_non_anchor = float(
        np.mean(
            [
                t4_t1_phase[label]["candidate_minus_reference_mean_mw"]
                for label in ("+15min", "+30min", "+45min")
            ]
        )
    )

    t4 = next(row for row in comparison_rows if row["comparison"].startswith("T4_"))
    t4_phase = {
        row["stratum"]: row
        for row in aggregate_rows
        if row["group"] == "T4_native_query"
        and row["stratum_type"] == "within_hour_phase"
    }
    non_anchor = float(
        np.mean(
            [t4_phase[label]["candidate_minus_t0_mean_mw"] for label in ("+15min", "+30min", "+45min")]
        )
    )
    decision = {
        "protocol": "3-seed validation screening; no 2024 test used",
        "primary_comparison": "T4_native_query_vs_T0_raw_linear",
        "overall_delta_mw": t4["candidate_minus_t0_mean_mw"],
        "overall_ci95": [t4["ci95_low_mw"], t4["ci95_high_mw"]],
        "seed_wins": t4["candidate_seed_wins"],
        "anchor_delta_mw": t4_phase["anchor"]["candidate_minus_t0_mean_mw"],
        "non_anchor_delta_mean_mw": non_anchor,
        "go_cross_resolution_story": bool(
            t4["candidate_minus_t0_mean_mw"] <= -1.0
            and t4["candidate_seed_wins"] >= 2
            and non_anchor < 0.0
        ),
        "go_rule": "T4 improves >=1 MW overall, wins >=2/3 seeds, and improves non-anchor phases",
        "forcing_variation_train_tertiles": thresholds.tolist(),
        "native_query_vs_latent_interpolation": {
            "comparison": "T4_native_query_vs_T1_latent_linear",
            "overall_delta_mw": t4_t1_point,
            "overall_ci95": [t4_t1_low, t4_t1_high],
            "seed_differences_mw": t4_t1_seed_deltas,
            "seed_wins": int(sum(value < 0 for value in t4_t1_seed_deltas)),
            "anchor_delta_mw": t4_t1_phase["anchor"][
                "candidate_minus_reference_mean_mw"
            ],
            "non_anchor_delta_mean_mw": t4_t1_non_anchor,
            "native_query_needed": bool(
                t4_t1_point <= -1.0
                and sum(value < 0 for value in t4_t1_seed_deltas) >= 2
                and t4_t1_non_anchor < 0.0
            ),
            "screening_rule": "T4 improves >=1 MW over T1, wins >=2/3 seeds, and improves non-anchor phases",
        },
    }
    if not decision["go_cross_resolution_story"]:
        decision["scientific_position"] = (
            "Do not use temporal cross-resolution as a method contribution; "
            "simple raw-space interpolation is sufficient under this screen."
        )
    elif decision["native_query_vs_latent_interpolation"]["native_query_needed"]:
        decision["scientific_position"] = (
            "Native-path target-time querying is supported beyond both raw- and "
            "latent-space interpolation; continuous-coordinate follow-up is justified."
        )
    else:
        decision["scientific_position"] = (
            "Raw-space interpolation is insufficient, but native-path attention is "
            "not necessary beyond latent response-space alignment; do not yet claim "
            "a continuous-query architecture contribution."
        )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "comparisons": comparison_rows,
                "focused_pairwise": focused_overall_rows,
                "focused_pairwise_stratified_summary": focused_stratified_rows,
                "stratified_summary": aggregate_rows,
                "t4_vs_t1_stratified_summary": t4_t1_aggregate,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
