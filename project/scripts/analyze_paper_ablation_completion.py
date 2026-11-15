from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from tqdm import tqdm

from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix
from scripts.analyze_incremental_mechanism_v2 import compare_groups, group_predictions, summarize_group


SEEDS = list(range(42, 52))
ARCHITECTURE = (
    "nearest_hour",
    "independent_weather_path",
    "no_wind_geometry",
    "fixed_bilinear",
    "no_aux_weather",
    "no_anchor",
    "frozen_baseline",
)
KD_TARGETS = ("kd_station_only", "kd_cluster_only", "kd_no_aux_weather", "kd_no_anchor")
HORIZON_BANDS = (("0.25-1h", slice(0, 4)), ("1.25-3h", slice(4, 12)), ("3.25-6h", slice(12, 24)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def paired_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    statistic = array.mean() / max(array.std(ddof=1) / np.sqrt(len(array)), 1e-12)
    return float(2 * student_t.sf(abs(statistic), len(array) - 1))


def apply_holm(rows: list[dict[str, Any]], family: str) -> None:
    indices = [index for index, row in enumerate(rows) if row["family"] == family]
    p = np.asarray([rows[index]["seedwise_p_two_sided"] for index in indices])
    order = np.argsort(p); adjusted = np.empty_like(p); running = 0.0
    for rank, local in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[local])); adjusted[local] = running
    for local, index in enumerate(indices): rows[index]["holm_p_within_family"] = float(adjusted[local])


def horizon_rows(candidate: Path, reference: Path, comparison: str, repeats: int, seed: int, split: str) -> list[dict[str, Any]]:
    _, cand = group_predictions(candidate, SEEDS, split); _, ref = group_predictions(reference, SEEDS, split)
    np.testing.assert_array_equal(cand["origin_unix"], ref["origin_unix"])
    delta = np.abs(cand["cluster_prediction_mw"] - cand["cluster_target_mw"]) - np.abs(ref["cluster_prediction_mw"] - ref["cluster_target_mw"])
    rows = []
    for index, (label, selection) in enumerate(HORIZON_BANDS):
        point, low, high, days = block_bootstrap_matrix(delta[:, selection].mean(axis=1, keepdims=True), cand["origin_unix"], repeats, seed + index)
        rows.append({"comparison": comparison, "horizon_band": label, "candidate_minus_reference_mw": float(point[0]), "ci95_low_mw": float(low[0]), "ci95_high_mw": float(high[0]), "utc_days": days})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the focused paper-ablation completion matrix")
    parser.add_argument("--root", default="artifacts/paper_ablation_completion_v1")
    parser.add_argument("--normal-reference", default="artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h")
    parser.add_argument("--kd-reference", default="artifacts/ensemble_stabilization_v1/groups/kd_050")
    parser.add_argument("--direct-reference", default="artifacts/incremental_mechanism_v2/groups/direct_d64")
    parser.add_argument("--full-history-reference", default="artifacts/incremental_mechanism_v2/groups/joint_full")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve(); groups_root = root / "groups"
    output = Path(args.output_dir).resolve() if args.output_dir else root / f"analysis_{args.split}"
    references = {
        "Direct": Path(args.direct_reference).resolve(),
        "FullHistory": Path(args.full_history_reference).resolve(),
        "Normal": Path(args.normal_reference).resolve(),
        "KD": Path(args.kd_reference).resolve(),
    }
    candidates = [("architecture", name, groups_root / name, references["Normal"]) for name in ARCHITECTURE]
    # A fixed bilinear sampler cannot use the learned wind-geometry features.  Its
    # clean primary contrast is therefore the otherwise matching no-geometry
    # model, not Normal (which would change two spatial mechanisms at once).
    candidates = [
        (family, name, path, groups_root / "no_wind_geometry" if name == "fixed_bilinear" else reference)
        for family, name, path, reference in candidates
    ]
    candidates += [("kd_targets", name, groups_root / name, references["KD"]) for name in KD_TARGETS]
    if args.skip_missing:
        marker = "COMPLETE" if args.split == "val" else "TEST_COMPLETE"
        candidates = [
            item
            for item in candidates
            if all((item[2] / f"seed{seed}/{marker}").is_file() for seed in SEEDS)
            and all((item[3] / f"seed{seed}/predictions_{args.split}.npz").is_file() for seed in SEEDS)
        ]
    else:
        marker = "COMPLETE" if args.split == "val" else "TEST_COMPLETE"
        missing = [str(path / f"seed{seed}/{marker}") for _, _, path, _ in candidates for seed in SEEDS if not (path / f"seed{seed}/{marker}").is_file()]
        missing += [str(reference / f"seed{seed}/predictions_{args.split}.npz") for _, _, _, reference in candidates for seed in SEEDS if not (reference / f"seed{seed}/predictions_{args.split}.npz").is_file()]
        if missing: raise FileNotFoundError("Missing formal ablations:\n" + "\n".join(missing))
    summaries = []
    for label, path in references.items():
        available = all(
            (path / f"seed{seed}/metrics_{args.split}.json").is_file()
            and (path / f"seed{seed}/predictions_{args.split}.npz").is_file()
            for seed in SEEDS
        )
        if not available and label == "FullHistory":
            continue
        if not available:
            raise FileNotFoundError(
                f"Incomplete {label} reference for split={args.split}: {path}"
            )
        row = summarize_group(path, SEEDS, args.split); row["group"] = label; row["source"] = "existing"; summaries.append(row)
    for family, name, path, _ in candidates:
        row = summarize_group(path, SEEDS, args.split); row["group"] = name; row["source"] = family; summaries.append(row)
    comparisons, horizons = [], []
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    leave = os.environ.get("WIND_PROGRESS_LEAVE", "1").lower() not in {"0", "false", "no"}
    candidate_progress = tqdm(
        candidates,
        total=len(candidates),
        desc=f"{prefix} ablation statistics".strip(),
        unit="comparison",
        dynamic_ncols=True,
        position=int(position_value) if position_value else None,
        leave=leave,
        disable=False,
    )
    for index, (family, name, path, reference) in enumerate(candidate_progress):
        candidate_progress.set_postfix_str(f"{name} vs {reference.name}", refresh=False)
        row = compare_groups(path, reference, SEEDS, args.bootstrap_repeats, 31000 + index, args.split)
        row.update({"comparison": f"{name}_vs_{reference.name}", "candidate": name, "reference": reference.name, "family": family})
        row["seedwise_p_two_sided"] = paired_p(row["seedwise_differences_mw"])
        comparisons.append(row)
        horizons.extend(horizon_rows(path, reference, row["comparison"], args.bootstrap_repeats, 32000 + 10 * index, args.split))
    for family in ("architecture", "kd_targets"): apply_holm(comparisons, family)
    decisions = []
    for row in comparisons:
        decisions.append({
            "comparison": row["comparison"],
            "candidate_needed": bool(row["seedwise_ci95_low_mw"] > 0 and row["ensemble_ci95_low_mw"] > 0 and row["holm_p_within_family"] < 0.05),
            "candidate_better": bool(row["seedwise_ci95_high_mw"] < 0 and row["ensemble_ci95_high_mw"] < 0 and row["holm_p_within_family"] < 0.05),
            "interpretation": "candidate_needed means removing/replacing the component is reliably worse",
        })
    write_csv(output / "summaries.csv", summaries); write_csv(output / "comparisons.csv", comparisons); write_csv(output / "horizon_bands.csv", horizons); write_csv(output / "decisions.csv", decisions)
    payload = {"protocol": {"split": args.split, "seeds": SEEDS, "bootstrap_repeats": args.bootstrap_repeats, "multiple_testing": "Holm within architecture and KD-target families", "negative_difference_means_candidate_better": True}, "summaries": summaries, "comparisons": comparisons, "horizon_bands": horizons, "decisions": decisions}
    output.mkdir(parents=True, exist_ok=True); (output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups_analyzed": len(candidates), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
