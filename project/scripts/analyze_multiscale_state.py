from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.analyze_final_route import (
        compare_groups,
        summarize_group,
        write_csv,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from analyze_final_route import (  # type: ignore[no-redef]
        compare_groups,
        summarize_group,
        write_csv,
    )


def analyze(
    experiment_root: Path,
    final_route_root: Path,
    paper_root: Path,
    output: Path,
    *,
    seeds: list[int],
    bootstrap_repeats: int,
) -> None:
    groups = {
        "selected_state": experiment_root / "confirmation" / "main_state",
        "matched_parallel": experiment_root / "confirmation" / "matched_parallel",
        "no_multiscale": experiment_root / "confirmation" / "no_multiscale",
        "no_ramp_loss": experiment_root / "confirmation" / "no_ramp_loss",
        "legacy_state": final_route_root / "candidates" / "m0_controlled_full",
        "legacy_parallel": paper_root / "controls" / "scratch_aux",
    }
    contrasts = (
        (
            "selected_state",
            "matched_parallel",
            "state_recurrence_under_matched_training",
        ),
        ("selected_state", "no_multiscale", "station_conditioned_spatial_scales"),
        ("selected_state", "no_ramp_loss", "quarter_hour_change_supervision"),
        ("selected_state", "legacy_state", "combined_route_update"),
        ("selected_state", "legacy_parallel", "updated_state_vs_legacy_parallel"),
    )
    output.mkdir(parents=True, exist_ok=True)
    group_rows = [summarize_group(name, path, seeds) for name, path in groups.items()]
    comparison_rows = [
        compare_groups(
            candidate,
            groups[candidate],
            reference,
            groups[reference],
            scientific_question=question,
            repeats=bootstrap_repeats,
            seed=72 + index,
        )
        for index, (candidate, reference, question) in enumerate(contrasts)
    ]
    write_csv(output / "validation_groups.csv", group_rows)
    write_csv(output / "paired_day_bootstrap.csv", comparison_rows)
    payload = {
        "selection_split": "val",
        "note": "Negative candidate_minus_reference_mae_mw favors the candidate.",
        "groups": group_rows,
        "comparisons": comparison_rows,
    }
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output / "analysis.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only analysis for multiscale controlled-state experiments"
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--final-route-root", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 2:
        raise ValueError("Analysis requires at least two seeds")
    analyze(
        Path(args.experiment_root).resolve(),
        Path(args.final_route_root).resolve(),
        Path(args.paper_root).resolve(),
        Path(args.output_dir).resolve(),
        seeds=seeds,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(f"[multiscale-state-analysis] output={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
