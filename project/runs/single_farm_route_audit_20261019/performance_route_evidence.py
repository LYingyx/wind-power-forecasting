"""Summarize known-farm performance without selecting farms by test gains."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from audit import paired

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREV = ROOT / "runs/single_farm_review_20261018"
BASELINES = ["TRANSFORMER_FC", "TIDE_FC", "NBEATSX_FC", "TCN_FC", "TFT_FC",
             "TIMEXER_FC", "DLINEAR_FC", "GRU_H", "LSTM_H", "PATCHTST_H",
             "ITRANSFORMER_H", "MODERNTCN_H", "CNN_W", "VIT_W", "SWIN_W"]


def main():
    frame = pd.read_csv(OUT / "source_all56.csv", dtype={"station": str})
    val = frame[frame.split.eq("val")].set_index("station")
    test = frame[frame.split.eq("test")].set_index("station")
    assert len(val) == len(test) == 56
    rows = []
    for split, data in [("val", val), ("test", test)]:
        for model in ["E2C", "CBH_SL"]:
            for baseline in BASELINES:
                row = paired(data[model] - data[baseline], f"{model}-{baseline}", split)
                row.update(model=model, baseline=baseline, model_mae=data[model].mean(),
                           baseline_mae=data[baseline].mean(),
                           gain_pct=100 * (1 - data[model].mean() / data[baseline].mean()))
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "source_advantage_vs_all15.csv", index=False)

    # Retrospective robustness comparison; family chosen on 2023 only.
    selected = val[BASELINES].idxmin(axis=1)
    result = test[["station_name", "capacity_mw", "E2C", "CBH_SL"]].copy()
    result["selected_baseline_2023"] = selected
    result["selected_baseline_mae_2024"] = [test.loc[s, b] for s, b in selected.items()]
    for model in ["E2C", "CBH_SL"]:
        result[f"{model}_gain_pct"] = 100 * (1 - result[model] / result.selected_baseline_mae_2024)
    result.reset_index().to_csv(OUT / "per_farm_validation_selected_baseline.csv", index=False)
    contrasts = [paired(result[m] - result.selected_baseline_mae_2024,
                        f"{m}-per_farm_validation_selected_baseline") for m in ["E2C", "CBH_SL"]]

    selection = json.loads((PREV / "farm_selection.json").read_text())
    ids = selection["source56"]["selected"]
    assert not selection["source56"]["uses_prediction_errors"]
    reps = test.loc[ids, ["station_name", "capacity_mw", "TRANSFORMER_FC", "E2C", "CBH_SL"]].copy()
    reps["e2c_gain_pct"] = 100 * (1 - reps.E2C / reps.TRANSFORMER_FC)
    reps["local_cbh_gain_pct"] = 100 * (1 - reps.CBH_SL / reps.TRANSFORMER_FC)
    reps.reset_index().to_csv(OUT / "representative_performance_advantage.csv", index=False)

    # Existing archived lead metrics; no new inference or selection by lead.
    leads = pd.read_csv(PREV / "tables/all_models_lead_seed.csv")
    leads = leads[leads.model.isin(["E0", "E2C", "TRANSFORMER_FC", "TIDE_FC", "NBEATSX_FC"])].copy()
    leads["horizon"] = pd.cut(leads.lead_hours, [0, 1, 3, 6], labels=["0-1h", "1-3h", "3-6h"])
    assert leads.groupby(["model", "split", "seed"]).size().eq(24).all()
    horizon = leads.groupby(["split", "horizon", "model"], observed=True).mae.mean().unstack()
    horizon["e2c_vs_transformer_gain_pct"] = 100 * (1 - horizon.E2C / horizon.TRANSFORMER_FC)
    horizon.reset_index().to_csv(OUT / "source_horizon_advantage.csv", index=False)
    metrics = pd.read_csv(PREV / "tables/source_full_year_system_comparison.csv")
    metrics[metrics.model.isin(["E2C", "E0", "TRANSFORMER_FC", "TIDE_FC", "NBEATSX_FC"])].to_csv(
        OUT / "source_mae_rmse_r2.csv", index=False)
    summary = dict(route="known-farm cross-year individual power forecasting; shared training",
                   main_model="E2C-CBH with regional history; local-input CBH-SL control",
                   held_out_transfer="excluded from main task and future experiment priorities; original evidence retained",
                   per_farm_val_selected_family_counts=selected.value_counts().to_dict(),
                   per_farm_val_selected_baseline_test_mae=result.selected_baseline_mae_2024.mean(),
                   paired_comparisons=contrasts,
                   representative_farms=ids,
                   representative_gain_range_pct=[reps.e2c_gain_pct.min(), reps.e2c_gain_pct.max()],
                   limitations=["2024 is a previously examined cross-year evaluation, not a new blind test.",
                                "E2C uses regional history and FP32; shared baselines use local inputs and historical BF16.",
                                "Per-farm baseline family selection is a retrospective robustness analysis on 2023, not preregistered.",
                                "New comparisons aggregate previously audited metrics; no new training or inference."])
    (OUT / "performance_route_evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(horizon.to_string())


if __name__ == "__main__":
    main()
