import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_original_cbh_20260917_v1")
sys.path.insert(0, str(ROOT / "code"))

from cbh_adapter import CBHAdapterSR  # noqa: E402
from train_adapter import CBHData, Ensemble, evaluate_samples  # noqa: E402
from paths import V2  # noqa: E402


def main():
    device = "cuda:5"
    station = 5
    seeds = list(range(42, 52))
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    data = CBHData("val", station, seeds, device)
    report = torch.tensor(np.load(V2 / "data/val_report_indices.npy"), device=device)
    models = [CBHAdapterSR("LINEAR", torch.ones(14), seed, 31) for seed in seeds]
    ensemble = Ensemble(models, device)
    vmap_prediction = evaluate_samples(ensemble, data, report, "LINEAR", 1)[:, :, 0]
    for row, (seed, model) in enumerate(zip(seeds, models)):
        model.to(device).eval()
        direct = []
        for begin in range(0, len(report), 128):
            selected = report[begin : begin + 128]
            index = selected[None]
            values = data.batch(index)
            zeros = torch.zeros(len(selected), 6, 3, 14, device=device)
            aux = torch.zeros(len(selected), 3, 3, 14, device=device)
            aux_t = torch.zeros(len(selected), 3, device=device)
            pred, _ = model(*(value[0] for value in values), zeros, aux, aux_t, aux)
            direct.append(pred.cpu())
        direct = torch.cat(direct)
        original = torch.tensor(np.load(ROOT / "original_eval" / f"seed{seed}" / "predictions_fp32.npy")[:, station])
        print(seed, "direct-original", float((direct-original).abs().max()), "vmap-direct", float((vmap_prediction[row]-direct).abs().max()), "vmap-original", float((vmap_prediction[row]-original).abs().max()))


if __name__ == "__main__":
    main()
