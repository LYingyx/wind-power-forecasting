"""Verify that the CBH power loss reaches each proxy-weather generator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_original_cbh_20260917_v1")
V2 = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_named_models_20260917_v2")
sys.path.insert(0, str(ROOT / "code"))

from artifact_utils import dump_json, sha256  # noqa: E402
from cbh_adapter import CBHAdapterSR  # noqa: E402
from train_adapter import CBHData  # noqa: E402


METHODS = ["DET", "ICFM", "VPCFM", "SINE_SI", "VP_EPS", "VP_V", "EDM", "MEANFLOW"]


def main() -> None:
    device = "cuda:5"
    seed = 42
    station = 0
    station_id = "1043019"
    data = CBHData("train", station, [seed], device)
    indices = torch.arange(8, device=device)
    grouped = data.batch(indices[None])
    inputs = tuple(value[0] for value in grouped)
    target = data.y[indices]
    rows = []
    for method in METHODS:
        pretrain = V2 / "results" / f"TiDE_SR_{method}" / station_id / f"seed{seed}" / "pretrained_generator.pt"
        payload = torch.load(pretrain, map_location="cpu", weights_only=False)
        model = CBHAdapterSR(method, payload["scale"], seed, station_index=17).to(device)
        assert model.generator is not None
        model.generator.load_state_dict(payload["model_state"])
        model.eval()
        random = torch.Generator(device=device).manual_seed(20260917)
        generated_noise = torch.randn(8, 6, 3, 14, device=device, generator=random)
        zeros = torch.zeros(8, 3, 3, 14, device=device)
        zero_time = torch.zeros(8, 3, device=device)
        prediction, _ = model(*inputs, generated_noise, zeros, zero_time, zeros)
        loss = (prediction - target).abs().mean()
        loss.backward()
        gradient_l1 = float(sum(
            parameter.grad.abs().sum()
            for parameter in model.generator.parameters()
            if parameter.grad is not None
        ))
        if not np.isfinite(gradient_l1) or gradient_l1 <= 0:
            raise RuntimeError(f"No downstream gradient for {method}: {gradient_l1}")
        rows.append({
            "method": method,
            "power_loss": float(loss),
            "downstream_power_to_generator_gradient_l1": gradient_l1,
            "pretrained_generator_sha256": sha256(pretrain),
        })
        del model
        torch.cuda.empty_cache()
    dump_json(ROOT / "status" / "CBH_GENERATOR_GRADIENT_AUDIT.json", {
        "passed": True,
        "scope": "power MAE only; auxiliary weather-proxy loss excluded from backward",
        "station_id": station_id,
        "seed": seed,
        "batch_size": len(indices),
        "checks": rows,
    })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
