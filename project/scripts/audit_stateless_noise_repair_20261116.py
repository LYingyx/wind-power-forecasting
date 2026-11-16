"""Exhaustive validation-noise regression and failed-checkpoint recovery checks."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import types

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--skip-exhaustive", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / "code"))
    from representation import stateless_noise
    from data import SharedData
    from model import SharedModel, configure
    configure()
    original = (run / "monitoring/noise_repair/representation_before.txt").read_text()
    before = types.ModuleType("original_representation")
    exec(compile(original, "original_representation", "exec"), before.__dict__)
    origin = np.load(run / "data/val/origin.npy")
    keys = (origin[:, None] // 900 * 55 + np.arange(55)[None]).reshape(-1)
    report = dict(timestamp=datetime.datetime.now().astimezone().isoformat(),
                  device=args.device, validation_samples=len(keys), seeds=[], checkpoints=[],
                  original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  repaired_sha256=hashlib.sha256((run / "code/representation.py").read_bytes()).hexdigest())
    if not args.skip_exhaustive:
        for seed in range(42, 47):
            bad = []
            for start in range(0, len(keys), 8192):
                key = torch.tensor(keys[start:start+8192], device=args.device, dtype=torch.long)
                old = before.stateless_noise(key, 8, seed, torch.float32)
                new = stateless_noise(key, 8, seed, torch.float32)
                finite = torch.isfinite(old)
                assert torch.isfinite(new).all()
                assert torch.equal(old[finite], new[finite]), "Originally finite noise changed"
                for a, b, c, d in (~finite).nonzero().cpu().tolist():
                    bad.append(dict(index=start+a, key=int(keys[start+a]), sample=b, segment=c, channel=d))
            report["seeds"].append(dict(seed=seed, repaired_coordinates=bad,
                                        originally_finite_values_identical=True, all_repaired_finite=True))
            print("seed", seed, "repaired coordinates", bad, flush=True)
    special = torch.tensor([102483043, 102277694, int(keys[0]), int(keys[-1])],
                           device=args.device, dtype=torch.long)
    torch.manual_seed(87654)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    for seed in range(42, 47):
        whole = stateless_noise(special, 8, seed, torch.float32)
        split = torch.cat([stateless_noise(k[None], 8, seed, torch.float32) for k in special])
        assert torch.equal(whole, split)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    report.update(batch_invariant=True, training_rng_unchanged=True)
    for jid in args.checkpoints:
        ck = torch.load(run / "jobs" / jid / "latest.pt", map_location="cpu", weights_only=False)
        spec = ck["spec"]
        model = SharedModel(spec).to(args.device)
        model.load_state_dict(ck["model"])
        model.eval()
        assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
        index = {44: 245853, 46: 50954}[spec["seed"]]
        data = SharedData("val", spec)
        batch, _, _, _ = data.batch(np.array([index, index+1]), args.device)
        with torch.no_grad():
            noise = before.stateless_noise(batch["sample_key"], 8, spec["seed"], torch.float32)
            assert not torch.isfinite(noise).all(), "Original failure did not reproduce"
            samples = model.prediction_samples(batch)
            assert torch.isfinite(samples).all(), "Failed checkpoint still invalid"
        report["checkpoints"].append(dict(job=jid, epoch=ck["epoch"], cursor=ck["cursor"],
                                          parameters_finite=True, failing_batch_now_finite=True))
        print("checkpoint passed", jid, flush=True)
        del model, data, ck
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("all checks passed", str(args.output), flush=True)


if __name__ == "__main__":
    main()
