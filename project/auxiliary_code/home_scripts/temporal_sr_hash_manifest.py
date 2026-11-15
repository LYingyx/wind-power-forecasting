from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_named_models_20260917_v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


code = sorted((ROOT / "code").glob("*.py"))
snapshot = sorted((ROOT / "vendor/final_code_snapshot").glob("*.py"))
code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code}
snapshot_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in snapshot}
matches = {
    path.name: sha256(path) == sha256(ROOT / "vendor/final_code_snapshot" / path.name)
    for path in code
}

support = [
    ROOT / "configs/protocol.json",
    ROOT / "requirements-lock.txt",
    ROOT / "data/selected_farms.json",
    ROOT / "status/FINAL_AUDIT.json",
    ROOT / "status/baseline_input_audit.json",
    ROOT / "status/generator_audit.json",
    ROOT / "status/proxy_condition_audit.json",
    ROOT / "analysis/FINAL_REPORT_CN.md",
    ROOT / "analysis/LITERATURE_SYNTHESIS_CN.md",
    ROOT / "analysis/METHOD_MATRIX_CN.md",
    ROOT / "analysis/RESEARCH_AND_PROTOCOL_CN.md",
    ROOT / "analysis/overall_seed_summary.csv",
    ROOT / "analysis/per_station_seed_summary.csv",
    ROOT / "analysis/paired_comparisons.csv",
    ROOT / "analysis/proxy_summary.csv",
]

payload = {
    "generated_at": datetime.now().astimezone().isoformat(),
    "hash_algorithm": "sha256",
    "code_file_count": len(code),
    "snapshot_file_count": len(snapshot),
    "all_snapshot_files_match_code": all(matches.values()) and len(code) == len(snapshot),
    "snapshot_matches": matches,
    "code": code_hashes,
    "snapshot": snapshot_hashes,
    "supporting_artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in support},
    "third_party_and_run_manifest": "references/corrected_run_source_hashes.json",
}

target = ROOT / "references/final_code_hashes.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
temporary.replace(target)
print(json.dumps({key: payload[key] for key in ["code_file_count", "snapshot_file_count", "all_snapshot_files_match_code"]}, indent=2))
