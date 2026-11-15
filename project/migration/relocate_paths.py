from __future__ import annotations

import os
from pathlib import Path


root = Path(os.environ["DEST_ROOT"])
env_root = Path(os.environ["ENV_ROOT"])
old_root = os.environ["OLD_ROOT"]
old_python = os.environ["OLD_PYTHON"]

for base in (
    root / "artifacts/cross_farm_shared_specific_20260920_v1",
    root / "artifacts/selected30_fewshot_baselines_20260920_v1",
):
    if not base.exists():
        continue
    for path in base.rglob("*.py"):
        text = path.read_text()
        rewritten = text.replace(old_root, str(root)).replace(old_python, str(env_root / "bin/python"))
        if rewritten != text:
            path.write_text(rewritten)

