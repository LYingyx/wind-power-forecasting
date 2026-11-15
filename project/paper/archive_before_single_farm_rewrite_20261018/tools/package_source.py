"""Package only the self-contained files required to compile main.tex."""
from pathlib import Path
import hashlib
import json
import zipfile

paper = Path(__file__).resolve().parents[1]
members = [
    "main.tex", "preamble.tex", "sections/introduction.tex", "sections/method.tex",
    "sections/results.tex", "figures/architecture_overview.png",
]
with zipfile.ZipFile(paper / "arxiv_source.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    for relative in members:
        path = paper / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        bundle.write(path, relative)
manifest = {
    "main_file": "main.tex",
    "compiler": "xelatex (TeX Live with ctex and Fandol)",
    "source_files": {
        relative: hashlib.sha256((paper / relative).read_bytes()).hexdigest()
        for relative in members
    },
    "scope": "Chinese Introduction, Method and Results draft; author metadata, abstract and references are not supplied.",
}
(paper / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(paper / "arxiv_source.zip")
