"""Copy the reviewed GPT image into the paper; this script does not draw."""
from pathlib import Path
import shutil

paper = Path(__file__).resolve().parents[1]
source = paper.parent / "output/imagegen/framework_research_v18/framework_research_v18_final.png"
if not source.is_file():
    raise FileNotFoundError(source)
shutil.copy2(source, paper / "figures/architecture_overview.png")
print(paper / "figures/architecture_overview.png")
