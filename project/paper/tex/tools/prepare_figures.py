"""Copy the reviewed GPT image into the paper; this script does not draw."""
from pathlib import Path
import shutil

paper = Path(__file__).resolve().parents[1]
source = paper.parents[1] / "output/imagegen/single_farm_framework_v19/framework_v19_final.png"
if not source.is_file():
    raise FileNotFoundError(source)
shutil.copy2(source, paper / "figures/architecture_single_farm_v19.png")
print(paper / "figures/architecture_single_farm_v19.png")
