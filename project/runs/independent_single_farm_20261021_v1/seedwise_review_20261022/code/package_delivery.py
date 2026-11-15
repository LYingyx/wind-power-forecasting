"""Package the current, validated deliverables, including independent audits."""
import hashlib,json,zipfile
from pathlib import Path
O=Path(__file__).resolve().parents[1]
for fn in ['VALIDATION.json','INDEPENDENT_VALIDATION.json','HTML_VALIDATION.json']:
    assert json.loads((O/fn).read_text())['passed'],fn
entries=[]
for p in sorted(O.rglob('*')):
    if not p.is_file() or p.name in ['MANIFEST.json','RESULTS_BUNDLE.zip'] or p.suffix=='.log' or '__pycache__' in p.parts:continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    entries.append(dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=h.hexdigest()))
(O/'MANIFEST.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2))
with zipfile.ZipFile(O/'RESULTS_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for x in entries:
        z.write(O/x['path'],arcname=x['path'],compress_type=zipfile.ZIP_STORED if x['path'].endswith(('.gz','.xlsx')) else zipfile.ZIP_DEFLATED)
    z.write(O/'MANIFEST.json',arcname='MANIFEST.json')
with zipfile.ZipFile(O/'RESULTS_BUNDLE.zip') as z:
    assert len(z.infolist())==len(entries)+1
    assert z.testzip() is None
print(json.dumps(dict(files=len(entries),archive_bytes=(O/'RESULTS_BUNDLE.zip').stat().st_size,zip_crc_passed=True)))
