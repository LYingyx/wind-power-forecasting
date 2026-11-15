"""Package active manuscript sources and audited table inputs; exclude obsolete drafts."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

paper = Path(__file__).resolve().parents[1]
active = ['main.tex','preamble.tex','sections/introduction.tex','sections/method.tex',
          'sections/results.tex','sections/discussion.tex','sections/appendix.tex']
members = set(active + ['build.sh','README_CN.md','AUTHOR_NOTES_CN.md',
                        'RESULTS_PROVENANCE_CN.md','BUILD_CHECK.json','VISUAL_REVIEW.json'])
for rel in active:
    text = (paper / rel).read_text()
    for match in re.finditer(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        members.add('figures/'+match[1])
members.update(str(p.relative_to(paper)) for p in (paper/'data/single_farm').glob('*') if p.is_file())
members.update(str(p.relative_to(paper)) for p in (paper/'figures/imagegen_provenance').glob('*') if p.is_file())
members.update('tools/'+name for name in ['package_source.py','prepare_figures.py',
                                         'update_single_farm_tables.py','verify_manuscript.py'])
manifest={
    'main_file':'main.tex', 'compiler':'XeLaTeX with ctex and Fandol',
    'scope':'Chinese single-farm manuscript: abstract, introduction, method, results, discussion, conclusion, appendices. Author metadata and literature references remain to be supplied.',
    'source_files':{rel:hashlib.sha256((paper/rel).read_bytes()).hexdigest() for rel in sorted(members)},
}
(paper/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
members.add('source_manifest.json')
for name in ['manuscript_source.zip','arxiv_source.zip']:
    with zipfile.ZipFile(paper/name,'w',zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(members): z.write(paper/rel,rel)
    with zipfile.ZipFile(paper/name) as z:
        assert z.testzip() is None
print(f'Packaged {len(members)} active files to {paper / "manuscript_source.zip"}')
