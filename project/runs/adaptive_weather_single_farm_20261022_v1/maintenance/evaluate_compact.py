"""Invoke the unchanged frozen evaluator, then compact only completed outputs."""
from pathlib import Path
import sys,argparse,json,traceback
RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
from evaluate import run
from config import atomic_json,now
from storage import compact
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);args=ap.parse_args();spec=json.loads(Path(args.spec).read_text())
 try:run(spec)
 except BaseException:
  atomic_json(RUN/'evaluation'/spec['id']/'FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
 # A compaction failure must not relabel a successful scientific evaluation as failed.
 try:compact(RUN/'evaluation'/spec['id'])
 except BaseException:
  atomic_json(RUN/'evaluation'/spec['id']/'STORAGE_FAILURE.json',dict(timestamp=now(),traceback=traceback.format_exc()));raise
