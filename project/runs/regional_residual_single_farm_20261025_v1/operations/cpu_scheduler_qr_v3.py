"""Use the verified QR ARX recovery within the existing resource-aware CPU queue."""
from pathlib import Path
import sys, argparse
OPS = Path(__file__).resolve().parent
sys.path.insert(0,str(OPS));sys.path.insert(0,str(OPS/'arx_recovery_20261027'))
import cpu_scheduler_v2 as scheduler
import sparse_arx
from recover_arx import run
sparse_arx.run = run
scheduler.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job-id');parser.add_argument('--max-workers',type=int,default=16);parser.add_argument('--reference-slots',type=int,default=4)
    args=parser.parse_args()
    scheduler.worker(args.job_id) if args.job_id else scheduler.schedule(args)
