import subprocess,sys,time,traceback
from prepare_age import OUT,RUN,atomic,stamp

try:
    while not all((OUT/f'AGE{x}_READY.json').exists() for x in [12,24]):
        atomic(OUT/'PIPELINE_STATUS.json',dict(timestamp=stamp(),stage='extracting_older_forecasts',gpu_jobs=0))
        time.sleep(10)
    atomic(OUT/'PIPELINE_STATUS.json',dict(timestamp=stamp(),stage='paired_D2_analysis',gpu_jobs=0))
    with (OUT/'analysis.log').open('a') as f:
        subprocess.run([sys.executable,'-u',str(RUN/'code/analyze_age.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    atomic(OUT/'PIPELINE_STATUS.json',dict(timestamp=stamp(),stage='completed',gpu_jobs=0))
except Exception:
    atomic(OUT/'FAILURE.json',dict(timestamp=stamp(),traceback=traceback.format_exc()))
    raise
