from settings import *

def freeze():
    assert json.loads((RUN/'CHECKS.json').read_text())['passed']
    assert json.loads((RUN/'SMOKE_CHECK.json').read_text())['passed']
    assert json.loads((RUN/'analysis/DIAGNOSTIC_METRIC_REPLAY.json').read_text())['passed']
    files=list((RUN/'code').glob('*.py'))+list((PREVIOUS/'code').rglob('*.py'))
    files += [RUN/'manifest.json',RUN/'PROTOCOL.json',PREPARED/'train/READY.json',PREPARED/'val/READY.json']
    files += list((CACHE/'static').glob('*.npy'))
    files += [COLUMN_DATA/'station_dem_srtm30m.json',COLUMN_DATA/'column_train_stats.json']
    atomic_json(RUN/'SOURCE_FREEZE.json',dict(timestamp=now(),files={str(p):sha(p) for p in sorted(set(files))},purpose='screening protocol/code freeze, not 2024 model selection freeze'))
    print('frozen',len(set(files)),'source/config/static files')
if __name__=='__main__':freeze()
