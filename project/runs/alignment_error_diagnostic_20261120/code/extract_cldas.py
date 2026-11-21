"""Extract CLDAS-2.0 (CMA land data assimilation, 0.0625 deg hourly analysis) over Jilin, 2022-2023.
Variables: WIN (10 m speed), TMP, PRS, RHU(QAIR), SRA(SWDN), PRE. Crop LAT idx 656..764, LON idx 963..1076.
2022 = retrospective product P_CLDAS_RE01; 2023 = real-time CLDAS_RT_ASI (to 2023-10-09) and NRT Z_NAFP_..._NRT
(from 2023-10-10, produced ~48 h after valid time); RT/NRT name humidity SHU and radiation SSRA.
Output output/cldas_crop.npy float32 [17520, 6, 109, 114] indexed by file-name hour (time zone verified separately)."""
import numpy as np, netCDF4 as nc, os, glob, sys
from multiprocessing import Pool
from common import *
SRC = '/shared_data_hdd1/All_Datasets/CLDAS_2.0'
VARS = [('WIN', 'WIND'), ('TMP', 'TAIR'), ('PRS', 'PAIR'), ('RHU', 'QAIR'), ('SRA', 'SWDN'), ('PRE', 'PRCP')]
I0, I1, J0, J1 = 656, 765, 963, 1077
n = (T1 - T0) // 3600


def one(h):
    import time
    t = time.gmtime(T0 + h * 3600)
    stamp = time.strftime('%Y%m%d%H', t)
    out = np.full((len(VARS), I1 - I0, J1 - J0), np.nan, np.float32)
    for k, (tag, var) in enumerate(VARS):
        dd = f'{SRC}/{stamp[:4]}/{stamp[:8]}/'
        rt = {'RHU': 'SHU', 'SRA': 'SSRA'}.get(tag, tag)
        cands = [dd + f'P_CLDAS_RE01_EA16_{tag}_HOUR_{stamp}.nc', dd + f'CLDAS_RT_ASI_0P0625_HOR-{rt}-{stamp}.nc'] +             sorted(glob.glob(dd + f'Z_NAFP_C_BABJ_*_P_CLDAS_NRT_ASI_0P0625_HOR-{rt}-{stamp}.nc'))
        p = next((q for q in cands if os.path.exists(q)), cands[0])
        try:
            with nc.Dataset(p) as f:
                x = f[var][I0:I1, J0:J1]
                out[k] = np.ma.filled(x.astype(np.float32), np.nan)
        except Exception:
            pass
    return h, out


if __name__ == '__main__':
    fill = 'fill' in sys.argv     # only re-read hours whose WIN is missing
    if fill:
        mm = np.load(OUT / 'cldas_crop.npy', mmap_mode='r+')
        todo = np.flatnonzero(np.isnan(mm[:, 0, 54, 57]))
    else:
        mm = np.lib.format.open_memmap(OUT / 'cldas_crop.npy', 'w+', np.float32, (n, len(VARS), I1 - I0, J1 - J0)); todo = range(n)
    with nc.Dataset(f'{SRC}/2022/20220101/P_CLDAS_RE01_EA16_WIN_HOUR_2022010100.nc') as f:
        np.save(OUT / 'cldas_lat.npy', np.asarray(f["LAT"][I0:I1])); np.save(OUT / 'cldas_lon.npy', np.asarray(f["LON"][J0:J1]))
    with Pool(32) as pool:
        for i, (h, x) in enumerate(pool.imap_unordered(one, todo, chunksize=16)):
            mm[h] = x
            if i % 2000 == 0: print(i, flush=True)
    mm.flush()
    miss = np.isnan(mm[:, :, 50, 50]).mean(0)
    print('missing frac per var', dict(zip([v[0] for v in VARS], miss.round(4).tolist())))
