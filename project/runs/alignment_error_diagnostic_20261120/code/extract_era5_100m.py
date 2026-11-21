"""Extract ERA5 100 m u/v (2022-2023, hourly) on the same 24x25 Jilin crop. Read-only on source GRIB."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np, eccodes as ec
from common import ERA5_SURF, OUT, T0, T1, LAT, LON, save_json

ROWS = slice(int(round((90 - LAT[0]) / .25)), int(round((90 - LAT[-1]) / .25)) + 1)
COLS = slice(int(round(LON[0] / .25)), int(round(LON[-1] / .25)) + 1)


def read_day(day):
    out = np.full((24, 2, 24, 25), np.nan, np.float32)
    for k, var in enumerate(['100m_u_component_of_wind', '100m_v_component_of_wind']):
        path = ERA5_SURF / var / f'{day:%Y}' / f'{day:%Y-%m-%d}.grib'
        with open(path, 'rb') as f:
            while True:
                g = ec.codes_grib_new_from_file(f)
                if g is None:
                    break
                try:
                    assert ec.codes_get(g, 'Ni') == 1440 and ec.codes_get(g, 'Nj') == 721
                    assert ec.codes_get(g, 'latitudeOfFirstGridPointInDegrees') == 90.0
                    assert ec.codes_get(g, 'longitudeOfFirstGridPointInDegrees') == 0.0
                    assert int(ec.codes_get(g, 'dataDate')) == int(f'{day:%Y%m%d}')
                    h = int(ec.codes_get(g, 'dataTime')) // 100
                    vals = ec.codes_get_values(g).reshape(721, 1440)
                    out[h, k] = vals[ROWS, COLS]
                finally:
                    ec.codes_release(g)
    return day, out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    n = (T1 - T0) // 3600
    arr = np.lib.format.open_memmap(OUT / 'era5_100m.npy', mode='w+', dtype=np.float32, shape=(n, 2, 24, 25))
    days = [datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(n // 24)]
    with ProcessPoolExecutor(24) as pool:
        for day, block in pool.map(read_day, days):
            i = (int(day.timestamp()) - T0) // 3600
            arr[i:i + 24] = block
    arr.flush()
    save_json('ERA5_100M_READY.json', dict(hours=int(n), nan_hours=int(np.isnan(arr[:, 0, 0, 0]).sum()),
                                          rows=[ROWS.start, ROWS.stop], cols=[COLS.start, COLS.stop]))
    print('done', np.isnan(arr[:, 0, 0, 0]).sum())
