"""Common observed-state groups, thresholds learned from 2022 only."""
from settings import *
from residual_inputs import topology, mapping_history
import numpy as np

def main():
    bounds = {}
    for split in ['train', 'val']:
        h = np.load(CACHE/split/'station_history.npy', mmap_mode='r')
        p = np.asarray(h[:, :, -2]); delta = p-np.asarray(h[:, :, -6])
        labels = np.zeros((len(h), 56, 3), dtype='i1')
        for site in range(56):
            ids, geo, valid, capacity = topology(site, True)
            weight = capacity*valid; weight = weight/max(weight.sum(), 1e-12)
            regional_delta = (delta[:, ids]*weight).sum(1)
            local_delta = np.asarray(h[:, site, -1]-h[:, site, -5])
            innovation = p[:, site]-mapping_history(split, site)[0][:, -2]
            if split == 'train':
                bounds[str(site)] = dict(ramp=float(np.quantile(abs(regional_delta), .75)),
                                         innovation=float(np.quantile(abs(innovation), .5)))
            b = bounds[str(site)]
            labels[:, site, 0] = np.where(regional_delta < -b['ramp'], 0,
                                         np.where(regional_delta > b['ramp'], 2, 1))
            labels[:, site, 1] = np.where(regional_delta*local_delta < 0, 0,
                                         np.where(regional_delta*local_delta > 0, 2, 1))
            labels[:, site, 2] = abs(innovation) > b['innovation']
        np.save(ASSETS/f'regional_labels_{split}.npy', labels)
        print('regional labels ready', split, labels.shape, flush=True)
    atomic_json(ASSETS/'REGIONAL_DIAGNOSTICS_READY.json', dict(timestamp=now(),
        fit_year=2022, selection_year=2023, threshold_source='2022 only', bounds=bounds,
        groups=['regional_change_1h', 'local_regional_trend', 'own_mapping_innovation'],
        interpretation='mapping innovation includes operational/mapping errors; no pure NWP-error claim',
        opens_2024=False))

if __name__ == '__main__': main()
