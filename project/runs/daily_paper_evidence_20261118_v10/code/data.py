"""Same frozen samples as R0, with the eight available forecast nodes exposed."""
from base_data import SharedData as OriginalData
from settings import DATA
from physics import operating_state
import numpy as np
import torch


class SharedData(OriginalData):
    def __init__(self, split, spec, rows=None):
        super().__init__(split, spec, rows)
        self.wrong_state = spec.get('wrong_state', False)
        if self.wrong_state:
            self.state_donor = np.load(self.root / f'donor_r{spec["seed"]}.npy')
            self.donor_history = np.load(DATA/'train/h.npy', mmap_mode='r')
            self.donor_rings = np.load(DATA/'train/rings.npy', mmap_mode='r')

    def batch(self, indices, device='cpu'):
        inputs, y, local_si, ti = super().batch(indices, device)
        si = self.sites[local_si]
        cy = self.a['cycle'][ti]
        inputs['hourly'] = torch.as_tensor(np.array(self.a['nodes'][cy, si]), device=device)
        inputs['offset'] = torch.as_tensor(np.array(self.a['offset'][ti]), device=device, dtype=torch.float32)
        inputs['sample_key'] = torch.as_tensor(np.asarray(self.a['origin'][ti])//900*55+si,
                                             device=device, dtype=torch.long)
        h = np.array(self.a['h'][ti, si])
        rings = np.array(self.a['rings'][ti, si])
        if self.wrong_state:
            donor = self.state_donor[ti]
            h = np.array(self.donor_history[donor, si])
            rings = np.array(self.donor_rings[donor, si])
        state = operating_state(h, self.flat[si, None])
        state = (state-self.sm[si, 15:22])/self.ss[si, 15:22]
        rings = (rings-self.rm[si])/self.rs[si]
        # The unchanged 24-point history spans 23 intervals = 5.75 hours.
        c = np.concatenate([h[:, -1:], 1-h[:, -1:], (h[:, -1:]-h[:, -5:-4]),
                            (h[:, -1:]-h[:, :1])/5.75, state,
                            rings[:, [0, 2, 4, 6, 8, 10]]], axis=-1).astype('f4')
        inputs['state_c'] = torch.as_tensor(c, device=device)
        # Only the new module context uses donor history. Backbone h and labels
        # stay unchanged, including in the wrong-state control.
        inputs['refinement_h'] = (inputs['h'].clone() if self.spec.get('wrong_state_only')
                                  else torch.as_tensor(np.array(h), device=device))
        return inputs, y, local_si, ti
