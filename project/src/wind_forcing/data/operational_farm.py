"""Single-farm BASE/SEL inputs from the operational bundle, with provenance guards.

No path to the former local-weather cache, prepared histories, or feature stats
is used. Regional observations can be read from this same cache's history array.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from .operational import interpolate_nodes, PROTOCOLS


def _physics():
    project = Path(__file__).resolve().parents[3]
    path = project / 'runs/regional_reliability_validation_20261027_v1/code/physics.py'
    spec = importlib.util.spec_from_file_location('operational_physics_functions', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperationalFarmData:
    def __init__(self, cache_dir, split, site, feature_set='SEL', stats=None, device='cpu', limit=None):
        if split not in ['train', 'val', 'test'] or feature_set not in ['BASE', 'SEL']:
            raise ValueError('Supported splits: train/val/test; features: BASE/SEL')
        root = Path(cache_dir).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('operational_protocol', {}).get('protocol') not in PROTOCOLS:
            raise ValueError('An operational data manifest is required')
        bundle = root.parent
        if not (bundle / 'SUCCESS.json').exists():
            raise ValueError('Operational dataset completion audit has not passed')
        fingerprint = manifest['preprocess_fingerprint']
        if stats is None and split != 'train':
            raise ValueError('Validation/test must use training feature statistics')
        if stats is not None:
            if stats.get('data_fingerprint') != fingerprint or stats.get('feature_set') != feature_set:
                raise ValueError('Feature statistics are from another data protocol or feature set')
        if not 0 <= site < 56:
            raise ValueError('Site index must be 0..55')
        source = root / split
        prepared = Path(manifest['physics_dir'])
        origin = np.load(source / 'origin_unix.npy')[:limit]
        cycle = np.load(source / 'weather_cycle_index.npy')[:limit]
        offset = np.load(source / 'weather_time_offset_hours.npy')[:limit]
        h = np.array(np.load(source / 'station_history.npy', mmap_mode='r')[:limit, site])
        y = np.array(np.load(source / 'station_future.npy', mmap_mode='r')[:limit, site])
        nodes = np.load(Path(manifest['local_weather_dir']) / f'{split}.npy', mmap_mode='r')[cycle, site]
        past = np.load(prepared / split / 'past_weather.npy', mmap_mode='r')[cycle, site]
        past_time = (np.load(prepared / split / 'past_valid_unix.npy')[cycle] - origin[:, None]) / 3600
        times = np.concatenate([past_time, offset], 1).astype('f4')
        allnodes = np.concatenate([past, nodes], 1)
        lead = np.arange(1, 25, dtype='f4') / 4
        weather = interpolate_nodes(nodes, offset, lead)
        reference = interpolate_nodes(nodes, offset, np.array([0.]))[:, 0]
        mean = np.load(root / 'static/nwp_mean.npy')
        std = np.load(root / 'static/nwp_std.npy')
        physical = weather * std + mean
        ref_physical = reference * std + mean
        physics = _physics()
        elevation = float(np.load(prepared / 'elevation.npy')[site])
        density, stability = physics.thermodynamics(physical, elevation)
        ref_density, _ = physics.thermodynamics(ref_physical, elevation)
        dyn_nodes = np.load(prepared / split / 'dynamics.npy', mmap_mode='r')[cycle, site]
        dynamics = interpolate_nodes(dyn_nodes, offset, lead)
        if stats is None:
            train_h = np.load(root / 'train/station_history.npy', mmap_mode='r')[:, site]
            flat = float(np.quantile(abs(np.diff(train_h, axis=1)), .1))
        else:
            flat = stats['flat_threshold']
        state = physics.operating_state(h, flat)
        repeated = np.broadcast_to(state[:, None], (len(h), 24, state.shape[-1]))
        raw = np.concatenate([density, dynamics, stability, repeated], -1) if feature_set == 'SEL' else np.zeros((len(h), 24, 0), 'f4')
        if stats is None:
            mu = raw.mean((0, 1), dtype=np.float64).astype('f4') if raw.shape[-1] else np.zeros(0, 'f4')
            sd = raw.std((0, 1), dtype=np.float64).astype('f4').clip(1e-4) if raw.shape[-1] else np.ones(0, 'f4')
            stats = dict(data_fingerprint=fingerprint, feature_set=feature_set, fitted_split='train',
                site=site, samples=len(h), mean=mu.tolist(), std=sd.tolist(), flat_threshold=flat,
                groups=['density', 'dynamics', 'stability', 'state'] if feature_set == 'SEL' else [])
        if stats.get('site') != site or stats.get('fitted_split') != 'train':
            raise ValueError('Feature statistics must come from this farm training split')
        x = (raw - np.asarray(stats['mean'], 'f4')) / np.asarray(stats['std'], 'f4')
        assert np.isfinite(x).all()
        tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.array(a), dtype=dtype, device=device)
        self.n = len(h)
        self.site, self.split, self.origin, self.stats = site, split, origin, stats
        self.h, self.y = tensor(h), tensor(y)
        self.phase = tensor(np.load(source / 'origin_phase.npy')[:limit], torch.long)
        for name, array in dict(weather=weather, reference=reference, physical=physical,
                reference_physical=ref_physical, density=density, ref_density=ref_density, x=x,
                times=offset, all_times=times, allnodes=allnodes,
                historical_input=np.concatenate([h, past.reshape(len(h), -1), past_time / 6], 1)).items():
            setattr(self, name, tensor(array))
        self.diagnostic = dict(density=tensor(density), stability=tensor(stability),
            dynamics=tensor(dynamics), state=tensor(repeated), calendar=tensor(physics.calendar(origin, lead)))
        self.data_fingerprint = fingerprint
        self.issue_unix = np.load(source / 'origin_issue_unix.npy')[:limit]
        self.available_unix = np.load(source / 'origin_available_unix.npy')[:limit]

    def batch(self, indices):
        return {key: getattr(self, key)[indices] for key in ['h', 'phase', 'weather', 'reference', 'physical',
            'reference_physical', 'density', 'ref_density', 'x', 'times', 'all_times', 'allnodes', 'historical_input']}
