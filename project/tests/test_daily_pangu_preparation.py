"""Prevent missing or unaudited daily forecasts from becoming active training data."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_operational_pangu.py'
    spec = importlib.util.spec_from_file_location('daily_preparation_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = module.operational.DAILY
    return module


def test_missing_source_stops_before_allocating_weather_store(builder, tmp_path):
    out = tmp_path / 'bundle'
    out.mkdir()
    keys = [(builder.operational.unix('2021-12-31 00:00:00+00:00'), 45)]
    with pytest.raises(FileNotFoundError, match='Pangu files unavailable'):
        builder.extract_raw(out, tmp_path / 'unmounted_source', keys, 1)
    assert not (out / 'raw_frames.npy').exists()
    assert not (out / 'SUCCESS.json').exists()
    coverage = json.loads((out / 'SOURCE_COVERAGE.json').read_text())
    assert coverage['missing_count'] == 1 and not coverage['root_accessible']
    assert '/2021/45/' in coverage['missing'][0]


def completed_bundle(builder, tmp_path):
    builder.PROJECT = tmp_path / 'project'
    (builder.PROJECT / 'data').mkdir(parents=True)
    out = tmp_path / 'bundle'
    (out / 'cache').mkdir(parents=True)
    identity = {'protocol': builder.PROTOCOL.name, 'root': '/archive'}
    builder.atomic_json(out / 'cache/manifest.json', {
        'preprocess_fingerprint': 'new-daily-fingerprint', 'operational_protocol': identity})
    builder.atomic_json(out / 'SUCCESS.json', {'preprocess_fingerprint': 'new-daily-fingerprint'})
    builder.atomic_json(out / 'AUDIT.json', {
        'passed': True, 'manifest_sha256': builder.digest(out / 'cache/manifest.json')})
    active = builder.PROJECT / 'data/ACTIVE_DATASET.json'
    active.write_text('{"protocol":"old"}\n')
    config = {'cache_dir': str(builder.PROJECT / 'data/daily_cache')}
    return out, config, active


@pytest.mark.parametrize('audit', [None, {'passed': True, 'preprocess_fingerprint': 'old-fingerprint'}])
def test_daily_activation_rejects_absent_or_stale_independent_audit(builder, tmp_path, audit):
    out, config, active = completed_bundle(builder, tmp_path)
    if audit is not None:
        builder.atomic_json(out / 'INDEPENDENT_AUDIT.json', audit)
    before = active.read_bytes()
    with pytest.raises(ValueError, match='audit'):
        builder.activate_bundle(out, config)
    assert active.read_bytes() == before
    assert not Path(config['cache_dir']).exists()


def test_daily_activation_uses_only_matching_completed_audit(builder, tmp_path):
    out, config, active = completed_bundle(builder, tmp_path)
    builder.atomic_json(out / 'INDEPENDENT_AUDIT.json', {
        'passed': True, 'preprocess_fingerprint': 'new-daily-fingerprint'})
    builder.activate_bundle(out, config)
    data = json.loads(active.read_text())
    assert data['protocol'] == builder.PROTOCOL.name
    assert data['preprocess_fingerprint'] == 'new-daily-fingerprint'
    assert Path(data['cache_dir']).resolve() == out / 'cache'
    assert data['require_new_training_and_normalization']
