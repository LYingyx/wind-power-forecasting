from pathlib import Path
import numpy as np
import pytest
from wind_forcing.data.operational import (
    unix, select_issue, plan_times, november_exclusions, source_path, interpolate_nodes,
)


@pytest.mark.parametrize('origin,issue', [
    ('2023-06-02 07:45:00', '2023-06-01 00:00:00+00:00'),
    ('2023-06-02 08:00:00', '2023-06-01 12:00:00+00:00'),
    ('2023-06-02 19:45:00', '2023-06-01 12:00:00+00:00'),
    ('2023-06-02 20:00:00', '2023-06-02 00:00:00+00:00'),
    ('2024-01-01 00:00:00', '2023-12-31 00:00:00+00:00'),
])
def test_business_release_boundaries(origin, issue):
    assert int(select_issue(unix(origin))) == unix(issue)


def test_all_phases_and_age_and_target_coverage():
    origins = unix('2023-06-01 00:00:00') + np.arange(192) * 900
    p = plan_times(origins)
    assert set((origins - p['issue']) // 60) == set(range(720, 1440, 15))
    assert np.max((origins + 21600 - p['issue']) / 3600) == 29.75
    assert np.max((p['future'] - p['issue'][:, None]) / 3600) == 30
    assert np.min((p['past'] - p['issue'][:, None]) / 3600) == 6
    # Values are valid-time hours, so every quarter-hour interpolation has known truth.
    nodes = p['future'][..., None] / 3600
    offsets = (p['future'] - origins[:, None]) / 3600
    q = np.arange(1, 25) / 4
    result = interpolate_nodes(nodes, offsets, q)
    np.testing.assert_allclose(result[..., 0], origins[:, None] / 3600 + q, rtol=0, atol=.04)
    with pytest.raises(ValueError, match='outside'):
        interpolate_nodes(nodes, offsets, np.array([8.]))


def test_no_double_timezone_shift_and_step_path():
    assert unix('2023-06-01 08:00:00') == unix('2023-06-01 00:00:00+00:00')
    assert source_path('/archive', unix('2022-12-31 23:00:00+00:00'), 86) == Path(
        '/archive/2022/86/2022-12-31 23:00:00+00:00-86.npy')
    assert source_path('/archive', unix('2022-12-31 23:00:00+00:00'), 6).name.endswith('-06.npy')


def test_november_and_cross_month_contamination():
    values = ['2024-10-31 16:45:00', '2024-10-31 18:00:00', '2024-11-15 12:00:00',
              '2024-12-01 08:00:00', '2024-12-01 19:45:00', '2024-12-01 20:00:00']
    excluded = november_exclusions(np.array([unix(x) for x in values]))
    bad = np.logical_or.reduce(list(excluded.values()))
    np.testing.assert_array_equal(bad, [False, True, True, True, True, False])
