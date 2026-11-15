"""Independent calendar and valid-time checks for the manuscript protocol."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from wind_forcing.data.operational import (
    DAILY, TWICE_DAILY, select_issue, plan_times, november_exclusions,
    interpolate_nodes, source_path, unix,
)


@pytest.mark.parametrize('origin,expected_issue', [
    ('2023-06-01 22:45:00', '2023-05-31 00:00:00+00:00'),
    ('2023-06-01 23:00:00', '2023-06-01 00:00:00+00:00'),
    ('2023-06-02 10:00:00', '2023-06-01 00:00:00+00:00'),
    ('2023-06-02 22:45:00', '2023-06-01 00:00:00+00:00'),
    ('2022-01-01 14:15:00', '2021-12-31 00:00:00+00:00'),
    ('2024-03-01 22:45:00', '2024-02-29 00:00:00+00:00'),
])
def test_release_and_year_leap_boundaries(origin, expected_issue):
    assert int(select_issue(unix(origin), DAILY)) == unix(expected_issue)


@pytest.mark.parametrize('year,days', [(2022, 365), (2023, 365), (2024, 366)])
def test_full_year_against_local_calendar_rule(year, days):
    bj = ZoneInfo('Asia/Shanghai')
    start = datetime(year, 1, 1, tzinfo=bj)
    origins = int(start.timestamp()) + np.arange(days * 96) * 900
    expected = []
    for t in origins:
        local = datetime.fromtimestamp(int(t), bj)
        issue_date = local.date() if local.hour >= 23 else local.date() - timedelta(days=1)
        expected.append(int(datetime(issue_date.year, issue_date.month, issue_date.day,
                                     tzinfo=timezone.utc).timestamp()))
    p = plan_times(origins, DAILY)
    np.testing.assert_array_equal(p['issue'], expected)
    switches = np.flatnonzero(np.diff(p['issue'])) + 1
    assert len(switches) == days
    assert np.all((origins[switches] + 8 * 3600) % 86400 == 23 * 3600)
    assert np.all(p['available'] <= origins)
    assert np.all(origins < p['available'] + 86400)
    assert set((origins - p['issue']) // 60) == set(range(900, 2340, 15))


def test_forecast_nodes_cover_quarter_hour_targets_without_extrapolation():
    origin = unix('2023-06-01 00:00:00') + np.arange(192) * 900
    p = plan_times(origin, DAILY)
    future_leads = (p['future'] - p['issue'][:, None]) // 3600
    past_leads = (p['past'] - p['issue'][:, None]) // 3600
    assert (future_leads.min(), future_leads.max()) == (15, 45)
    # Six strict historical nodes are floor(t)-6h through floor(t)-1h.
    assert (past_leads.min(), past_leads.max()) == (9, 37)
    offsets = (p['future'] - origin[:, None]) / 3600
    nodes = ((p['future'] - origin[0]) / 3600)[..., None]
    q = np.arange(1, 25) / 4
    actual = interpolate_nodes(nodes, offsets, q)[..., 0]
    np.testing.assert_allclose(actual, (origin[:, None] - origin[0]) / 3600 + q,
                               rtol=0, atol=1e-6)
    assert (origin + 21600 - p['issue']).max() / 3600 == 44.75


def test_missing_previous_year_must_not_select_newer_forecast():
    p = plan_times(np.array([unix('2022-01-01 14:15:00')]), DAILY)
    path = source_path('/archive', int(p['issue'][0]), 45)
    assert str(path) == '/archive/2021/45/2021-12-31 00:00:00+00:00-45.npy'


def test_november_issue_exclusion_ends_at_daily_release():
    values = ['2024-12-01 08:00:00', '2024-12-01 22:45:00', '2024-12-01 23:00:00']
    bad = np.logical_or.reduce(list(november_exclusions(
        np.array([unix(x) for x in values]), DAILY).values()))
    np.testing.assert_array_equal(bad, [True, True, False])


def test_old_protocol_is_not_silently_relabelled():
    t = unix('2023-06-02 20:00:00')
    assert int(select_issue(t)) == int(select_issue(t, TWICE_DAILY))
    assert int(select_issue(t)) != int(select_issue(t, DAILY))
