"""Availability-aware Pangu selection. All stored timestamps are UTC Unix seconds."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

HOUR = 3600
LATENCY = 12 * HOUR
PERIOD = 12 * HOUR
LOCAL = ZoneInfo('Asia/Shanghai')
BAD_START = int(datetime(2024, 11, 1, tzinfo=LOCAL).timestamp())
BAD_END = int(datetime(2024, 12, 1, tzinfo=LOCAL).timestamp())
BAD_UTC_START = int(datetime(2024, 11, 1, tzinfo=timezone.utc).timestamp())
BAD_UTC_END = int(datetime(2024, 12, 1, tzinfo=timezone.utc).timestamp())


def unix(value: str, timezone_name: str = 'Asia/Shanghai') -> int:
    """Localize naive power timestamps; preserve timezone-aware NWP timestamps."""
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ZoneInfo(timezone_name))
    return int(stamp.timestamp())


def select_issue(origin):
    origin = np.asarray(origin, dtype=np.int64)
    return ((origin - LATENCY) // PERIOD) * PERIOD


def plan_times(origin):
    origin = np.asarray(origin, dtype=np.int64)
    if np.any(origin % 900):
        raise ValueError('Power origins must lie on the 15-minute time grid')
    issue = select_issue(origin)
    base = (origin // HOUR) * HOUR
    future = base[:, None] + np.arange(8, dtype=np.int64)[None] * HOUR
    past = base[:, None] + np.arange(-6, 0, dtype=np.int64)[None] * HOUR
    available = issue + LATENCY
    assert np.all(available <= origin) and np.all(origin < available + PERIOD)
    assert np.all(issue % PERIOD == 0)
    assert np.all(future[:, 0] <= origin + 900)
    assert np.all(future[:, -1] >= origin + 6 * HOUR)
    assert np.all((future - issue[:, None]) <= 30 * HOUR)
    return dict(issue=issue, available=available, base=base, future=future, past=past)


def november_exclusions(origin):
    """Conservative exclusion of local power/weather windows and bad issue files.

    Covers the UTC filename month as well as the business calendar month. This
    also prevents December boundary samples from reusing a November issue.
    """
    origin = np.asarray(origin, dtype=np.int64)
    p = plan_times(origin)
    overlaps = lambda a, b: (a < BAD_END) & (b >= BAD_START)
    return {
        'power_window_touches_local_november': overlaps(origin - 23 * 900, origin + 24 * 900),
        'weather_window_touches_local_november': overlaps(p['past'][:, 0], p['future'][:, -1]),
        'issue_in_utc_november': (p['issue'] >= BAD_UTC_START) & (p['issue'] < BAD_UTC_END),
        'issue_in_local_november': (p['issue'] >= BAD_START) & (p['issue'] < BAD_END),
    }


def source_path(root: str | Path, issue: int, lead: int) -> Path:
    if lead < 0 or lead > 96:
        raise ValueError(f'Unsupported Pangu lead: {lead}')
    stamp = datetime.fromtimestamp(int(issue), tz=timezone.utc)
    step = f'{int(lead):02d}'
    return Path(root) / str(stamp.year) / step / f'{stamp:%Y-%m-%d %H:%M:%S}+00:00-{step}.npy'


def interpolate_nodes(nodes, offsets, queries):
    """Hourly linear interpolation without silent endpoint extrapolation."""
    nodes = np.asarray(nodes)
    offsets = np.asarray(offsets)
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = np.broadcast_to(queries, (len(nodes), len(queries)))
    if not np.allclose(np.diff(offsets, axis=1), 1):
        raise ValueError('Expected contiguous hourly nodes')
    x = queries - offsets[:, :1]
    if np.any(x < -1e-6) or np.any(x > nodes.shape[1] - 1 + 1e-6):
        raise ValueError('Required forecast valid time is outside available nodes')
    lo = np.minimum(np.floor(x).astype(np.int64), nodes.shape[1] - 2)
    frac = (x - lo)[..., None]
    rows = np.arange(len(nodes))[:, None]
    return (nodes[rows, lo] * (1 - frac) + nodes[rows, lo + 1] * frac).astype(np.float32)
