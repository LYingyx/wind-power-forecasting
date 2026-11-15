"""Direct weather-projection replacements for the frozen P0I/CBH backbone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wind_forcing.models.p0i_transfer import P0IConditionTransfer

CHANNELS = ("msl", "u10", "v10", "t2m", "z1000", "z925", "q1000", "q925",
            "t1000", "t925", "u1000", "u925", "v1000", "v925")
G0 = 9.80665
KAPPA = 0.2854


def column_coordinates(physical: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    """Invertible 14-weather-plus-height representation, ordered in three groups."""
    p = physical
    elev = elevation
    theta1000 = p[..., 8]
    theta925 = p[..., 9] * (1000.0 / 925.0) ** KAPPA
    return torch.stack((
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], elev,
        p[..., 10] - p[..., 1], p[..., 12] - p[..., 2], theta1000, p[..., 6],
        p[..., 4] / G0 - elev,
        p[..., 11] - p[..., 10], p[..., 13] - p[..., 12], theta925 - theta1000,
        p[..., 7] - p[..., 6], (p[..., 5] - p[..., 4]) / G0,
    ), dim=-1)


def invert_column_coordinates(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Used by the numeric information-preservation test."""
    v = values
    elev = v[..., 4]
    u1000, v1000 = v[..., 1] + v[..., 5], v[..., 2] + v[..., 6]
    theta1000 = v[..., 7]
    z1000 = (elev + v[..., 9]) * G0
    p = torch.stack((
        v[..., 0], v[..., 1], v[..., 2], v[..., 3], z1000,
        z1000 + v[..., 14] * G0, v[..., 8], v[..., 8] + v[..., 13],
        theta1000, (theta1000 + v[..., 12]) / ((1000.0 / 925.0) ** KAPPA),
        u1000, u1000 + v[..., 10], v1000, v1000 + v[..., 11],
    ), dim=-1)
    return p, elev


class WeatherProjection(nn.Module):
    def __init__(self, variant: str, static: dict, dem_path: str, stats_path: str, width: int):
        super().__init__()
        if variant not in {"E1", "E2L", "E2N", "E2", "E2C", "E3"}:
            raise ValueError(variant)
        if tuple(map(str, static["nwp_channel_name"])) != CHANNELS:
            raise ValueError("Unexpected NWP channel ordering")
        nwp_mean = np.asarray(static["nwp_mean"], dtype=float)
        if not (8e4 < nwp_mean[0] < 1.2e5 and 180 < nwp_mean[3] < 330
                and 180 < nwp_mean[8] < 330 and 180 < nwp_mean[9] < 330
                and 0 < nwp_mean[6] < .1 and 0 < nwp_mean[7] < .1
                and 0 < nwp_mean[4] < nwp_mean[5] < 2e4):
            raise ValueError("Weather physical-unit sanity check failed")
        dem = json.loads(Path(dem_path).read_text())
        records = dem["records"]
        ids = list(map(str, static["station_id"]))
        if len(records) != len(ids) or [r["station_id"] for r in records] != ids:
            raise ValueError("DEM station ordering differs from training cache")
        heights = np.array([r["terrain_elevation_m"] for r in records], dtype=np.float32)
        if not np.isfinite(heights).all():
            raise ValueError("DEM contains nonfinite heights")
        stats = json.loads(Path(stats_path).read_text())
        if stats["station_ids"] != ids or stats["weather_channels"] != list(CHANNELS):
            raise ValueError("Feature statistics do not match cache")
        self.variant = variant
        self.register_buffer("elevation", torch.as_tensor(heights).reshape(1, -1, 1))
        self.register_buffer("nwp_mean", torch.as_tensor(static["nwp_mean"], dtype=torch.float32))
        self.register_buffer("nwp_std", torch.as_tensor(static["nwp_std"], dtype=torch.float32))
        self.register_buffer("column_mean", torch.as_tensor(stats["mean"], dtype=torch.float32))
        self.register_buffer("column_std", torch.as_tensor(stats["std"], dtype=torch.float32))
        self.register_buffer("elevation_mean", torch.tensor(stats["mean"][4], dtype=torch.float32))
        self.register_buffer("elevation_std", torch.tensor(stats["std"][4], dtype=torch.float32))
        if variant == "E1":
            self.network = nn.Linear(15, width)
        elif variant == "E2L":
            # Exact depth/parameter-count control for E2N.  With no
            # intervening activation, the three affine maps compose into one
            # affine weather projection despite the overparameterization.
            self.network = nn.Sequential(nn.Linear(14, width), nn.Linear(width, width),
                                         nn.Linear(width, width))
        elif variant == "E2N":
            # Capacity-matched nonlinear control with exactly the original
            # 14 normalized weather channels.  This arm changes only the
            # linear-vs-MLP encoder factor relative to E0; it deliberately
            # does not expose terrain elevation to the model.
            self.network = nn.Sequential(nn.Linear(14, width), nn.GELU(), nn.Linear(width, width),
                                         nn.GELU(), nn.Linear(width, width))
        elif variant in {"E2", "E2C"}:
            self.network = nn.Sequential(nn.Linear(15, width), nn.GELU(), nn.Linear(width, width),
                                         nn.GELU(), nn.Linear(width, width))
        else:
            if width != 64:
                raise ValueError("E3 group widths are specified for d_model=64")
            self.surface = nn.Sequential(nn.Linear(5, 16), nn.GELU())
            self.lower = nn.Sequential(nn.Linear(5, 24), nn.GELU())
            self.upper = nn.Sequential(nn.Linear(5, 24), nn.GELU())
            self.network = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        if weather.ndim != 4 or weather.shape[-1] != 14 or weather.shape[1] != self.elevation.shape[1]:
            raise ValueError(f"Expected [B,N,T,14], received {tuple(weather.shape)}")
        elev = self.elevation.to(weather).expand(weather.shape[:3])
        if self.variant in {"E2L", "E2N"}:
            return self.network(weather)
        if self.variant in {"E1", "E2"}:
            value = torch.cat((weather, ((elev - self.elevation_mean) / self.elevation_std)[..., None]), -1)
            return self.network(value)
        physical = weather.float() * self.nwp_std + self.nwp_mean
        column = column_coordinates(physical, elev.float())
        value = ((column - self.column_mean) / self.column_std).to(weather)
        if self.variant == "E2C":
            return self.network(value)
        merged = torch.cat((self.surface(value[..., :5]), self.lower(value[..., 5:10]),
                            self.upper(value[..., 10:])), -1)
        return self.network(merged)


class ColumnWeatherCBH(P0IConditionTransfer):
    def __init__(self, cfg: dict, static: dict, forecast_steps: int = 24):
        super().__init__(cfg, static, forecast_steps)
        self.native_projection = WeatherProjection(
            cfg["weather_encoder_variant"], static, cfg["station_dem_path"],
            cfg["column_stats_path"], int(cfg["d_model"]),
        )


def install_builder() -> None:
    """Experiment-local registration; does not edit the frozen source snapshot."""
    import wind_forcing.models as models

    original = models.build_forecast_model

    def build(cfg, static, *, history_steps=24, forecast_steps=24):
        if cfg.get("weather_encoder_variant") in {"E1", "E2L", "E2N", "E2", "E2C", "E3"}:
            return ColumnWeatherCBH(cfg, static, forecast_steps)
        return original(cfg, static, history_steps=history_steps, forecast_steps=forecast_steps)

    models.build_forecast_model = build
