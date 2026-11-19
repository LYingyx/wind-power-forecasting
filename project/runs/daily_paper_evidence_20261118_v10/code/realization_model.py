"""State-conditioned modulation of forecast changes; frozen v2 backbones."""
from parent_model import SharedModel as Parent, configure, seed_all
from representation import lerp, kernels
from settings import ASSETS
import numpy as np
import torch
from torch import nn


class Realization(nn.Module):
    """Two scalar responses per hourly segment, never a nonlinear latent-weather input."""
    def __init__(self, use_state=True):
        super().__init__()
        self.use_state = use_state
        # State 17 + two wind-speed changes at each of two levels + time 1.
        # noS retains exactly the same parameter count and receives zero state.
        self.network = nn.Sequential(nn.Linear(22, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, c, wind_features, offset):
        c = c if self.use_state else torch.zeros_like(c)
        time = (offset[:, :-1] + .5)[..., None] / 6
        x = torch.cat([c[:, None].expand(-1, 7, -1), wind_features, time], -1)
        segment = .5 * self.network(x).tanh()
        return kernels(offset) @ segment


class SharedModel(Parent):
    def __init__(self, spec):
        super().__init__(spec)
        self.generative = False
        self.variant = spec.get('realization', 'none')
        self.register_buffer('wind_mean', torch.from_numpy(np.load(ASSETS/'nwp_mean.npy')))
        self.register_buffer('wind_std', torch.from_numpy(np.load(ASSETS/'nwp_std.npy')))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(spec['seed'] + 20261117)
            if self.variant != 'none':
                self.realization = Realization(spec.get('use_state', True))
                if self.variant == 'additive':
                    # Same conditioning information, but no multiplication by D.
                    # This control has 2*d additional parameters, reported explicitly.
                    self.state_basis = nn.Linear(2, self.d, bias=False)

    @staticmethod
    def speed(physical):
        return torch.stack([torch.linalg.vector_norm(physical[..., [1, 2]], dim=-1),
                            torch.linalg.vector_norm(physical[..., [10, 12]], dim=-1)], -1)

    def weather_terms(self, b):
        projection = self.weather_projection if self.family == 'W0' else self.original.w_in.weather
        # Preserve the reference's numerical interpolation order at beta=0.
        base = projection(b['weather'])
        origin_weather = lerp(b['hourly'], b['offset'], b['offset'].new_zeros(1))
        origin = projection(origin_weather)
        physical = b['hourly'] * self.wind_std + self.wind_mean
        origin_speed = self.speed(origin_weather*self.wind_std+self.wind_mean)
        midpoint_speed = self.speed((physical[:, :-1]+physical[:, 1:])/2)
        segment_change = self.speed(physical[:, 1:])-self.speed(physical[:, :-1])
        features = torch.cat([midpoint_speed-origin_speed, segment_change], -1)/10
        # Forecast wind direction, never realized future power direction.
        wind_change = self.speed(b['physical'])-origin_speed
        return base, origin, features, wind_change

    def modified_weather(self, b):
        base, origin, features, wind_change = self.weather_terms(b)
        pair = self.realization(b['state_c'], features, b['offset'])
        if self.variant == 'additive':
            return base+self.state_basis(pair)
        if self.variant == 'symmetric':
            beta = pair.mean(-1, keepdim=True)
        else:
            # u1000/v1000 speed determines which scalar branch is used.
            direction = wind_change[..., 1:2]
            beta = torch.where(direction >= 0, pair[..., :1], pair[..., 1:])
        return base+beta*(base-origin)

    def forward(self, b):
        if self.variant == 'none':
            return super().forward(b)
        return self.decode(b, self.modified_weather(b))
