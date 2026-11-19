"""Power-supervised temporal queries conditioned on the complete past trajectory."""
from realization_model import SharedModel as RealizationModel, configure, seed_all
import torch
from torch import nn


class HistoryRefinement(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode = mode
        self.history = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(16*24, 16), nn.GELU())
        # 17 state + 16 history + 30 wind changes + 8 node times + 2 query times.
        self.network = nn.Sequential(nn.Linear(73, 32), nn.GELU(), nn.Linear(32, 2))
        self.basis = nn.Linear(2, width, bias=False)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, h, c, window, offset):
        if getattr(self, 'history_order', 'original') == 'reverse_past':
            h = torch.cat([h[:, :-1].flip(-1), h[:, -1:]], -1)
        if self.mode in ['summary', 'no_state']:
            h = torch.zeros_like(h)
        if self.mode == 'no_state':
            c = torch.zeros_like(c)
        if getattr(self, 'state_only_off', False):
            c = torch.zeros_like(c)
        clock = torch.arange(-23, 1, device=h.device, dtype=h.dtype)/24
        delta = torch.diff(h, dim=-1, prepend=h[:, :1])
        encoded = self.history(torch.stack([
            h-h[:, -1:], delta, clock[None].expand_as(h)], 1))
        lead = torch.arange(1, 25, device=h.device, dtype=h.dtype)/4
        windows = window[:, None].expand(-1, 24, -1)
        if getattr(self, 'window_mode', 'full') == 'none':
            windows = torch.cat([torch.zeros_like(windows[..., :30]), windows[..., 30:]], -1)
        if getattr(self, 'window_mode', 'full') == 'local':
            segment = ((lead[None, :, None] >= offset[:, None, :]).sum(-1)-1).clamp(0, 6)
            node = torch.arange(8, device=h.device)[None, None]
            pair = ((node == segment[..., None]) | (node == segment[..., None]+1))
            diff = torch.arange(7, device=h.device)[None, None] == segment[..., None]
            mask = torch.cat([pair.repeat_interleave(2, -1), diff.repeat_interleave(2, -1),
                              torch.ones_like(pair)], -1)
            windows = windows * mask
        context = torch.cat([c[:, None].expand(-1, 24, -1),
                             encoded[:, None].expand(-1, 24, -1), windows], -1)
        phase = (lead[None]-offset[:, :1]).remainder(1)
        time = torch.stack([lead[None].expand(len(h), -1)/6, phase], -1)
        coefficients = .5*self.network(torch.cat([
            context, time], -1)).tanh()
        return self.basis(coefficients)


class SharedModel(RealizationModel):
    def __init__(self, spec):
        self.history_variant = spec.get('refinement', False)
        super().__init__(dict(spec, realization='none') if self.history_variant else spec)
        self.spec = spec
        if self.history_variant:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(spec['seed']+20261118)
                self.refiner = HistoryRefinement(self.d, spec['history_mode'])

    def window_features(self, b):
        from representation import lerp
        physical = b['hourly']*self.wind_std+self.wind_mean
        speed = self.speed(physical)
        origin = lerp(b['hourly'], b['offset'], b['offset'].new_zeros(1))
        origin_speed = self.speed(origin*self.wind_std+self.wind_mean)
        return torch.cat([(speed-origin_speed).flatten(1)/10,
                          torch.diff(speed, dim=1).flatten(1)/10,
                          b['offset']/6], -1)

    def modified_weather(self, b):
        if not self.history_variant:
            return super().modified_weather(b)
        projection = self.weather_projection if self.family == 'W0' else self.original.w_in.weather
        base = projection(b['weather'])
        return base+self.refiner(b['refinement_h'], b['state_c'],
                                 self.window_features(b), b['offset'])

    def forward(self, b):
        if not self.history_variant:
            return super().forward(b)
        return self.decode(b, self.modified_weather(b))
