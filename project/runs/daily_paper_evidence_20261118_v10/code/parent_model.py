"""V2 changes only the weather-query path, plus the explicit R0+H control."""
from backbone import SharedModel as Backbone, configure, seed_all, zero_last
from representation import Window, Teacher, Flow, lerp, kernels, stateless_noise
import torch
from torch import nn


class WeatherTimeE2L(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weather = nn.Sequential(nn.Linear(14, d), nn.Identity(), nn.Linear(d, d),
                                     nn.Identity(), nn.Linear(d, d))
        self.time = nn.Linear(4, d, bias=False)

    def forward(self, x):
        return self.weather(x[..., :14])+self.time(x[..., 14:])


class SharedModel(Backbone):
    def __init__(self, spec):
        super().__init__(spec)
        self.representation = spec.get('representation', 'R0')
        self.generative = self.representation.startswith('R4')
        # Preserve all existing backbone parameters and its initialization RNG.
        with torch.random.fork_rng(devices=[]):
            # Seed CPU initialization only: torch.manual_seed also resets CUDA,
            # which fork_rng(devices=[]) intentionally does not snapshot.
            torch.default_generator.manual_seed(spec['seed']+20261116)
            if self.family == 'TF' and self.representation != 'TF_R3':
                self.original.w_in = WeatherTimeE2L(self.d)
            if self.representation == 'R0+H':
                old = self.head[0]
                new = nn.Linear(self.d+2, old.out_features)
                with torch.no_grad():
                    new.weight[:, :self.d].copy_(old.weight)
                    new.weight[:, self.d:].zero_()
                    new.bias.copy_(old.bias)
                self.head[0] = new
            if self.representation in ['R2', 'R3'] or self.generative:
                self.window = Window(self.d, self.representation not in ['R2', 'R4-noS'])
                self.residual_projection = nn.Linear(4, self.d, bias=False)
            if self.generative:
                self.teacher = Teacher(self.d)
                self.flow = Flow()
                # At initialization, stochastic models also exactly recover R1.
                nn.init.zeros_(self.residual_projection.weight)

    def latent(self, b):
        projection = self.weather_projection if self.family == 'W0' else self.original.w_in.weather
        z = projection(b['hourly'])
        return z, lerp(z, b['offset'])

    def decode(self, b, weather):
        h = b['h']; n = len(h)
        site = torch.cat([self.static[b['site']], self.embedding(b['site'])], -1)
        sv = self.site_projection(site)
        lead = torch.arange(1, 25, device=h.device, dtype=h.dtype)[None].expand(n, -1)/24
        extra = sv[:, None].expand(-1, 24, -1)
        if hasattr(self, 'forecast_window_projection'):
            extra = extra+self.forecast_window_projection(torch.cat([
                b['hourly'].flatten(1), b['offset']/6], -1))[:, None]
        if self.has_region:
            extra = extra+self.sel(b['x'])+self.regional(b['region'])[:, None]+self.neighbor(b)
        if self.family == 'W0':
            time = torch.stack([lead, torch.sin(2*torch.pi*lead/4), torch.cos(2*torch.pi*lead/4)], -1)
            context = torch.cat([time, site[:, None].expand(-1, 24, -1),
                                 b['phase'][:, None, None].to(h).expand(-1, 24, 1)/3], -1)
            clock = torch.arange(-23, 1, device=h.device, dtype=h.dtype).expand_as(h)/24
            memory = self.history_encoder(self.history_projection(torch.stack([h, clock], -1)))
            w = self.align_out(self.align_v(weather))
            q = self.context_projection(context)+w+self.condition_response(b, w, memory, lead)+extra
            hidden = self.decoder(q, memory)
            if self.representation == 'R0+H':
                level = torch.stack([h[:, -1], 1-h[:, -1]], -1)
                hidden = torch.cat([hidden, level[:, None].expand(-1, 24, -1)], -1)
            return h[:, -1:]+self.head(hidden).squeeze(-1)
        m = self.original
        clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype).expand_as(h)
        memory = m.h_enc(m.pe(m.h_in(torch.stack([h, clock], -1))+sv[:, None]))
        tf = torch.stack([lead, torch.sin(2*torch.pi*lead), torch.cos(2*torch.pi*lead),
                          b['phase'][:, None].to(h).expand(-1, 24)/3], -1)
        q = m.pe(weather+m.w_in.time(tf)+extra)
        out = m.dec(q, memory)
        return h[:, -1:]+m.head(torch.cat([out, h[:, -1:, None].expand(-1, 24, 1)], -1)).squeeze(-1)

    def with_parameters(self, b, base, params):
        return self.decode(b, base+self.residual_projection(kernels(b['offset'])@params))

    def sampled_power(self, b, base, params):
        n, count = params.shape[:2]
        repeated = {k: v[:, None].expand(-1, count, *v.shape[1:]).reshape(n*count, *v.shape[1:])
                    for k, v in b.items()}
        power = self.with_parameters(repeated, base[:, None].expand(-1, count, -1, -1).reshape(n*count, 24, self.d),
                                     params.reshape(n*count, 7, 4))
        return power.reshape(n, count, 24)

    def forward(self, b):
        if self.representation in ['R0', 'TF_R3', 'TF_E2L']:
            return super().forward(b)
        if self.representation == 'R0+H':
            return self.decode(b, self.weather_projection(b['weather']))
        z, base = self.latent(b)
        if self.representation == 'R1':
            return self.decode(b, base)
        if not self.generative:
            return self.with_parameters(b, base, self.window(z, b['offset'], b['state_c']))
        return self.prediction_samples(b, z, base).mean(1)

    def prediction_samples(self, b, z=None, base=None):
        if z is None:
            z, base = self.latent(b)
        context = self.window.features(z, b['offset'], b['state_c'])
        noise = stateless_noise(b['sample_key'], 8, self.spec['seed'], z.dtype)
        params = self.flow.sample(noise, context)
        return self.sampled_power(b, base, params)

    def training_loss(self, b, y):
        mse_weight = self.spec.get('mse_weight', 0.0)
        def point(p):
            return (p-y).abs().mean()+mse_weight*(p-y).square().mean()
        if not self.generative:
            return point(self(b))
        z, base = self.latent(b)
        at_origin = lerp(z, b['offset'], z.new_zeros(1))
        target, kl = self.teacher(y-b['h'][:, -1:], base-at_origin, b['offset'])
        teacher_power = self.with_parameters(b, base, target)
        context = self.window.features(z, b['offset'], b['state_c'])
        noise = torch.randn_like(target)
        t = torch.rand(len(y), device=y.device, dtype=y.dtype)
        detached = target.detach()
        mixed = (1-t[:, None, None])*noise+t[:, None, None]*detached
        matching = (self.flow(mixed, t, context)-(detached-noise)).square().mean()
        student_noise = torch.randn(len(y), 4, 7, 4, device=y.device, dtype=y.dtype)
        student_params = self.flow.sample(student_noise, context)
        student_power = self.sampled_power(b, base, student_params).mean(1)
        return point(teacher_power)+point(student_power)+0.1*matching+0.001*kl
