"""Hourly-window residuals and a training-only posterior for conditional flow matching."""
import math
import torch
from torch import nn


def lerp(nodes, offset, queries=None):
    if queries is None:
        queries = torch.arange(1, 25, device=nodes.device, dtype=nodes.dtype)/4
    x = queries[None]-offset[:, :1]
    lo = x.floor().long().clamp(0, nodes.shape[1]-2)
    frac = (x-lo)[..., None]
    left = torch.gather(nodes, 1, lo[..., None].expand(-1, -1, nodes.shape[-1]))
    right = torch.gather(nodes, 1, (lo+1)[..., None].expand(-1, -1, nodes.shape[-1]))
    return left*(1-frac)+right*frac


def kernels(offset):
    query = torch.arange(1, 25, device=offset.device, dtype=offset.dtype)/4
    centers = offset[:, :-1]+0.5
    weights = torch.exp(-0.5*((query[None, :, None]-centers[:, None])/0.75)**2)
    return weights/weights.sum(-1, keepdim=True)


def encoder(width=32):
    layer = nn.TransformerEncoderLayer(width, 2, width*2, dropout=0.0,
                                      activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)


class Window(nn.Module):
    def __init__(self, d, state):
        super().__init__()
        self.use_state = state
        self.input = nn.Linear(2*d+1+(17 if state else 0), 32)
        self.encoder = encoder()
        self.output = nn.Linear(32, 4)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def features(self, z, offset, c):
        dz = torch.cat([z[:, 1:]-z[:, :-1], torch.zeros_like(z[:, :1])], 1)
        items = [z, dz, offset[..., None]/6]
        if self.use_state:
            items.append(c[:, None].expand(-1, 8, -1))
        hidden = self.encoder(self.input(torch.cat(items, -1)))
        return (hidden[:, :-1]+hidden[:, 1:])/2

    def forward(self, z, offset, c):
        return self.output(self.features(z, offset, c))


class Teacher(nn.Module):
    """Posterior sees labels only in training_loss; forward inference never calls it."""
    def __init__(self, d):
        super().__init__()
        self.input = nn.Linear(d+2, 32)
        self.encoder = encoder()
        self.output = nn.Linear(32, 8)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, change, latent_change, offset):
        clock = torch.arange(1, 25, device=change.device, dtype=change.dtype)/24
        x = torch.cat([change[..., None], latent_change,
                       clock[None, :, None].expand(len(change), -1, -1)], -1)
        hidden = self.encoder(self.input(x))
        weights = kernels(offset).transpose(1, 2)
        weights = weights/weights.sum(-1, keepdim=True)
        mu, logvar = self.output(weights@hidden).chunk(2, -1)
        logvar = logvar.clamp(-6, 2)
        # Posterior prior N(0, 0.1^2). KL is applied to standardized parameters.
        sample = 0.1*(mu+torch.exp(0.5*logvar)*torch.randn_like(mu))
        kl = 0.5*(mu.square()+logvar.exp()-1-logvar).mean()
        return sample, kl


def stateless_noise(key, count, seed, dtype):
    """Batch/order invariant evaluation noise, with no change to the training RNG."""
    s = torch.arange(count, device=key.device)[None, :, None, None]
    j = torch.arange(28, device=key.device).reshape(1, 1, 7, 4)
    m = 2147483647
    x = (key[:, None, None, None]*104729+s*13007+j*7919+seed*15485863+1) % m
    x = ((x ^ (x >> 16))*73244475) % m
    x = ((x ^ (x >> 16))*73244475) % m
    x = x ^ (x >> 16)
    u1 = (x.to(torch.float64)+0.5)/m
    x = ((x ^ (x >> 15))*73244475+17) % m
    x = x ^ (x >> 16)
    u2 = (x.to(torch.float64)+0.5)/m
    return (torch.sqrt(-2*torch.log(u1))*torch.cos(2*math.pi*u2)).to(dtype)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(32+4+1, 32)
        self.encoder = encoder()
        self.output = nn.Linear(32, 4)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, t, context):
        clock = t.reshape(-1, 1, 1).expand(-1, 7, 1)
        return self.output(self.encoder(self.input(torch.cat([x, context, clock], -1))))

    def sample(self, noise, context, steps=8):
        n, count = noise.shape[:2]
        x = noise.reshape(n*count, 7, 4)
        context = context[:, None].expand(-1, count, -1, -1).reshape(n*count, 7, 32)
        for i in range(steps):
            t = x.new_full((len(x),), i/steps)
            x = x+self(x, t, context)/steps
        return x.reshape(n, count, 7, 4)
