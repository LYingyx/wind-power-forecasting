"""Compact implementations of the requested forecasting baselines.

Every invocation is a single station and a single random seed.  The classes are
deliberately kept dependency free so that the experiment is reproducible on the
existing environment.  The names follow the published model families: the
Transformer/TFT/TimeXer versions consume known future weather covariates, the
_*_H versions consume only the 24 observed power points, and the *_W versions
consume only the issued weather trajectory.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = [
    "TRANSFORMER_FC", "TFT_FC", "TIMEXER_FC",
    "TIDE_FC", "DLINEAR_FC", "NBEATSX_FC", "TCN_FC",
    "LSTM_H", "GRU_H", "PATCHTST_H", "ITRANSFORMER_H", "MODERNTCN_H",
    "CNN_W", "VIT_W", "SWIN_W",
]


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None], persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype, device=x.device)


class MLP(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def _encoder(d, layers=2, heads=4):
    layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                       batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, layers)


def _interpolate_weather(nodes, times):
    """Linear interpolation from seven hourly Pangu nodes to 24 quarter hours."""
    b, n, c = nodes.shape
    q = torch.arange(1, 25, device=nodes.device, dtype=nodes.dtype)[None] / 4.0
    q = q.expand(b, -1)
    # searchsorted accepts a batch of sorted rows on current torch versions.
    right = torch.searchsorted(times.contiguous(), q.contiguous()).clamp(1, n - 1)
    left = right - 1
    gather_l = left[..., None].expand(-1, -1, c)
    gather_r = right[..., None].expand(-1, -1, c)
    t0 = times.gather(1, left)
    t1 = times.gather(1, right)
    alpha = ((q - t0) / (t1 - t0).clamp_min(1e-6))[..., None]
    l = nodes.gather(1, gather_l)
    r = nodes.gather(1, gather_r)
    return l + alpha * (r - l)


class HistoryLSTM(nn.Module):
    def __init__(self, d=64, cell="lstm"):
        super().__init__(); self.cell = cell
        self.inp = nn.Linear(2, d)
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = R(d, d, num_layers=2, batch_first=True, dropout=0.0)
        self.head = MLP(d + 1, d, 24)

    def forward(self, h, weather, phase):
        clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None, :, None].expand(h.shape[0], -1, -1)
        z, _ = self.rnn(self.inp(torch.cat([h[..., None], clock], -1)))
        return h[:, -1, None] + self.head(torch.cat([z[:, -1], h[:, -1, None]], -1))


class PatchTSTHistory(nn.Module):
    def __init__(self, d=64, patch=4):
        super().__init__(); self.patch = patch
        self.embed = nn.Linear(patch + 2, d); self.pe = PositionalEncoding(d, 16)
        self.enc = _encoder(d, 3); self.head = MLP(d, d, 24)

    def forward(self, h, weather, phase):
        b = h.shape[0]; x = h.reshape(b, 6, 4)
        phasef = (phase.float() / 3.0)[:, None, None].expand(-1, 6, 1)
        p = torch.linspace(0, 1, 6, device=h.device, dtype=h.dtype)[None, :, None].expand(b, -1, -1)
        z = self.pe(self.embed(torch.cat([x, phasef, p], -1)))
        return h[:, -1, None] + self.head(self.enc(z).mean(1))


class ITransformerHistory(nn.Module):
    """Inverted Transformer: the one observed power variate is a token."""
    def __init__(self, d=64):
        super().__init__(); self.embed = nn.Linear(24 + 1, d); self.enc = _encoder(d, 2)
        self.head = MLP(d, d, 24)

    def forward(self, h, weather, phase):
        z = self.embed(torch.cat([h, phase.float()[:, None] / 3.0], -1))[:, None]
        return h[:, -1, None] + self.head(self.enc(z)[:, 0])


class ModernTCNHistory(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.inp = nn.Conv1d(2, d, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2, groups=d), nn.Conv1d(d, d, 1), nn.GELU(),
                                                   nn.Conv1d(d, d, 3, padding=2, dilation=2, groups=d), nn.Conv1d(d, d, 1), nn.GELU()) for _ in range(3)])
        self.norm = nn.BatchNorm1d(d); self.head = MLP(d + 1, d, 24)

    def forward(self, h, weather, phase):
        clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None].expand(h.shape[0], -1)
        z = self.inp(torch.stack([h, clock], 1))
        for block in self.blocks: z = z + block(z)
        z = self.norm(z).mean(-1)
        return h[:, -1, None] + self.head(torch.cat([z, h[:, -1, None]], -1))


class TransformerFutureCov(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.h_in = nn.Linear(2, d); self.w_in = nn.Linear(14 + 4, d)
        self.pe = PositionalEncoding(d, 64); self.h_enc = _encoder(d, 2)
        dec_layer = nn.TransformerDecoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dec_layer, 2); self.head = MLP(d + 1, d, 1)

    def forward(self, h, weather, phase):
        b = h.shape[0]; hist_clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None, :, None].expand(b, -1, -1)
        mem = self.h_enc(self.pe(self.h_in(torch.cat([h[..., None], hist_clock], -1))))
        lead = torch.arange(1, 25, device=h.device, dtype=h.dtype)[None].expand(b, -1) / 24
        feat = torch.stack([lead, torch.sin(2 * math.pi * lead), torch.cos(2 * math.pi * lead), phase[:, None].float() / 3.0 + lead * 0], -1)
        q = self.pe(self.w_in(torch.cat([weather, feat], -1)))
        out = self.dec(q, mem)
        return h[:, -1, None] + self.head(torch.cat([out, h[:, -1, None, None].expand(-1, 24, -1)], -1)).squeeze(-1)


class TFFutureCov(nn.Module):
    """TFT-style variable selection, gated recurrent history and attention."""
    def __init__(self, d=64):
        super().__init__(); self.hvars = nn.ModuleList([nn.Linear(1, d) for _ in range(2)])
        self.wvars = nn.ModuleList([nn.Linear(1, d) for _ in range(14)])
        self.selector = nn.Linear(14 * d, 14); self.h_rnn = nn.LSTM(d, d, 2, batch_first=True)
        self.w_rnn = nn.LSTM(d, d, 2, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid()); self.head = MLP(d + 1, d, 1)

    def forward(self, h, weather, phase):
        b = h.shape[0]; clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None].expand(b, -1)
        hh = 0.5 * (self.hvars[0](h[..., None]) + self.hvars[1](clock[..., None]))
        hm, _ = self.h_rnn(hh); tokens = torch.stack([m(weather[..., i:i+1]) for i, m in enumerate(self.wvars)], 2)
        weights = torch.softmax(self.selector(tokens.flatten(-2)), -1)[..., None]
        ww = (tokens * weights).sum(2); wm, _ = self.w_rnn(ww)
        gate = self.gate(torch.cat([hm[:, -1:, :].expand(-1, 24, -1), wm], -1)); z = gate * wm + (1 - gate) * hm[:, -1:, :]
        return h[:, -1, None] + self.head(torch.cat([z, h[:, -1, None, None].expand(-1, 24, -1)], -1)).squeeze(-1)


class TimeXerFutureCov(nn.Module):
    """TimeXer-style endogenous history token and exogenous weather patches."""
    def __init__(self, d=64):
        super().__init__(); self.h_patch = nn.Linear(4, d); self.w_patch = nn.Linear(14 * 4, d)
        self.pe = PositionalEncoding(d, 16); self.exo = _encoder(d, 2)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True); self.head = MLP(d + 1, d, 1)

    def forward(self, h, weather, phase):
        b = h.shape[0]; hp = self.pe(self.h_patch(h.reshape(b, 6, 4))); wp = self.pe(self.w_patch(weather.reshape(b, 6, 14 * 4)))
        ex = self.exo(wp); q = hp.mean(1, keepdim=True).expand(-1, 24, -1)
        z, _ = self.cross(q, ex, ex); z = z + torch.nn.functional.interpolate(ex.transpose(1, 2), 24, mode="linear", align_corners=False).transpose(1, 2)
        return h[:, -1, None] + self.head(torch.cat([z, h[:, -1, None, None].expand(-1, 24, -1)], -1)).squeeze(-1)


class TiDEFutureCov(nn.Module):
    """TiDE-style MLP encoder/decoder for long-horizon known covariates.

    The past target trajectory and future covariate trajectory are encoded into
    a fixed representation, passed through residual dense blocks, and decoded
    to all 24 forecast leads.  This is the dependency-free equivalent used for
    the controlled baseline, with no autoregressive access to future targets.
    """
    def __init__(self, d=64):
        super().__init__(); self.past = nn.Sequential(nn.Linear(24 + 24, d), nn.GELU(), nn.Linear(d, d))
        self.future = nn.Sequential(nn.Linear(24 * (14 + 4), 2*d), nn.GELU(), nn.Linear(2*d, 2*d))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(3*d, 3*d), nn.GELU(), nn.Linear(3*d, 3*d)) for _ in range(3)])
        self.decoder = nn.Sequential(nn.Linear(3*d, 2*d), nn.GELU(), nn.Linear(2*d, 24))
        self.skip = nn.Linear(24, 24)

    def forward(self, h, weather, phase):
        b = h.shape[0]; clock = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None].expand(b, -1)
        lead = torch.arange(1, 25, device=h.device, dtype=h.dtype)[None].expand(b, -1) / 24
        feat = torch.stack([lead, torch.sin(2*math.pi*lead), torch.cos(2*math.pi*lead), phase[:, None].float().expand(-1, 24)/3.], -1)
        p = self.past(torch.cat([h, clock], -1)); f = self.future(torch.cat([weather, feat], -1).flatten(1)); z = torch.cat([p, f], -1)
        for block in self.blocks: z = z + block(z)
        return h[:, -1, None] + self.skip(h - h.mean(-1, keepdim=True)) + self.decoder(z)


class DLinearFutureCov(nn.Module):
    """DLinear-style trend/level linear baseline plus future-weather regression."""
    def __init__(self, d=64):
        super().__init__(); self.level = nn.Linear(24, 24); self.trend = nn.Linear(24, 24)
        self.weather = nn.Sequential(nn.Linear(14 + 4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, weather, phase):
        b = h.shape[0]; t = torch.linspace(-1, 0, 24, device=h.device, dtype=h.dtype)[None].expand(b, -1)
        lead = torch.arange(1, 25, device=h.device, dtype=h.dtype)[None].expand(b, -1) / 24
        feat = torch.stack([lead, torch.sin(2*math.pi*lead), torch.cos(2*math.pi*lead), phase[:, None].float().expand(-1, 24)/3.], -1)
        centered = h - h.mean(-1, keepdim=True); w = self.weather(torch.cat([weather, feat], -1)).squeeze(-1)
        return h.mean(-1, keepdim=True) + self.level(centered) + self.trend(t) + w


class NBeatsXFutureCov(nn.Module):
    """Small N-BEATSx-style residual/backcast blocks with exogenous weather."""
    def __init__(self, d=64):
        super().__init__(); self.inp = nn.Linear(24 + 24*14, d)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, 24+24)) for _ in range(4)])
        self.updates = nn.ModuleList([nn.Linear(24, d) for _ in range(4)])
        self.exo = nn.Sequential(nn.Linear(14+4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, weather, phase):
        b = h.shape[0]; lead = torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/24
        feat = torch.stack([lead,torch.sin(2*math.pi*lead),torch.cos(2*math.pi*lead),phase[:,None].float().expand(-1,24)/3.],-1)
        z=self.inp(torch.cat([h,weather.flatten(1)],-1)); forecast=torch.zeros(b,24,device=h.device,dtype=h.dtype)
        residual=h
        for block, update in zip(self.blocks, self.updates):
            out=block(z); residual=residual-out[:,:24]; forecast=forecast+out[:,24:]; z=z+0.05*torch.tanh(update(out[:,:24]))
        return h[:,-1,None]+forecast+self.exo(torch.cat([weather,feat],-1)).squeeze(-1)


class TCNFutureCov(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.inp=nn.Conv1d(14+2,d,1); self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=1,dilation=1,groups=d),nn.Conv1d(d,d,1),nn.GELU(),nn.Conv1d(d,d,3,padding=2,dilation=2,groups=d),nn.Conv1d(d,d,1),nn.GELU()) for _ in range(3)]); self.head=nn.Conv1d(d,1,1)
    def forward(self,h,weather,phase):
        b=h.shape[0]; lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/24; ph=phase[:,None].float().expand(-1,24)/3.; z=torch.cat([weather,h[:,-1,None,None].expand(-1,24,1),ph[...,None]],-1).transpose(1,2); z=self.inp(z)
        for block in self.blocks:z=z+block(z)
        return h[:,-1,None]+self.head(z).squeeze(1)


class CNNWeather(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.inp = nn.Conv1d(14, d, 3, padding=1); self.net = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.GELU(), nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.GELU())
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, h, weather, phase): return self.head(self.net(self.inp(weather.transpose(1, 2)))).squeeze(1)


class ViTWeather(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.inp = nn.Linear(14 + 2, d); self.pe = PositionalEncoding(d, 32); self.enc = _encoder(d, 3); self.head = nn.Linear(d, 1)

    def forward(self, h, weather, phase):
        b = weather.shape[0]; lead = torch.arange(1, 25, device=weather.device, dtype=weather.dtype)[None].expand(b, -1) / 24
        z = self.pe(self.inp(torch.cat([weather, lead[..., None], phase[:, None, None].float().expand(-1, 24, 1) / 3], -1)))
        return self.head(self.enc(z)).squeeze(-1)


class SwinWeather(nn.Module):
    """1-D Swin-style alternating local windows and shifted windows."""
    def __init__(self, d=64):
        super().__init__(); self.inp = nn.Linear(14 + 2, d); self.pe = PositionalEncoding(d, 32)
        self.local = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 2*d, dropout=0., batch_first=True, norm_first=True, activation="gelu") for _ in range(3)])
        self.head = nn.Linear(d, 1)

    def forward(self, h, weather, phase):
        b = weather.shape[0]; lead = torch.arange(1, 25, device=weather.device, dtype=weather.dtype)[None].expand(b, -1) / 24
        z = self.pe(self.inp(torch.cat([weather, lead[..., None], phase[:, None, None].float().expand(-1, 24, 1) / 3], -1)))
        for i, layer in enumerate(self.local):
            if i % 2 == 0:
                z = layer(z)
            else:
                shift = 2; zs = torch.roll(z, shift, 1); zs = layer(zs); z = torch.roll(zs, -shift, 1)
        return self.head(z).squeeze(-1)


def build_model(family, d=64):
    if family == "LSTM_H": return HistoryLSTM(d, "lstm")
    if family == "GRU_H": return HistoryLSTM(d, "gru")
    if family == "PATCHTST_H": return PatchTSTHistory(d)
    if family == "ITRANSFORMER_H": return ITransformerHistory(d)
    if family == "MODERNTCN_H": return ModernTCNHistory(d)
    if family == "TRANSFORMER_FC": return TransformerFutureCov(d)
    if family == "TFT_FC": return TFFutureCov(d)
    if family == "TIMEXER_FC": return TimeXerFutureCov(d)
    if family == "TIDE_FC": return TiDEFutureCov(d)
    if family == "DLINEAR_FC": return DLinearFutureCov(d)
    if family == "NBEATSX_FC": return NBeatsXFutureCov(d)
    if family == "TCN_FC": return TCNFutureCov(d)
    if family == "CNN_W": return CNNWeather(d)
    if family == "VIT_W": return ViTWeather(d)
    if family == "SWIN_W": return SwinWeather(d)
    raise ValueError(family)


def input_group(family):
    if family.endswith("_H"): return "history_only"
    if family.endswith("_FC"): return "future_covariate"
    return "weather_only"
