"""Paper controls. All historical code is retained under history_model.py."""
import torch
from torch import nn
from history_model import SharedModel as HistoryModel, configure, seed_all

class CovariateLinear(nn.Module):
    """Direct future-covariate linear regression; not a DLinear replica."""
    generative = False
    def __init__(self, spec):
        super().__init__(); self.spec = spec; seed_all(spec['seed'])
        self.readout = nn.Linear(1101, 24)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
    def features(self, b):
        return torch.cat([b['h'], b['near'], b['region'], b['weather'].flatten(1),
            b['hourly'].flatten(1), b['offset']/6, b['x'].flatten(1),
            nn.functional.one_hot(b['site'],55).to(b['h']),
            nn.functional.one_hot(b['phase'],4).to(b['h'])],-1)
    def forward(self,b):
        return b['h'][:,-1:] + self.readout(self.features(b))
    def training_loss(self,b,y):
        return (self(b)-y).abs().mean()

class PaperModel(HistoryModel):
    def __init__(self,spec):
        super().__init__(spec)
        if self.history_variant:
            self.refiner.window_mode=spec.get('window_mode','full')
            self.refiner.history_order=spec.get('history_order','original')
            self.refiner.state_only_off=spec.get('state_only_off',False)
        if spec.get('full_window_baseline'):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(spec['seed']+20261119)
                self.forecast_window_projection=nn.Linear(120,self.d,bias=False)
                nn.init.zeros_(self.forecast_window_projection.weight)
    def controlled_inputs(self,b):
        b=dict(b); control=self.spec.get('input_ablation')
        if control in ['operating','region','atmosphere']:
            b['x']=b['x'].clone(); b['state_c']=b['state_c'].clone()
        if control=='operating':
            b['x'][...,15:]=0; b['state_c'][...,:11]=0
        if control=='region':
            b['region']=torch.zeros_like(b['region']); b['near']=torch.zeros_like(b['near'])
            b['state_c'][...,11:]=0
        if control=='atmosphere': b['x'][...,:15]=0
        return b
    def decode(self,b,weather):
        if self.spec.get('decoder_state_off'):
            b=dict(b); b['x']=b['x'].clone(); b['x'][...,15:]=0
            b['region']=torch.zeros_like(b['region']); b['near']=torch.zeros_like(b['near'])
        return super().decode(b,weather)
    def forward(self,b):
        b=self.controlled_inputs(b)
        if self.spec.get('decoder_state_off') and not self.history_variant:
            return self.decode(b,self.weather_projection(b['weather']))
        return super().forward(b)

def SharedModel(spec):
    return CovariateLinear(spec) if spec['family']=='LINEAR' else PaperModel(spec)
