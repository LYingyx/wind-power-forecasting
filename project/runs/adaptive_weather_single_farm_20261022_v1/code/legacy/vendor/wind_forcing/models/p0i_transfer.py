"""Controlled transfer to the original factorized, pointwise P0I backbone.

PA/PJ are separable/joint *input adapters*. Their complete Transformer forecasts
both already mix history and weather. PA is not a strict additive-output model.
"""
import torch
from torch import nn
from wind_forcing.models.p0i_legacy import TemporalAlignedBaseline,interpolate
from wind_forcing.models.forced_evolution import build_forced_evolution_model

class P0IConditionTransfer(TemporalAlignedBaseline):
    def __init__(self,cfg,static,forecast_steps=24):
        super().__init__(cfg,static,forecast_steps)
        self.transfer_variant=cfg['transfer_variant']
        if self.transfer_variant not in {'P0','PB','PA','PJ','CBH','CB','CH','CNEUTRAL'}:raise ValueError('Unknown transfer variant')
        if self.variant!='P0I' or self.scope!='native7':raise ValueError('Fixed P0I/native7 required')
        self.baseline_model=build_forced_evolution_model(cfg['history_baseline_model_config'],static,history_steps=24,forecast_steps=forecast_steps)
        self.baseline_model.requires_grad_(False);self.baseline_model.eval()
        self.condition_mode=cfg.get('condition_mode','full')
        if self.condition_mode not in {'full','baseline','history','neutral'}: raise ValueError('Unknown condition_mode')
        self.freeze_history_baseline=True
        self.baseline_config=cfg['history_baseline_model_config']
        width=int(cfg['d_model'])
        # State uses the *same* memory that the downstream decoder already sees,
        # plus the frozen b and delta-b. No future target is an input.
        self.condition_projection=nn.Sequential(nn.Linear(width+3,width),nn.GELU(),nn.LayerNorm(width))
        # No adapter dropout: PA's two evaluations do not consume extra random
        # draws that would change the common decoder's dropout masks.
        self.transfer_adapter=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        nn.init.zeros_(self.transfer_adapter[-1].weight);nn.init.zeros_(self.transfer_adapter[-1].bias)
        self.condition_bias=nn.Parameter(torch.zeros(1,1,forecast_steps,width))

    def train(self,mode=True):
        super().train(mode)
        self.baseline_model.eval()
        return self

    def load_history_checkpoint(self,checkpoint):
        if checkpoint['model_config']!=self.baseline_config:raise ValueError('Baseline model config mismatch')
        self.baseline_model.load_state_dict(checkpoint['model_state'],strict=True)
        self.baseline_model.requires_grad_(False);self.baseline_model.eval()

    def adapter_response(self,weather,state):
        if self.transfer_variant in {'CBH','CB','CH','CNEUTRAL'}:
            return self.transfer_adapter(torch.cat((weather,state),-1))
        return torch.zeros_like(weather)

    def forward(self,*,station_history,cluster_history,weather,weather_batch_index,weather_time_offset_hours,origin_phase,**unused):
        b,n=station_history.shape[:2];h=station_history[...,-self.visible:];origin=h[...,-1:]
        with torch.no_grad():
            base=self.baseline_model(station_history=station_history,cluster_history=cluster_history,origin_phase=origin_phase)
        baseline=base['station_factor']
        anchor_baseline=baseline
        leads=torch.arange(1,self.forecast_steps+1,device=h.device,dtype=h.dtype)[None].expand(b,-1)/4
        time=torch.stack((leads/6,torch.sin(2*torch.pi*leads/24),torch.cos(2*torch.pi*leads/24)),-1)
        phase=origin_phase.to(h).reshape(b,-1)[:,:1,None].expand(-1,self.forecast_steps,-1)/3
        context=torch.cat((time[:,None].expand(-1,n,-1,-1),self.site[None,:,None].expand(b,-1,self.forecast_steps,-1),phase[:,None].expand(-1,n,-1,-1)),-1)
        nodes=self.station_weather(weather).index_select(0,weather_batch_index)
        ht=torch.arange(-self.visible+1,1,device=h.device,dtype=h.dtype)/(4*6)
        hist=torch.stack((h,ht.expand_as(h)),-1).reshape(b*n,self.visible,2)
        memory=self.history_encoder(self.history_projection(hist))
        q=self.context_projection(context).reshape(b,n,self.forecast_steps,-1)
        tokens=interpolate(nodes[:,:,:7],weather_time_offset_hours[:,:7],leads)
        times=leads[:,None,:].expand(b,n,-1)
        w=self._align_masked(q,tokens,times,leads,True,'point')
        q=q+w
        if self.transfer_variant in {'CBH','CB','CH','CNEUTRAL'}:
            delta=anchor_baseline-torch.cat((origin,anchor_baseline[...,:-1]),-1)
            summary=memory.mean(1).reshape(b,n,-1)[:,:,None,:].expand(-1,-1,self.forecast_steps,-1)
            if self.condition_mode in {'baseline','neutral'}: summary=torch.zeros_like(summary)
            if self.condition_mode in {'history','neutral'}: baseline=torch.zeros_like(baseline); delta=torch.zeros_like(delta)
            if self.condition_mode=='neutral': delta=torch.zeros_like(delta)
            raw=torch.cat((summary,baseline[...,None],delta[...,None],(leads[:,None,:,None]/6).expand(-1,n,-1,-1)),-1)
            condition=self.condition_projection(raw)+self.condition_bias.to(raw)
            q=q+self.adapter_response(w,condition)
        x=self.decoder(q.reshape(b*n,self.forecast_steps,-1),memory)
        correction=self.readout(x).reshape(b,n,self.forecast_steps)
        anchor=origin if self.transfer_variant=='P0' else anchor_baseline
        result=self.hierarchical_output(anchor+correction)
        result['baseline_station_factor']=anchor_baseline
        result['baseline_cluster_factor']=base['cluster_factor']
        return result
