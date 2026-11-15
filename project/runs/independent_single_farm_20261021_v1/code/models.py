"""Single-farm models. Published families are explicitly labeled adaptations."""
from common import *
import copy
from torch import nn
from torch.nn import functional as F
import reference_original as ref
import baselines_original as old
from column_original import WeatherProjection
from wind_forcing.models.p0i_transfer import P0IConditionTransfer

def baseline(site, seed):
    return ref.build_baseline(site,seed,static())

class CachedCorrection(P0IConditionTransfer):
    """Exact point-alignment CBH graph with precomputed frozen B0 and weather.

    The singleton attention softmax is identically one; only its v/out maps
    affect the result. Inactive source modules stay in checkpoints for audit.
    """
    def forward(self,h,weather,phase,base):
        b=h.shape[0]; leads=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(b,-1)/4
        time=torch.stack((leads/6,torch.sin(2*torch.pi*leads/24),torch.cos(2*torch.pi*leads/24)),-1)
        context=torch.cat((time,self.site[0][None,None].expand(b,24,-1),phase[:,None,None].to(h).expand(-1,24,-1)/3),-1)
        ht=torch.arange(-23,1,device=h.device,dtype=h.dtype)/24
        memory=self.history_encoder(self.history_projection(torch.stack((h,ht.expand_as(h)),-1)))
        tokens=self.native_projection(weather[:,None])[:,0]
        w=self.align_out(self.align_v(tokens))
        summary=memory.mean(1)[:,None].expand(-1,24,-1)
        delta=base-torch.cat((h[:,-1:],base[:,:-1]),-1)
        conditioned=base
        if self.condition_mode in ('baseline','neutral'): summary=torch.zeros_like(summary)
        if self.condition_mode in ('history','neutral'):
            conditioned=torch.zeros_like(base);delta=torch.zeros_like(delta)
        raw=torch.cat((summary,conditioned[...,None],delta[...,None],leads[...,None]/6),-1)
        cond=self.condition_projection(raw)+self.condition_bias[:,0]
        q=self.context_projection(context)+w+self.transfer_adapter(torch.cat((w,cond),-1))
        correction=self.readout(self.decoder(q,memory)).squeeze(-1)
        anchor=base
        if self.output_mode=='direct': anchor=torch.zeros_like(base)
        elif self.output_mode=='last': anchor=h[:,-1:]
        return anchor+correction

def correction(site,seed,width,arm,state,fast=True):
    s=static(); seed_all(seed); cfg=copy.deepcopy(ref.CBH_CONFIG);cfg['d_model']=width
    cfg['condition_mode']={'CB':'baseline','CH':'history','CNEUTRAL':'neutral','NEUTRAL_LAST':'neutral'}.get(arm,'full')
    cls=CachedCorrection if fast else P0IConditionTransfer
    m=cls(cfg,ref.single_site_static(s,site),24)
    m.load_history_checkpoint(dict(model_config=ref.BASELINE_CONFIG,model_state=state))
    q,sv=ref.canonical_static_vectors(s)
    with torch.no_grad():
        m.site.copy_(sv[site:site+1]);m.baseline_model.forcing_encoder.station_query.station_static.copy_(q[site:site+1])
        m.indices.zero_();m.weights.zero_();m.weights[...,0]=1
    if arm!='E0':
        # Validate global ordering, then slice only the fixed elevation buffer.
        proj=WeatherProjection('E2C',s,str(COLUMN/'station_dem_srtm30m.json'),str(COLUMN/'column_train_stats.json'),width)
        proj.elevation=proj.elevation[:,site:site+1].clone();m.native_projection=proj
    m.output_mode={'DIRECT':'direct','NEUTRAL_LAST':'last'}.get(arm,'baseline')
    return m

class DLinearFC(nn.Module):
    """Moving-average decomposition + linear future-exogenous regression."""
    def __init__(self,d=64):
        super().__init__();self.trend=nn.Linear(24,24);self.season=nn.Linear(24,24);self.weather=nn.Linear(24*14+1,24)
    def forward(self,h,w,p):
        tr=F.avg_pool1d(F.pad(h[:,None],(2,2),mode='replicate'),5,stride=1)[:,0]
        return self.trend(tr)+self.season(h-tr)+self.weather(torch.cat((w.flatten(1),p[:,None].float()/3),-1))

class NBeatsXFC(nn.Module):
    """Generic N-BEATSx adaptation with actual residual backcast chaining."""
    def __init__(self,d=64):
        super().__init__();self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(24+24*14+1,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU(),nn.Linear(d,48)) for _ in range(4)])
    def forward(self,h,w,p):
        level=h[:,-1:];r=h-level;f=torch.zeros_like(h); ex=torch.cat((w.flatten(1),p[:,None].float()/3),-1)
        for block in self.blocks:
            o=block(torch.cat((r,ex),-1));r=r-o[:,:24];f=f+o[:,24:]
        return level+f

class TCNFC(nn.Module):
    """Temporal residual convs over all observed history + issued future NWP."""
    def __init__(self,d=64):
        super().__init__();self.h=nn.Sequential(nn.Conv1d(2,d,3,padding=1),nn.GELU(),nn.Conv1d(d,d,3,padding=2,dilation=2),nn.GELU(),nn.Flatten(),nn.Linear(24*d,d))
        self.inp=nn.Conv1d(14+d+2,d,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=2**i,dilation=2**i),nn.GELU(),nn.Conv1d(d,d,1)) for i in range(4)])
        self.head=nn.Conv1d(d,1,1)
    def forward(self,h,w,p):
        clock=torch.linspace(-1,0,24,device=h.device).expand_as(h)
        hs=self.h(torch.stack((h,clock),1))[:,:,None].expand(-1,-1,24)
        lead=torch.arange(1,25,device=h.device).float().expand_as(h)/24
        z=self.inp(torch.cat((w.transpose(1,2),hs,lead[:,None],p[:,None,None].float().expand(-1,1,24)/3),1))
        for block in self.blocks:z=z+block(z)
        return h[:,-1:]+self.head(z)[:,0]

class TimeXerFC(nn.Module):
    """Univariate TimeXer adaptation: endogenous patches/global token,
    exogenous variate tokens, alternating endogenous self/global cross attention.
    Future weather is known at forecast issuance; no future power input.
    """
    def __init__(self,d=64):
        super().__init__();self.patch=nn.Linear(4,d);self.pe=old.PositionalEncoding(d,16)
        self.global_token=nn.Parameter(torch.randn(1,1,d)*.02);self.exo=nn.Linear(24,d)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,4,2*d,dropout=0,batch_first=True,norm_first=True,activation='gelu') for _ in range(2)])
        self.cross=nn.ModuleList([nn.MultiheadAttention(d,4,batch_first=True) for _ in range(2)])
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)]);self.head=nn.Linear(7*d,24)
    def forward(self,h,w,p):
        center=h.mean(-1,keepdim=True);scale=h.std(-1,keepdim=True,unbiased=False).clamp_min(1e-3)
        z=self.pe(self.patch(((h-center)/scale).reshape(-1,6,4)))
        z=torch.cat((z,self.global_token.expand(len(h),-1,-1)),1);ex=self.exo(w.transpose(1,2))
        for layer,cross,norm in zip(self.layers,self.cross,self.norm):
            z=layer(z);g=z[:,-1:];a,_=cross(g,ex,ex,need_weights=False);z=torch.cat((z[:,:-1],norm(g+a)),1)
        return self.head(z.flatten(1))*scale+center

class TFTFC(nn.Module):
    """TFT family adaptation with selection, recurrent context, causal attention,
    gated residual paths; point-MAE objective replaces quantile heads.
    """
    def __init__(self,d=64):
        super().__init__();self.hp=nn.Linear(2,d);self.wp=nn.Parameter(torch.randn(14,d)*.02);self.wb=nn.Parameter(torch.zeros(14,d))
        self.select=nn.Sequential(nn.Linear(14,d),nn.ELU(),nn.Linear(d,14))
        self.hr=nn.LSTM(d,d,2,batch_first=True);self.wr=nn.LSTM(d,d,2,batch_first=True)
        self.attn=nn.MultiheadAttention(d,4,batch_first=True)
        self.gate=nn.Sequential(nn.Linear(d,2*d),nn.GLU());self.norm=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.ELU(),nn.Linear(2*d,d));self.gate2=nn.Sequential(nn.Linear(d,2*d),nn.GLU());self.norm2=nn.LayerNorm(d)
        self.head=nn.Linear(d,1)
    def forward(self,h,w,p):
        clock=torch.linspace(-1,0,24,device=h.device).expand_as(h)
        mem,state=self.hr(self.hp(torch.stack((h,clock),-1)))
        tokens=w[...,None]*self.wp+self.wb;weight=self.select(w).softmax(-1)
        z,_=self.wr((tokens*weight[...,None]).sum(2),state)
        full=torch.cat((mem,z),1);mask=torch.arange(48,device=h.device)[None,:]>(24+torch.arange(24,device=h.device)[:,None])
        a,_=self.attn(z,full,full,attn_mask=mask,need_weights=False)
        z=self.norm(z+self.gate(a));z=self.norm2(z+self.gate2(self.ff(z)))
        return h[:,-1:]+self.head(z).squeeze(-1)

class PatchTSTH(nn.Module):
    """Univariate patch transformer with reversible instance normalization."""
    def __init__(self,d=64):
        super().__init__();self.embed=nn.Linear(4,d);self.pe=old.PositionalEncoding(d,16);self.enc=old._encoder(d,3);self.head=nn.Linear(6*d,24)
    def forward(self,h,w,p):
        mu=h.mean(-1,keepdim=True);sd=h.std(-1,keepdim=True,unbiased=False).clamp_min(1e-3)
        z=self.enc(self.pe(self.embed(((h-mu)/sd).reshape(-1,6,4))))
        return self.head(z.flatten(1))*sd+mu

def named(family,width):
    constructors={'TRANSFORMER_FC':old.TransformerFutureCov,'TIDE_FC':old.TiDEFutureCov,
                  'NBEATSX_FC':NBeatsXFC,'TIMEXER_FC':TimeXerFC,'TFT_FC':TFTFC,'TCN_FC':TCNFC,
                  'DLINEAR_FC':DLinearFC,'GRU_H':lambda d:old.HistoryLSTM(d,'gru'),
                  'PATCHTST_H':PatchTSTH,'CNN_W':old.CNNWeather}
    return constructors[family](width)
