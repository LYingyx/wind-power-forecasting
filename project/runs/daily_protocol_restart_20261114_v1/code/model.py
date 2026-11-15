"""Shared per-farm forecasts. No mixing of examples or future power labels."""
from settings import *
import sys,math,random
import numpy as np,torch
from torch import nn
sys.path.insert(0,str(RUN/'code/legacy'))
sys.path.insert(0,str(RUN/'code/legacy/vendor'))
import baselines_original as baselines
from column_original import column_coordinates
from wind_forcing.models.p0i_legacy import ResidualConv

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def configure():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
def zero_last(m):
    z=[x for x in m.modules() if isinstance(x,nn.Linear)][-1];nn.init.zeros_(z.weight)
    if z.bias is not None:nn.init.zeros_(z.bias)

class NeighborReadout(nn.Module):
    """Original N3 readout: pooled 8-neighbor history with neighbor geometry keys."""
    def __init__(self,width):
        super().__init__();topo=np.load(ASSETS/'topology.npz')
        self.register_buffer('geometry',torch.from_numpy(topo['geometry']));self.register_buffer('mask',torch.from_numpy(topo['mask']))
        self.query=nn.Sequential(nn.Linear(30,32),nn.GELU(),nn.Linear(32,32));self.key=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32));self.value=nn.Linear(2,32,bias=False);self.out=nn.Linear(32,width,bias=False);nn.init.zeros_(self.out.weight)
    def forward(self,b):
        h=b['h'];n=len(h);geo=self.geometry[b['site']];k=geo.shape[1]
        p=b['near'][:,None].expand(-1,k,-1);delta=torch.diff(p,dim=-1,prepend=p[:,:,:1]);t=p.shape[-1]
        age=torch.arange(-21,1,device=h.device,dtype=h.dtype)/4
        inp=torch.cat([p[...,None],delta[...,None],age[None,None,:,None].expand(n,k,t,1)/6,geo[:,:,None].expand(-1,-1,t,-1)/100],-1)
        key=self.key(inp.flatten(1,2));value=self.value(torch.stack([p,delta],-1).flatten(1,2))
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)/4;time=torch.stack([lead/6,torch.sin(2*torch.pi*lead/24),torch.cos(2*torch.pi*lead/24)],-1)
        q=self.query(torch.cat([h[:,None].expand(-1,24,-1),time[None].expand(n,-1,-1),b['phase'][:,None,None].to(h).expand(-1,24,1)/3,b['physical'][...,[1,2]]/10],-1))
        key=torch.cat([key,h.new_zeros(n,1,32)],1);value=torch.cat([value,h.new_zeros(n,1,32)],1)
        mask=torch.cat([self.mask[b['site']][:,:,None].expand(-1,-1,t).flatten(1),torch.ones(n,1,device=h.device,dtype=torch.bool)],1)
        logits=(q@key.transpose(1,2))/math.sqrt(32);weights=logits.masked_fill(~mask[:,None],-1e4).softmax(-1)
        return self.out(weights@value)

class SharedModel(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec;self.family=spec['family'];self.condition=spec.get('condition','neutral');cfg=spec['config'];d=cfg['width'];drop=cfg['dropout'];self.d=d
        seed_all(spec['seed'])
        static=np.load(ASSETS/'site_static.npy');self.register_buffer('static',torch.from_numpy(static));self.embedding=nn.Embedding(len(static),8);self.site_projection=nn.Linear(12,d)
        self.rm=spec.get('rm',False)
        self.has_region=spec.get('input','R3')=='R3' and self.family!='B0'
        if self.family=='B0':
            self.history_projection=nn.Linear(2,d);self.history_encoder=nn.Sequential(*[ResidualConv(d,2**i,drop) for i in range(3)])
            self.query=nn.Linear(4,d);self.attn=nn.MultiheadAttention(d,4,dropout=drop,batch_first=True);self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1));zero_last(self.head);return
        if self.family=='W0':
            self.history_projection=nn.Linear(12 if self.rm else 2,d)
            self.history_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
            self.weather_projection=nn.Sequential(nn.Linear(15,d),nn.GELU(),nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
            self.align_v=nn.Linear(d,d,bias=False);self.align_out=nn.Linear(d,d);self.context_projection=nn.Linear(16,d)
            self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d,4,2*d,drop,batch_first=True,norm_first=True),2)
            self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,1));zero_last(self.head)
            stats=json.loads((ASSETS/'normalization.json').read_text())
            for name,value in [('column_mean',stats['column_mean']),('column_std',stats['column_std']),('elevation',np.load(ASSETS/'elevation.npy'))]:self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        elif self.family=='TF':
            self.original=baselines.TransformerFutureCov(d);zero_last(self.original.head)
            if self.rm:self.original.h_in=nn.Linear(12,d)
            for m in self.original.modules():
                if isinstance(m,nn.Dropout):m.p=drop
                if isinstance(m,nn.MultiheadAttention):m.dropout=drop
        elif self.family=='TIDE':
            self.original=baselines.TiDEFutureCov(d);self.future_extra=nn.Linear(24*d,2*d,bias=False);nn.init.zeros_(self.future_extra.weight);self.dropout=nn.Dropout(drop)
        elif self.family=='TFT':
            self.hp=nn.Linear(2,d);self.wp=nn.Parameter(torch.randn(14,d)*.02);self.wb=nn.Parameter(torch.zeros(14,d));self.select=nn.Sequential(nn.Linear(14,d),nn.ELU(),nn.Linear(d,14))
            self.hr=nn.LSTM(d,d,2,dropout=drop,batch_first=True);self.wr=nn.LSTM(d,d,2,dropout=drop,batch_first=True);self.attn=nn.MultiheadAttention(d,4,dropout=drop,batch_first=True)
            self.gate=nn.Sequential(nn.Linear(d,2*d),nn.GLU());self.norm=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.ELU(),nn.Dropout(drop),nn.Linear(2*d,d));self.gate2=nn.Sequential(nn.Linear(d,2*d),nn.GLU());self.norm2=nn.LayerNorm(d);self.head=nn.Linear(d,1);zero_last(self.head)
        else:raise ValueError(self.family)
        if self.family=='W0' or self.condition!='neutral':
            self.condition_projection=nn.Sequential(nn.Linear(d+3,d),nn.GELU(),nn.LayerNorm(d));self.transfer_adapter=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d));zero_last(self.transfer_adapter);self.condition_bias=nn.Parameter(torch.zeros(1,24,d))
        if self.has_region:
            self.sel=nn.Linear(22,d,bias=False);nn.init.zeros_(self.sel.weight)
            if not self.rm:
                self.regional=nn.Linear(12,d,bias=False);nn.init.zeros_(self.regional.weight);self.neighbor=NeighborReadout(d)

    def condition_response(self,b,w,memory,lead):
        n=len(w);summary=w.new_zeros(n,24,self.d);base=w.new_zeros(n,24);delta=base
        if self.condition in ['cb','cbh','anchor','wrong_cb','wrong_cbh']:
            base=b['base'];delta=b['base_delta']
        if self.condition in ['cbh','wrong_cbh']:
            # Both real and wrong CBH perform the same extra encoder call.
            h=b['condition_h'];clock=torch.arange(-23,1,device=h.device,dtype=h.dtype).expand_as(h)/24
            summary=self.history_encoder(self.history_projection(torch.stack([h,clock],-1))).mean(1)[:,None].expand(-1,24,-1)
        raw=torch.cat([summary,base[...,None],delta[...,None],lead[...,None]],-1)
        cond=self.condition_projection(raw)+self.condition_bias
        return self.transfer_adapter(torch.cat([w,cond],-1))

    def forward(self,b):
        h=b['h'];n=len(h);site=torch.cat([self.static[b['site']],self.embedding(b['site'])],-1);sv=self.site_projection(site)
        lead=torch.arange(1,25,device=h.device,dtype=h.dtype)[None].expand(n,-1)/24
        tf=torch.stack([lead,torch.sin(2*torch.pi*lead),torch.cos(2*torch.pi*lead),b['phase'][:,None].to(h).expand(-1,24)/3],-1)
        if self.family=='B0':
            clock=torch.arange(-23,1,device=h.device,dtype=h.dtype).expand_as(h)/24;memory=self.history_encoder(self.history_projection(torch.stack([h,clock],-1))+sv[:,None])
            q=self.query(tf)+sv[:,None];a,_=self.attn(q,memory,memory,need_weights=False);return h[:,-1:]+self.head(torch.cat([q,a],-1)).squeeze(-1)
        extra=sv[:,None].expand(-1,24,-1)
        if self.has_region:
            extra=extra+self.sel(b['x'])
            if not self.rm:extra=extra+self.regional(b['region'])[:,None]+self.neighbor(b)
        if self.family=='W0':
            time=torch.stack([lead,torch.sin(2*torch.pi*lead/4),torch.cos(2*torch.pi*lead/4)],-1)
            context=torch.cat([time,site[:,None].expand(-1,24,-1),b['phase'][:,None,None].to(h).expand(-1,24,1)/3],-1)
            clock=torch.arange(-23,1,device=h.device,dtype=h.dtype).expand_as(h)/24
            hin=torch.stack([h,clock],-1)
            if self.rm:hin=torch.cat([hin,b['rm_history']],-1)
            memory=self.history_encoder(self.history_projection(hin)+(sv[:,None] if self.rm else 0))
            col=column_coordinates(b['physical'],self.elevation[b['site']][:,None].expand(-1,24));w=self.align_out(self.align_v(self.weather_projection((col-self.column_mean)/self.column_std)))
            q=self.context_projection(context)+w+self.condition_response(b,w,memory,lead)+extra
            out=self.head(self.decoder(q,memory)).squeeze(-1);anchor=b['base'] if self.condition=='anchor' else h[:,-1:];return anchor+out
        if self.family=='TF':
            m=self.original;clock=torch.linspace(-1,0,24,device=h.device,dtype=h.dtype).expand_as(h)
            hin=torch.stack([h,clock],-1)
            if self.rm:hin=torch.cat([hin,b['rm_history']],-1)
            memory=m.h_enc(m.pe(m.h_in(hin)+sv[:,None]))
            w=m.w_in(torch.cat([b['weather'],tf],-1));q=m.pe(w+extra)
            if hasattr(self,'condition_projection'):q=q+self.condition_response(b,w,memory,lead)
            out=m.dec(q,memory);return h[:,-1:]+m.head(torch.cat([out,h[:,-1:,None].expand(-1,24,1)],-1)).squeeze(-1)
        if self.family=='TIDE':
            m=self.original;clock=torch.linspace(-1,0,24,device=h.device,dtype=h.dtype).expand_as(h);p=m.past(torch.cat([h,clock],-1))+sv
            f=m.future(torch.cat([b['weather'],tf],-1).flatten(1))+self.future_extra(extra.flatten(1));z=torch.cat([p,f],-1)
            for block in m.blocks:z=z+self.dropout(block(z))
            return h[:,-1:]+m.skip(h-h.mean(-1,keepdim=True))+m.decoder(z)
        clock=torch.linspace(-1,0,24,device=h.device,dtype=h.dtype).expand_as(h);mem,state=self.hr(self.hp(torch.stack([h,clock],-1))+sv[:,None]);w=b['weather'];tokens=w[...,None]*self.wp+self.wb;weight=self.select(w).softmax(-1)
        z,_=self.wr((tokens*weight[...,None]).sum(2)+extra,state);full=torch.cat([mem,z],1);mask=torch.arange(48,device=h.device)[None]>(24+torch.arange(24,device=h.device)[:,None])
        a,_=self.attn(z,full,full,attn_mask=mask,need_weights=False);z=self.norm(z+self.gate(a));z=self.norm2(z+self.gate2(self.ff(z)));return h[:,-1:]+self.head(z).squeeze(-1)
