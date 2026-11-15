import torch
from torch import nn
from wind_forcing.models.baselines import ForecastModelBase
from wind_forcing.models.components import build_station_patch_metadata
FAMILIES=("early_tcn","covariate_transformer","pamf_adapted","history_tcn","weather_anchor_tcn")

def interpolate(nodes, times, leads):
    # nodes [B,N,T,C]; interpolation uses actual NWP valid-time offsets.
    if times.ndim != 2 or nodes.shape[0] != times.shape[0] or nodes.shape[2] != times.shape[1]:
        raise ValueError('Incompatible NWP offsets')
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError('NWP valid times must increase')
    right = torch.searchsorted(times.contiguous(), leads.contiguous()).clamp(1, times.shape[1] - 1)
    left = right - 1
    t0, t1 = times.gather(1, left), times.gather(1, right)
    alpha = ((leads - t0) / (t1 - t0)).clamp(0, 1)
    shape = (nodes.shape[0], nodes.shape[1], leads.shape[1], nodes.shape[3])
    v0 = nodes.gather(2, left[:, None, :, None].expand(shape))
    v1 = nodes.gather(2, right[:, None, :, None].expand(shape))
    return v0 + alpha[:, None, :, None] * (v1 - v0)

class ResidualConv(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                                 nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, width, 1))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.net(x.transpose(1, 2)).transpose(1, 2))

class CovariateBaseline(ForecastModelBase):
    def __init__(self, cfg, static, forecast_steps=24):
        super().__init__(static['station_capacity_mw'], forecast_steps)
        self.family = cfg['baseline_family']
        if self.family not in FAMILIES:
            raise ValueError(self.family)
        self.visible = int(cfg.get('history_visible_steps', 12))
        width, dropout = int(cfg['d_model']), float(cfg['dropout'])
        indices, _, weights = build_station_patch_metadata(static['weather_latitude'], static['weather_longitude'],
            static['station_latitude'], static['station_longitude'], 'bilinear', 1)
        self.register_buffer('indices', torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer('weights', torch.as_tensor(weights, dtype=torch.float32))
        lat, lon = map(torch.as_tensor, (static['station_latitude'], static['station_longitude']))
        # All baselines get the same location/capacity information as the main model.
        site = torch.stack(((lat-lat.mean())/lat.std(), (lon-lon.mean())/lon.std(),
                            self.capacity_mw / self.capacity_mw.mean()), -1)
        self.register_buffer('site', site.float())
        channels = int(cfg.get('weather_channels', 14))
        self.is_attention = self.family in {'covariate_transformer', 'pamf_adapted'}
        if self.is_attention:
            self.history_projection = nn.Linear(2, width)
            self.future_projection = nn.Linear(channels + 7, width)
            self.history_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, 4, 2*width,
                dropout, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
            if self.family == 'covariate_transformer':
                self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(width, 4, 2*width,
                    dropout, batch_first=True, norm_first=True), 2)
            else:
                self.cross = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
                self.gate = nn.Linear(2*width, width)
                self.decoder = nn.Sequential(*(ResidualConv(width, 2**i, dropout) for i in range(3)))
        else:
            hist_channels = 1 if self.family == 'weather_anchor_tcn' else self.visible
            weather_channels = 0 if self.family == 'history_tcn' else channels
            self.projection = nn.Linear(hist_channels + weather_channels + 7, width)
            self.decoder = nn.Sequential(*(ResidualConv(width, 2**i, dropout) for i in range(3)))
        self.readout = nn.Sequential(nn.Linear(width, width//2), nn.GELU(), nn.Linear(width//2, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def station_weather(self, weather):
        b, t, c = weather.shape[:3]
        values = weather.flatten(-2)[..., self.indices.flatten()]
        values = values.reshape(b, t, c, len(self.capacity_mw), -1)
        return (values * self.weights[None, None, None]).sum(-1).permute(0, 3, 1, 2)

    def forward(self, *, station_history, weather=None, weather_batch_index=None,
                weather_time_offset_hours=None, origin_phase, **unused):
        b, n = station_history.shape[:2]
        h = station_history[..., -self.visible:]
        anchor = h[..., -1:]
        leads = torch.arange(1, self.forecast_steps+1, device=h.device, dtype=h.dtype)[None].expand(b, -1) / 4
        time = torch.stack((leads / 6, torch.sin(2*torch.pi*leads/24), torch.cos(2*torch.pi*leads/24)), -1)
        # origin_phase is a quarter-hour index 0..3 in the shared loader.
        phase = origin_phase.to(h).reshape(b, -1)[:, :1, None].expand(-1, self.forecast_steps, -1) / 3
        context = torch.cat((time[:, None].expand(-1,n,-1,-1), self.site[None,:,None].expand(b,-1,self.forecast_steps,-1),
                             phase[:,None].expand(-1,n,-1,-1)), -1)
        if self.family != 'history_tcn':
            nodes = self.station_weather(weather).index_select(0, weather_batch_index)
            w = interpolate(nodes, weather_time_offset_hours, leads)
        if self.is_attention:
            ht = torch.arange(-self.visible+1, 1, device=h.device, dtype=h.dtype) / (4*6)
            history = torch.stack((h, ht.expand_as(h)), -1).reshape(b*n, self.visible, 2)
            memory = self.history_encoder(self.history_projection(history))
            future = torch.cat((w, context), -1)
            query = self.future_projection(future).reshape(b*n,self.forecast_steps,-1)
            if self.family == 'covariate_transformer':
                x = self.decoder(query, memory)
            else:
                attended, _ = self.cross(query, memory, memory, need_weights=False)
                gate = torch.sigmoid(self.gate(torch.cat((query, attended), -1)))
                x = self.decoder(query + gate * attended)
        else:
            history = anchor if self.family == 'weather_anchor_tcn' else h
            pieces = [history[:,:,None].expand(-1,-1,self.forecast_steps,-1), context]
            if self.family != 'history_tcn':
                pieces.append(w)
            x = self.decoder(self.projection(torch.cat(pieces, -1)).reshape(b*n,self.forecast_steps,-1))
        correction = self.readout(x).reshape(b,n,self.forecast_steps)
        return self.hierarchical_output(anchor + correction)

def nearest_interpolate(nodes,times,leads):
    # nodes [B,N,M,C], times [B,M], leads [B,L]
    d=(times[:,:,None]-leads[:,None,:]).abs(); idx=d.argmin(1)
    shape=(nodes.shape[0],nodes.shape[1],leads.shape[1],nodes.shape[3])
    return nodes.gather(2,idx[:,None,:,None].expand(shape))

def monotone_cubic(nodes,times,leads):
    # Fritsch-Carlson monotone Hermite interpolation along the native NWP time axis.
    B,N,M,C=nodes.shape; L=leads.shape[1]
    h=(times[:,1:]-times[:,:-1]).clamp_min(1e-5)
    d=(nodes[:,:,1:]-nodes[:,:,:-1])/h[:,None,:,None]
    m=torch.zeros_like(nodes); m[:,:,0]=d[:,:,0]; m[:,:,-1]=d[:,:,-1]
    hp=h[:,:-1]; hn=h[:,1:]; dp=d[:,:,:-1]; dn=d[:,:,1:]
    good=(dp*dn)>0
    w1=2*hn+hp; w2=hn+2*hp
    dp_safe=torch.where(dp.abs()>1e-6,dp,torch.ones_like(dp)); dn_safe=torch.where(dn.abs()>1e-6,dn,torch.ones_like(dn))
    mm=(w1[:,None,:,None]+w2[:,None,:,None])/(w1[:,None,:,None]/dp_safe+w2[:,None,:,None]/dn_safe)
    m[:,:,1:-1]=torch.where(good,mm,torch.zeros_like(mm))
    right=torch.searchsorted(times.contiguous(),leads.contiguous()).clamp(1,M-1); left=right-1
    t0=times.gather(1,left); t1=times.gather(1,right); s=((leads-t0)/(t1-t0).clamp_min(1e-5)).clamp(0,1)
    shape=(B,N,L,C)
    y0=nodes.gather(2,left[:,None,:,None].expand(shape)); y1=nodes.gather(2,right[:,None,:,None].expand(shape))
    m0=m.gather(2,left[:,None,:,None].expand(shape)); m1=m.gather(2,right[:,None,:,None].expand(shape))
    hh=(t1-t0)[:,None,:,None]; s=s[:,None,:,None]
    h00=2*s**3-3*s**2+1; h10=s**3-2*s**2+s; h01=-2*s**3+3*s**2; h11=s**3-s**2
    return h00*y0+h10*hh*m0+h01*y1+h11*hh*m1

class TemporalAlignedBaseline(CovariateBaseline):
    def __init__(self,cfg,static,forecast_steps=24):
        base=dict(cfg);base['baseline_family']='covariate_transformer'
        super().__init__(base,static,forecast_steps)
        self.variant=str(cfg.get('alignment_variant','A1'))
        self.scope=str(cfg.get('native_scope','full'))
        width=int(cfg['d_model']); channels=int(cfg.get('weather_channels',14)); drop=float(cfg.get('dropout',.05))
        self.native_projection=nn.Linear(channels,width)
        self.align_q=nn.Linear(width,width,bias=False); self.align_k=nn.Linear(width,width,bias=False); self.align_v=nn.Linear(width,width,bias=False)
        self.align_out=nn.Linear(width,width)
        self.relative_bias=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,4))
        self.align_alpha=nn.Parameter(torch.zeros(()))
        self.context_projection=nn.Linear(7,width)
        self.pre_future=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,2*width,drop,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
        self.weather_readout=nn.Sequential(nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,1))
        self.history_readout=nn.Sequential(nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,1))
        self.query_only_cross=nn.MultiheadAttention(width,4,dropout=drop,batch_first=True)
        # Matched retrieval controls.  P0/P1/P2 use the 24 interpolated
        # tokens (pointwise / local-4 / global); N0/N1/N2 use the same
        # attention with native 7 / local-2 / global-7 tokens.  The mask is
        # applied before softmax for every target, rather than flattening a
        # local token pool across targets.
        self.add_history_readout=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width//1,1))
        # All new residual/readout paths start at zero where appropriate.
        nn.init.zeros_(self.align_out.weight);nn.init.zeros_(self.align_out.bias)
        for h in (self.weather_readout[-1],self.history_readout[-1]): nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    def _native_select(self,nodes,times,leads):
        if self.scope=='native7':
            # Matched-input control: retain exactly the first seven native
            # nodes used by A1's fixed interpolation (the eighth node is never
            # accessed by its 0.25--6 h targets).
            return nodes[:,:,:7],times[:,:7]
        if self.scope=='full': return nodes,times
        k=2 if self.scope=='local2' else 4
        dist=(times[:,:,None]-leads[:,None,:]).abs(); idx=dist.topk(min(k,times.shape[1]),dim=1,largest=False).indices
        B,N,M,C=nodes.shape; L=leads.shape[1]; kk=idx.shape[1]
        idx_l=idx.permute(0,2,1)  # [B,L,k]
        ids=idx_l[:,None,:,:,None].expand(B,N,L,kk,C)
        vals=nodes[:, :, None, :, :].expand(B,N,L,M,C).gather(3,ids).reshape(B,N,L*kk,C)
        tids=idx_l[:,None,:,:].expand(B,N,L,kk)
        st=times[:,None,None,:].expand(B,N,L,M).gather(3,tids).reshape(B,N,L*kk)
        return vals,st
    def _align(self,q,nodes,times,leads,relative):
        # q [B,N,L,D], nodes [B,N,M,C], times [B,M], leads [B,L]
        B,N,L,D=q.shape; M=nodes.shape[2]; H=4; dh=D//H
        k=self.align_k(self.native_projection(nodes)); v=self.align_v(self.native_projection(nodes)); qq=self.align_q(q)
        qq=qq.reshape(B*N,L,H,dh).permute(0,2,1,3)
        k=k.reshape(B*N,M,H,dh).permute(0,2,1,3); v=v.reshape(B*N,M,H,dh).permute(0,2,1,3)
        scores=torch.matmul(qq,k.transpose(-1,-2))/dh**0.5
        if relative:
            if times.ndim==2:
                dt=(leads[:,:,None]-times[:,None,:]).abs() # B,L,M
                bias=self.relative_bias(dt.reshape(-1,1)).reshape(B,L,M,H).permute(0,3,1,2)
                scores=scores+bias.repeat_interleave(N,0)
            else:
                dt=(leads[:,None,:,None]-times[:,:,None,:]).abs() # B,N,L,M
                bias=self.relative_bias(dt.reshape(-1,1)).reshape(B*N,L,M,H).permute(0,3,1,2)
                scores=scores+bias
        att=torch.softmax(scores,-1); out=torch.matmul(att,v).permute(0,2,1,3).reshape(B*N,L,D)
        return self.align_out(out).reshape(B,N,L,D)
    def _align_masked(self,q,nodes,times,leads,relative,mode):
        B,N,L,D=q.shape; M=nodes.shape[2]; H=4; dh=D//H
        k=self.align_k(self.native_projection(nodes)); v=self.align_v(self.native_projection(nodes)); qq=self.align_q(q)
        qq=qq.reshape(B*N,L,H,dh).permute(0,2,1,3)
        k=k.reshape(B*N,M,H,dh).permute(0,2,1,3); v=v.reshape(B*N,M,H,dh).permute(0,2,1,3)
        scores=torch.matmul(qq,k.transpose(-1,-2))/dh**0.5
        dt=(leads[:,None,:,None]-times[:,:,None,:]).abs() # B,N,L,M
        bias=self.relative_bias(dt.reshape(-1,1)).reshape(B*N,L,M,H).permute(0,3,1,2)
        if relative: scores=scores+bias
        dist=dt
        if mode in {'point','local'}:
            # Match local retrieval capacity across interpolated and native
            # rows: both receive the two nearest available weather tokens.
            kk=1 if mode=='point' else min(2,M)
            ids=dist.topk(kk,dim=-1,largest=False).indices
            allowed=torch.zeros_like(dist,dtype=torch.bool).scatter_(-1,ids,True)
            scores=scores.masked_fill(~allowed.reshape(B*N,1,L,M),torch.finfo(scores.dtype).min)
        att=torch.softmax(scores,-1); out=torch.matmul(att,v).permute(0,2,1,3).reshape(B*N,L,D)
        return self.align_out(out).reshape(B,N,L,D)
    def forward(self,*,station_history,weather=None,weather_batch_index=None,weather_time_offset_hours=None,origin_phase,**unused):
        b,n=station_history.shape[:2]; h=station_history[...,-self.visible:]; anchor=h[...,-1:]
        leads=torch.arange(1,self.forecast_steps+1,device=h.device,dtype=h.dtype)[None].expand(b,-1)/4
        time=torch.stack((leads/6,torch.sin(2*torch.pi*leads/24),torch.cos(2*torch.pi*leads/24)),-1)
        phase=origin_phase.to(h).reshape(b,-1)[:,:1,None].expand(-1,self.forecast_steps,-1)/3
        context=torch.cat((time[:,None].expand(-1,n,-1,-1),self.site[None,:,None].expand(b,-1,self.forecast_steps,-1),phase[:,None].expand(-1,n,-1,-1)),-1)
        nodes=self.station_weather(weather).index_select(0,weather_batch_index)
        times=weather_time_offset_hours
        w=interpolate(nodes,times,leads)
        if self.variant=='A0': w=nearest_interpolate(nodes,times,leads)
        elif self.variant=='A2': w=monotone_cubic(nodes,times,leads)
        ht=torch.arange(-self.visible+1,1,device=h.device,dtype=h.dtype)/(4*6)
        hist=torch.stack((h,ht.expand_as(h)),-1).reshape(b*n,self.visible,2)
        memory=self.history_encoder(self.history_projection(hist))
        if self.variant in {'A6','A6N8','A6N7','R3','R4N','R4A','P0I','P1I','P2I','N0','N1','N2','H0','H1','H2'}:
            q=self.context_projection(context).reshape(b*n,self.forecast_steps,-1).reshape(b,n,self.forecast_steps,-1)
            context_q=q
            native_nodes,native_times=self._native_select(nodes,times,leads)
            if self.variant in {'P0I','P1I','P2I'}:
                native_nodes=interpolate(nodes[:,:,:7],times[:,:7],leads)
                native_times=leads[:,None,:].expand(b,n,-1)
                mode={'P0I':'point','P1I':'local','P2I':'global'}[self.variant]
                q=q+self._align_masked(q,native_nodes,native_times,leads,True,mode)
            elif self.variant in {'N0','N1','N2'}:
                native_nodes=native_nodes[:,:,:7]; native_times=native_times[:,:7]
                native_times=native_times[:,None,:].expand(b,n,-1)
                mode={'N0':'point','N1':'local','N2':'global'}[self.variant]
                q=q+self._align_masked(q,native_nodes,native_times,leads,True,mode)
            elif self.variant in {'H0','H1','H2'}:
                q=q+self._align(q,native_nodes[:,:,:7],native_times[:,:7],leads,True)
            else:
                # Existing A6/R3/R4 branches are retained unchanged.
                pass
            if self.variant=='R3':
                # Same Q/K/V/output/history/decoder as A6, but K/V are the
                # 24 linear target tokens reconstructed from the same seven
                # native nodes. This is the matched representation control.
                native_nodes=interpolate(native_nodes,native_times,leads)
                native_times=leads[:,None,:].expand(b,n,-1)
            if self.variant in {'A6','A6N8','A6N7','R3','R4N','R4A'}:
                q=q+self._align(q,native_nodes,native_times,leads,True)
            if self.variant=='R4A':
                # Shared native weather projection; lambda=1 anchor control.
                anchor_nodes=interpolate(native_nodes if self.variant=='R4A' and native_nodes.shape[2] == 7 else nodes[:,:,:7],
                                         native_times if self.variant=='R4A' and native_times.shape[1] == 7 else times[:,:7],leads)
                q=q+self.native_projection(anchor_nodes)
        else:
            q=self.future_projection(torch.cat((w,context),-1)).reshape(b,n,self.forecast_steps,-1)
            if self.variant=='A3': q=self.pre_future(q.reshape(b*n,self.forecast_steps,-1)).reshape(b,n,self.forecast_steps,-1)
            if self.variant in {'A4','A5','B1','B2','B3','C0','C1','C2','C3'}:
                native_nodes,native_times=self._native_select(nodes,times,leads)
                q=q+self._align(q,native_nodes,native_times,leads,self.variant in {'A5','B1','B2','B3','C0','C1','C2','C3'})
        qflat=q.reshape(b*n,self.forecast_steps,-1); mem=memory
        if self.variant in {'H0'}:
            correction=self.weather_readout(qflat).reshape(b,n,self.forecast_steps)
        elif self.variant=='H1':
            weather_corr=self.weather_readout(qflat).reshape(b,n,self.forecast_steps)
            hs=mem.mean(1)[:,None,:].expand(-1,self.forecast_steps,-1)
            # Strict additive control: the history branch sees only history
            # and target/site context, never the weather-retrieval output.
            cq=context_q.reshape(b*n,self.forecast_steps,-1)
            history_corr=self.add_history_readout(torch.cat((hs,cq),-1)).reshape(b,n,self.forecast_steps)
            correction=weather_corr+history_corr
        elif self.variant=='C0':
            correction=self.weather_readout(qflat).reshape(b,n,self.forecast_steps)
        elif self.variant=='C1':
            attended,_=self.query_only_cross(qflat,mem,mem,need_weights=False)
            correction=self.readout(attended).reshape(b,n,self.forecast_steps)
        elif self.variant=='C2':
            weather_corr=self.weather_readout(qflat).reshape(b,n,self.forecast_steps)
            hist_summary=mem.mean(1).reshape(b,n,-1)[:, :, None, :].expand(-1,-1,self.forecast_steps,-1).reshape(b*n,self.forecast_steps,-1)
            history_corr=self.history_readout(hist_summary).reshape(b,n,self.forecast_steps)
            correction=weather_corr+history_corr
        else:
            x=self.decoder(qflat,mem)
            correction=self.readout(x).reshape(b,n,self.forecast_steps)
        if self.variant=='A6': anchor_term=anchor
        else: anchor_term=anchor
        return self.hierarchical_output(anchor_term+correction)
