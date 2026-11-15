"""Explicit units and validity masks; diagnostic proxies, not measured hub physics."""
import numpy as np

G = 9.80665
RD = 287.05
KAPPA = 0.2854
OMEGA = 7.2921159e-5

def thermodynamics(weather, elevation_m, hub_height_m=100.):
    """Physical channels in Pa/K/m²s⁻²/m s⁻¹. No CF upper bound is imposed."""
    w = np.asarray(weather, dtype=np.float64)
    t = w[..., 3]
    z = np.asarray(elevation_m, dtype=np.float64)
    # Dry hypsometric approximation; MSLP is NOT local surface pressure.
    column_t = np.maximum(t + 0.0065*z/2, 180.)
    ps = w[..., 0] * np.exp(-G*z/(RD*column_t))
    thub = np.maximum(t - 0.0065*hub_height_m, 180.)
    phub = ps*np.exp(-G*hub_height_m/(RD*np.maximum((t+thub)/2, 180.)))
    rho = phub/(RD*thub)
    theta_s = t*(1e5/np.maximum(ps, 1.))**KAPPA
    theta_925 = w[..., 9]*(1000./925.)**KAPPA
    height_925 = w[..., 5]/G-z
    valid_925 = height_925 > hub_height_m
    du, dv = w[..., 11]-w[..., 1], w[..., 13]-w[..., 2]
    height = np.maximum(height_925-10., 1.)
    theta_gradient = np.where(valid_925, 100.*(theta_925-theta_s)/height, 0.)
    shear = np.where(valid_925, 100.*np.hypot(du,dv)/height, 0.)
    rb = np.where(valid_925, G/np.maximum(theta_s,180.)*(theta_925-theta_s)*height/(du*du+dv*dv+1.), 0.)
    # Neutral logarithmic interpolation is only a proxy; no inferred roughness/flux.
    alpha = np.where(valid_925, np.log(hub_height_m/10.)/np.log(np.maximum(height_925,hub_height_m+1.)/10.), 0.)
    uh, vh = w[...,1]+alpha*du, w[...,2]+alpha*dv
    v10 = np.hypot(w[...,1],w[...,2]); vhub=np.hypot(uh,vh)
    density = np.stack((rho/1.225, (rho/1.225)**(1/3)*v10/10., rho*v10**3/1000.),-1)
    stability = np.stack((theta_gradient, shear, np.clip(rb,-5,5), valid_925.astype(float), vhub/10.),-1)
    return density.astype('f4'), stability.astype('f4')

def dynamic_fields(patch, latitude, latitude_step, longitude_step, weights):
    """[cycle,node,row,col,14] physical weather -> station differential features.

    z is geopotential, so geostrophic wind uses grad(z)/f, with no extra g.
    Signed grid spacing handles north-to-south latitude ordering.
    """
    w=np.asarray(patch,dtype=np.float64)
    dx=np.deg2rad(longitude_step)*6371000.*np.cos(np.deg2rad(latitude))
    dy=np.deg2rad(latitude_step)*6371000.
    coriolis=2*OMEGA*np.sin(np.deg2rad(latitude))
    if abs(coriolis)<1e-5: raise ValueError('Geostrophic proxy is invalid near the equator')
    wt=np.asarray(weights).reshape(w.shape[-3:-1])
    at=lambda a:(a*wt).sum((-2,-1))
    ug=-at(np.gradient(w[...,5],dy,axis=-2,edge_order=2))/coriolis
    vg=at(np.gradient(w[...,5],dx,axis=-1,edge_order=2))/coriolis
    dtdx=at(np.gradient(w[...,9],dx,axis=-1,edge_order=2))
    dtdy=at(np.gradient(w[...,9],dy,axis=-2,edge_order=2))
    point=(w*wt[None,None,:,:,None]).sum((-3,-2))
    adv=-(point[...,11]*dtdx+point[...,13]*dtdy)*3600.
    dpdt=np.gradient(point[...,0],1.,axis=1)/100.
    dvdt=np.gradient(np.hypot(point[...,1],point[...,2]),1.,axis=1)
    return np.stack((ug/10.,vg/10.,(point[...,1]-ug)/10.,(point[...,2]-vg)/10.,adv,dpdt,dvdt),-1).astype('f4')

def calendar(origin_unix, leads_hours):
    valid=np.asarray(origin_unix)[:,None]+np.rint(np.asarray(leads_hours)[None]*3600).astype('i8')
    local=valid+8*3600
    days=local.astype('datetime64[s]').astype('datetime64[D]')
    year_start=days.astype('datetime64[Y]').astype('datetime64[D]')
    year_end=(days.astype('datetime64[Y]')+1).astype('datetime64[D]')
    year_fraction=((days-year_start).astype(float)+(local%86400)/86400.)/(year_end-year_start).astype(float)
    angle=2*np.pi*(local%86400)/86400.
    return np.stack((np.sin(angle),np.cos(angle),np.sin(2*np.pi*year_fraction),np.cos(2*np.pi*year_fraction)),-1).astype('f4')

def operating_state(history, flat_threshold):
    """History-only plateau descriptors. No curtailment/icing labels inferred."""
    h=np.asarray(history); d=np.diff(h,axis=-1); flat=np.abs(d)<=flat_threshold
    duration=np.cumprod(flat[:,::-1],axis=1).sum(1)/23.
    return np.stack((h[:,-4:].std(1),h[:,-8:].std(1),np.ptp(h[:,-4:],axis=1),
                     h[:,-1]-h[:,-5],h[:,-1]-h[:,0],flat[:,-4:].mean(1),duration),-1).astype('f4')

def neighbor_geometry(lat, lon, maximum_k=8, radius_km=150.):
    lat=np.asarray(lat);lon=np.asarray(lon);n=len(lat)
    phi=np.deg2rad(lat);lam=np.deg2rad(lon)
    dphi=phi[None,:]-phi[:,None];dlam=lam[None,:]-lam[:,None]
    a=np.sin(dphi/2)**2+np.cos(phi[:,None])*np.cos(phi[None,:])*np.sin(dlam/2)**2
    distance=2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))
    np.fill_diagonal(distance,np.inf)
    ids=np.argsort(distance,axis=1,kind='stable')[:,:maximum_k]
    dist=np.take_along_axis(distance,ids,axis=1)
    # Vector FROM neighbor TO target, east/north in km.
    dx=(lon[:,None]-lon[ids])*np.pi/180*6371*np.cos(phi[:,None])
    dy=(lat[:,None]-lat[ids])*np.pi/180*6371
    return ids,np.stack((dx,dy,dist),-1).astype('f4'),dist<=radius_km
