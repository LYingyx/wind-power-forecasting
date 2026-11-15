"""Independent GRIB point/time/parameter audit of labels actually used in loss."""
from pathlib import Path
import sys,datetime,json
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'code'))
from settings import *
import numpy as np
import eccodes as ec

def main():
    lat=np.load(CACHE/'static/station_latitude.npy');lon=np.load(CACHE/'static/station_longitude.npy')
    mu=np.load(CACHE/'static/nwp_mean.npy');sd=np.load(CACHE/'static/nwp_std.npy')
    assert (sd>0).all();records=[]
    for split in ['train','val']:
        c=DATA/'common'/split;origin=np.load(c/'origin_unix.npy');off=np.load(c/'weather_time_offset_hours.npy');cycle=np.load(c/'cycle.npy')
        label=np.load(DATA/'labels'/split/'future_uv.npy',mmap_mode='r')
        issue=np.load(c/'issue0.npy');age=(origin-issue)/3600
        assert (age>=12).all() and (age<24).all()
        for i,node in [(0,0),(len(origin)//2,3),(len(origin)-1,7)]:
            t=int(origin[i]+off[i,node]*3600);dt=datetime.datetime.fromtimestamp(t,datetime.timezone.utc)
            assert dt.year in [2022,2023]
            for comp,(folder,param) in enumerate([('10m_u_component_of_wind',165),('10m_v_component_of_wind',166)]):
                path=Path('/hj102_data_large/ERA5/surface_new')/folder/str(dt.year)/f'{dt:%Y-%m-%d}.grib';found=False
                with path.open('rb') as f:
                    while True:
                        g=ec.codes_grib_new_from_file(f)
                        if g is None:break
                        try:
                            if ec.codes_get(g,'validityDate')!=int(dt.strftime('%Y%m%d')) or ec.codes_get(g,'validityTime')!=dt.hour*100:continue
                            assert ec.codes_get(g,'paramId')==param
                            assert ec.codes_get(g,'Ni')==1440 and ec.codes_get(g,'Nj')==721
                            assert ec.codes_get(g,'iScansNegatively')==0 and ec.codes_get(g,'jScansPositively')==0
                            lat0=ec.codes_get(g,'latitudeOfFirstGridPointInDegrees');lon0=ec.codes_get(g,'longitudeOfFirstGridPointInDegrees')
                            ry=(lat0-lat)/.25;cx=((lon-lon0)%360)/.25;rr=np.floor(ry).astype(int);cc=np.floor(cx).astype(int)
                            fy=ry-rr;fx=cx-cc
                            idx=np.stack([rr*1440+cc,(rr+1)*1440+cc,rr*1440+(cc+1)%1440,(rr+1)*1440+(cc+1)%1440],-1)
                            values=np.asarray(ec.codes_get_double_elements(g,'values',idx.ravel().tolist())).reshape(56,4)
                            weights=np.stack([(1-fy)*(1-fx),fy*(1-fx),(1-fy)*fx,fy*fx],-1)
                            physical=(values*weights).sum(1)
                            actual=label[cycle[i],:,node,comp]*sd[comp+1]+mu[comp+1]
                            error=float(abs(actual-physical).max());assert error<2e-5,error
                            records.append(dict(split=split,origin_utc=str(np.datetime64(int(origin[i]),'s')),valid_utc=dt.isoformat(),node=node,param_id=param,path=str(path),max_absolute_mps=error));found=True;break
                        finally:ec.codes_release(g)
                assert found,(path,dt)
    atomic_json(RUN/'operations/LABEL_AUDIT.json',dict(timestamp=now(),passed=True,source='ERA5 final surface GRIB',farms_per_frame=56,frames=len(records),normalization='original 2022 Pangu mean/std, invertible',test_read=False,checks=records))
    print('LABEL AUDIT PASSED',len(records),max(r['max_absolute_mps'] for r in records),flush=True)
if __name__=='__main__':main()
