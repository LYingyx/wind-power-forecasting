from settings import *
import numpy as np,torch
from physics import thermodynamics,operating_state
from operational import interpolate_nodes

class SharedData:
    def __init__(self,split,spec,rows=None):
        assert split in ['train','val']
        self.split=split;self.spec=spec;self.root=DATA/split
        self.a={k:np.load(self.root/(k+'.npy'),mmap_mode='r') for k in ['h','y','origin','phase','cycle','offset','nodes','dynamics','near','rings']}
        self.rows=np.arange(len(self.a['h'])) if rows is None else np.asarray(rows)
        self.farms=self.a['h'].shape[1];self.n=len(self.rows)*self.farms
        stats=json.loads((ASSETS/'normalization.json').read_text())['sites']
        self.flat=np.array([stats[str(s)]['flat_threshold'] for s in range(self.farms)],'f4')
        self.sm=np.array([stats[str(s)]['sel_mean'] for s in range(self.farms)],'f4');self.ss=np.array([stats[str(s)]['sel_std'] for s in range(self.farms)],'f4')
        self.rm=np.array([stats[str(s)]['ring_mean'] for s in range(self.farms)],'f4');self.rs=np.array([stats[str(s)]['ring_std'] for s in range(self.farms)],'f4')
        self.mu=np.load(ASSETS/'nwp_mean.npy');self.sd=np.load(ASSETS/'nwp_std.npy');self.elevation=np.load(ASSETS/'elevation.npy')
        self.rm_history=np.load(RUN/'rm_cache'/f'{split}.npy',mmap_mode='r') if spec.get('rm') else None
        self.condition=spec.get('condition','neutral');self.reference=None
        if self.condition in ['cb','cbh','anchor','wrong_cb','wrong_cbh']:
            seed=spec['seed'];self.reference=np.load(DATA/f'reference_r{seed}/{split}.npy',mmap_mode='r')
            if self.condition.startswith('wrong'):
                self.donor=np.load(self.root/f'donor_r{seed}.npy');self.donor_base=np.load(DATA/f'reference_r{seed}/train.npy',mmap_mode='r');self.donor_h=np.load(DATA/'train/h.npy',mmap_mode='r')
    def batch(self,indices,device='cpu'):
        idx=np.asarray(indices);ti=self.rows[idx//self.farms];si=idx%self.farms
        h=np.asarray(self.a['h'][ti,si]);y=np.asarray(self.a['y'][ti,si]);b=dict(h=h,phase=np.asarray(self.a['phase'][ti]),site=si)
        if self.spec['family']!='B0':
            cy=self.a['cycle'][ti];offset=self.a['offset'][ti]
            weather=interpolate_nodes(self.a['nodes'][cy,si],offset,np.arange(1,25)/4).astype('f4')
            dyn=interpolate_nodes(self.a['dynamics'][cy,si],offset,np.arange(1,25)/4).astype('f4')
            p=weather*self.sd+self.mu
            den,stab=thermodynamics(p,self.elevation[si,None]);state=operating_state(h,self.flat[si,None])
            x=np.concatenate([den,dyn,stab,np.broadcast_to(state[:,None],(len(h),24,7))],-1)
            b.update(weather=weather,physical=p,x=((x-self.sm[si,None])/self.ss[si,None]).astype('f4'),region=((self.a['rings'][ti,si]-self.rm[si])/self.rs[si]).astype('f4'),near=np.asarray(self.a['near'][ti,si]))
        if self.rm_history is not None:b['rm_history']=np.asarray(self.rm_history[ti,si])
        if self.reference is not None:
            base=np.asarray(self.reference[ti,si]);ch=h
            if self.condition.startswith('wrong'):
                donor=self.donor[ti];base=np.asarray(self.donor_base[donor,si]);ch=np.asarray(self.donor_h[donor,si])
            b.update(base=base,base_delta=base-np.concatenate([ch[:,-1:],base[:,:-1]],-1),condition_h=ch)
        inputs={k:torch.as_tensor(np.array(v),dtype=torch.long if k in ['site','phase'] else torch.float32,device=device) for k,v in b.items()}
        return inputs,torch.as_tensor(np.array(y),device=device),si,ti
