"""The completed affine-plus-state FSPC mean, generalized to component studies."""
from dataclasses import dataclass
import numpy as np
from portable_dc import CompiledCalibrator,summaries
from granularity import fit_partition
from mean_anchor import fit_bias,select_strength,centered_forecasts,state_ids
from location_scale import fit_location,select_blend
from mean_improved_all import split_coordinates

VERSION='fspc-unified-affine-mean-v1-20261110'


@dataclass
class UnifiedMean:
    cal: object
    bias: np.ndarray
    strength: float
    location: dict
    weight: float

    def predict(self,rows,horizon,weight=None):
        mean,_,_=summaries(rows);ids=state_ids(self.cal,rows,horizon)
        base=centered_forecasts(self.cal,rows,horizon,self.bias,self.strength)
        eta=self.weight if weight is None else weight
        if eta==0:return base
        p=self.location;candidate=p['intercept']+p['slope']*mean+p['state_shift'][ids]
        reference=mean+self.strength*self.bias[ids]
        return base+eta*(candidate-reference)[:,None]

    def uncentered(self,rows,horizon):
        mean,_,scale=summaries(rows)
        return mean[:,None]+scale[:,None]*self.cal.quantiles[self.cal.template_ids(mean,scale,horizon)]

    def to_record(self):
        return dict(version=VERSION,mean_edges=self.cal.mean_edges.tolist(),scale_edges=self.cal.scale_edges.tolist(),
            quantiles=self.cal.quantiles.tolist(),state_to_template=self.cal.state_to_template.tolist(),
            horizon_length=int(self.cal.horizon_length),n_horizon_bins=int(self.cal.n_horizon_bins),
            bias=self.bias.tolist(),strength=float(self.strength),weight=float(self.weight),
            location={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.location.items()})

    @classmethod
    def from_record(cls,r):
        assert r['version']==VERSION
        cal=CompiledCalibrator(np.array(r['mean_edges']),np.array(r['scale_edges']),np.array(r['quantiles']),
            np.array(r['state_to_template'],dtype=int),r['horizon_length'],r['n_horizon_bins'])
        p=dict(r['location']);p['state_shift']=np.array(p['state_shift']);p['counts']=np.array(p['counts'])
        return cls(cal,np.array(r['bias']),r['strength'],p,r['weight'])

    @classmethod
    def from_existing(cls,blend):
        base=blend.reference;cal,bias,strength=base.reference
        return cls(cal,bias,float(strength),blend.candidate.location_parameters,float(blend.weight))


def fit_unified(train,train_y,train_h,val,val_y,val_h,window,n_windows,stride,horizon,
                bins=6,horizon_groups=1,budget=50000,ordering_seed=0,template_split='validation'):
    if template_split not in ('validation','train'):raise ValueError('Unknown template split')
    take=np.arange(len(val_y)) if len(val_y)<=budget else np.random.default_rng(ordering_seed).permutation(len(val_y))[:budget]
    v,y,h,w=val[take],val_y[take],val_h[take],window[take]
    fit,select,split=split_coordinates(w,n_windows,horizon,stride)
    if fit is None:fr,fy,fh,sr,sy,sh=train,train_y,train_h,v,y,h
    else:fr,fy,fh,sr,sy,sh=v[fit],y[fit],h[fit],v[select],y[select],h[select]
    def fitted(rows,targets,hids):
        tr,ty,th=(train,train_y,train_h) if template_split=='train' else (rows,targets,hids)
        empirical,metadata=fit_partition(tr,ty,th,horizon,bins,horizon_groups)
        cal=empirical.compile();bias,_=fit_bias(cal,train,train_y,train_h)
        strength,_=select_strength(cal,rows,targets,hids,bias)
        mean,_,_=summaries(rows);groups=state_ids(cal,rows,hids)
        p=fit_location(mean,targets,groups,n_states=bins*bins*horizon_groups)
        return UnifiedMean(cal,bias,strength,p,1.),metadata
    blocked,_=fitted(fr,fy,fh)
    weight,selection=select_blend(blocked.predict(sr,sh,weight=0.),blocked.predict(sr,sh,weight=1.),sy)
    full,meta=fitted(v,y,h);full.weight=weight
    meta.update(version=VERSION,template_split=template_split,split=split,weight=weight,selection=selection,
        requested_budget=budget,calibration_coordinates=len(y),test_targets_used=False,
        parameters_frozen_before_test=True,mean_fit='global affine plus shrunk state residual',
        pseudo_count=128,refit='all calibration coordinates after blend selection; weight frozen')
    return full,meta


def complete_diffusion(record):
 return (record.get('version')==VERSION and record.get('full_test_split') is True and
  record.get('original_corrected_scores_reproduced') is True and
  {'Native','FSPCMean','GlobalDC','GlobalPIT','ConditionalPIT','DC36'}.issubset(record.get('methods',{})))


def atomic_npz(path,**arrays):
 from pathlib import Path
 path=Path(path);temp=path.with_name(path.stem+'.tmp.npz')
 np.savez_compressed(temp,**arrays);temp.replace(path)


def missing_internal_controls(record):
 methods=(record or {}).get('methods',{})
 return [name for name in ['GlobalDC','GlobalPIT','ConditionalPIT'] if name not in methods]
