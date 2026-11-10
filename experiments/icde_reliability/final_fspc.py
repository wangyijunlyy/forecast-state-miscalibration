"""Complete frozen FSPC deployment, with reference-preserving arithmetic."""
from dataclasses import dataclass
import numpy as np
from portable_dc import CompiledCalibrator,EmpiricalCalibrator,summaries

VERSION='fspc-final-deploy-v1-20261110'
ARMS=('Native','MeanOnly','ShapeOnly','FSPC')


def save_parity_probe(path,rows,horizon,thresholds,reference):
    # Slicing a Fortran reservoir is noncontiguous. A default NPZ copy would
    # silently switch its reduction order to C and invalidate bitwise replay.
    order='F' if rows.flags.f_contiguous else 'C'
    np.savez_compressed(path,rows=np.array(rows[:128],order=order),horizon=horizon[:128],
        thresholds=thresholds[:128],reference=reference[:128],source_row_order=np.array(order))


@dataclass
class CompiledFSPC:
    cal: CompiledCalibrator
    bias: np.ndarray
    strength: float
    intercept: float
    slope: float
    state_shift: np.ndarray
    weight: float

    @classmethod
    def from_model(cls,model):
        c=model.cal;p=model.location
        centered=c.quantiles-c.quantiles.mean(1,keepdims=True)
        cal=CompiledCalibrator(c.mean_edges.copy(),c.scale_edges.copy(),centered,
            c.state_to_template.copy(),c.horizon_length,c.n_horizon_bins)
        return cls(cal,model.bias.copy(),float(model.strength),float(p['intercept']),
                   float(p['slope']),np.asarray(p['state_shift']).copy(),float(model.weight))

    def parts(self,mean,scale,horizon):
        keys=EmpiricalCalibrator.keys(self.cal,mean,scale,horizon)
        ids=(keys[:,0]*(len(self.cal.mean_edges)-1)+keys[:,1])*(len(self.cal.scale_edges)-1)+keys[:,2]
        template=self.cal.state_to_template[ids]
        reference=mean+self.strength*self.bias[ids]
        delta=None
        if self.weight!=0:
            candidate=self.intercept+self.slope*mean+self.state_shift[ids]
            delta=self.weight*(candidate-reference)
        return template,reference,delta

    def transform_summaries(self,mean,scale,horizon):
        ids,reference,delta=self.parts(mean,scale,horizon)
        result=reference[:,None]+scale[:,None]*self.cal.quantiles[ids]
        # Keep the two additions in precisely the reference implementation's order.
        if delta is not None:result=result+delta[:,None]
        return result

    def transform_rows(self,rows,horizon):
        mean,_,scale=summaries(rows)
        return self.transform_summaries(mean,scale,horizon)

    def probabilities_summaries(self,mean,scale,horizon,thresholds):
        mean=np.asarray(mean,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
        threshold=np.asarray(thresholds,dtype=np.float64)
        if threshold.ndim==1:threshold=threshold[:,None]
        if mean.ndim!=1 or scale.shape!=mean.shape or threshold.ndim!=2 or threshold.shape[0]!=len(mean) or np.any(scale<=0):
            raise ValueError('Expected matching positive-scale summaries and per-coordinate thresholds')
        ids,reference,delta=self.parts(mean,scale,horizon)
        lo=np.zeros(threshold.shape,dtype=int);hi=np.full(threshold.shape,100,dtype=int)
        while np.any(lo<hi):
            active=lo<hi;mid=(lo+hi)//2
            value=reference[:,None]+scale[:,None]*self.cal.quantiles[ids[:,None],np.minimum(mid,99)]
            if delta is not None:value=value+delta[:,None]
            right=value<=threshold
            lo=np.where(active&right,mid+1,lo);hi=np.where(active&~right,mid,hi)
        return (100-lo)/100

    def probabilities_rows(self,rows,horizon,thresholds):
        mean,_,scale=summaries(rows)
        return self.probabilities_summaries(mean,scale,horizon,thresholds)

    def save(self,path):
        c=self.cal
        np.savez(path,final_fspc_format_version=np.array(1),mean_edges=c.mean_edges,
            scale_edges=c.scale_edges,centered_quantiles=c.quantiles,state_to_template=c.state_to_template,
            horizon_length=np.array(c.horizon_length),n_horizon_bins=np.array(c.n_horizon_bins),
            bias=self.bias,strength=np.array(self.strength),intercept=np.array(self.intercept),
            slope=np.array(self.slope),state_shift=self.state_shift,weight=np.array(self.weight))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if int(d['final_fspc_format_version'])!=1:raise ValueError('Unsupported full FSPC state')
            c=CompiledCalibrator(d['mean_edges'].copy(),d['scale_edges'].copy(),d['centered_quantiles'].copy(),
                d['state_to_template'].copy(),int(d['horizon_length']),int(d['n_horizon_bins']))
            return cls(c,d['bias'].copy(),float(d['strength']),float(d['intercept']),float(d['slope']),
                d['state_shift'].copy(),float(d['weight']))


def factorial_forecasts(model,rows,horizon):
    rows=np.asarray(rows,dtype=np.float64);mean,_,scale=summaries(rows)
    compiled=CompiledFSPC.from_model(model)
    ids,reference,delta=compiled.parts(mean,scale,horizon)
    location=reference if delta is None else reference+delta
    return dict(Native=rows,MeanOnly=rows+(location-mean)[:,None],
        ShapeOnly=mean[:,None]+scale[:,None]*compiled.cal.quantiles[ids],FSPC=model.predict(rows,horizon))
