"""Experiment gates and full-stream common metrics."""
from pathlib import Path
import json
import numpy as np


def severe_trigger(ratio, amplitude, nonfinite=False):
    return bool(nonfinite or (ratio>=5 and amplitude>=100))


def stage_ready(rows, expected):
    return len(rows)==expected and all(r.get('training_complete') and Path(r.get('checkpoint','')).is_file() for r in rows)


def seed_statistics(values):
    if set(values)!={1,2,3}:raise ValueError('Exactly independent training seeds 1,2,3 required')
    a=np.array([values[k] for k in (1,2,3)],dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nonfinite metric')
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),n_seeds=3)


class CommonMetrics:
    def __init__(self):
        self.n=0;self.crps=0.;self.se=0.;self.ae=0.;self.medae=0.;self.bins=np.zeros(10,dtype=np.int64)
    def update(self,samples,target):
        x=np.asarray(samples,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
        assert x.ndim==4 and x.shape[:1]+x.shape[2:]==y.shape
        assert np.isfinite(x).all() and np.isfinite(y).all()
        rows=x.transpose(0,2,3,1).reshape(-1,x.shape[1]);y=y.reshape(-1);s=rows.shape[1]
        # Bound memory for high-dimensional complete-test windows.
        for start in range(0,len(y),16384):
            a=rows[start:start+16384];b=y[start:start+16384];ordered=np.sort(a,axis=1)
            weights=(2*np.arange(1,s+1)-s-1)/(s*s)
            self.crps+=float((np.mean(np.abs(a-b[:,None]),axis=1)-ordered@weights).sum())
            err=a.mean(1)-b;self.se+=float((err**2).sum());self.ae+=float(np.abs(err).sum())
            self.medae+=float(np.abs(np.median(a,axis=1)-b).sum())
            q=np.percentile(a,np.arange(11)*10,axis=1)
            membership=np.clip(((b[None,:]-q)>0).sum(0),1,10)-1
            self.bins+=np.bincount(membership,minlength=10);self.n+=len(b)
    def result(self):
        if not self.n:raise ValueError('Empty stream')
        mse=self.se/self.n;qice=float(np.mean(np.abs(self.bins/self.n-.1)))
        return dict(CRPS=self.crps/self.n,QICE=qice,QICE_percent=100*qice,MSE=mse,MAE=self.ae/self.n,
                    RMSE=float(np.sqrt(mse)),MedianMAE=self.medae/self.n,coordinate_count=self.n)


def atomic(path,obj):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
