"""Vectorized empirical CRPS and original NsDiff-family QICE semantics."""
import numpy as np


class ScoreAccumulator:
    def __init__(self):
        self.crps_sum=0.;self.points=0
        self.bin_counts=np.zeros(10,dtype=np.int64)

    def update(self,samples,target):
        values=np.asarray(samples,dtype=np.float64)
        target=np.asarray(target,dtype=np.float64)
        assert values.shape[0]==target.shape[0] and values.shape[2:]==target.shape[1:]
        rows=values.transpose(0,2,3,1).reshape(-1,values.shape[1])
        y=target.reshape(-1);n=rows.shape[1]
        ordered=np.sort(rows,axis=1)
        weights=(2*np.arange(1,n+1,dtype=np.float64)-n-1)/(n*n)
        scores=np.mean(np.abs(rows-y[:,None]),axis=1)-ordered@weights
        self.crps_sum+=float(scores.sum());self.points+=len(y)
        quantiles=np.percentile(rows,np.arange(11)*10,axis=1)
        memberships=((y[None,:]-quantiles)>0).sum(axis=0)
        bins=np.clip(memberships,1,10)-1
        self.bin_counts+=np.bincount(bins,minlength=10)

    def result(self):
        assert self.points>0 and self.bin_counts.sum()==self.points
        qice=float(np.mean(np.abs(self.bin_counts/self.points-.1)))
        return dict(CRPS=self.crps_sum/self.points,QICE=qice,QICE_percent=100*qice,
                    coordinate_count=self.points)
