"""Full QRF with ragged bootstrap-leaf storage instead of dense zero padding.

Forest, bootstrap multiplicities, inverse-leaf weighting and finite-sample
linear interpolation match quantile-forest 1.4.2. Nothing is subsampled.
Interpolation convention follows its BSD-3-Clause calc_weighted_quantile.
"""
import numpy as np
from numba import njit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices

@njit(cache=True,nogil=True)
def _evaluate(leaves,node_start,offsets,members,y,levels):
    result=np.empty((len(leaves),len(levels)));n=len(y);trees=leaves.shape[1]
    for row in range(len(leaves)):
        weights=np.zeros(n);total_members=0
        for t in range(trees):
            node=node_start[t]+leaves[row,t]
            total_members+=offsets[node+1]-offsets[node]
        for t in range(trees):
            node=node_start[t]+leaves[row,t];begin=offsets[node];end=offsets[node+1]
            weight=1./(end-begin);weight*=total_members;weight/=trees
            for p in range(begin,end):weights[members[p]]+=weight
        cumulative=np.cumsum(weights);factor=cumulative[-1]-1.;floor=0;ceil=1
        for j in range(len(levels)):
            q=levels[j];position=q*factor+1.;idx=floor
            while idx<n and position>=cumulative[idx]:
                if weights[idx]>0:floor=idx;ceil=idx+1
                idx+=1
            ceil=min(ceil,n-1)
            while weights[ceil]==0:
                ceil+=1
                if ceil>=n:ceil=floor;break
            if floor>=n-1:result[row,j]=y[-1];continue
            if ceil<=0:result[row,j]=y[0];continue
            if floor==ceil or y[floor]==y[ceil]:result[row,j]=y[floor];continue
            p0=min(q,(cumulative[floor]-1.)/factor)
            p1=max(q,(cumulative[ceil]-1.)/factor)
            p1=p0+(p1-p0)/weights[ceil]
            if q>=p1:result[row,j]=y[ceil];continue
            fraction=(q-p0)/(p1-p0)
            result[row,j]=y[floor]+min(1.,max(0.,fraction))*(y[ceil]-y[floor])
    return result

class RaggedQuantileForest:
    max_samples_leaf=None
    def __init__(self,**kwargs):self.forest=RandomForestRegressor(**kwargs)
    def fit(self,x,y):
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=float)
        self.forest.fit(x,y);n=len(y);trees=len(self.forest.estimators_)
        sorter=np.argsort(y);rank=np.argsort(sorter);self.y=y[sorter]
        sizes=[e.tree_.node_count+1 for e in self.forest.estimators_]
        self.node_start=np.r_[0,np.cumsum(sizes)].astype(np.int64)
        self.offsets=np.empty(int(self.node_start[-1]),np.int64)
        self.members=np.empty(n*trees,np.int32)
        for t,estimator in enumerate(self.forest.estimators_):
            bootstrap=_generate_sample_indices(estimator.random_state,n,n)
            leaf=estimator.apply(x[bootstrap]);order=np.argsort(leaf,kind='stable')
            start=t*n;self.members[start:start+n]=rank[bootstrap[order]]
            counts=np.bincount(leaf,minlength=estimator.tree_.node_count)
            at=self.node_start[t];self.offsets[at:at+len(counts)+1]=start+np.r_[0,np.cumsum(counts)]
        return self
    def predict(self,x,quantiles,weighted_quantile=True,weighted_leaves=True,
                aggregate_leaves_first=True,interpolation='linear'):
        if not (weighted_quantile and weighted_leaves and aggregate_leaves_first and interpolation=='linear'):
            raise ValueError('Adapter implements the prespecified full weighted-leaf QRF configuration')
        self.forest.n_jobs=1
        # Identical leaf assignments define exactly the same weighted CDF.
        # Evaluate each such distribution once, then restore query order.
        leaves=self.forest.apply(np.asarray(x,dtype=np.float32))
        unique,inverse=np.unique(leaves,axis=0,return_inverse=True)
        values=_evaluate(unique,self.node_start,self.offsets,self.members,self.y,np.asarray(quantiles,dtype=float))
        return values[inverse]
