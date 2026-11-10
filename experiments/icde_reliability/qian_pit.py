"""Qian et al. (Technometrics 2025), Section 3.3 soft-weighted CRPS PIT.

For the paper's Gaussian input CDFs, d_ij = sigma_i * Delta Phi^-1(tau_j).
The Delta term cancels. The optimal PIT map is therefore the conditional
weighted empirical CDF of Gaussian-standardized errors, with weights
sigma_i * Epanechnikov((mu_i-mu)/h). A persistent moment tree evaluates this
exact weighted distribution without constructing an N-by-N quantile array.
Ill-conditioned moment queries use direct local kernel sums. Coincident boundary
neighbors use the continuous outward-bandwidth limit. No PIT knots, residual-source
exclusions, or empirical-pool truncation occur.
"""
import numpy as np
try:
    from numba import njit,prange
except ImportError:
    # A slow reference route supports algorithm checks without optional JIT.
    def njit(*args,**kwargs):
        if args and callable(args[0]):return args[0]
        return lambda f:f
    prange=range

@njit(cache=True)
def _build(mu,weight,rank,nz):
    n=len(mu);depth=int(np.ceil(np.log2(max(2,nz))))+2
    size=n*depth+1;left=np.zeros(size,np.int32);right=np.zeros(size,np.int32)
    sums=np.zeros((size,3),np.float64);roots=np.zeros(n+1,np.int32);used=1
    for i in range(n):
        old=roots[i];new=used;used+=1;roots[i+1]=new;l=0;r=nz
        while True:
            left[new]=left[old];right[new]=right[old]
            sums[new,0]=sums[old,0]+weight[i]
            sums[new,1]=sums[old,1]+weight[i]*mu[i]
            sums[new,2]=sums[old,2]+weight[i]*mu[i]*mu[i]
            if r-l==1:break
            mid=(l+r)//2;child=used;used+=1
            if rank[i]<mid:
                left[new]=child;old=left[old];r=mid
            else:
                right[new]=child;old=right[old];l=mid
            new=child
    return roots,left[:used],right[:used],sums[:used]

@njit(cache=True)
def _radius(values,q,k):
    n=len(values);count=max(1,n//k);split=np.searchsorted(values,q)
    lo=max(0,count-(n-split));hi=min(count,split)
    while lo<=hi:
        a=(lo+hi)//2;b=count-a
        al=q-values[split-a] if a else -np.inf
        an=q-values[split-a-1] if a<split else np.inf
        bl=values[split+b-1]-q if b else -np.inf
        bn=values[split+b]-q if b<n-split else np.inf
        if al>bn:hi=a-1
        elif bl>an:lo=a+1
        else:return max(al,bl)
    raise ValueError('Nearest-neighbor radius is not identifiable')

@njit(cache=True)
def _mass(sums,a,b,q,h):
    s0=sums[b,0]-sums[a,0]
    if h==0:return max(0.,s0)
    s1=sums[b,1]-sums[a,1];s2=sums[b,2]-sums[a,2]
    return max(0.,s0-(s2-2*q*s1+q*q*s0)/(h*h))

@njit(cache=True)
def _bounds(mu,q,k):
    h=_radius(mu,q,k)
    return np.searchsorted(mu,q-h,'left'),np.searchsorted(mu,q+h,'right'),h

@njit(cache=True,parallel=True)
def _quantiles(mu,z,roots,left,right,sums,query,k,levels):
    out=np.empty((len(query),len(levels)))
    for i in prange(len(query)):
        q=query[i];lo,hi,h=_bounds(mu,q,k);aa=roots[lo];bb=roots[hi]
        total=_mass(sums,aa,bb,q,h)
        if total<=0:
            out[i]=np.nan
            continue
        for j in range(len(levels)):
            a=aa;b=bb;l=0;r=len(z);target=levels[j]*total
            while r-l>1:
                mid=(l+r)//2;mass=_mass(sums,left[a],left[b],q,h)
                if target<=mass:
                    a=left[a];b=left[b];r=mid
                else:
                    target-=mass;a=right[a];b=right[b];l=mid
            out[i,j]=z[l]
    return out

@njit(cache=True,parallel=True)
def _cdf(mu,z,roots,left,right,sums,query,values,k):
    out=np.empty(len(query))
    for i in prange(len(query)):
        q=query[i];lo,hi,h=_bounds(mu,q,k);a=roots[lo];b=roots[hi]
        total=_mass(sums,a,b,q,h)
        if total<=0:
            out[i]=np.nan
            continue
        cut=np.searchsorted(z,values[i],'right');l=0;r=len(z);mass=0.
        while r-l>1:
            mid=(l+r)//2
            if cut>=mid:
                mass+=_mass(sums,left[a],left[b],q,h);a=right[a];b=right[b];l=mid
            else:a=left[a];b=left[b];r=mid
        if cut>l:mass+=_mass(sums,a,b,q,h)
        out[i]=min(1.,max(0.,mass/total))
    return out

@njit(cache=True,parallel=True)
def _unstable_queries(mu,roots,sums,query,k):
    mask=np.zeros(len(query),np.bool_)
    for i in prange(len(query)):
        q=query[i];lo,hi,h=_bounds(mu,q,k)
        if h==0:continue
        a=roots[lo];b=roots[hi];mass=_mass(sums,a,b,q,h)
        magnitude=(abs(sums[a,2])+abs(sums[b,2])+
            2*abs(q)*(abs(sums[a,1])+abs(sums[b,1]))+
            q*q*(abs(sums[a,0])+abs(sums[b,0])))
        roundoff=64*np.finfo(np.float64).eps*magnitude/(h*h)
        mask[i]=(h*h<1e-6*(1+q*q) or mass<=0 or roundoff>1e-8*mass)
    return mask

@njit(cache=True,parallel=True)
def _direct_cdf(mu,weight,rank,z,query,values,k):
    out=np.empty(len(query))
    for i in prange(len(query)):
        q=query[i];lo,hi,h=_bounds(mu,q,k);total=0.;covered=0.
        for j in range(lo,hi):
            w=weight[j] if h==0 else weight[j]*max(0.,1-((mu[j]-q)/h)**2)
            total+=w
            if z[rank[j]]<=values[i]:covered+=w
        if total<=0:
            # When all nearest means coincide on the kernel boundary, take
            # h -> h+; their common kernel factor cancels from normalization.
            for j in range(lo,hi):
                total+=weight[j]
                if z[rank[j]]<=values[i]:covered+=weight[j]
        out[i]=covered/total if total>0 else np.nan
    return out

@njit(cache=True,parallel=True)
def _coincident_representatives(mu,query,k):
    representative=query.copy()
    for i in prange(len(query)):
        lo,hi,h=_bounds(mu,query[i],k)
        if hi>lo and mu[lo]==mu[hi-1]:representative[i]=mu[lo]
    return representative

@njit(cache=True,parallel=True)
def _direct_quantiles(mu,weight,rank,z,query,k,levels):
    out=np.empty((len(query),len(levels)))
    for i in prange(len(query)):
        q=query[i];lo,hi,h=_bounds(mu,q,k);order=np.argsort(rank[lo:hi])
        cumulative=np.empty(hi-lo);total=0.
        for t in range(hi-lo):
            j=lo+order[t]
            w=weight[j] if h==0 else weight[j]*max(0.,1-((mu[j]-q)/h)**2)
            total+=w;cumulative[t]=total
        if total<=0:
            for t in range(hi-lo):
                total+=weight[lo+order[t]];cumulative[t]=total
        if total<=0:out[i]=np.nan;continue
        for t in range(len(levels)):
            index=min(np.searchsorted(cumulative,levels[t]*total,'left'),hi-lo-1)
            out[i,t]=z[rank[lo+order[index]]]
    return out

class SoftCRPSPIT:
    @classmethod
    def from_summaries(cls,mu,sigma,y):
        obj=cls();mu=np.asarray(mu,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)
        if np.any(sigma<=0):raise ValueError('Gaussian scales must be positive')
        obj.offset=float(mu.mean());obj.unit=max(float(mu.std()),1e-6)
        order=np.argsort(mu,kind='stable');obj.mu=(mu[order]-obj.offset)/obj.unit
        error=(np.asarray(y,dtype=np.float64)-mu)/sigma
        obj.z,rank=np.unique(error[order],return_inverse=True)
        weight=sigma[order]/sigma.max()
        obj.tree=_build(obj.mu,weight,rank,len(obj.z));obj.k=None
        obj.source_mu=mu[order].copy();obj.source_weight=weight.copy();obj.source_rank=rank.copy()
        obj.stable_cdf_queries=0;obj.stable_quantile_queries=0
        return obj

    @classmethod
    def fit(cls,rows,y,seed,kmax=20):
        mu=np.mean(rows,axis=1);sigma=np.std(rows,axis=1,ddof=1).clip(1e-6)
        obj=cls.from_summaries(mu,sigma,y);z=(y-mu)/sigma
        levels=np.sort(np.random.default_rng(seed).uniform(0,1,3000));history=[]
        for k in range(1,min(kmax,max(1,len(y)//2))+1):
            cpit=np.sort(obj.cdf(mu,z,k))
            pce=float(np.mean((np.searchsorted(cpit,levels,'right')/len(cpit)-levels)**2))
            history.append(dict(k=k,calibration_PCE=pce))
        obj.k=min(history,key=lambda a:(a['calibration_PCE'],a['k']))['k'];obj.history=history
        obj.metadata=dict(method='Qian2025 soft-weighted CRPS, Gaussian input',selected_k=obj.k,
            selection='minimum calibration PCE over K=1..20 and 3000 fixed random probability levels',
            candidates=history,calibration_coordinates=len(y),source_filter=False,
            evaluation='exact kernel-weighted empirical residual CDF; no probability clipping',
            gaussian_scale_floor=1e-6,numerical_revision='stable-local-kernel-20261109',
            stable_fit_cdf_queries=obj.stable_cdf_queries,
            bandwidth_ties='continuous outward bandwidth limit for coincident boundary neighbors')
        return obj

    def standardized_quantiles(self,means,k=None,levels=None):
        if levels is None:levels=(np.arange(100)+.5)/100
        query=(np.asarray(means,dtype=float)-self.offset)/self.unit
        selected=int(k or self.k);levels=np.asarray(levels,dtype=float)
        mask=_unstable_queries(self.mu,self.tree[0],self.tree[3],query,selected)
        result=np.empty((len(query),len(levels)))
        if np.any(~mask):result[~mask]=_quantiles(self.mu,self.z,*self.tree,query[~mask],selected,levels)
        if np.any(mask):
            representative=_coincident_representatives(self.source_mu,np.asarray(means,dtype=float)[mask],selected)
            unique,inverse=np.unique(representative,return_inverse=True)
            values=_direct_quantiles(self.source_mu,self.source_weight,self.source_rank,self.z,unique,selected,levels)
            result[mask]=values[inverse]
            self.stable_quantile_queries+=int(mask.sum())
        if not np.isfinite(result).all():
            i=int(np.flatnonzero(~np.isfinite(result).all(axis=1))[0])
            detail={'mean':float(np.asarray(means)[i]),'k':selected,'direct':bool(mask[i])}
            if hasattr(self,'source_mu'):
                lo,hi,h=_bounds(self.source_mu,float(np.asarray(means)[i]),selected)
                detail.update(lo=int(lo),hi=int(hi),h=float(h),source_min=float(self.source_mu[lo]),source_max=float(self.source_mu[max(lo,hi-1)]))
            raise ValueError('Zero soft-kernel mass or nonfinite quantiles: '+str(detail))
        return result

    def cdf(self,means,z,k=None):
        query=(np.asarray(means,dtype=float)-self.offset)/self.unit
        selected=int(k or self.k);z=np.asarray(z,dtype=float)
        mask=_unstable_queries(self.mu,self.tree[0],self.tree[3],query,selected)
        result=np.empty(len(query))
        if np.any(~mask):result[~mask]=_cdf(self.mu,self.z,*self.tree,query[~mask],z[~mask],selected)
        if np.any(mask):
            result[mask]=_direct_cdf(self.source_mu,self.source_weight,self.source_rank,self.z,
                np.asarray(means,dtype=float)[mask],z[mask],selected)
            self.stable_cdf_queries+=int(mask.sum())
        if not np.isfinite(result).all():raise ValueError('Zero soft-kernel mass or nonfinite CDF')
        return result

    def predict(self,rows):
        mean=rows.mean(1);sigma=rows.std(1,ddof=1).clip(1e-6)
        return mean[:,None]+sigma[:,None]*self.standardized_quantiles(mean)
