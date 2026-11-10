"""State-conditional empirical location/scale calibration; no test access."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls
from portable_dc import edges, summaries, EmpiricalCalibrator
from mean_anchor import fit_bias, select_strength, centered_forecasts

VERSION='fspc-location-scale-v2-20261109'
METHODS=['FSPCBlocked','FSPCMean','FSPCScale','FSPCJoint']
LEVELS=(np.arange(100)+.5)/100


def complete_result(record,task,mode,external=False):
    required={'Native','FSPCBlocked','FSPCRefit','FSPCMean','FSPCScale','FSPCJoint',
        'FSPCMeanBlocked','FSPCScaleBlocked','FSPCJointBlocked'}
    if external:required.update(['EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016'])
    return (record.get('version')==VERSION and record.get('task')==task and record.get('mode')==mode
        and record.get('full_test_split') is True and required.issubset(record.get('methods',{}))
        and (not external or record.get('external_baselines_refitted') is True))


def crps_rows(values,targets):
    a=np.asarray(values,dtype=float);y=np.asarray(targets,dtype=float)
    q=a.shape[1];weights=(2*np.arange(1,q+1)-q-1)/(q*q)
    return np.abs(a-y[:,None]).mean(1)-a@weights


def variance_objective(parameters,mean,sigma2,targets,shape):
    a,b=parameters
    variance=a*sigma2+b;scale=np.sqrt(np.maximum(variance,1e-12))
    values=mean[:,None]+scale[:,None]*shape
    q=shape.shape[1];weights=(2*np.arange(1,q+1)-q-1)/(q*q)
    pair=shape@weights
    loss=np.mean(np.abs(values-targets[:,None]).mean(1)-scale*pair)
    derivative=(np.mean(np.sign(values-targets[:,None])*shape,axis=1)-pair)/(2*scale)
    derivative=np.where(variance>=1e-12,derivative,0.)
    return float(loss),np.array([np.mean(derivative*sigma2),np.mean(derivative)])


def select_blend(reference,candidate,targets,mse_tolerance=0.):
    reference=np.asarray(reference,dtype=float);candidate=np.asarray(candidate,dtype=float)
    targets=np.asarray(targets,dtype=float)
    if reference.shape!=candidate.shape or reference.shape[0]!=len(targets):
        raise ValueError('Selection forecasts must share coordinates')
    if mse_tolerance<0 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Expected finite forecasts and a nonnegative prespecified tolerance')
    delta=candidate-reference;error=reference.mean(1)-targets;shift=delta.mean(1)
    quadratic=float(np.mean(shift*shift));linear=float(np.mean(error*shift))
    limit=float(np.mean(error*error))+mse_tolerance
    if quadratic<=1e-28:
        high=1.
    elif linear>=0 and mse_tolerance==0:
        high=0.
    else:
        high=float(np.clip((-linear+np.sqrt(linear*linear+quadratic*mse_tolerance))/quadratic,0.,1.))
    # Both ordered inputs imply an ordered quantile blend. Pair term is linear.
    q=reference.shape[1];weights=(2*np.arange(1,q+1)-q-1)/(q*q)
    pair0=float(np.mean(reference@weights));pair_delta=float(np.mean(delta@weights))
    residual=reference-targets[:,None]
    def objective(w):return float(np.abs(residual+w*delta).mean()-pair0-w*pair_delta)
    choices=[0.,high]
    if high>1e-12:
        result=minimize_scalar(objective,bounds=(0.,high),method='bounded',options={'xatol':1e-7})
        if not result.success:raise RuntimeError('CRPS blend minimization failed')
        choices.append(float(result.x))
    best=min(choices,key=objective)
    selected_mse=float(np.mean((error+best*shift)**2))
    if selected_mse>limit+1e-10*max(1.,limit):raise RuntimeError('MSE constraint violated')
    return best,dict(weight=best,maximum_feasible_weight=high,mse_tolerance=mse_tolerance,
        reference_mse=float(np.mean(error*error)),selected_mse=selected_mse,
        reference_crps=objective(0.),selected_crps=objective(best),coordinates=len(targets))


def replay_windows(n_windows,horizon,channels,capacity,seed,batch_sizes=None):
    """Replay the original priority reservoir exactly, preserving retained order."""
    sizes=[1]*n_windows if batch_sizes is None else batch_sizes
    if sum(sizes)!=n_windows:raise ValueError('Batch sizes do not cover calibration windows')
    rng=np.random.default_rng(seed);priority=np.empty(0);identities=np.empty(0,dtype=np.int64);seen=0
    for batch in sizes:
        count=batch*horizon*channels;p=rng.random(count);idx=np.arange(seen,seen+count);seen+=count
        if count>capacity:
            take=np.argpartition(p,-capacity)[-capacity:];p=p[take];idx=idx[take]
        priority=np.concatenate([priority,p]);identities=np.concatenate([identities,idx])
        if len(priority)>capacity:
            take=np.argpartition(priority,-capacity)[-capacity:];priority=priority[take];identities=identities[take]
    return identities//(horizon*channels),(identities//channels)%horizon,identities%channels


def chronological_split(window,n_windows,horizon,stride):
    cut=max(1,2*n_windows//3)
    # Fit forecast ending at origin+H-1 must precede selection origin.
    fit_last=cut-int(np.ceil(horizon/stride))
    fit=window<=fit_last;selection=window>=cut
    if fit.sum()<128 or selection.sum()<1:
        # Tiny mathematical fixtures are allowed; campaigns enforce real budgets.
        if not fit.any() or not selection.any():raise ValueError('Insufficient independent calibration windows')
    return fit,selection,dict(cut_window=cut,fit_last_window=fit_last,
        purged_windows=max(0,cut-fit_last-1),fit_windows=int(len(np.unique(window[fit]))),
        selection_windows=int(len(np.unique(window[selection]))),horizon=horizon,stride=stride)


def fit_location(mean,targets,groups,n_states=36,pseudo_count=128):
    mean=np.asarray(mean,dtype=float);targets=np.asarray(targets,dtype=float)
    center=float(mean.mean());spread=float(mean.std())
    if spread>1e-10:
        slope=float(np.dot(mean-center,targets-targets.mean())/np.dot(mean-center,mean-center))
    else:slope=0.
    intercept=float(targets.mean()-slope*center)
    residual=targets-(intercept+slope*mean)
    counts=np.bincount(groups,minlength=n_states)
    shifts=np.bincount(groups,weights=residual,minlength=n_states)/(counts+pseudo_count)
    return dict(intercept=intercept,slope=slope,state_shift=shifts,counts=counts)


def standardized_shapes(residual,groups,n_states=36,min_count=128):
    def normalized(values):
        q=np.quantile(values,LEVELS);q=q-q.mean();sd=float(np.sqrt(np.mean(q*q)))
        if sd<1e-10:
            # No residual variation: retain an identifiable unit shape; scale carries width.
            q=np.linspace(-1,1,100);sd=float(np.sqrt(np.mean(q*q)))
        return q/sd
    global_shape=normalized(residual)
    shapes=np.tile(global_shape,(n_states,1));counts=np.bincount(groups,minlength=n_states)
    for g in range(n_states):
        if counts[g]>=min_count:shapes[g]=normalized(residual[groups==g])
    return shapes,counts


@dataclass
class EmpiricalLocationScale:
    mean_edges: np.ndarray
    scale_edges: np.ndarray
    reference: object=None
    location_parameters: object=None
    variance_parameters: object=None
    shapes: object=None
    metadata: object=None

    def groups(self,rows):
        mean,_,scale=summaries(rows)
        return 6*np.searchsorted(self.mean_edges[1:-1],mean,side='right')+np.searchsorted(self.scale_edges[1:-1],np.log(scale),side='right')

    def location(self,rows):
        mean,_,_=summaries(rows)
        if self.location_parameters is not None:
            p=self.location_parameters
            return p['intercept']+p['slope']*mean+p['state_shift'][self.groups(rows)]
        if isinstance(self.reference,tuple):
            _,bias,strength=self.reference
            return mean+strength*bias[self.groups(rows)]
        return self.reference.location(rows)

    def predict(self,rows):
        mean,_,scale=summaries(rows);groups=self.groups(rows)
        if self.variance_parameters is not None:
            p=self.variance_parameters[groups]
            width=np.sqrt(np.maximum(p[:,0]*scale*scale+p[:,1],1e-12))
            return self.location(rows)[:,None]+width[:,None]*self.shapes[groups]
        if isinstance(self.reference,tuple):
            cal,bias,strength=self.reference
            return centered_forecasts(cal,rows,np.zeros(len(mean),dtype=int),bias,strength)
        return self.reference.predict(rows)+(self.location(rows)-self.reference.location(rows))[:,None]

    @classmethod
    def baseline(cls,rows,targets,train_rows,train_targets,horizon):
        cal=EmpiricalCalibrator.fit_rows(rows,targets,np.zeros(len(targets),dtype=int),horizon,summary_bins=6).compile()
        bias,_=fit_bias(cal,train_rows,train_targets,np.zeros(len(train_targets),dtype=int))
        strength,selection=select_strength(cal,rows,targets,np.zeros(len(targets),dtype=int),bias)
        return cls(cal.mean_edges,cal.scale_edges,reference=(cal,bias,strength),
            metadata=dict(kind='split-matched FSPC',source_rows_used=len(targets),strength_selection=selection))

    @classmethod
    def fit(cls,rows,targets,new_mean,new_scale,reference=None):
        mean,_,sigma=summaries(rows);targets=np.asarray(targets,dtype=float)
        obj=cls(edges(mean,6) if reference is None else reference.mean_edges,
            edges(np.log(sigma),6) if reference is None else reference.scale_edges,reference=reference)
        if reference is None and not new_mean:raise ValueError('A reference mean is required')
        groups=obj.groups(rows);counts=np.bincount(groups,minlength=36)
        if new_mean:obj.location_parameters=fit_location(mean,targets,groups)
        obj.metadata=dict(kind='empirical location-scale',new_mean=new_mean,new_scale=new_scale,
            source_rows_used=len(targets),collapsed_sources_used=int(np.sum(sigma<=1.01e-6)),
            state_counts=counts.tolist(),pseudo_count=128,template_min_count=128)
        if not new_scale:return obj
        location=obj.location(rows);errors=targets-location;sigma2=sigma*sigma
        variance_unit=max(float(np.mean(errors*errors)),1e-8)
        sigma_unit=max(float(np.mean(sigma2)),1e-8)
        design=np.column_stack([sigma2/sigma_unit,np.ones(len(mean))])
        global_coef=nnls(design,errors*errors/variance_unit)[0]
        coef=np.tile(global_coef,(36,1))
        for g in range(36):
            take=groups==g;n=int(take.sum())
            if n>=128:
                local=nnls(design[take],errors[take]*errors[take]/variance_unit)[0]
                coef[g]=(n*local+128*global_coef)/(n+128)
        # Coefficients are dimensionless for stable constrained optimization.
        effective=np.sqrt(np.maximum((coef[groups,0]*sigma2/sigma_unit+coef[groups,1])*variance_unit,1e-12))
        shapes,_=standardized_shapes(errors/effective,groups)
        failures=[]
        for g in range(36):
            take=groups==g;n=int(take.sum())
            if n<128:continue
            m=location[take]/np.sqrt(variance_unit);yy=targets[take]/np.sqrt(variance_unit)
            xx=sigma2[take]/sigma_unit;z=np.tile(shapes[g],(n,1));prior=global_coef
            def objective(p):
                loss,gradient=variance_objective(p,m,xx,yy,z)
                regularizer=128./n*.01
                return loss+regularizer*np.sum((p-prior)**2),gradient+2*regularizer*(p-prior)
            result=minimize(objective,coef[g],jac=True,method='L-BFGS-B',bounds=[(0.,None),(1e-8,None)],
                options={'maxiter':100,'ftol':1e-10,'gtol':1e-6,'maxls':40})
            if np.isfinite(result.fun) and result.fun<=objective(coef[g])[0]+1e-10:coef[g]=result.x
            if not result.success:failures.append(dict(state=g,message=str(result.message)))
        obj.variance_parameters=coef*np.array([variance_unit/sigma_unit,variance_unit])
        p=obj.variance_parameters[groups]
        final_scale=np.sqrt(np.maximum(p[:,0]*sigma2+p[:,1],1e-12))
        obj.shapes,_=standardized_shapes(errors/final_scale,groups)
        obj.metadata.update(variance_unit=variance_unit,sigma_unit=sigma_unit,
            variance_parameters=obj.variance_parameters.tolist(),optimizer_warnings=failures,
            variance_objective='empirical CRPS with 128/n * .01 global-coefficient shrinkage',
            template_refit='standardized residuals using final effective scale; centered unit variance')
        return obj


@dataclass
class QuantileBlend:
    reference: EmpiricalLocationScale
    candidate: EmpiricalLocationScale
    weight: float

    def predict(self,rows):
        base=self.reference.predict(rows)
        if self.weight==0:return base
        return base+self.weight*(self.candidate.predict(rows)-base)
