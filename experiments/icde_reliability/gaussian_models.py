"""Low-parameter, non-binned calibration fit only on supplied calibration data."""
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import ndtr,ndtri


def gaussian_crps(mean,scale,target):
    scale=np.maximum(np.asarray(scale,dtype=float),1e-6)
    z=(np.asarray(target)-np.asarray(mean))/scale
    phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return scale*(z*(2*ndtr(z)-1)+2*phi-1/np.sqrt(np.pi))


def gaussian_loss_gradient(params,mu,sigma,target,affine=False):
    if affine:c0,c1,A,B=params;location=c0+c1*mu
    else:A,B=params;location=mu
    variance=A+B*sigma*sigma
    scale=np.sqrt(np.maximum(variance,1e-12))
    z=(target-location)/scale
    phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    dlocation=1-2*ndtr(z)
    dvariance=(2*phi-1/np.sqrt(np.pi))/(2*scale)
    dvariance=np.where(variance>=1e-12,dvariance,0.)
    grad=[float(dvariance.mean()),float(np.mean(dvariance*sigma*sigma))]
    if affine:grad=[float(dlocation.mean()),float(np.mean(dlocation*mu))]+grad
    return float(gaussian_crps(location,scale,target).mean()),np.asarray(grad)


def ensemble_summaries(rows,target):
    rows=np.asarray(rows,dtype=np.float64);mu=rows.mean(1)
    delta=rows-mu[:,None];error=np.asarray(target,dtype=float)-mu
    S=rows.shape[1];weights=(2*np.arange(1,S+1)-S-1)/(S*S)
    pair=float(np.mean(np.sort(rows,axis=1)@weights))
    return delta,error,max(0.,pair)


def fit_global_scale(rows,target):
    delta,error,pair=ensemble_summaries(rows,target)
    def objective(a):return float(np.mean(np.abs(a*delta-error[:,None]))-a*pair)
    def derivative(a):return float(np.mean(np.sign(a*delta-error[:,None])*delta)-pair)
    if np.mean(np.abs(delta))<1e-12:
        return dict(kind='global_scale',scale=1.,fit_objective=objective(1.),optimizer_success=True,
            optimizer_message='No within-coordinate native spread; width coefficient unidentifiable',iterations=0)
    high=1.
    for _ in range(60):
        if derivative(high)>=0:break
        high*=2
    else:raise RuntimeError('Failed to bracket global-width CRPS minimum')
    result=minimize_scalar(objective,bounds=(0.,high),method='bounded',options={'xatol':1e-8,'maxiter':200})
    candidates=[0.,1.,float(result.x)]
    best=min(candidates,key=objective)
    return dict(kind='global_scale',scale=float(best),fit_objective=objective(best),
        optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nfev))


def fit_global_affine(rows,target):
    delta,error,pair=ensemble_summaries(rows,target)
    initial=fit_global_scale(rows,target)
    def objective(params):
        shift,scale=params;residual=shift+scale*delta-error[:,None]
        sign=np.sign(residual)
        return float(np.mean(np.abs(residual))-scale*pair),np.array([sign.mean(),np.mean(sign*delta)-pair])
    result=minimize(objective,[0.,initial['scale']],jac=True,method='L-BFGS-B',
        bounds=[(None,None),(0.,None)],options={'maxiter':200,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    return dict(kind='global_affine',shift=float(result.x[0]),scale=float(result.x[1]),
        fit_objective=float(result.fun),optimizer_success=bool(result.success),
        optimizer_message=str(result.message),iterations=int(result.nit))


def fit_gaussian(mu,sigma,target,affine=False):
    mu=np.asarray(mu,dtype=float);sigma=np.asarray(sigma,dtype=float);target=np.asarray(target,dtype=float)
    residual_var=max(float(np.var(target-mu)),1e-6)
    candidates=[]
    for fraction,slope in [(.01,1.),(.5,.5)]:
        initial=[max(1e-8,fraction*residual_var),slope]
        bounds=[(0.,None),(0.,None)]
        if affine:initial=[0.,1.]+initial;bounds=[(None,None),(None,None)]+bounds
        result=minimize(gaussian_loss_gradient,initial,args=(mu,sigma,target,affine),jac=True,
            method='L-BFGS-B',bounds=bounds,
            options={'maxiter':200,'ftol':1e-12,'gtol':1e-7,'maxls':50})
        candidates.append(result)
    successful=[r for r in candidates if r.success]
    result=min(successful or candidates,key=lambda r:r.fun)
    if affine:c0,c1,A,B=result.x
    else:A,B=result.x;c0=0.;c1=1.
    return dict(kind='gaussian_affine' if affine else 'gaussian_variance',mean_intercept=float(c0),
        mean_multiplier=float(c1),variance_floor=float(A),variance_multiplier=float(B),
        fit_analytic_gaussian_crps=float(result.fun),optimizer_success=bool(result.success),
        optimizer_message=str(result.message),iterations=int(result.nit),
        initializations=[dict(success=bool(r.success),objective=float(r.fun)) for r in candidates])


def predict(model,rows):
    rows=np.asarray(rows,dtype=float);mu=rows.mean(1)
    if model['kind'] in ['global_scale','global_affine']:
        return mu[:,None]+model.get('shift',0.)+model['scale']*(rows-mu[:,None])
    sigma=rows.std(1,ddof=1).clip(1e-6)
    mean=model['mean_intercept']+model['mean_multiplier']*mu
    scale=np.sqrt(np.maximum(model['variance_floor']+model['variance_multiplier']*sigma*sigma,1e-12))
    grid=ndtri((np.arange(100)+.5)/100)
    return mean[:,None]+scale[:,None]*grid
