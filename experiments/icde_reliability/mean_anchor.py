"""Mean-preserving and controlled-mean ablations of compiled empirical FSDC."""
import numpy as np
from portable_dc import EmpiricalCalibrator, summaries


def state_ids(cal, rows, horizon):
    mean, _, scale = summaries(rows)
    keys = EmpiricalCalibrator.keys(cal, mean, scale, horizon)
    return (keys[:,0]*(len(cal.mean_edges)-1)+keys[:,1])*(len(cal.scale_edges)-1)+keys[:,2]


def centered_forecasts(cal, rows, horizon, bias=None, strength=0.):
    mean, _, scale = summaries(rows)
    ids = cal.template_ids(mean,scale,horizon)
    quantiles = cal.quantiles-cal.quantiles.mean(1,keepdims=True)
    location = mean if bias is None else mean+strength*np.asarray(bias)[state_ids(cal,rows,horizon)]
    return location[:,None]+scale[:,None]*quantiles[ids]


def fit_bias(cal, rows, targets, horizon, pseudo_count=128, min_count=128):
    mean, _, _ = summaries(rows); targets=np.asarray(targets,dtype=float)
    if targets.shape!=mean.shape or not np.isfinite(targets).all():
        raise ValueError('One finite training target is required per coordinate')
    ids=state_ids(cal,rows,horizon)
    states=cal.n_horizon_bins*(len(cal.mean_edges)-1)*(len(cal.scale_edges)-1)
    counts=np.bincount(ids,minlength=states)
    sums=np.bincount(ids,weights=targets-mean,minlength=states)
    bias=np.where(counts>=min_count,sums/(counts+pseudo_count),0.)
    return bias,counts


def select_strength(cal, rows, targets, horizon, bias):
    mean, _, _ = summaries(rows);targets=np.asarray(targets,dtype=float)
    if targets.shape!=mean.shape or not np.isfinite(targets).all():
        raise ValueError('One finite validation target is required per coordinate')
    shift=np.asarray(bias)[state_ids(cal,rows,horizon)];error=targets-mean
    denominator=float(np.dot(shift,shift));numerator=float(np.dot(error,shift))
    strength=float(np.clip(numerator/denominator,0.,1.)) if denominator>0 else 0.
    native=float(np.mean(error**2));selected=float(np.mean((error-strength*shift)**2))
    if selected>native+1e-12*max(1.,native):raise ValueError('Validation strength failed Native fallback')
    return strength,dict(native_mse=native,selected_mse=selected,numerator=numerator,
        denominator=denominator,strength=strength,selection_split='validation',
        bias_fit_split='train',test_targets_used=False,coordinates=len(mean))
