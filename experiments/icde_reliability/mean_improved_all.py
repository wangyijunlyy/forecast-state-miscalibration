"""Frozen mean-only protocol shared by all backbones; no model training."""
import numpy as np
from location_scale import EmpiricalLocationScale, QuantileBlend, chronological_split, replay_windows, select_blend

VERSION='fspc-mean-all-v1-20261110'
METHODS=('Native','FSPCRefit','FSPCMean')


def split_coordinates(window,n_windows,horizon,stride):
    reason='too few independent validation coordinates'
    if n_windows>=2:
        try:
            fit,select,meta=chronological_split(window,n_windows,horizon,stride)
            if fit.sum()>=128 and select.sum()>=128:
                meta['protocol']='chronological first two thirds fit, final third selects; overlapping targets purged'
                return fit,select,meta
        except ValueError as error:reason=str(error)
    return None,None,dict(protocol='training forecasts fit; entire validation selects',reason=reason)


def replay_provenance(targets,batch_sizes,h,c,y,capacity,seed):
    n,H,C=targets.shape
    w,hh,cc=replay_windows(n,H,C,capacity,seed,batch_sizes)
    np.testing.assert_array_equal(hh,h);np.testing.assert_array_equal(cc,c)
    np.testing.assert_array_equal(targets[w,h,c],y)
    return w


def fit_mean_models(task,rows,y,window,train,train_y,n_windows,stride):
    fit,select,split=split_coordinates(window,n_windows,task['horizon'],stride)
    if fit is None:fit_rows,fit_y,select_rows,select_y=train,train_y,rows,y
    else:fit_rows,fit_y,select_rows,select_y=rows[fit],y[fit],rows[select],y[select]
    base=EmpiricalLocationScale.baseline(fit_rows,fit_y,train,train_y,task['horizon'])
    candidate=EmpiricalLocationScale.fit(fit_rows,fit_y,True,False,base)
    eta,selection=select_blend(base.predict(select_rows),candidate.predict(select_rows),select_y,mse_tolerance=0.)
    full=EmpiricalLocationScale.baseline(rows,y,train,train_y,task['horizon'])
    full_candidate=EmpiricalLocationScale.fit(rows,y,True,False,full)
    split.update(fit_coordinates=len(fit_y),selection_coordinates=len(select_y))
    meta=dict(version=VERSION,split=split,selection=selection,weight=eta,
        calibration_coordinates=len(y),training_coordinates=len(train_y),
        test_targets_used=False,parameters_frozen_before_test=True,
        mean_parameters='global affine mean plus state residual sum/(count+128)',
        width_and_shape='unchanged from FSPCRefit',refit='all calibration coordinates; blend weight frozen',
        constraint='validation MSE <= split-matched FSPC; no test constraint')
    return {'FSPCRefit':full,'FSPCMean':QuantileBlend(full,full_candidate,eta)},meta


def valid_result(r,task):
    if not (r.get('version')==VERSION and r.get('task')==task and r.get('full_test_split') is True
        and r.get('backbone_retrained') is False and set(METHODS).issubset(r.get('methods',{}))
        and r.get('test_windows',0)>0 and r.get('test_batches',0)>0):return False
    fit=r.get('fit',{})
    if fit.get('test_targets_used') is not False or fit.get('parameters_frozen_before_test') is not True:return False
    counts=[r['methods'][m].get('coordinate_count',0) for m in METHODS]
    return counts[0]>0 and len(set(counts))==1


def coverage(tasks,states):
    return dict(expected=len(tasks),complete=sum(states.get(t['id']) is True for t in tasks),
        invalid=[t['id'] for t in tasks if states.get(t['id']) is False],
        missing=[t['id'] for t in tasks if t['id'] not in states])


def reuse_diffusion_record(old,task,path):
    from location_scale import complete_result
    if task.get('backbone')!='DiffusionTS' or not complete_result(old,task,'unclipped',True):
        raise ValueError('Reuse requires exact task and complete unclipped prior protocol')
    fit=old['fit'];selection=fit['selection']['FSPCMean']
    assert selection['training']['new_mean'] is True and selection['training']['new_scale'] is False
    new_fit=dict(fit,version=VERSION,selection=selection,weight=selection['weight'],source_version=fit.get('version'))
    result=dict(version=VERSION,task=task,full_test_split=True,test_batches=old['test_batches'],test_windows=old['test_windows'],
        checkpoint=old['checkpoint'],backbone_retrained=False,methods={m:old['methods'][m] for m in METHODS},fit=new_fit,
        provenance=old.get('provenance',{}),source_result=str(path),source_version=old['version'],sampling='unclipped',
        evaluation_execution='reused completed full-test paired mean-only experiment; no inference repeated',
        native_parity='prior corrected-sampling Native shared by both methods',original_fspc_parity='original operator refitted on corrected sampling')
    if not valid_result(result,task):raise ValueError('Prior result does not satisfy full paired evidence')
    return result


def regular_origin_stride(origins):
    origins=np.asarray(origins,dtype=np.int64)
    delta=np.diff(origins)
    if not len(delta):raise ValueError('At least two forecast origins required')
    if delta[0]<=0 or not np.all(delta==delta[0]):raise ValueError('Expected strictly increasing regular forecast origins')
    return int(delta[0])


def designated_pilots(tasks,metadata):
    pilots={}
    for task in tasks:
        if task['backbone']!='DiffusionTS':pilots.setdefault(task['backbone'],task)
    for name,record in metadata.items():
        matches=[t for t in tasks if t['id']==record['task'] and t['backbone']==name and name!='DiffusionTS']
        if len(matches)!=1:raise ValueError('Pilot metadata must identify a canonical task of its own backbone')
        pilots[name]=matches[0]
    return pilots
