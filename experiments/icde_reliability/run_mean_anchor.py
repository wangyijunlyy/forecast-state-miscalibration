"""Paired cached-cohort and complete-native-test evaluation of mean anchoring."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path
import numpy as np
from portable_dc import CompiledCalibrator, edges, summaries
from mean_anchor import centered_forecasts, fit_bias, select_strength, state_ids as bias_ids
from reliability_metrics import ReliabilityMetrics, state_ids, assert_traversal
from mean_anchor_campaign import version_for, validate_record, METHODS, LEGACY_VERSION as VERSION
# VERSION remains the immutable pilot identifier for its existing aggregator.


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def require_parity(actual,expected):
    for key in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-9,err_msg=key)


def values(name,cal,rows,h,bias,strength):
    if name=='Native':return rows
    if name=='DC36':return cal.transform_rows(rows,h)
    return centered_forecasts(cal,rows,h,bias if name=='DC36ControlledMean' else None,
                              strength if name=='DC36ControlledMean' else 0.)


def run(task,root,source):
    version=version_for(task)
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():
        old=json.loads((out/'results.json').read_text());validate_record(old,task);return
    start=time.monotonic();old=json.loads((source/'runs'/task['id']/'results.json').read_text())
    assert old['task']==task and old['version']=='icde-fsdc36-v1-20261103'
    cal=CompiledCalibrator.load(source/'compiled_deployment/runs'/task['id']/'compiled.npz')
    assert len(cal.mean_edges)==len(cal.scale_edges)==7 and cal.n_horizon_bins==1
    trainpath=source/'train_calibration/runs'/task['id']/'calibration.npz'
    valpath=source/'runs'/task['id']/'calibration.npz'
    with np.load(trainpath) as d:train,y_train,h_train=[d[k] for k in ['rows','targets','horizon']]
    bias,counts=fit_bias(cal,train,y_train,h_train)
    mean,_,scale=summaries(train);mean_edges=edges(mean,6);scale_edges=edges(np.log(scale),6)
    del train,y_train,h_train,mean,scale
    with np.load(valpath) as d:validation,y_val,h_val,thresholds=[d[k] for k in ['rows','targets','horizon','thresholds']]
    strength,selection=select_strength(cal,validation,y_val,h_val,bias)
    del validation,y_val,h_val
    fit=dict(version=version,task=task,summary_bins=6,horizon_groups=1,output_members=100,
        pseudo_count=128,min_bias_count=128,state_bias=bias.tolist(),training_state_counts=counts.tolist(),
        selection=selection,training_source=str(trainpath.resolve()),validation_source=str(valpath.resolve()),
        template_source=str(source/'compiled_deployment/runs'/task['id']/'compiled.npz'),
        alpha_frozen_before_test=True,neural_training_performed=False)
    atomic(out/'fit.json',fit)
    np.savez_compressed(out/'mean_parameters.npz',bias=bias,counts=counts,strength=np.array(strength))
    # Every fitted parameter is frozen before this first access to held-out targets.
    with np.load(source/'runs'/task['id']/'test_reservoir.npz') as d:
        rows,y,h,channels=[d[k] for k in ['rows','targets','horizon','channel']]
    groups=state_ids(rows.T[None,:,None,:],cal.mean_edges,cal.scale_edges)
    previous=json.loads((source/'cohort/runs'/task['id']/'results.json').read_text())
    cohort={'Native':previous['native'],'DC36':previous['methods']['b6_h1_n50000']}
    evaluated=METHODS[2:]
    for name in evaluated:
        forecast=values(name,cal,rows,h,bias,strength)
        acc=ReliabilityMetrics(thresholds[channels],n_states=36)
        acc.update(forecast.T[None,:,None,:],y[None,None,:],groups,ordered=name!='Native')
        cohort[name]=acc.result();del forecast
    np.testing.assert_allclose(cohort['DC36MeanPreserving']['MSE'],cohort['Native']['MSE'],rtol=1e-10,atol=1e-9)
    atomic(out/'cohort.json',dict(version=version,task=task,evaluation='shared retained test cohort',
        coordinates=len(y),methods=cohort,baseline_parity=True,fit=fit,
        baseline_metrics_recomputed=False,newly_evaluated_methods=evaluated))
    del rows,y,h,channels,cohort
    print('MEAN_ANCHOR_COHORT_COMPLETE',task['id'],'alpha',strength,flush=True)
    sys.path.insert(0,str(source/'code'))
    import torch
    import run_task as legacy
    import run_fsdc36 as original
    torch.set_num_threads(2)
    # Native loaders may re-extract a shared archive on every initialization.
    # Serialize preparation for a dataset so another worker cannot read a partial file.
    locks=root/'data_prepare_locks';locks.mkdir(exist_ok=True)
    with (locks/(task['dataset'].lower()+'.lock')).open('a') as data_lock:
        fcntl.flock(data_lock,fcntl.LOCK_EX)
        interface=original.prepare(task,out)
    accumulators={name:ReliabilityMetrics(thresholds,n_states=36) for name in evaluated}
    generation=0.;transform={name:0. for name in evaluated};mean_error={name:0. for name in evaluated}
    legacy.seed(task['seed']+30000);steps=windows=0
    for step,batch in enumerate(interface['test'],1):
        begin=time.monotonic();torch.cuda.synchronize()
        with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
        torch.cuda.synchronize();generation+=time.monotonic()-begin
        b,s,h,c=x.shape;rows=x.transpose(0,2,3,1).reshape(-1,s)
        horizon=np.broadcast_to(np.arange(h)[None,:,None],(b,h,c)).reshape(-1)
        groups=state_ids(x,mean_edges,scale_edges)
        native_mean,_,_=summaries(rows)
        for name in evaluated:
            begin=time.monotonic();forecast=values(name,cal,rows,horizon,bias,strength)
            transform[name]+=time.monotonic()-begin
            if name in mean_error:
                expected=native_mean.copy()
                if name=='DC36ControlledMean':expected+=strength*bias[bias_ids(cal,rows,horizon)]
                mean_error[name]=max(mean_error[name],float(np.max(abs(forecast.mean(1)-expected))))
            tensor=x if name=='Native' else forecast.reshape(b,h,c,s).transpose(0,3,1,2)
            accumulators[name].update(tensor,y,groups,ordered=name!='Native')
            del forecast,tensor
        windows+=len(y);steps=step
        if step%10==0:print('MEAN_ANCHOR_TEST',task['id'],step,windows,round(time.monotonic()-start,1),flush=True)
    assert_traversal(interface['test'],steps,windows)
    assert steps==old['test_batches'] and windows==old['test_windows']
    scores={name:old['methods'][name] for name in METHODS[:2]}
    scores.update({name:acc.result() for name,acc in accumulators.items()})
    for name in evaluated:
        np.testing.assert_array_equal(scores[name]['state_coverage']['counts'],scores['Native']['state_coverage']['counts'])
    np.testing.assert_allclose(scores['DC36MeanPreserving']['MSE'],scores['Native']['MSE'],atol=1e-9,rtol=1e-10)
    result=dict(version=version,task=task,methods=scores,fit=fit,full_test_split=True,
        test_batches=steps,test_windows=windows,test_seed=task['seed']+30000,
        source_primary=str(source/'runs'/task['id']/'results.json'),checkpoint=interface['checkpoint'],
        baseline_parity=True,baseline_metrics_recomputed=False,newly_evaluated_methods=evaluated,
        diagnostic_partition='original training-fitted 6x6 partition',
        maximum_mean_deviation=mean_error,native_generation_seconds=generation,
        transform_seconds=transform,total_seconds=time.monotonic()-start,neural_training_performed=False)
    atomic(out/'results.json',result)
    print('MEAN_ANCHOR_FULL_COMPLETE',task['id'],round(result['total_seconds'],2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();task=json.loads(a.task.read_text());out=a.root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(task,a.root,a.source)
