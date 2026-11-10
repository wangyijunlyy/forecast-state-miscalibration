"""Exact full-test replay when the original reservoir retained every coordinate.

Large-task reservoirs are rejected, rather than promoted to full-test evidence.
The sampled-native and published-FSPC scores must both match full-iterator records.
"""
import argparse
from dataclasses import fields
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from run_location_scale import atomic,load,cached_calibration,fit_models
from location_scale import VERSION,complete_result
from portable_dc import summaries,edges
from reliability_metrics import ReliabilityMetrics


def evaluate(task,root,source,mean_root):
    old=load(source/'runs'/task['id']/'results.json')
    with np.load(source/'runs'/task['id']/'test_reservoir.npz') as d:
        test,test_y,h,c=[d[k] for k in ['rows','targets','horizon','channel']]
    if len(test_y)!=old['methods']['Native']['coordinate_count']:
        print('CACHE_REJECT_SUBSET',task['id'],len(test_y),old['methods']['Native']['coordinate_count'],flush=True);return
    assert old['full_test_split']
    out=root/'original/runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'results.json').exists():
        assert complete_result(load(out/'results.json'),task,'original',False);return
    start=time.monotonic();torch.set_num_threads(2)
    # Only construct the deterministic data loaders; no model or CUDA allocation.
    sys.path.insert(0,task['repo']);os.chdir(task['repo'])
    cls=getattr(importlib.import_module('src.experiments.DiffusionTS_SPARC_PIT'),'DiffusionTSSPARCPITForecast')
    names={f.name for f in fields(cls)};args={k:v for k,v in task['training_args'].items() if k in names}
    args.update(device='cpu',num_worker=0,batch_size=1,data_path=str(Path(task['repo'])/'data'))
    data=cls(**args);data._init_data_loader(shuffle=False,fast_val=True,fast_test=True)
    rows,y,window,thresholds,provenance=cached_calibration(task,source,data)
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:train,train_y=[d[k] for k in ['rows','targets']]
    mu,_,sigma=summaries(train);mean_edges=edges(mu,6);scale_edges=edges(np.log(sigma),6)
    tm,_,ts=summaries(test)
    groups=6*np.searchsorted(mean_edges[1:-1],tm,side='right')+np.searchsorted(scale_edges[1:-1],np.log(ts),side='right')
    def score(a,ordered):
        accumulator=ReliabilityMetrics(thresholds[c],n_states=36)
        accumulator.update(a.T[None,:,None,:],test_y[None,None,:],groups,ordered=ordered)
        return accumulator.result()
    native=score(test,False)
    for metric in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
        np.testing.assert_allclose(native[metric],old['methods']['Native'][metric],atol=1e-9,rtol=1e-10)
    np.testing.assert_array_equal(native['state_coverage']['counts'],old['methods']['Native']['state_coverage']['counts'])
    np.testing.assert_allclose(native['events']['brier'],old['methods']['Native']['events']['brier'],atol=1e-12)
    models,fit=fit_models(task,out,rows,y,window,train,train_y,provenance['calibration_windows'],provenance['origin_stride'])
    # Native replay parity is checked independently before fitting; no test value
    # enters fitting or selection. State the timing accurately for this executor.
    fit=dict(fit);fit.pop('parameters_frozen_before_test',None)
    fit.update(native_parity_checked_before_fit=True,
        parameters_frozen_before_calibrated_test_evaluation=True,test_targets_used=False)
    atomic(out/'fit.json',fit)
    scores={'Native':native};times={}
    for name,model in models.items():
        begin=time.monotonic();forecast=model.predict(test);times[name]=time.monotonic()-begin;scores[name]=score(forecast,True)
    anchor=load(mean_root/'runs'/task['id']/'results.json')['methods']['DC36ControlledMean']
    for metric in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
        np.testing.assert_allclose(scores['FSPCRefit'][metric],anchor[metric],atol=1e-9,rtol=1e-10)
    scores['FSPCPublished']=anchor
    atomic(out/'task.json',task)
    atomic(out/'results.json',dict(version=VERSION,task=task,mode='original',full_test_split=True,
        test_batches=old['test_batches'],test_windows=old['test_windows'],checkpoint=old['checkpoint'],
        native_parity={'Native':True,'FSPCRefit':True},methods=scores,fit=fit,provenance=provenance,
        evaluation_execution='all original full-test coordinates retained; exact cached native forecasts replayed',
        source_full_iterator_record=str(source/'runs'/task['id']/'results.json'),
        native_generation_seconds=0.,transform_seconds=times,total_seconds=time.monotonic()-start,
        native_collapsed_coordinates=int(np.sum(ts<=1.01e-6)),native_coordinate_count=len(test_y),
        native_collapsed_fraction=float(np.mean(ts<=1.01e-6)),external_baselines_refitted=False,
        diagnostic_partition='original training-derived 6x6 states, fixed across methods and sampling pipelines'))
    print('CACHE_FULL_COMPLETE',task['id'],{n:round(v['CRPS'],5) for n,v in scores.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();root=a.root.resolve();mean=a.mean_root.resolve()
    for task in load(source/'manifest.json'):
        if task['backbone']=='DiffusionTS':evaluate(task,root,source,mean)
