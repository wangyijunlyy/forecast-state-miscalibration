"""Fit four external baselines, then stream their shared complete native test."""
import argparse
import fcntl
import json
import os
import pickle
import gzip
from pathlib import Path
import sys
import time
import resource
import numpy as np
import torch
from external_baselines import VERSION,METHODS,EMOS,SoftCRPSPIT,DistributionRecalibration,QRF,verify_checkpoint_reference
from portable_dc import edges,summaries
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix('.tmp')
    p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)

def load_json(path):return json.loads(path.read_text())

def existing_emos(task,rows,y,source):
    root=Path(task['source36_root'])
    candidates=[root/'gaussian_runs'/task['id']/'fitting.json',
                root/'deepstuq_backbone/gaussian_runs'/task['id']/'fitting.json']
    for p in candidates:
        if not p.exists():continue
        fit=load_json(p);candidate=fit['models'].get('GaussianAffine4P')
        if candidate is None:continue
        cache=Path(fit['coordinate_source']) if 'coordinate_source' in fit else None
        if cache is None:cache=root/'deepstuq_backbone/runs'/task['id']/'calibration.npz'
        with np.load(cache) as d:
            np.testing.assert_array_equal(d['rows'],rows);np.testing.assert_array_equal(d['targets'],y)
        return candidate,str(p)
    return None,None


def fitting(task,out,source):
    record=out/'fit.json'
    if record.exists() and all((out/(n+'.pkl')).exists() for n in METHODS):
        meta=load_json(record);assert meta['version']==VERSION and meta['task']==task
        models={}
        for name in METHODS:
            with gzip.open(out/(name+'.pkl'),'rb') as f:models[name]=pickle.load(f)
        return models,meta
    with np.load(source/'runs'/task['id']/'calibration.npz') as d:
        rows=d['rows'];y=d['targets']
    old,old_path=existing_emos(task,rows,y,source);models={};times={}
    definitions=[('EMOS',lambda:EMOS.fit(rows,y,old)),
        ('Qian2025SoftCRPS',lambda:SoftCRPSPIT.fit(rows,y,task['seed']+71000)),
        ('DistributionRecalibration2022',lambda:DistributionRecalibration.fit(rows,y,task['seed']+72000)),
        ('QRF2016',lambda:QRF.fit(rows,y,task['seed']+73000))]
    for name,fit in definitions:
        file=out/(name+'.pkl');t=time.monotonic()
        if file.exists():
            with gzip.open(file,'rb') as f:models[name]=pickle.load(f)
        else:
            print('EXTERNAL_FIT_START',task['id'],name,len(y),flush=True)
            models[name]=fit()
            models[name].metadata['fit_seconds']=time.monotonic()-t
            with gzip.open(file.with_suffix('.tmp'),'wb',compresslevel=3) as f:pickle.dump(models[name],f,protocol=4)
            file.with_suffix('.tmp').replace(file)
        if name=='DistributionRecalibration2022':models[name].metadata['architecture_source']='paper Section 6.1'
        times[name]=models[name].metadata['fit_seconds']
        print('EXTERNAL_FIT_DONE',task['id'],name,round(times[name],2),flush=True)
    meta=dict(version=VERSION,task=task,calibration_split='validation',calibration_coordinates=len(y),
        calibration_source=str(source/'runs'/task['id']/'calibration.npz'),
        all_parameters_frozen_before_test_access=True,methods={n:m.metadata for n,m in models.items()},
        fit_seconds=times,emos_reuse_source=old_path,backbone_retrained=False)
    atomic(record,meta)
    return models,meta


def run(task,root,source,mean_root):
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target=out/'results.json'
        if target.exists():
            v=load_json(target);assert v['version']==VERSION and v['task']==task;return
        start=time.monotonic();models,fit=fitting(task,out,source)
        old=load_json(source/'runs'/task['id']/'results.json')
        anchor=load_json(mean_root/'runs'/task['id']/'results.json')
        assert old['task']==anchor['task']==task
        with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:
            mu,_,sigma=summaries(d['rows']);mean_edges=edges(mu,6);scale_edges=edges(np.log(sigma),6)
        with np.load(source/'runs'/task['id']/'calibration.npz') as d:thresholds=d['thresholds']
        # Import the exact previously verified native preparation code.
        sys.path.insert(0,str(source/'code'))
        import run_fsdc36 as original
        import run_task as legacy
        locks=root/'data_prepare_locks';locks.mkdir(exist_ok=True)
        with (locks/(task['dataset'].lower()+'.lock')).open('a') as dl:
            fcntl.flock(dl,fcntl.LOCK_EX);interface=original.prepare(task,out)
        verify_checkpoint_reference(interface['checkpoint'])
        accum={n:ReliabilityMetrics(thresholds,n_states=36) for n in ['Native']+METHODS}
        legacy.seed(task['seed']+30000);steps=windows=0;generation=0.
        transform={n:0. for n in METHODS};metric_seconds={n:0. for n in ['Native']+METHODS}
        for step,batch in enumerate(interface['test'],1):
            begin=time.monotonic();torch.cuda.synchronize()
            with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
            torch.cuda.synchronize();generation+=time.monotonic()-begin
            b,s,h,c=x.shape;rows=x.transpose(0,2,3,1).reshape(-1,s)
            groups=state_ids(x,mean_edges,scale_edges)
            begin=time.monotonic();accum['Native'].update(x,y,groups);metric_seconds['Native']+=time.monotonic()-begin
            for name in METHODS:
                begin=time.monotonic();forecast=models[name].predict(rows);transform[name]+=time.monotonic()-begin
                assert forecast.shape==rows.shape and np.isfinite(forecast).all()
                if np.any(np.diff(forecast,axis=1)<-1e-10):raise ValueError('Unsorted marginal '+name)
                tensor=forecast.reshape(b,h,c,s).transpose(0,3,1,2);begin=time.monotonic()
                accum[name].update(tensor,y,groups,ordered=True);metric_seconds[name]+=time.monotonic()-begin
                del tensor,forecast
            windows+=len(y);steps=step
            if step%10==0 or step==1:
                print('EXTERNAL_TEST',task['id'],step,windows,round(time.monotonic()-start,1),flush=True)
                atomic(out/'progress.json',dict(task=task['id'],phase='full_test',batches=steps,windows=windows,
                    expected_batches=old['test_batches'],elapsed_seconds=time.monotonic()-start))
        assert_traversal(interface['test'],steps,windows)
        assert steps==old['test_batches'] and windows==old['test_windows']
        scores={n:a.result() for n,a in accum.items()};parity={}
        for metric in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
            actual=scores['Native'][metric];reference=old['methods']['Native'][metric]
            np.testing.assert_allclose(actual,reference,rtol=1e-10,atol=1e-9,err_msg=metric)
            parity[metric]=float(actual-reference)
        for n in METHODS:np.testing.assert_array_equal(scores[n]['state_coverage']['counts'],scores['Native']['state_coverage']['counts'])
        for n in ['GlobalDC','GlobalPIT','ConditionalPIT']:scores[n]=old['methods'][n]
        scores['FSPC']=anchor['methods']['DC36ControlledMean']
        raw=models['DistributionRecalibration2022']
        atomic(target,dict(version=VERSION,task=task,full_test_split=True,test_batches=steps,test_windows=windows,
            checkpoint=interface['checkpoint'],test_sampling_seed=task['seed']+30000,methods=scores,fit=fit,
            native_parity=parity,diagnostic_partition='original training-derived mean6 x scale6',
            existing_scores_reused=['GlobalDC','GlobalPIT','ConditionalPIT','FSPC'],
            native_generation_seconds=generation,transform_seconds=transform,metric_seconds=metric_seconds,
            total_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            raw_distribution_recalibration_crossings=raw.raw_crossings,
            raw_distribution_recalibration_adjacent_pairs=raw.raw_comparisons,
            serialized_model_bytes={n:(out/(n+'.pkl')).stat().st_size for n in METHODS},backbone_retrained=False))
        print('EXTERNAL_COMPLETE',task['id'],{n:round(scores[n]['CRPS'],6) for n in METHODS},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root','task']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    run(load_json(a.task),a.root.resolve(),a.source.resolve(),a.mean_root.resolve())
