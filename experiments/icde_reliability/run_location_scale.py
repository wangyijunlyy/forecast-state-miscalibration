"""Blocked validation and complete paired native-test calibration experiments."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch
from location_scale import VERSION,METHODS,EmpiricalLocationScale,QuantileBlend,select_blend,replay_windows,chronological_split,complete_result
from portable_dc import summaries,edges
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal
from audit_diffusion_sampling import find_native_model,dataset_stride,disable_clamp,audit


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def load(path):return json.loads(path.read_text())


def cached_calibration(task,source,model):
    record=load(source/'runs'/task['id']/'validation.json')
    with np.load(source/'runs'/task['id']/'calibration.npz') as d:
        rows,y,h,c,thresholds=[d[k] for k in ['rows','targets','horizon','channel','thresholds']]
    n=record['calibration_windows'];channels=thresholds.shape[0]
    w,hh,cc=replay_windows(n,task['horizon'],channels,50000,task['seed']+40000)
    np.testing.assert_array_equal(hh,h);np.testing.assert_array_equal(cc,c)
    # Check reconstructed chronology against actual normalized dataset targets.
    dataset=model.val_loader.dataset
    targets=np.stack([np.asarray(dataset[i][1],dtype=np.float32)[-task['horizon']:] for i in range(n)])
    np.testing.assert_array_equal(targets[w,h,c],y)
    return rows,y,w,thresholds,dict(calibration_windows=n,coordinate_provenance='exact priority replay; horizon/channel/actual target equality verified',origin_stride=dataset_stride(dataset))


def collect(task,interface,legacy,out,split,capacity=50000):
    # New sampling requires new forecasts for every fitted model and template.
    from reservoir import CalibrationReservoir
    phase='train' if split=='train' else 'val';sampling_offset=10000 if split=='train' else 20000
    reservoir_seed=task['seed'] if split=='train' else task['seed']+40000
    path=out/(split+'_calibration.npz')
    if path.exists():
        with np.load(path) as d:return tuple(d[k] for k in ['rows','targets','window'])
    legacy.seed(task['seed']+sampling_offset);pool=CalibrationReservoir(capacity,reservoir_seed);windows=0
    batch_sizes=[]
    for step,batch in enumerate(interface[phase],1):
        if step>interface['fit_limit']:break
        with torch.no_grad():x,y,_=interface['predict'](batch,'calibration' if split=='train' else 'validation',step)
        pool.update(x,y);batch_sizes.append(len(y));windows+=len(y)
        if step%20==0:print('LS_COLLECT',task['id'],split,step,flush=True)
    rows,y,h,c=pool.arrays()
    w,hh,cc=replay_windows(windows,task['horizon'],x.shape[-1],capacity,reservoir_seed,batch_sizes)
    np.testing.assert_array_equal(hh,h);np.testing.assert_array_equal(cc,c)
    np.savez_compressed(path,rows=rows,targets=y,window=w,horizon=h,channel=c)
    return rows,y,w


def fit_models(task,out,rows,y,window,train,train_y,n_windows,stride):
    cache=out/'models.pkl.gz';record=out/'fit.json'
    if cache.exists() and record.exists():
        meta=load(record)
        if meta['version']!=VERSION:raise ValueError('Incompatible fitting cache')
        with gzip.open(cache,'rb') as f:return pickle.load(f),meta
    fit,selection,split=chronological_split(window,n_windows,task['horizon'],stride) if n_windows>=2 else (None,None,None)
    if fit is None or fit.sum()<128 or selection.sum()<128:
        # A one-window validation split cannot be divided without sharing targets.
        fit_rows,fit_y=train,train_y;select_rows,select_y=rows,y
        split=dict(protocol='training forecasts fit; entire validation selects',reason='too few independent validation windows',
            fit_coordinates=len(train_y),selection_coordinates=len(y))
    else:
        fit_rows,fit_y=rows[fit],y[fit];select_rows,select_y=rows[selection],y[selection]
        split.update(protocol='chronological first two thirds fit, final third selects; overlapping targets purged',
            fit_coordinates=len(fit_y),selection_coordinates=len(select_y))
    begin=time.monotonic()
    base=EmpiricalLocationScale.baseline(fit_rows,fit_y,train,train_y,task['horizon'])
    reference=base.predict(select_rows);selected={};selection_meta={};blocked={}
    options={'FSPCMean':(True,False),'FSPCScale':(False,True),'FSPCJoint':(True,True)}
    for name,(new_mean,new_scale) in options.items():
        start=time.monotonic();candidate=EmpiricalLocationScale.fit(fit_rows,fit_y,new_mean,new_scale,base)
        weight,meta=select_blend(reference,candidate.predict(select_rows),select_y,mse_tolerance=0.)
        selected[name]=weight;meta.update(fit_seconds=time.monotonic()-start,training=candidate.metadata)
        selection_meta[name]=meta
        blocked[name]=QuantileBlend(base,candidate,weight)
        print('LS_SELECTED',task['id'],name,round(weight,6),meta['reference_crps'],meta['selected_crps'],flush=True)
    # Freeze selection weights, then refit only the fitted distributions on all calibration data.
    full_base=EmpiricalLocationScale.baseline(rows,y,train,train_y,task['horizon'])
    models={'FSPCBlocked':base,'FSPCRefit':full_base}
    # Keep the held-out factorial models intact as a clean causal comparison.
    for name,(new_mean,new_scale) in options.items():
        candidate_full=EmpiricalLocationScale.fit(rows,y,new_mean,new_scale,full_base)
        models[name]=QuantileBlend(full_base,candidate_full,selected[name])
    models.update({name+'Blocked':value for name,value in blocked.items()})
    meta=dict(version=VERSION,task=task,split=split,selection=selection_meta,
        fit_seconds=time.monotonic()-begin,calibration_coordinates=len(y),training_coordinates=len(train_y),
        test_targets_used=False,parameters_frozen_before_test=True,refit='all calibration coordinates after selection; weights frozen',
        held_out_constraint='selected forecast MSE <= split-matched FSPC MSE, tolerance zero',
        blocked_models='retained unchanged for factorial analysis; refitted models reported separately')
    with gzip.open(cache.with_suffix('.tmp'),'wb') as f:pickle.dump(models,f,protocol=4)
    cache.with_suffix('.tmp').replace(cache);atomic(record,meta)
    return models,meta


def run(task,root,source,mean_root,mode,external=False):
    out=root/mode/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'results.json').exists():
        result=load(out/'results.json')
        if complete_result(result,task,mode,external):return
        raise ValueError('Existing result does not satisfy requested protocol; preserve it and use a fresh run directory')
    atomic(out/'task.json',task);start=time.monotonic()
    sys.path.insert(0,str(source/'code'))
    import run_fsdc36 as original
    import run_task as legacy
    torch.set_num_threads(2)
    locks=root/'data_prepare_locks';locks.mkdir(exist_ok=True)
    with (locks/(task['dataset'].lower()+'.lock')).open('a') as dl:
        fcntl.flock(dl,fcntl.LOCK_EX);interface=original.prepare(task,out)
    # The original interface modifies sys.path; our isolated modules remain pre-imported.
    model=find_native_model(interface)
    rows,y,window,thresholds,provenance=cached_calibration(task,source,model)
    atomic(out/'sampling_audit.json',audit(model,rows,y))
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:train,train_y=[d[k] for k in ['rows','targets']]
    mu,_,sigma=summaries(train);mean_edges=edges(mu,6);scale_edges=edges(np.log(sigma),6)
    if mode=='unclipped':
        disable_clamp(model)
        train,train_y,_=collect(task,interface,legacy,out,'train')
        rows,y,window=collect(task,interface,legacy,out,'val')
    models,fit=fit_models(task,out,rows,y,window,train,train_y,provenance['calibration_windows'],provenance['origin_stride'])
    # Fit all external baselines only on new sampling, with their complete fixed protocols.
    if external:
        sys.path.insert(0,'/home/wyj/icde_external_baselines_20261105/code')
        from external_baselines import EMOS,SoftCRPSPIT,DistributionRecalibration,QRF
        for name,fn in [('EMOS',lambda:EMOS.fit(rows,y)),('Qian2025SoftCRPS',lambda:SoftCRPSPIT.fit(rows,y,task['seed']+71000)),
            ('DistributionRecalibration2022',lambda:DistributionRecalibration.fit(rows,y,task['seed']+72000)),
            ('QRF2016',lambda:QRF.fit(rows,y,task['seed']+73000))]:
            path=out/(name+'.pkl.gz')
            if path.exists():
                with gzip.open(path,'rb') as f:models[name]=pickle.load(f)
            else:
                models[name]=fn()
                with gzip.open(path,'wb') as f:pickle.dump(models[name],f,protocol=4)
    old=load(source/'runs'/task['id']/'results.json')
    anchor=load(mean_root/'runs'/task['id']/'results.json')
    # Freeze all fitted parameters before the first access to held-out targets.
    atomic(out/'frozen.json',dict(version=VERSION,mode=mode,fit=fit,provenance=provenance,
        checkpoint=interface['checkpoint'],methods=list(models),external_baselines_refitted=external))
    accum={n:ReliabilityMetrics(thresholds,n_states=36) for n in ['Native']+list(models)}
    generation=0.;transform={n:0. for n in models};collapse_count=coordinates=steps=windows=0
    legacy.seed(task['seed']+30000)
    for step,batch in enumerate(interface['test'],1):
        torch.cuda.synchronize();begin=time.monotonic()
        with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
        torch.cuda.synchronize();generation+=time.monotonic()-begin
        b,s,h,c=x.shape;flat=x.transpose(0,2,3,1).reshape(-1,s)
        groups=state_ids(x,mean_edges,scale_edges)
        accum['Native'].update(x,y,groups)
        collapse_count+=int(np.sum(flat.std(1,ddof=1)<=1.01e-6));coordinates+=len(flat)
        for name,calibrator in models.items():
            begin=time.monotonic();forecast=calibrator.predict(flat);transform[name]+=time.monotonic()-begin
            if not np.isfinite(forecast).all() or np.any(np.diff(forecast,axis=1)<-1e-10):raise ValueError('Invalid sorted forecast: '+name)
            accum[name].update(forecast.reshape(b,h,c,s).transpose(0,3,1,2),y,groups,ordered=True)
        windows+=len(y);steps=step
        if step==1 or step%10==0:
            print('LS_TEST',mode,task['id'],step,old['test_batches'],flush=True)
            atomic(out/'progress.json',dict(batches=step,expected=old['test_batches'],elapsed=time.monotonic()-start))
    assert_traversal(interface['test'],steps,windows)
    assert windows==old['test_windows'] and steps==old['test_batches']
    scores={n:a.result() for n,a in accum.items()};parity={}
    if mode=='original':
        for name,ref in [('Native',old['methods']['Native']),('FSPCRefit',anchor['methods']['DC36ControlledMean'])]:
            for metric in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
                np.testing.assert_allclose(scores[name][metric],ref[metric],atol=1e-9,rtol=1e-10)
            parity[name]=True
        scores['FSPCPublished']=anchor['methods']['DC36ControlledMean']
    for n in accum:np.testing.assert_array_equal(scores[n]['state_coverage']['counts'],scores['Native']['state_coverage']['counts'])
    result=dict(version=VERSION,task=task,mode=mode,full_test_split=True,test_batches=steps,test_windows=windows,
        checkpoint=interface['checkpoint'],native_parity=parity,methods=scores,fit=fit,provenance=provenance,
        native_generation_seconds=generation,transform_seconds=transform,total_seconds=time.monotonic()-start,
        native_collapsed_coordinates=collapse_count,native_coordinate_count=coordinates,
        native_collapsed_fraction=collapse_count/coordinates,external_baselines_refitted=external,
        diagnostic_partition='original training-derived 6x6 states, fixed across methods and sampling pipelines')
    atomic(out/'results.json',result)
    print('LS_COMPLETE',mode,task['id'],{n:round(v['CRPS'],5) for n,v in scores.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root','task']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=['original','unclipped'],default='original');p.add_argument('--external',action='store_true')
    a=p.parse_args();run(load(a.task),a.root.resolve(),a.source.resolve(),a.mean_root.resolve(),a.mode,a.external)
