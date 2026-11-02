"""Paired complete-test correction suite for freshly trained192-step NsDiff tasks."""
import argparse,json,pickle,resource,time
from pathlib import Path
import numpy as np
import torch
import run_task as primary
from gaussian_models import fit_gaussian,predict as gaussian_predict
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal
from core import CommonMetrics


def fit_training(task,root,interface):
    out=root/'train_calibration/runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'fit_metadata.json').exists():
        with (out/'calibrators.pkl').open('rb') as f:return pickle.load(f)
    primary.seed(task['seed']+10000);pool=primary.CalibrationReservoir(50000,task['seed']);windows=steps=0
    for step,batch in enumerate(interface['train'],1):
        if step>interface['fit_limit']:break
        with torch.no_grad():x,y,_=interface['predict'](batch,'calibration',step)
        assert x.shape[1:3]==(100,192) and y.shape[1]==192
        pool.update(x,y);windows+=len(y);steps=step
        if step%20==0:print('TRAIN_FIT',task['id'],step,flush=True)
    rows,y,h,c=pool.arrays();models={};meta={}
    for name,bins in [('DC16',1),('DC64',4)]:models[name],meta[name]=primary.fit_dc(rows,y,h,192,n_horizon=bins)
    with (out/'calibrators.pkl').open('wb') as f:pickle.dump(models,f)
    np.savez_compressed(out/'calibration.npz',rows=rows,targets=y,horizon=h,channel=c)
    primary.atomic(out/'fit_metadata.json',dict(fit_split='train',fit_windows=windows,fit_batches=steps,retained_points=len(y),seen_points=pool.seen_points,prediction_seed=task['seed']+10000,models=meta,source_scale_threshold=1.01e-6))
    return models


def fit_gaussians(task,out,source):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'fitting.json').exists():return json.loads((out/'fitting.json').read_text())
    with np.load(source) as d:rows,y=d['rows'],d['targets']
    mu=rows.mean(1);sigma=rows.std(1,ddof=1).clip(1e-6);models={};seconds={}
    for name,affine in [('GaussianVariance2P',False),('GaussianAffine4P',True)]:
        start=time.monotonic();models[name]=fit_gaussian(mu,sigma,y,affine);seconds[name]=time.monotonic()-start
        if not models[name]['optimizer_success']:raise ValueError('Gaussian optimization failed: '+name)
    result=dict(version='full-gaussian-controls-v1-20261102',task=task,fitting_split='validation',calibration_coordinates=len(y),coordinate_source=str(source),models=models,fit_seconds=seconds)
    primary.atomic(out/'fitting.json',result);return result


def run(task,root):
    assert task['backbone']=='NsDiff' and task['horizon']==192 and task['model_args']['pred_len']==task['training_args']['pred_len']==192
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True);primary.atomic(out/'task.json',task)
    if not (out/'results.json').exists():
        interface=primary.prepare(task,out);train_models=fit_training(task,root,interface);partition=train_models['DC16']
        if (out/'validation.json').exists():
            validation=json.loads((out/'validation.json').read_text())
            with (out/'calibrators.pkl').open('rb') as f:models=pickle.load(f)
            with np.load(out/'calibration.npz') as d:thresholds=d['thresholds']
        else:models,thresholds,validation=primary.collect_validation(task,out,interface,partition)
        models.update(TrainDC16=partition,TrainDC64=train_models['DC64'])
        gaussian_dir=root/'gaussian_runs'/task['id'];fitting=fit_gaussians(task,gaussian_dir,out/'calibration.npz')
        names=primary.METHODS+list(fitting['models'])+['TrainDC64'];metrics={n:ReliabilityMetrics(thresholds) for n in names}
        timings={n:0. for n in names};metric_seconds={n:0. for n in names};native_reference=CommonMetrics()
        pool=primary.CalibrationReservoir(50000,task['seed']+70000);primary.seed(task['seed']+30000);windows=steps=0;start=time.monotonic()
        for step,batch in enumerate(interface['test'],1):
            torch.cuda.synchronize();begin=time.monotonic()
            with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
            torch.cuda.synchronize();timings['Native']+=time.monotonic()-begin
            assert x.shape[1:3]==(100,192) and y.shape[1]==192
            groups=state_ids(x,partition.mean_edges,partition.scale_edges);pool.update(x,y);native_reference.update(x,y)
            for name in names:
                begin=time.monotonic()
                if name=='Native':values=x
                elif name in models:values=models[name].transform(x)
                else:
                    b,s,h,c=x.shape;rows=x.transpose(0,2,3,1).reshape(-1,s)
                    values=gaussian_predict(fitting['models'][name],rows).reshape(b,h,c,s).transpose(0,3,1,2)
                if name!='Native':timings[name]+=time.monotonic()-begin
                begin=time.monotonic();metrics[name].update(values,y,groups,ordered=name!='Native');metric_seconds[name]+=time.monotonic()-begin
                del values
            windows+=len(y);steps=step
            if step%20==0:print('H192_TEST',task['id'],step,windows,round(time.monotonic()-start,1),flush=True)
        assert_traversal(interface['test'],steps,windows);scores={n:a.result() for n,a in metrics.items()};reference=native_reference.result()
        deltas={k:scores['Native'][k]-reference[k] for k in ['CRPS','MSE','QICE_percent']}
        if max(abs(v) for v in deltas.values())>1e-9:raise ValueError('Independent native metric accumulation differs')
        rows,y,h,c=pool.arrays();np.savez_compressed(out/'test_reservoir.npz',rows=rows,targets=y,horizon=h,channel=c)
        primary.controls(task,out,pool,partition,thresholds)
        comparison=dict(task=task,full_test_split=True,test_windows=windows,test_batches=steps,test_seed=task['seed']+30000,methods={n:scores[n] for n in ['Native','TrainDC16','TrainDC64']})
        primary.atomic(out/'training_partition_comparison.json',comparison)
        train_out=root/'train_calibration/runs'/task['id']
        primary.atomic(train_out/'results.json',dict(task=task,full_test_split=True,test_windows=windows,test_batches=steps,test_seed=task['seed']+30000,common_normalized={new:scores[old] for old,new in [('Native','Native'),('TrainDC16','DC16'),('TrainDC64','DC64')]},native_reference=reference))
        record=dict(version=primary.VERSION,task=task,checkpoint=interface['checkpoint'],full_test_split=True,test_windows=windows,test_batches=steps,test_seed=task['seed']+30000,methods={n:scores[n] for n in primary.METHODS},validation=validation,native_score_deltas=deltas,native_reference_metric_implementation='independent CommonMetrics accumulation on identical native forecasts',native_reference=reference,generation_or_transform_seconds={n:timings[n] for n in primary.METHODS},metric_seconds=metric_seconds,native_generation_excludes_preparation=True,total_test_seconds=time.monotonic()-start,maximum_resident_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),diagnostic_partition='training-fitted DC16 mean/scale boundaries shared by all methods',native_output_shape=[100,192,int(thresholds.shape[0])])
        primary.atomic(gaussian_dir/'results.json',dict(version='full-gaussian-controls-v1-20261102',task=task,full_test_split=True,checkpoint=interface['checkpoint'],test_windows=windows,test_batches=steps,test_seed=task['seed']+30000,methods={n:scores[n] for n in ['Native','GaussianVariance2P','GaussianAffine4P']},fitting=fitting,native_score_deltas={k:0. for k in deltas},diagnostic_partition=record['diagnostic_partition'],same_primary_prediction_pass=True))
        primary.atomic(out/'results.json',record)
    import benchmark_compiled,run_state_controls
    benchmark_compiled.run(task,root);run_state_controls.ROOT=root;run_state_controls.TRAIN=root/'train_calibration/runs';run_state_controls.run(task)
    print('NSDIFF_H192_COMPLETE',task['id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);run(json.loads(a.task.read_text()),a.root)
