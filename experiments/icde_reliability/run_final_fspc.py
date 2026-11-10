"""Run full FSPC factorial accuracy, complete fitting cost and exact deployment."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import time
import numpy as np
import torch
from final_fspc import VERSION,ARMS,CompiledFSPC,factorial_forecasts,save_parity_probe
from fspc_unified import UnifiedMean,fit_unified
from location_scale import replay_windows
from portable_dc import summaries,edges
from run_mean_improved_all import atomic,load,check_parity
from run_unified_components import score

OPERATIONS=('reference_decode','compiled_decode','reference_decode_and_queries',
            'compiled_decode_and_queries','direct_queries_from_rows','direct_queries_from_summaries')


def initialize(counter,cpus):
    with counter.get_lock():slot=counter.value;counter.value+=1
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,cpus[2*slot:2*slot+2])
    torch.set_num_threads(2)


def inputs(task,source,means,aux,prior):
    ident=task['id'];diff=task['backbone']=='DiffusionTS'
    previous=load(means/'runs'/ident/'results.json')
    provenance=dict(previous['provenance'],reference_selection=previous['fit']['selection'])
    val_path=prior/'unclipped/runs'/ident/'val_calibration.npz' if diff else source/'runs'/ident/'calibration.npz'
    train_path=prior/'unclipped/runs'/ident/'train_calibration.npz' if diff else source/'train_calibration/runs'/ident/'calibration.npz'
    test_path=aux/'diffusion/runs'/ident/'test_reservoir.npz' if diff else source/'runs'/ident/'test_reservoir.npz'
    with np.load(val_path) as d:
        val,vy,vh,vc=[d[k] for k in ['rows','targets','horizon','channel']]
        if 'window' in d:window=d['window']
        else:
            window,hh,cc=replay_windows(provenance['calibration_windows'],task['horizon'],int(vc.max())+1,
                50000,task['seed']+40000,provenance['batch_sizes'])
            np.testing.assert_array_equal(hh,vh);np.testing.assert_array_equal(cc,vc)
    with np.load(train_path) as d:train,ty,th=[d[k] for k in ['rows','targets','horizon']]
    model=UnifiedMean.from_record(load(aux/'components/runs'/ident/'parameters.json'))
    return model,(train,ty,th,val,vy,vh,window),provenance,test_path,dict(validation=str(val_path),training=str(train_path),
        test=str(test_path),frozen_parameters=str(aux/'components/runs'/ident/'parameters.json'))


def task_run(task,root,source,means,aux,prior,workers):
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'results.json').exists():
        r=load(out/'results.json');assert r['version']==VERSION and r['task']==task
        assert r['parity']['exact_reconstructed_values'] and r['parity']['exact_event_probabilities'];return task['id']
    start=time.monotonic();model,fitargs,provenance,test_path,sources=inputs(task,source,means,aux,prior)
    atomic(out/'parameters.json',model.to_record())
    fit_times=[];fit_cpu=[];fit_maximum=0.;fit_meta=None;fit_checks=[]
    validation=fitargs[3];vh=fitargs[5];probe=np.arange(min(512,len(validation)))
    for repeat in range(3):
        before=time.perf_counter();cpu=time.process_time()
        fitted,fit_meta=fit_unified(*fitargs,provenance['calibration_windows'],provenance['origin_stride'],task['horizon'])
        fit_cpu.append(time.process_time()-cpu);fit_times.append(time.perf_counter()-before)
        np.testing.assert_array_equal(fitted.cal.state_to_template,model.cal.state_to_template)
        for key in ['mean_edges','scale_edges','quantiles']:
            np.testing.assert_allclose(getattr(fitted.cal,key),getattr(model.cal,key),rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(fitted.bias,model.bias,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(fitted.strength,model.strength,rtol=1e-10,atol=1e-9)
        # Finite-ensemble CRPS is piecewise linear in a location blend. Its
        # minimizer need not be numerically unique. Check the actual historical
        # selection objective and MSE feasibility, rather than demanding equal
        # optimizer arguments on a flat optimum.
        weight_difference=abs(fitted.weight-model.weight)
        old_selection=provenance['reference_selection'];selection=fit_meta['selection']
        for key in ['reference_crps','selected_crps','reference_mse']:
            np.testing.assert_allclose(selection[key],old_selection[key],rtol=1e-10,atol=1e-10)
        assert selection['selected_mse']<=selection['reference_mse']+1e-10*max(1.,selection['reference_mse'])
        for key in ['intercept','slope','state_shift']:
            np.testing.assert_allclose(fitted.location[key],model.location[key],rtol=1e-10,atol=1e-9)
        got=fitted.predict(validation[probe],vh[probe]);expected=model.predict(validation[probe],vh[probe])
        span=model.predict(validation[probe],vh[probe],weight=1.)-model.predict(validation[probe],vh[probe],weight=0.)
        prediction_difference=float(np.max(abs(got-expected)))
        prediction_bound=weight_difference*float(np.max(abs(span)))+1e-9+1e-10*float(np.max(abs(expected)))
        assert prediction_difference<=prediction_bound,(task['id'],prediction_difference,prediction_bound)
        fit_checks.append(dict(weight_difference=weight_difference,prediction_difference=prediction_difference,prediction_bound=prediction_bound,
            selection_crps_difference=abs(selection['selected_crps']-old_selection['selected_crps']),
            reference_crps_difference=abs(selection['reference_crps']-old_selection['reference_crps']),
            selected_mse=selection['selected_mse'],reference_mse=selection['reference_mse']))
        fit_maximum=max(fit_maximum,prediction_difference)
        atomic(out/'progress.json',dict(stage='complete_fit',repetitions=repeat+1,elapsed=time.monotonic()-start))
    del fitted,got,expected,validation,fitargs
    # All fitting and numerical parameter checks precede reading any test target.
    atomic(out/'frozen.json',dict(version=VERSION,task=task,fit=fit_meta,parameters_frozen_before_test=True,
        test_targets_used_for_fitting=False,sources=sources))
    with np.load(test_path) as d:rows,y,h,c=[d[k] for k in ['rows','targets','horizon','channel']]
    with np.load(source/'runs'/task['id']/'calibration.npz') as d:all_thresholds=d['thresholds']
    threshold=all_thresholds[c]
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:original_train=d['rows']
    m,_,s=summaries(original_train);me=edges(m,6);se=edges(np.log(s),6);del original_train
    mean,_,scale=summaries(rows)
    groups=6*np.searchsorted(me[1:-1],mean,side='right')+np.searchsorted(se[1:-1],np.log(scale),side='right')
    arms=factorial_forecasts(model,rows,h)
    mean_error=float(np.max(abs(arms['ShapeOnly'].mean(1)-mean)))
    mean_pair_error=float(np.max(abs(arms['MeanOnly'].mean(1)-arms['FSPC'].mean(1))))
    spread_error=float(np.max(abs(arms['MeanOnly'].std(1,ddof=1)-rows.std(1,ddof=1))))
    scale_limit=1e-10*max(1.,float(abs(rows).max()))
    assert max(mean_error,mean_pair_error,spread_error)<=scale_limit
    metrics={name:score(arms[name],y,c,groups,all_thresholds,name not in ('Native','MeanOnly')) for name in ARMS}
    control=load(aux/'components/runs'/task['id']/'results.json');assert len(y)==control['coordinates']
    check_parity(metrics['Native'],control['native']);check_parity(metrics['FSPC'],control['methods']['FSPCMean'])
    for name,key in [('Native','native'),('FSPC','FSPCMean')]:
        wanted=control['native'] if name=='Native' else control['methods'][key]
        np.testing.assert_allclose(metrics[name]['MedianMAE'],wanted['MedianMAE'],atol=1e-9,rtol=1e-10)
    reference=arms['FSPC'];del arms
    compile_times=[]
    for _ in range(3):
        before=time.perf_counter();compiled=CompiledFSPC.from_model(model);compile_times.append(time.perf_counter()-before)
    compiled.save(out/'compiled.npz');compiled=CompiledFSPC.load(out/'compiled.npz')
    decoded=compiled.transform_rows(rows,h)
    difference=float(np.max(abs(decoded-reference)));exact=bool(np.array_equal(decoded,reference))
    assert exact,(task['id'],difference)
    def queries(values,thresholds=threshold):
        return np.stack([(values>thresholds[:,j,None]).mean(1) for j in range(3)],axis=1)
    expected=queries(reference);direct=compiled.probabilities_rows(rows,h,threshold)
    cached=compiled.probabilities_summaries(mean,scale,h,threshold)
    np.testing.assert_array_equal(direct,expected);np.testing.assert_array_equal(cached,expected)
    brier=((direct-(y[:,None]>threshold))**2).mean(0)
    np.testing.assert_allclose(brier,metrics['FSPC']['events']['brier'],atol=1e-12,rtol=1e-12)
    ties=reference[:,[0,49,99]]
    np.testing.assert_array_equal(compiled.probabilities_rows(rows,h,ties),queries(reference,ties))
    save_parity_probe(out/'parity_probe.npz',rows,h,threshold,reference)
    del decoded,direct,cached,expected,reference,ties
    operations=dict(reference_decode=lambda:model.predict(rows,h),compiled_decode=lambda:compiled.transform_rows(rows,h),
        reference_decode_and_queries=lambda:queries(model.predict(rows,h)),
        compiled_decode_and_queries=lambda:queries(compiled.transform_rows(rows,h)),
        direct_queries_from_rows=lambda:compiled.probabilities_rows(rows,h,threshold),
        direct_queries_from_summaries=lambda:compiled.probabilities_summaries(mean,scale,h,threshold))
    # Every operation receives one unmeasured warm-up, then randomized paired repetitions.
    for operation in operations.values():value=operation();del value
    times={key:[] for key in OPERATIONS};rng=np.random.default_rng(20261110+task['seed'])
    for repeat in range(3):
        for name in rng.permutation(OPERATIONS):
            before=time.perf_counter();value=operations[name]();times[name].append(time.perf_counter()-before);del value
    record=dict(version=VERSION,task=task,coordinates=len(y),samples=100,thresholds_per_coordinate=3,
        full_test_split=False,evaluation='shared retained test coordinates',
        backbone_retrained=False,native_inference_repeated=False,sources=sources,methods=metrics,
        fit=dict(wall_seconds=fit_times,cpu_seconds=fit_cpu,median_seconds=float(np.median(fit_times)),
            full_protocol=True,includes_mean_and_blend_selection=True,excludes_native_generation_and_file_io=True,
            replay_maximum_prediction_difference=fit_maximum,metadata=fit_meta),
        parity=dict(exact_reconstructed_values=exact,maximum_value_difference=difference,exact_event_probabilities=True,
            exact_boundary_tie_probabilities=True,serialized_reload_verified=True,brier=brier.tolist(),
            original_cohort_native_and_fspc_scores_reproduced=True,shape_only_mean_error=mean_error,
            shared_mean_error=mean_pair_error,mean_only_spread_error=spread_error),
        compilation_seconds=compile_times,median_compilation_seconds=float(np.median(compile_times)),
        reference_pickle_bytes=len(pickle.dumps(model,protocol=4)),compiled_npz_bytes=(out/'compiled.npz').stat().st_size,
        repeats=3,times=times,median_seconds={k:float(np.median(v)) for k,v in times.items()},
        environment=dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            torch_cpu_threads=torch.get_num_threads(),workers=workers,thread_limit_environment={k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
        elapsed_seconds=time.monotonic()-start)
    record['fit'].update(replay_checks=fit_checks,optimizer_xatol=1e-7,
        replay_criterion='same historical selection CRPS and feasible MSE; nonunique argmin recorded')
    atomic(out/'results.json',record);print('FINAL_FSPC_COMPLETE',task['id'],round(record['elapsed_seconds'],2),flush=True)
    return task['id']


def main():
    p=argparse.ArgumentParser()
    for key in ['root','source','means','aux','prior']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--task-id');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);controller=(a.root/'controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=load(a.source/'manifest.json')
    if a.task_id:tasks=[t for t in tasks if t['id']==a.task_id];assert len(tasks)==1
    atomic(a.root/'manifest.json',tasks);counter=mp.Value('i',0);cpus=sorted(os.sched_getaffinity(0))
    assert a.workers>0 and 2*a.workers<=len(cpus)
    pending=[t for t in tasks if not (a.root/'runs'/t['id']/'results.json').exists()];failures=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(counter,cpus)) as pool:
        active={}
        while pending or active:
            while pending and len(active)<a.workers:
                t=pending.pop(0);active[pool.submit(task_run,t,a.root,a.source,a.means,a.aux,a.prior,a.workers)]=t
            ready,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
            for future in ready:
                t=active.pop(future)
                try:future.result()
                except Exception as error:
                    import traceback
                    traceback.print_exc();failures.append(dict(task=t['id'],error=repr(error)))
            atomic(a.root/'status.json',dict(version=VERSION,expected=len(tasks),complete=sum((a.root/'runs'/t['id']/'results.json').exists() for t in tasks),
                pending=len(pending),active=[t['id'] for t in active.values()],failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
