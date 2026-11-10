"""True 36-state assignment controls and exact compiled-query measurements."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
from portable_dc import EmpiricalCalibrator, summaries, bounded
from reliability_metrics import ReliabilityMetrics

VERSION='fsdc36-assignment-compiled-v1-20261103'


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run(task,root):
    identity=task['id']; directory=root/'runs'/identity
    output=root/'state_controls/runs'/identity; benchmark=root/'compiled_deployment/runs'/identity
    if (output/'results.json').exists() and (benchmark/'results.json').exists():return
    output.mkdir(parents=True,exist_ok=True);benchmark.mkdir(parents=True,exist_ok=True)
    primary=json.loads((directory/'results.json').read_text())
    with (directory/'calibrators.pkl').open('rb') as f:models=pickle.load(f)
    with (root/'train_calibration/runs'/identity/'calibrators.pkl').open('rb') as f:training=pickle.load(f)['DC36']
    with np.load(directory/'calibration.npz') as d:
        fitting,target,h,thresholds=[d[k] for k in ['rows','targets','horizon','thresholds']]
    with np.load(directory/'test_reservoir.npz') as d:
        rows,y,test_h,channels=[d[k] for k in ['rows','targets','horizon','channel']]
    original=models['DC36'];fit_mean,raw,fit_scale=summaries(fitting)
    valid=raw>1.01e-6;residual=(target-fit_mean)/fit_scale
    cal=EmpiricalCalibrator(original.mean_edges,original.scale_edges,original.global_residuals,
        dict(original.bin_residuals),task['horizon'],1,int(valid.sum()),int((~valid).sum()))
    np.testing.assert_array_equal(bounded(residual[valid]),cal.global_pool)
    mu,_,sigma=summaries(rows)
    groups=6*np.searchsorted(training.mean_edges[1:-1],mu,side='right')
    groups+=np.searchsorted(training.scale_edges[1:-1],np.log(sigma),side='right')
    coordinate_thresholds=thresholds[channels]
    def score(values,ordered):
        acc=ReliabilityMetrics(coordinate_thresholds,n_states=36)
        acc.update(values.T[None,:,None,:],y[None,None,:],groups,ordered=ordered)
        return acc.result()
    begin=time.perf_counter();compiled=cal.compile();compile_seconds=time.perf_counter()-begin
    compiled.save(benchmark/'compiled.npz');compiled.save(output/'genuine.npz')
    decoded=original.transform(rows[:,:,None,None])[:,:,0,0]
    actual=compiled.transform_rows(rows,test_h)
    np.testing.assert_array_equal(actual,decoded)
    def queries(values):
        return np.stack([(values>coordinate_thresholds[:,j,None]).mean(1) for j in range(3)],axis=1)
    probabilities=queries(decoded)
    np.testing.assert_array_equal(compiled.probabilities_rows(rows,test_h,coordinate_thresholds),probabilities)
    np.testing.assert_array_equal(compiled.probabilities_summaries(mu,sigma,test_h,coordinate_thresholds),probabilities)
    brier=((probabilities-(y[:,None]>coordinate_thresholds))**2).mean(0)
    cohort=json.loads((root/'cohort/runs'/identity/'results.json').read_text())
    np.testing.assert_allclose(brier,cohort['methods']['b6_h1_n50000']['events']['brier'],atol=1e-10,rtol=1e-10)
    scores=dict(Native=score(rows,False),DC36=score(actual,True))
    for key in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
        np.testing.assert_allclose(scores['DC36'][key],cohort['methods']['b6_h1_n50000'][key],atol=1e-10,rtol=1e-10)
    global_q=np.quantile(models['GlobalDC'].global_residuals,(np.arange(100)+.5)/100)
    scores['GlobalDC']=score(mu[:,None]+sigma[:,None]*global_q,True)
    keys,inverse=np.unique(cal.keys(fit_mean,fit_scale,h),axis=0,return_inverse=True)
    for repeat in range(5):
        seed=task['seed']+80000+repeat
        shuffled=residual.copy();shuffled[valid]=residual[valid][np.random.default_rng(seed).permutation(valid.sum())]
        pools={tuple(key):bounded(shuffled[(inverse==i)&valid]) for i,key in enumerate(keys)
               if np.sum((inverse==i)&valid)>=128}
        if pools.keys()!=cal.local_pools.keys():raise ValueError('Shuffle changed source occupancy')
        changed=EmpiricalCalibrator(cal.mean_edges,cal.scale_edges,cal.global_pool,pools,task['horizon'],1,
                                   cal.retained_sources,cal.excluded_sources).compile()
        np.testing.assert_array_equal(changed.state_to_template,compiled.state_to_template)
        changed.save(output/f'shuffled{repeat}.npz')
        scores[f'Shuffled{repeat}']=score(changed.transform_rows(rows,test_h),True)
    atomic(output/'results.json',dict(version=VERSION,task=task,default_states=36,methods=scores,
        coordinates=len(y),fitting_coordinates=len(target),retained_sources=int(valid.sum()),
        excluded_sources=int((~valid).sum()),fixed_source_mask=True,identical_global_pool=True,
        identical_state_counts=True,genuine_matches_existing_control=True,
        permutation_seeds=[task['seed']+80000+i for i in range(5)],
        diagnostic_partition='training-fitted 6x6 boundaries shared by all methods'))
    del actual,decoded,probabilities
    operations=dict(reference_decode=lambda:original.transform(rows[:,:,None,None])[:,:,0,0],
        compiled_decode=lambda:compiled.transform_rows(rows,test_h),
        reference_decode_and_queries=lambda:queries(original.transform(rows[:,:,None,None])[:,:,0,0]),
        direct_queries_from_rows=lambda:compiled.probabilities_rows(rows,test_h,coordinate_thresholds),
        direct_queries_from_summaries=lambda:compiled.probabilities_summaries(mu,sigma,test_h,coordinate_thresholds))
    times={n:[] for n in operations};rng=np.random.default_rng(20261102+task['seed'])
    for _ in range(3):
        for name in rng.permutation(list(operations)):
            begin=time.perf_counter();value=operations[name]();times[name].append(time.perf_counter()-begin);del value
    atomic(benchmark/'results.json',dict(version=VERSION,task=task,default_states=36,
        coordinates=len(y),samples=100,thresholds_per_coordinate=3,repeats=3,
        complete_test_coordinates=primary['methods']['Native']['coordinate_count'],
        exact_reconstructed_values=True,exact_event_probabilities=True,maximum_value_difference=0.,
        brier=brier.tolist(),brier_matches_existing_control=True,times=times,
        median_seconds={n:float(np.median(v)) for n,v in times.items()},
        compilation_seconds=compile_seconds,original_pickle_bytes=len(pickle.dumps(original)),
        compiled_npz_bytes=(benchmark/'compiled.npz').stat().st_size,
        compiled_template_count=len(compiled.quantiles)))
    print('FSDC36_CONTROLS_COMPLETE',identity,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    tasks=json.loads((a.root/'manifest.json').read_text());failed={}
    while True:
        ready=[t for t in tasks if (a.root/'runs'/t['id']/'results.json').exists()
               and (a.root/'cohort/runs'/t['id']/'results.json').exists()
               and not (a.root/'compiled_deployment/runs'/t['id']/'results.json').exists() and t['id'] not in failed]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run,t,a.root):t['id'] for t in ready}
            for future in as_completed(futures):
                try:future.result()
                except Exception as error:
                    failed[futures[future]]=repr(error);print('FSDC36_CONTROLS_FAILED',futures[future],repr(error),flush=True)
        count=sum((a.root/'compiled_deployment/runs'/t['id']/'results.json').exists() for t in tasks)
        atomic(a.root/'controls_status.json',dict(complete=count,expected=249,failures=failed))
        status=json.loads((a.root/'status.json').read_text())
        remaining_ready = any((a.root/'runs'/t['id']/'results.json').exists()
            and (a.root/'cohort/runs'/t['id']/'results.json').exists()
            and not (a.root/'compiled_deployment/runs'/t['id']/'results.json').exists()
            and t['id'] not in failed for t in tasks)
        if count==249 or (not status['active'] and not status['pending'] and not remaining_ready):break
        time.sleep(15)
    if failed or count != 249:raise SystemExit(1)
