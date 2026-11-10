"""Evaluate the thirteen Table V settings on frozen, paired forecast caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
from pathlib import Path
import time
import numpy as np
from fspc_granularity import VERSION,VARIANTS,fit_variant
from granularity import method_name
from mean_anchor import centered_forecasts, state_ids
from reliability_metrics import ReliabilityMetrics
from portable_dc import summaries
from mean_anchor_campaign import validate_tasks


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run(task,root,source,mean_root):
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target=out/'results.json'
        if target.exists():
            saved=json.loads(target.read_text())
            assert saved['task']==task and saved['version']==VERSION
            assert set(saved['methods'])=={method_name(*v) for v in VARIANTS}
            return task['id']
        begin=time.monotonic()
        paths=dict(training=source/'train_calibration/runs'/task['id']/'calibration.npz',
                   validation=source/'runs'/task['id']/'calibration.npz',
                   test=source/'runs'/task['id']/'test_reservoir.npz')
        with np.load(paths['training']) as d:train,train_y,train_h=[d[k] for k in ['rows','targets','horizon']]
        with np.load(paths['validation']) as d:val,val_y,val_h,thresholds=[d[k] for k in ['rows','targets','horizon','thresholds']]
        # Fit every variant before reading any held-out observation.
        fitted={}
        for b,g,budget in VARIANTS:
            fitted[method_name(b,g,budget)]=fit_variant(train,train_y,train_h,val,val_y,val_h,
                task['horizon'],b,g,budget,task['seed']+60000)
        del train,train_y,train_h,val,val_y,val_h
        with np.load(paths['test']) as d:rows,y,h,channels=[d[k] for k in ['rows','targets','horizon','channel']]
        original=json.loads((source/'cohort/runs'/task['id']/'results.json').read_text())
        reference=json.loads((mean_root/'runs'/task['id']/'cohort.json').read_text())
        assert reference['task']==original['task']==task
        assert reference['coordinates']==original['coordinates']==len(y)
        groups=np.zeros(len(y),dtype=int)
        methods={};resources={};maximum_mean_error=0.
        native_mean,_,_=summaries(rows)
        # CRPS, QICE and point scores do not depend on diagnostic partition choice.
        for name,(cal,bias,strength,meta) in fitted.items():
            forecasts=centered_forecasts(cal,rows,h,bias,strength)
            ids=state_ids(cal,rows,h)
            discrepancy=float(np.max(abs(forecasts.mean(1)-(native_mean+strength*bias[ids]))))
            assert discrepancy<=1e-8*max(1.,float(np.max(abs(native_mean))))
            maximum_mean_error=max(maximum_mean_error,discrepancy)
            acc=ReliabilityMetrics(thresholds[channels],n_states=1)
            acc.update(forecasts.T[None,:,None,:],y[None,None,:],groups,ordered=True)
            score=acc.result()
            methods[name]={k:score[k] for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE','coordinate_count']}
            active=np.asarray(meta['source_counts'])>=128
            meta['fallback_percent']=100*float(np.mean(~active[ids]))
            meta['maximum_mean_error']=discrepancy
            resources[name]=meta
        default=methods['b6_h1_n50000'];parity={}
        for metric in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
            difference=abs(default[metric]-reference['methods']['DC36ControlledMean'][metric])
            assert np.isclose(default[metric],reference['methods']['DC36ControlledMean'][metric],rtol=1e-10,atol=1e-9),(task['id'],metric,difference)
            parity[metric]=difference
        atomic(target,dict(version=VERSION,task=task,evaluation='same retained test coordinates as Table IV',
            coordinates=len(y),native=original['native'],methods=methods,resources=resources,
            training_budget='fixed original training reservoir (at most 50000 coordinates)',
            validation_budgets=[2000,10000,50000],all_variants_fit_before_test_access=True,
            existing_default_parity=parity,maximum_mean_error=maximum_mean_error,
            source_paths={k:str(v.resolve()) for k,v in paths.items()},neural_training_performed=False,
            elapsed_seconds=time.monotonic()-begin))
        print('FSPC_GRANULARITY_COMPLETE',task['id'],round(time.monotonic()-begin,2),flush=True)
        return task['id']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--mean-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--task-id')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    controller=(a.root/'controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((a.source/'manifest.json').read_text())
    validate_tasks(tasks,json.loads((a.mean_root/'manifest.json').read_text()))
    if a.task_id:tasks=[t for t in tasks if t['id']==a.task_id];assert len(tasks)==1
    atomic(a.root/'manifest.json',tasks)
    completed=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,t,a.root,a.source,a.mean_root):t['id'] for t in tasks}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as error:
                errors.append(dict(task=futures[future],error=repr(error)))
                print('FSPC_GRANULARITY_FAILED',futures[future],repr(error),flush=True)
            atomic(a.root/'status.json',dict(expected=len(tasks),complete=len(completed),errors=errors))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
