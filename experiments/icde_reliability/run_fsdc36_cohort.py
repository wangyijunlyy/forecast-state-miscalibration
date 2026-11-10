"""36-based budget/horizon/source controls and the paired granularity sweep."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
from granularity import fit_partition, reconstruct, method_name
from reliability_metrics import ReliabilityMetrics
from aggregation import close

VERSION = 'fsdc36-cohort-v1-20261103'
PARTITIONS = [(b, 1) for b in [1,2,3,4,6,8,12,16]]+[(6,g) for g in [2,4,8]]
BUDGETS = [2000,10000,50000]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def run(task, root):
    output = root/'cohort/runs'/task['id']/'results.json'
    if output.exists():
        if json.loads(output.read_text())['version'] != VERSION:
            raise ValueError('Cohort version mismatch')
        return
    start = time.monotonic(); source = Path(task['source36_root'])/'runs'/task['id']
    with np.load(source/'calibration.npz') as d:
        fitting, target, h, thresholds = [d[k] for k in ['rows','targets','horizon','thresholds']]
    with np.load(source/'test_reservoir.npz') as d:
        rows, y, test_h, channels = [d[k] for k in ['rows','targets','horizon','channel']]
    order = np.random.default_rng(task['seed']+60000).permutation(len(target))
    diagnostic, _ = fit_partition(fitting,target,h,task['horizon'],6,1)
    mean = rows.mean(1); scale = rows.std(1,ddof=1).clip(1e-6)
    groups = 6*np.searchsorted(diagnostic.mean_edges[1:-1],mean,side='right')
    groups += np.searchsorted(diagnostic.scale_edges[1:-1],np.log(scale),side='right')
    def score(values, ordered):
        acc = ReliabilityMetrics(thresholds[channels],n_states=36)
        acc.update(values.T[None,:,None,:],y[None,None,:],groups,ordered=ordered)
        return acc.result()
    native = score(rows,False); methods = {}; resources = {}
    previous_granularity = Path(task['source36_root'])
    if task['backbone'] == 'DeepSTUQ':
        previous_granularity = previous_granularity.parent
    old = json.loads((previous_granularity/'granularity/runs'/task['id']/'results.json').read_text())
    checks = []
    for budget in BUDGETS:
        take = order[:min(budget,len(target))]
        fitted = {}
        for b,g in PARTITIONS:
            begin = time.monotonic()
            cal, meta = fit_partition(fitting[take],target[take],h[take],task['horizon'],b,g)
            meta['fit_seconds'] = time.monotonic()-begin; fitted[b,g] = cal,meta
        for (b,g),(cal,meta) in fitted.items():
            name = method_name(b,g,budget); begin = time.monotonic()
            values, fallback = reconstruct(cal,rows,test_h)
            meta.update(transform_seconds=time.monotonic()-begin,requested_budget=budget,
                fallback_test_coordinates=fallback,fallback_percent=100*fallback/len(y))
            methods[name] = score(values,True); resources[name] = meta
            if g == 1:
                for key in ['CRPS','QICE_percent','MSE']:
                    close(methods[name][key],old['methods'][name][key],'Historical granularity parity failed')
                checks.append(name)
    take = order
    cal, meta = fit_partition(fitting[take],target[take],h[take],task['horizon'],6,1,source_filter=False)
    values, fallback = reconstruct(cal,rows,test_h)
    meta.update(requested_budget=50000,fallback_test_coordinates=fallback,fallback_percent=100*fallback/len(y))
    methods['UnfilteredDC36'] = score(values,True); resources['UnfilteredDC36'] = meta
    atomic(output,dict(version=VERSION,task=task,default_states=36,
        evaluation='shared retained test cohort after full native traversal',fitting_split='validation',
        coordinates=len(y),available_calibration_coordinates=len(target),
        diagnostic_partition='validation-fitted 6x6 boundaries shared across cohort controls',
        native=native,methods=methods,resources=resources,historical_parity_checks=checks,
        elapsed_seconds=time.monotonic()-start))
    print('FSDC36_COHORT_COMPLETE',task['id'],round(time.monotonic()-start,2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    tasks=json.loads((args.root/'manifest.json').read_text());errors=[];complete=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,t,args.root):t['id'] for t in tasks}
        for future in as_completed(futures):
            try:future.result();complete+=1
            except Exception as error:
                errors.append(dict(task=futures[future],error=repr(error)))
                print('FSDC36_COHORT_FAILED',futures[future],repr(error),flush=True)
            atomic(args.root/'cohort/status.json',dict(controller_pid=os.getpid(),expected=249,complete=complete,errors=errors))
    if errors:raise SystemExit(1)
