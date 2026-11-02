"""Execute the preregistered 33-fit partition/budget sweep for all 249 tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
from granularity import VERSION, PARTITIONS, BUDGETS, fit_partition, reconstruct, method_name
from reliability_metrics import ReliabilityMetrics
from aggregation import close


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def run(task, source, output):
    start = time.monotonic(); out = output/'runs'/task['id']/'results.json'
    if out.exists():
        value = json.loads(out.read_text())
        if value['version'] != VERSION or value['task'] != task:
            raise ValueError('Existing granularity result has incompatible identity')
        return task['id']
    directory = source/'runs'/task['id']
    with np.load(directory/'calibration.npz') as data:
        fitting, target, h, thresholds = [data[k] for k in ['rows', 'targets', 'horizon', 'thresholds']]
    with np.load(directory/'test_reservoir.npz') as data:
        rows, y, test_h, channels = [data[k] for k in ['rows', 'targets', 'horizon', 'channel']]
    control = json.loads((directory/'controls.json').read_text())
    if len(y) != control['retained_test_coordinates']:
        raise ValueError('Different retained test population')
    order = np.random.default_rng(task['seed']+60000).permutation(len(target))
    diagnostic, _ = fit_partition(fitting, target, h, task['horizon'], 4, 1)
    mean = rows.mean(1); scale = rows.std(1, ddof=1).clip(1e-6)
    groups = 4*np.searchsorted(diagnostic.mean_edges[1:-1], mean, side='right')
    groups += np.searchsorted(diagnostic.scale_edges[1:-1], np.log(scale), side='right')

    def score(values, ordered):
        accumulator = ReliabilityMetrics(thresholds[channels])
        accumulator.update(values.T[None, :, None, :], y[None, None, :], groups, ordered=ordered)
        return accumulator.result()

    native = score(rows, False); methods = {}; resources = {}; checks = []
    # Fit every partition in a budget before using any test target to score it.
    for budget in BUDGETS:
        take = order[:min(budget, len(target))]
        fitted = {}
        for b, g in PARTITIONS:
            begin = time.monotonic()
            cal, meta = fit_partition(fitting[take], target[take], h[take], task['horizon'], b, g)
            meta['fit_seconds'] = time.monotonic()-begin
            fitted[b, g] = cal, meta
        for (b, g), (cal, meta) in fitted.items():
            name = method_name(b, g, budget)
            begin = time.monotonic(); values, fallback = reconstruct(cal, rows, test_h)
            meta.update(transform_seconds=time.monotonic()-begin, requested_budget=budget,
                fallback_test_coordinates=fallback, fallback_percent=100*fallback/len(y))
            methods[name] = score(values, True); resources[name] = meta
            reference = f'Budget{budget}' if (b, g) == (4, 1) else 'DC64' if (b, g, budget) == (4, 4, 50000) else None
            if reference:
                for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE']:
                    close(methods[name][key], control['methods'][reference][key], 'Historical control parity failed: '+name+'/'+key)
                close(methods[name]['events']['brier'], control['methods'][reference]['events']['brier'], 'Brier parity failed')
                checks.append(name)
    atomic(out, dict(version=VERSION, task=task,
        evaluation='shared uniform retained coordinates after complete native test traversal',
        fitting_split='validation', source_directory=str(directory),
        full_test_coordinates_seen=control['full_test_coordinates_seen'], coordinates=len(y),
        available_calibration_coordinates=len(target), calibration_permutation_seed=task['seed']+60000,
        diagnostic_partition='validation-fitted 4x4 boundaries shared by all partitions and budgets',
        native=native, methods=methods, resources=resources, historical_parity_checks=checks,
        elapsed_seconds=time.monotonic()-start))
    print('GRANULARITY_COMPLETE', task['id'], round(time.monotonic()-start, 2), flush=True)
    return task['id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--task-id')
    args = parser.parse_args()
    tasks = []
    for source in [args.root, args.root/'deepstuq_backbone']:
        tasks.extend((task, source) for task in json.loads((source/'manifest.json').read_text()))
    if args.task_id:
        tasks = [(t, s) for t, s in tasks if t['id'] == args.task_id]
        if not tasks:
            raise ValueError('Unknown task')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic(args.output/'manifest.json', [t for t, _ in tasks])
    errors = []; completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, t, s, args.output):t['id'] for t, s in tasks}
        for future in as_completed(futures):
            try:
                future.result(); completed += 1
            except Exception as error:
                errors.append(dict(task=futures[future], error=repr(error)))
                print('GRANULARITY_FAILED', futures[future], repr(error), flush=True)
            atomic(args.output/'status.json', dict(version=VERSION, controller_pid=os.getpid(),
                expected_tasks=len(tasks), complete=completed, errors=errors))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
