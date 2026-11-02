"""Independent arithmetic and state-template audit for retained-cohort interventions."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, backbone_statistics, configuration_statistics, write_csv, close
from portable_dc import CompiledCalibrator

VERSION = 'state-assignment-controls-v1-20261102'
METHODS = {'Native', 'DC16', 'GlobalDC'} | {f'Shuffled{i}' for i in range(5)}
METRICS = ['CRPS', 'QICE_percent', 'state_error_pp', 'BrierMean', 'crps_reduction_percent']


def audit(path, expected, primary, compiled_runs):
    value = json.loads(path.read_text()); task = value['task']; identity = task['id']
    if path.parent.name != identity or identity not in expected or value['version'] != VERSION:
        raise ValueError('State-control identity/version mismatch')
    for key in ['backbone', 'dataset', 'horizon', 'seed']:
        if task[key] != expected[identity][key]:
            raise ValueError('Manifest mismatch')
    if set(value['methods']) != METHODS or value['permutation_seeds'] != [task['seed']+80000+i for i in range(5)]:
        raise ValueError('Incomplete intervention protocol')
    if not all(value[key] for key in ['fixed_source_mask', 'identical_global_pool', 'identical_state_counts', 'genuine_matches_existing_control']):
        raise ValueError('Intervention changed a controlled factor')
    old = json.loads((primary/identity/'controls.json').read_text())
    if value['coordinates'] != old['retained_test_coordinates']:
        raise ValueError('Retained test population mismatch')
    genuine = CompiledCalibrator.load(path.parent/'genuine.npz')
    reference = CompiledCalibrator.load(compiled_runs/identity/'compiled.npz')
    for key in ['mean_edges', 'scale_edges', 'quantiles', 'state_to_template']:
        np.testing.assert_array_equal(getattr(genuine, key), getattr(reference, key))
    for i in range(5):
        shuffled = CompiledCalibrator.load(path.parent/f'shuffled{i}.npz')
        for key in ['mean_edges', 'scale_edges', 'state_to_template']:
            np.testing.assert_array_equal(getattr(genuine, key), getattr(shuffled, key))
        if not np.isfinite(shuffled.quantiles).all() or np.any(np.diff(shuffled.quantiles, axis=1) < 0):
            raise ValueError('Invalid shuffled empirical quantiles')
    native = value['methods']['Native']; rows = []
    for method, metric in value['methods'].items():
        audit_metric(metric)
        if metric['coordinate_count'] != value['coordinates'] or metric['state_coverage']['counts'] != native['state_coverage']['counts']:
            raise ValueError('Unpaired intervention populations')
        close(metric['events']['observed_frequencies'], native['events']['observed_frequencies'], 'Events changed')
        if method == 'DC16':
            for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE']:
                close(metric[key], old['methods']['Budget50000'][key], 'Genuine score differs from earlier control')
        row = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
        row.update(method=method, CRPS=metric['CRPS'], QICE_percent=metric['QICE_percent'],
            state_error_pp=None if metric['state_coverage']['state_error'] is None else 100*metric['state_coverage']['state_error'],
            BrierMean=float(np.mean(metric['events']['brier'])),
            crps_reduction_percent=100*(native['CRPS']-metric['CRPS'])/native['CRPS'])
        rows.append(row)
    # Average five permutations within a task before aggregating training seeds.
    shuffled_rows = [r for r in rows if r['method'].startswith('Shuffled')]
    average = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
    average.update(method='ShuffledMean')
    for key in METRICS:
        average[key] = None if any(r[key] is None for r in shuffled_rows) else float(np.mean([r[key] for r in shuffled_rows]))
    rows.append(average)
    return identity, rows


def main():
    parser = argparse.ArgumentParser()
    for name in ['runs', 'manifest', 'primary-runs', 'compiled-runs', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); expected = {t['id']:t for t in json.loads(args.manifest.read_text())}
    rows = []; completed = []; errors = []
    for path in sorted(args.runs.glob('*/results.json')):
        try:
            identity, values = audit(path, expected, args.primary_runs, args.compiled_runs)
            completed.append(identity); rows.extend(values)
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'raw_task_metrics.csv', rows)
    write_csv(args.output/'configuration_statistics.csv', configuration_statistics(rows, METRICS))
    write_csv(args.output/'backbone_statistics.csv', backbone_statistics(rows, METRICS))
    missing = sorted(set(expected)-set(completed))
    report = dict(expected_tasks=len(expected), audited_tasks=len(completed), all_tasks_complete=not missing and not errors,
        missing_tasks=missing, errors=errors, evaluation='shared retained test coordinates, not complete-test scores',
        aggregation='five permutations averaged per task; configurations equally averaged within seeds; sample SD over three seeds')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
