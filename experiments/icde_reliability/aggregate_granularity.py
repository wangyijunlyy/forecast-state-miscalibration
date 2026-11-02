"""Audit complete partition sweeps and summarize paired seed-level evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, backbone_statistics, configuration_statistics, write_csv, close
from granularity import VERSION, PARTITIONS, BUDGETS, method_name

METRICS = ['CRPS', 'QICE_percent', 'MSE', 'BrierMean', 'fallback_percent',
           'crps_change_vs_dc16_percent', 'eligible_local_pools', 'eligible_state_percent',
           'retained_sources', 'median_occupied_source_count']


def audit(value, expected):
    task = value['task']; identity = task['id']
    if task != expected[identity] or value['version'] != VERSION or value['fitting_split'] != 'validation':
        raise ValueError('Task/protocol identity mismatch')
    names = {method_name(b, g, n) for b, g in PARTITIONS for n in BUDGETS}
    if set(value['methods']) != names or set(value['resources']) != names:
        raise ValueError('Incomplete partition/budget sweep')
    checks = {method_name(4, 1, n) for n in BUDGETS} | {method_name(4, 4, 50000)}
    if set(value['historical_parity_checks']) != checks:
        raise ValueError('Missing historical parity checks')
    if value['calibration_permutation_seed'] != task['seed']+60000:
        raise ValueError('Changed nested subset seed')
    native = value['native']; audit_metric(native)
    if native['coordinate_count'] != value['coordinates'] or value['coordinates'] > 50000:
        raise ValueError('Evaluation population mismatch')
    rows = []
    for budget in BUDGETS:
        masks = set()
        for b, g in PARTITIONS:
            name = method_name(b, g, budget); metric = value['methods'][name]; meta = value['resources'][name]
            audit_metric(metric)
            if metric['coordinate_count'] != value['coordinates'] or metric['state_coverage']['counts'] != native['state_coverage']['counts']:
                raise ValueError('Unpaired evaluation coordinates or diagnostic boundaries')
            close(metric['events']['observed_frequencies'], native['events']['observed_frequencies'], 'Event definitions changed')
            n = min(budget, value['available_calibration_coordinates'])
            if (meta['mean_bins'], meta['scale_bins'], meta['horizon_bins'], meta['requested_budget']) != (b, b, g, budget):
                raise ValueError('Wrong partition or budget')
            counts = np.array(meta['source_counts'])
            if len(counts) != b*b*g or meta['nominal_states'] != b*b*g or counts.sum() != meta['retained_sources']:
                raise ValueError('Source-count conservation failed')
            if np.any(counts < 0) or int((counts >= 128).sum()) != meta['eligible_local_pools']:
                raise ValueError('128-source fallback rule changed')
            if int((counts > 0).sum()) != meta['occupied_states']:
                raise ValueError('Occupied state count mismatch')
            if meta['calibration_coordinates'] != n or meta['retained_sources']+meta['excluded_sources'] != n:
                raise ValueError('Calibration budget/source population mismatch')
            masks.add((meta['retained_sources'], meta['excluded_sources']))
            fallback = meta['fallback_test_coordinates']
            if not 0 <= fallback <= value['coordinates']:
                raise ValueError('Invalid test fallback count')
            close(meta['fallback_percent'], 100*fallback/value['coordinates'], 'Fallback percentage mismatch')
            base = value['methods'][method_name(4, 1, budget)]['CRPS']
            row = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
            row.update(method=name, mean_bins=b, horizon_bins=g, nominal_states=b*b*g,
                requested_budget=budget, actual_budget=n, coordinate_count=value['coordinates'],
                CRPS=metric['CRPS'], QICE_percent=metric['QICE_percent'], MSE=metric['MSE'],
                BrierMean=float(np.mean(metric['events']['brier'])),
                fallback_percent=meta['fallback_percent'],
                crps_change_vs_dc16_percent=100*(metric['CRPS']-base)/base,
                eligible_local_pools=meta['eligible_local_pools'],
                eligible_state_percent=100*meta['eligible_local_pools']/(b*b*g),
                retained_sources=meta['retained_sources'],
                median_occupied_source_count=float(np.median(counts[counts > 0])))
            rows.append(row)
        if len(masks) != 1:
            raise ValueError('Partitions do not share the same source-validity population')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {t['id']:t for t in json.loads((args.results/'manifest.json').read_text())}
    rows = []; completed = []; errors = []
    for path in sorted((args.results/'runs').glob('*/results.json')):
        try:
            value = json.loads(path.read_text())
            if path.parent.name != value['task']['id']:
                raise ValueError('Directory identity mismatch')
            rows.extend(audit(value, expected)); completed.append(value['task']['id'])
        except Exception as error:
            errors.append(dict(path=str(path), error=repr(error)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'raw_task_metrics.csv', rows)
    config = configuration_statistics(rows, METRICS)
    write_csv(args.output/'configuration_statistics.csv', config)
    write_csv(args.output/'backbone_statistics.csv', backbone_statistics(rows, METRICS))
    configuration_wins = []
    by_config = defaultdict(dict)
    for row in config:
        by_config[row['backbone'], row['dataset'], row['horizon']][row['method']] = row
    for budget in BUDGETS:
        for b, g in PARTITIONS:
            name = method_name(b, g, budget); base = method_name(4, 1, budget)
            differences = [v[name]['CRPS_mean']-v[base]['CRPS_mean'] for v in by_config.values()]
            configuration_wins.append(dict(method=name, configurations=len(differences),
                lower_than_dc16=int(sum(x < 0 for x in differences)),
                higher_than_dc16=int(sum(x > 0 for x in differences)),
                tied_with_dc16=int(sum(x == 0 for x in differences))))
    write_csv(args.output/'configuration_comparisons.csv', configuration_wins)
    report = dict(version=VERSION, expected_tasks=len(expected), audited_tasks=len(completed),
        all_tasks_complete=set(completed) == set(expected) and not errors and len(expected) == 249,
        missing_tasks=sorted(set(expected)-set(completed)), errors=errors,
        fits_per_task=33, audited_fits=len(rows), configurations=len(by_config),
        evaluation='shared retained test coordinates; separate from complete-test primary scores',
        aggregation='equal configuration weights within each seed; sample SD over seeds 1,2,3',
        metric_direction='CRPS change is positive for worse than same-budget DC16')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['all_tasks_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
