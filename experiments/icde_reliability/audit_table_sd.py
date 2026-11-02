"""Check every published mean/SD cell against independently pooled raw seed scores."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from decimal import Decimal, ROUND_HALF_UP


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def estimate(rows, backbone, method, metric, dataset=None):
    groups = defaultdict(dict)
    for row in rows:
        if row['backbone'] != backbone or row['method'] != method or (dataset and row['dataset'] != dataset):
            continue
        if row.get(metric) and math.isfinite(float(row[metric])):
            groups[row['dataset'], row['horizon']][int(row['seed'])] = float(row[metric])
    complete = [values for values in groups.values() if set(values) == {1, 2, 3}]
    if not complete:
        raise ValueError(f'No complete three-seed evidence: {backbone}/{method}/{metric}')
    seed_averages = [mean(values[seed] for values in complete) for seed in (1, 2, 3)]
    return mean(seed_averages), stdev(seed_averages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args(); root = args.results; paper = args.paper
    backbones = ['K2VAE', 'PPM', 'TMDM', 'DiffusionTS', 'TimeDiff', 'NsDiff', 'DeepSTUQ']
    methods = ['Native', 'TrainDC16', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT',
               'GaussianVariance2P', 'GaussianAffine4P']
    primary = read(root/'summary/raw_task_metrics.csv')
    gaussian = read(root/'gaussian/summary/raw_task_metrics.csv')
    # Gaussian Native rows duplicate the paired primary reference population.
    combined = primary + [r for r in gaussian if r['method'] != 'Native']
    expected = {}
    for filename, metrics in [('full_distribution', ['CRPS', 'QICE_percent']),
                              ('full_reliability', ['state_error_pp', 'BrierMean'])]:
        expected[filename] = [estimate(combined, b, m, k) for k in metrics for m in methods for b in backbones]
    expected['point_mse'] = [estimate(primary, b, m, 'MSE') for b in backbones for m in ['Native', 'TrainDC16', 'DC16']]
    controls = read(root/'summary/raw_reservoir_controls.csv')
    expected['reservoir_controls'] = [estimate(controls, b, m, k) for b in backbones
        for k in ['CRPS', 'QICE_percent'] for m in ['Budget2000', 'Budget10000', 'Budget50000', 'DC64', 'UnfilteredDC16']]
    state = read(root/'state_controls/summary/raw_task_metrics.csv')
    expected['state_assignment'] = [estimate(state, b, m, k) for b in backbones
        for k in ['CRPS', 'QICE_percent'] for m in ['Native', 'GlobalDC', 'ShuffledMean', 'DC16']]
    compiled = read(root/'compiled/summary/raw_task_metrics.csv')
    expected['compiled_execution'] = [estimate(compiled, b, 'CompiledDC16', k) for b in backbones
        for k in ['compile_ms', 'compiled_kib', 'state_size_ratio', 'decode_speedup', 'direct_rows_query_speedup', 'cached_summary_query_speedup']]
    resources = read(root/'summary/raw_resource_metrics.csv')
    expected['execution_cost'] = [tuple(v*f for v in estimate(resources, b, 'DC16', k)) for b in backbones
        for k,f in [('dc16_fit_seconds', 1), ('dc16_calibrator_bytes', 1/1024),
                    ('dc16_transform_us_per_coordinate', 1), ('native_generation_us_per_coordinate', 1),
                    ('dc16_transform_to_generation_ratio', 100)]]
    diagnosis = read(paper/'figures/source_data/validation_native_task_diagnostics.csv')
    expected['problem_diagnostics'] = [estimate(diagnosis, b, 'Native', k) for b in backbones
        for k in ['pooled_median_error_pp', 'state_median_error_pp', 'state_median_range_pp']]
    counts = {}
    for name, values in expected.items():
        source = (paper/'tables'/f'{name}.tex').read_text()
        actual = re.findall(r'\$(-?\d+\.\d{3})\\pm(\d+\.\d{3})\$', source)
        target = [tuple(str(Decimal(f'{v:.12f}').quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
                        for v in (m, s)) for m, s in values]
        if actual != target or 'TBD' in source:
            raise ValueError(f'Mean/SD cells differ from independent raw-seed recomputation: {name}')
        counts[name] = len(actual)
    record = dict(all_passed=True, numeric_tables=len(counts), mean_sd_cells=sum(counts.values()),
                  per_table=counts, decimal_places=3, sample_sd_denominator=2,
                  pooling='equal configurations within each seed, then sample SD over seeds 1,2,3')
    (paper/'submission').mkdir(exist_ok=True)
    (paper/'submission/table_sd_audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
