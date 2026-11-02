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
    methods = ['Native', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT',
               'GaussianVariance2P', 'GaussianAffine4P']
    primary = read(root/'summary/raw_task_metrics.csv')
    gaussian = read(root/'gaussian/summary/raw_task_metrics.csv')
    # Gaussian Native rows duplicate the paired primary reference population.
    combined = primary + [r for r in gaussian if r['method'] != 'Native']
    expected = {}
    for filename, metrics in [('full_distribution', ['CRPS', 'QICE_percent']),
                              ('full_reliability', ['state_error_pp', 'BrierMean'])]:
        expected[filename] = [estimate(combined, b, m, k) for k in metrics for m in methods for b in backbones]
    expected['temporal_fitting'] = [estimate(primary, b, m, k)
        for k in ['CRPS', 'QICE_percent', 'state_error_pp', 'BrierMean']
        for m in ['TrainDC16', 'DC16'] for b in backbones]
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
    main_file = paper/'main.tex'
    sources = (main_file.read_text() if main_file.exists() else '') + ''.join(p.read_text() for p in (paper/'sections').glob('*.tex'))
    if r'\input{tables/granularity.tex}' in sources or (not main_file.exists() and (paper/'tables/granularity.tex').exists()):
        granularity = read(root/'granularity/summary/raw_task_metrics.csv')
        expected['granularity'] = [estimate(granularity,b,f'b{bins}_h1_n50000','CRPS')
            for b in backbones for bins in [1,2,3,4,6,8,12,16]]
        for b in backbones:
            expected['granularity'] += [estimate(granularity,b,m,'CRPS') for m in
                ['b4_h1_n2000','b4_h1_n10000','b4_h2_n50000','b4_h4_n50000','b4_h8_n50000']]
            expected['granularity'].append(estimate(controls,b,'UnfilteredDC16','CRPS'))
        if r'\input{tables/reservoir_controls.tex}' not in sources:
            expected.pop('reservoir_controls')
    best_flags = {name: [False]*len(values) for name,values in expected.items()}
    for name, n_methods in [('full_distribution',7),('full_reliability',7),('temporal_fitting',2)]:
        for start in range(0,len(expected[name]),n_methods*len(backbones)):
            for b in range(len(backbones)):
                indices=[start+b+len(backbones)*m for m in range(n_methods)]
                optimum=min(expected[name][i][0] for i in indices)
                for i in indices:
                    best_flags[name][i]=math.isclose(expected[name][i][0],optimum,rel_tol=0,abs_tol=1e-12)
    for name,n_methods in [('point_mse',3),('reservoir_controls',5),('state_assignment',4)]:
        if name not in expected:
            continue
        for start in range(0,len(expected[name]),n_methods):
            optimum=min(v[0] for v in expected[name][start:start+n_methods])
            for i in range(start,start+n_methods):
                best_flags[name][i]=math.isclose(expected[name][i][0],optimum,rel_tol=0,abs_tol=1e-12)
    if 'granularity' in expected:
        for start,end,size in [(0,56,8),(56,98,6)]:
            for offset in range(start,end,size):
                optimum=min(v[0] for v in expected['granularity'][offset:offset+size])
                for i in range(offset,offset+size):
                    best_flags['granularity'][i]=math.isclose(expected['granularity'][i][0],optimum,rel_tol=0,abs_tol=1e-12)
    counts = {}
    for name, values in expected.items():
        source = (paper/'tables'/f'{name}.tex').read_text()
        cells = re.findall(r'\\(beststat|stat)\{(-?\d+\.\d{3})\}\{(\d+\.\d{3})\}', source)
        actual = [(m,s) for _,m,s in cells]
        target = [tuple(str(Decimal(f'{v:.12f}').quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
                        for v in (m, s)) for m, s in values]
        if actual != target or 'TBD' in source:
            raise ValueError(f'Mean/SD cells differ from independent raw-seed recomputation: {name}')
        if [kind=='beststat' for kind,_,_ in cells] != best_flags[name]:
            raise ValueError(f'Bold means differ from full-precision within-backbone comparison: {name}')
        if '\\pm' in source or not any(arrow in source for arrow in ['\\downarrow','\\uparrow']):
            raise ValueError(f'Subscript SD or metric-direction formatting missing: {name}')
        counts[name] = len(actual)
    record = dict(all_passed=True, numeric_tables=len(counts), mean_sd_cells=sum(counts.values()),
                  per_table=counts, decimal_places=3, sample_sd_denominator=2,
                  pooling='equal configurations within each seed, then sample SD over seeds 1,2,3',
                  sd_format='right subscript with plus/minus',
                  best_mean_cells=sum(sum(flags) for flags in best_flags.values()),
                  best_rule='unrounded mean within the same backbone, metric and displayed comparison; numerical ties within 1e-12')
    (paper/'submission').mkdir(exist_ok=True)
    (paper/'submission/table_sd_audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
