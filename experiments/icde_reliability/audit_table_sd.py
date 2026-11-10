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
    primary = read(root/'summary/raw_task_metrics.csv')
    states = 36 if any(r['method']=='DC36' for r in primary) else 16
    default, train = f'DC{states}', f'TrainDC{states}'
    horizon = f'DC{states*4}'
    bins = 6 if states == 36 else 4
    methods = ['Native', default, 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']
    combined = primary
    expected = {}
    for filename, metrics in [('full_distribution', ['CRPS', 'QICE_percent']),
                              ('full_reliability', ['state_error_pp', 'BrierMean'])]:
        expected[filename] = []
        for k in metrics:
            for m in methods:
                expected[filename] += [estimate(combined,b,m,k) for b in backbones]
                if filename == 'full_distribution':
                    seed_ratios = []
                    for seed in (1,2,3):
                        ratios = []
                        for b in backbones:
                            reference = [float(r[k]) for r in primary
                                if r['backbone']==b and r['method']=='Native' and int(r['seed'])==seed]
                            values = [float(r[k]) for r in combined
                                if r['backbone']==b and r['method']==m and int(r['seed'])==seed]
                            if len(values)!=len(reference) or not values or mean(reference)<=0:
                                raise ValueError('Unpaired or invalid Native-normalized average')
                            ratios.append(mean(values)/mean(reference))
                        seed_ratios.append(mean(ratios))
                    expected[filename].append((mean(seed_ratios),stdev(seed_ratios)))
    expected['temporal_fitting'] = [estimate(primary, b, m, k)
        for k in ['CRPS', 'QICE_percent', 'state_error_pp', 'BrierMean']
        for m in [train, default] for b in backbones]
    expected['point_mse'] = [estimate(primary, b, m, 'MSE') for b in backbones for m in ['Native', train, default]]
    controls = read(root/'summary/raw_reservoir_controls.csv')
    expected['reservoir_controls'] = [estimate(controls, b, m, k) for b in backbones
        for k in ['CRPS', 'QICE_percent'] for m in ['Budget2000', 'Budget10000', 'Budget50000', horizon, 'Unfiltered'+default]]
    state = read(root/'state_controls/summary/raw_task_metrics.csv')
    expected['state_assignment'] = [estimate(state, b, m, k) for b in backbones
        for k in ['CRPS', 'QICE_percent'] for m in ['Native', 'GlobalDC', 'ShuffledMean', default]]
    compiled = read(root/'compiled/summary/raw_task_metrics.csv')
    expected['compiled_execution'] = [estimate(compiled, b, 'Compiled'+default, k) for b in backbones
        for k in ['compile_ms', 'compiled_kib', 'state_size_ratio', 'decode_speedup', 'direct_rows_query_speedup', 'cached_summary_query_speedup']]
    resources = read(root/'summary/raw_resource_metrics.csv')
    expected['execution_cost'] = [tuple(v*f for v in estimate(resources, b, default, k)) for b in backbones
        for k,f in [(f'dc{states}_fit_seconds', 1), (f'dc{states}_calibrator_bytes', 1/1024),
                    (f'dc{states}_transform_us_per_coordinate', 1), ('native_generation_us_per_coordinate', 1),
                    (f'dc{states}_transform_to_generation_ratio', 100)]]
    diagnosis = read(paper/'figures/source_data/validation_native_task_diagnostics.csv')
    expected['problem_diagnostics'] = [estimate(diagnosis, b, 'Native', k) for b in backbones
        for k in ['pooled_median_error_pp', 'state_median_error_pp', 'state_median_range_pp']]
    main_file = paper/'main.tex'
    sources = (main_file.read_text() if main_file.exists() else '') + ''.join(p.read_text() for p in (paper/'sections').glob('*.tex'))
    if r'\input{tables/granularity.tex}' in sources or (not main_file.exists() and (paper/'tables/granularity.tex').exists()):
        granularity = read(root/'granularity/summary/raw_task_metrics.csv')
        def retained_native_ratio(method):
            averages = []
            for seed in (1, 2, 3):
                ratios = []
                for backbone in backbones:
                    reference = {r['id']: float(r['CRPS']) for r in state
                        if r['backbone'] == backbone and r['method'] == 'Native' and int(r['seed']) == seed}
                    corrected = {r['id']: float(r['CRPS']) for r in granularity + controls
                        if r['backbone'] == backbone and r['method'] == method and int(r['seed']) == seed}
                    if not reference or reference.keys() != corrected.keys() or mean(reference.values()) <= 0:
                        raise ValueError('Unpaired retained-cohort ratio: '+backbone+'/'+method)
                    ratios.append(mean(corrected.values()) / mean(reference.values()))
                averages.append(mean(ratios))
            return mean(averages), stdev(averages)
        expected['granularity'] = [estimate(granularity,b,f'b{bins}_h1_n50000','CRPS')
            for b in backbones for bins in [1,2,3,4,6,8,12,16]]
        expected['granularity'] += [retained_native_ratio(f'b{bins}_h1_n50000')
            for bins in [1,2,3,4,6,8,12,16]]
        for b in backbones:
            expected['granularity'] += [estimate(granularity,b,m,'CRPS') for m in
                [f'b{bins}_h1_n2000',f'b{bins}_h1_n10000',f'b{bins}_h2_n50000',f'b{bins}_h4_n50000',f'b{bins}_h8_n50000']]
            expected['granularity'].append(estimate(controls,b,'Unfiltered'+default,'CRPS'))
        expected['granularity'] += [retained_native_ratio(m) for m in
            [f'b{bins}_h1_n2000',f'b{bins}_h1_n10000',f'b{bins}_h2_n50000',f'b{bins}_h4_n50000',f'b{bins}_h8_n50000','Unfiltered'+default]]
        if r'\input{tables/reservoir_controls.tex}' not in sources:
            expected.pop('reservoir_controls')
    best_flags = {name: [False]*len(values) for name,values in expected.items()}
    for name, n_methods in [('full_distribution',len(methods)),('full_reliability',len(methods)),('temporal_fitting',2)]:
        columns = len(backbones)+(name=='full_distribution')
        for start in range(0,len(expected[name]),n_methods*columns):
            for b in range(columns):
                indices=[start+b+columns*m for m in range(n_methods)]
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
        for start,end,size in [(0,64,8),(64,112,6)]:
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
    record = dict(all_passed=True, default_states=states, numeric_tables=len(counts), mean_sd_cells=sum(counts.values()),
                  per_table=counts, decimal_places=3, sample_sd_denominator=2,
                  pooling='equal configurations within each seed, then sample SD over seeds 1,2,3',
                  sd_format='right subscript with plus/minus',
                  best_mean_cells=sum(sum(flags) for flags in best_flags.values()),
                  distribution_average='equal backbone weights after dividing each seed-wise configuration mean by the corresponding Native mean',
                  best_rule='unrounded mean within the same backbone, metric and displayed comparison; numerical ties within 1e-12')
    (paper/'submission').mkdir(exist_ok=True)
    (paper/'submission/table_sd_audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
