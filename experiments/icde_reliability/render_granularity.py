"""Render audited granularity/budget/source controls in one paper table."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from render_tables import BACKBONES, LABELS, read, require_complete, comparison_cells
from granularity import method_name


def average_ratios(results, primary, native_directory, methods):
    """Normalize within each backbone/run before equally averaging backbones."""
    require_complete(native_directory, 249)
    native = defaultdict(dict)
    for row in read(native_directory/'raw_task_metrics.csv'):
        if row['method'] == 'Native':
            native[row['backbone'], int(row['seed'])][row['id']] = float(row['CRPS'])
    scores = defaultdict(dict)
    rows = [r for r in read(results/'raw_task_metrics.csv') if not r['method'].startswith('Unfiltered')] + read(primary/'raw_reservoir_controls.csv')
    for row in rows:
        if row['method'] in methods:
            key = row['method'], row['backbone'], int(row['seed'])
            if row['id'] in scores[key]:
                raise ValueError('Duplicate task in ratio inputs')
            scores[key][row['id']] = float(row['CRPS'])
    output = {}
    for method in methods:
        values = []
        for seed in (1, 2, 3):
            ratios = []
            for backbone in BACKBONES:
                reference = native[backbone, seed]
                corrected = scores[method, backbone, seed]
                if not reference or reference.keys() != corrected.keys():
                    raise ValueError('Ratio inputs use different task populations: '+method+'/'+backbone)
                denominator = mean(reference.values())
                if denominator <= 0:
                    raise ValueError('Native CRPS must be positive')
                ratios.append(mean(corrected.values())/denominator)
            values.append(mean(ratios))
        output[method] = dict(CRPS_mean=mean(values), CRPS_std=stdev(values),
            run1=values[0], run2=values[1], run3=values[2])
    return output


def render(results, primary, output, native_directory=None, default_bins=None):
    require_complete(results, 249)
    audit = json.loads((results/'audit.json').read_text())
    fspc = audit.get('method') == 'FSPC'
    if not fspc: require_complete(primary, 249)
    display_method = 'FSPC' if fspc else 'w/o Mean Control'
    if default_bins is None:
        default_bins = 6 if json.loads((results/'audit.json').read_text()).get('default_states') == 36 else 4
    stats = {(r['backbone'], r['method']):r for r in read(results/'backbone_statistics.csv')}
    methods_a = [method_name(b, 1, 50000) for b in [1, 2, 3, 4, 6, 8, 12, 16]]
    methods_b = [method_name(default_bins, 1, n) for n in [2000, 10000]]
    methods_b += [method_name(default_bins, g, 50000) for g in [2, 4, 8]]
    native_directory = native_directory or results.parent.parent/'state_controls/summary'
    if fspc:
        ratios = {r['method']:dict(CRPS_mean=float(r['CRPS_mean']),CRPS_std=float(r['CRPS_std']))
                  for r in read(results/'average_ratio_statistics.csv')}
    else:
        ratios = average_ratios(results, primary, native_directory, methods_a+methods_b)

    def score_record(b, bins, g=1, budget=50000):
        return stats[b, method_name(bins, g, budget)]

    caption = (rf'FSPC CRPS ($\downarrow$) on shared retained test cohorts (249 tasks; at most 50k coordinates per task). Each setting fits validation templates, training mean biases and a validation coefficient. Panel (a): 50k validation-fitting cap; panel (b): smaller validation budgets and horizon groups with ${default_bins}\times{default_bins}$ summary groups. The training reservoir is fixed at up to 50k coordinates; source filtering is fixed. Subscripts: sample SD across three run averages; bold: lowest unrounded row mean. Avg. ratio equally averages within-run backbone ratios to Native on the same test cohort. DeepSTUQ uses native traffic-flow units; others use model-normalized units.' if fspc else
               rf'Validation-fitted CRPS ($\downarrow$) on shared retained test cohorts (249 tasks; at most 50k coordinates per task). Panel (a): 50k budget cap; panel (b): budgets and horizon groups based on the default ${default_bins}\times{default_bins}$ partition. Source filtering is fixed throughout. Subscripts: sample SD across three run averages; bold: lowest unrounded row mean. Avg. ratio equally averages Native-normalized backbone scores within runs (Native = 1). DeepSTUQ uses original traffic-flow units; others use model-normalized units.')
    if fspc and audit.get('mean_version')=='fspc-unified-affine-mean-v1-20261110':
        caption=caption.replace('Each setting fits validation templates, training mean biases and a validation coefficient.', 'Each setting fits validation residual templates and refits the affine-plus-state mean using a training-bias reference and the common blend-selection protocol.')
    lines = [r'\begin{table*}[t]',
        rf'\caption{{{caption}}}\label{{tab:granularity}}\label{{tab:reservoir-controls}}',
        r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
        r'\multicolumn{9}{l}{(a) Mean--scale states $b^2$, with one horizon group} \\',
        ' & '.join(['Backbone']+[rf'\shortstack{{{b*b}\\({display_method})}}' if b==default_bins else str(b*b)
                                for b in [1,2,3,4,6,8,12,16]])+r' \\', r'\midrule']
    for b in BACKBONES:
        lines.append(' & '.join([LABELS[b]]+comparison_cells([score_record(b, bins) for bins in [1, 2, 3, 4, 6, 8, 12, 16]],'CRPS'))+r' \\')
    lines += [r'\midrule', ' & '.join([r'Avg. ratio $\downarrow$']+
        comparison_cells([ratios[m] for m in methods_a], 'CRPS'))+r' \\']
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\medskip',
        r'\begin{tabular}{lrrrrr}', r'\toprule',
        rf'\multicolumn{{6}}{{l}}{{(b) {display_method} at smaller budgets; ${default_bins}\times{default_bins}\times r$ states at the full budget}} \\',
        rf'Backbone & {display_method}: 2k & {display_method}: 10k & $r=2$ ({default_bins**2*2}) & $r=4$ ({default_bins**2*4}) & $r=8$ ({default_bins**2*8}) \\',
        r'\midrule']
    for b in BACKBONES:
        records = [score_record(b, default_bins, budget=n) for n in [2000, 10000]]
        records += [score_record(b, default_bins, g=g) for g in [2, 4, 8]]
        values = comparison_cells(records,'CRPS')
        lines.append(' & '.join([LABELS[b]]+values)+r' \\')
    lines += [r'\midrule', ' & '.join([r'Avg. ratio $\downarrow$']+
        comparison_cells([ratios[m] for m in methods_b], 'CRPS'))+r' \\']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    output.write_text('\n'.join(lines))
    ratio_output=results/('display_average_ratio_statistics.csv' if fspc else 'average_ratio_statistics.csv')
    with ratio_output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'CRPS_mean', 'CRPS_std', 'run1', 'run2', 'run3'])
        writer.writeheader()
        writer.writerows(dict(method=m, **r) for m, r in ratios.items())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', type=Path, help='Audited state-control summary with paired retained-cohort Native scores')
    args = parser.parse_args(); render(args.results, args.primary, args.output, args.native)
