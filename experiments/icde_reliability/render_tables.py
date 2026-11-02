"""Render audited complete suites into paper tables, with measured sample SD to three decimals."""
import argparse
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

BACKBONES = ['K2VAE', 'PPM', 'TMDM', 'DiffusionTS', 'TimeDiff', 'NsDiff', 'DeepSTUQ']
LABELS = {'K2VAE': r'$K^2$VAE', 'PPM':'PPM', 'TMDM':'TMDM', 'DiffusionTS':'Diffusion-TS',
          'TimeDiff':'TimeDiff', 'NsDiff':'NsDiff', 'DeepSTUQ':'DeepSTUQ'}
VENUES = {'K2VAE':"ICML '25", 'PPM':"ICML '26", 'TMDM':"ICLR '24", 'DiffusionTS':"ICLR '24",
          'TimeDiff':"ICML '23", 'NsDiff':"ICML '25", 'DeepSTUQ':"ICDE '23"}
METHODS = ['Native', 'TrainDC16', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']
HEADINGS = ['Native', 'Train DC16', 'Val DC16', 'Global DC', 'Global PIT', 'Cond. PIT']


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require_complete(directory, expected):
    audit = json.loads((directory/'audit.json').read_text())
    if not audit.get('all_tasks_complete') or audit['expected_tasks'] != expected or audit['audited_tasks'] != expected:
        raise ValueError(f'Refusing final table generation: {directory} has not passed the complete {expected}-task audit')


def cell(value, std, best=False):
    if value is None or value == '':
        return r'--'
    if std is None or std == '':
        raise ValueError('A reported mean requires its measured sample SD')
    # Guard binary summation noise at an exact three-decimal rounding boundary.
    rounded = [Decimal(f'{float(v):.12f}').quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
               for v in (value, std)]
    command = 'beststat' if best else 'stat'
    return rf'\{command}{{{rounded[0]}}}{{{rounded[1]}}}'


def best_mean(stats, backbone, method, metric, methods):
    value = float(stats[backbone,method][metric+'_mean'])
    optimum = min(float(stats[backbone,m][metric+'_mean']) for m in methods)
    return math.isclose(value, optimum, rel_tol=0, abs_tol=1e-12)


def comparison_cells(records, metric):
    optimum = min(float(r[metric+'_mean']) for r in records)
    return [cell(r[metric+'_mean'], r[metric+'_std'],
                 math.isclose(float(r[metric+'_mean']), optimum, rel_tol=0, abs_tol=1e-12))
            for r in records]


def table(caption, label, header, rows, wide=True, numeric_from=2):
    environment = 'table*' if wide else 'table'
    alignment = 'l'*numeric_from+'r'*(len(header)-numeric_from)
    return '\n'.join([rf'\begin{{{environment}}}[t]', rf'\caption{{{caption}}}\label{{{label}}}',
        r'\centering\small\setlength{\tabcolsep}{4pt}', rf'\begin{{tabular}}{{{alignment}}}', r'\toprule',
        ' & '.join(header)+r' \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', rf'\end{{{environment}}}', ''])


def primary(directory):
    stats = {(r['backbone'],r['method']):r for r in read(directory/'partial_backbone_statistics.csv')}
    controls = {(r['backbone'],r['method']):r for r in read(directory/'partial_reservoir_control_statistics.csv')}
    resources = {r['backbone']:r for r in read(directory/'partial_resource_statistics.csv')}
    for b in BACKBONES:
        sizes = {stats[b,m]['n_configurations'] for m in METHODS}
        eligible = {stats[b,m]['state_error_pp_n_configurations'] for m in METHODS}
        if len(sizes) != 1 or len(eligible) != 1:
            raise ValueError('Unpaired backbone/configuration summary populations')
    distribution = []; reliability = []; budgets = []; costs = []; points = []
    for b in BACKBONES:
        label = rf'\shortstack[l]{{{LABELS[b]}\\({VENUES[b]})}}'
        for i,(key,metric) in enumerate([('CRPS','CRPS'), ('QICE_percent','QICE (pp)')]):
            model = label if i == 0 else ''
            distribution.append(' & '.join([model, metric+r' $\downarrow$']+comparison_cells([stats[b,m] for m in METHODS],key))+r' \\')
        distribution.append(r'\midrule')
        points.append(' & '.join([LABELS[b]]+comparison_cells([stats[b,m] for m in ['Native', 'TrainDC16', 'DC16']],'MSE'))+r' \\')
        n = stats[b,'Native']['state_error_pp_n_configurations']; total = stats[b,'Native']['n_configurations']
        for i,(key,metric) in enumerate([('state_error_pp','State (pp)'), ('BrierMean','Brier')]):
            model = rf'{LABELS[b]} ({n}/{total})' if i == 0 else ''
            reliability.append(' & '.join([model, metric+r' $\downarrow$']+comparison_cells([stats[b,m] for m in METHODS],key))+r' \\')
        reliability.append(r'\midrule')
        budget_methods = ['Budget2000','Budget10000','Budget50000','DC64','UnfilteredDC16']
        for i,(key,metric) in enumerate([('CRPS','CRPS'), ('QICE_percent','QICE (pp)')]):
            budgets.append(' & '.join([LABELS[b] if i == 0 else '', metric+r' $\downarrow$']+
                comparison_cells([controls[b,m] for m in budget_methods],key))+r' \\')
        budgets.append(r'\midrule')
        resource = resources[b]
        keys = ['dc16_fit_seconds', 'dc16_calibrator_bytes', 'dc16_transform_us_per_coordinate',
            'native_generation_us_per_coordinate', 'dc16_transform_to_generation_ratio']
        factors = [1, 1/1024, 1, 1, 100]
        costs.append(' & '.join([LABELS[b]]+[cell(float(resource[k+'_mean'])*f,
            float(resource[k+'_std'])*f) for k,f in zip(keys,factors)])+r' \\')
    common = 'Subscripts give sample SD across three seed averages of configurations. DeepSTUQ retains traffic-flow units; other backbones use model-normalized units.'
    return {
        'full_distribution.tex':table('Complete-test distribution scores. Validation-fitted methods share fitting coordinates and native test forecasts; Train DC16 uses the earlier training segment. '+common,
            'tab:full-distribution', ['Backbone','Metric']+HEADINGS, distribution[:-1]),
        'full_reliability.tex':table('Complete-test state and event reliability. Parentheses give configurations with eligible states in all three seeds over all configurations. State errors use that common subset; Brier scores retain all tasks and equally average threshold levels 0.8,0.9,0.95. '+common,
            'tab:full-reliability', ['Backbone (n)','Metric']+HEADINGS, reliability[:-1]),
        'reservoir_controls.tex':table('Calibration-budget and mechanism controls on shared uniformly retained test coordinates. Budgets are coordinate caps; DC64 and unfiltered DC16 use the full available validation reservoir. These scores are separate from complete-test scores. '+common,
            'tab:reservoir-controls', ['Backbone','Metric','DC16: 2k','DC16: 10k','DC16: 50k','DC64','Unfiltered DC16'], budgets[:-1]),
        'execution_cost.tex':table('Validation-fitted DC16 execution cost. Transform and generation are microseconds per coordinate; the ratio is CPU transform time divided by synchronized native GPU generation time. State size is KiB. Subscripts give sample SD over three seed averages of configurations.',
            'tab:execution-cost', ['Backbone',r'Fit (s) $\downarrow$',r'State (KiB) $\downarrow$',r'Transform ($\mu$s) $\downarrow$',r'Native ($\mu$s) $\downarrow$',r'Ratio (\%) $\downarrow$'], costs, numeric_from=1),
        'point_mse.tex':table('DC16 mean-forecast MSE ($\\downarrow$) across fitting segments. Subscripts give sample SD over three seed averages. DeepSTUQ uses flow units; others use model-normalized units.',
            'tab:point', ['Backbone','Native','Train DC16','Val DC16'], points, wide=False, numeric_from=1).replace(r'\tabcolsep}{4pt}',r'\tabcolsep}{2pt}'),
    }


def complete_method_matrices(primary_directory, gaussian_directory):
    """Compare correction families under the same validation-fitting protocol."""
    require_complete(gaussian_directory,249)
    stats={(r['backbone'],r['method']):r for r in read(primary_directory/'partial_backbone_statistics.csv')}
    stats.update({(r['backbone'],r['method']):r for r in read(gaussian_directory/'backbone_statistics.csv')})
    methods=[m for m in METHODS if m != 'TrainDC16']+['GaussianVariance2P','GaussianAffine4P']
    labels=[name for m,name in zip(METHODS,HEADINGS) if m != 'TrainDC16']+['Gaussian2P','Gaussian4P']
    for b in BACKBONES:
        for key in ['n_configurations','state_error_pp_n_configurations']:
            if len({stats[b,m][key] for m in methods})!=1:
                raise ValueError('Gaussian and primary summary populations differ')
    distribution=[];reliability=[]
    for metric,label in [('CRPS','CRPS'),('QICE_percent','QICE (pp)')]:
        distribution.append(r'\multicolumn{8}{l}{'+label+r' $\downarrow$} \\')
        for method,name in zip(methods,labels):
            distribution.append(' & '.join([name]+[cell(stats[b,method][metric+'_mean'], stats[b,method][metric+'_std'],best_mean(stats,b,method,metric,methods)) for b in BACKBONES])+r' \\')
        distribution.append(r'\midrule')
    for metric,label in [('state_error_pp','State coverage error (pp)'),('BrierMean','Mean Brier score')]:
        reliability.append(r'\multicolumn{8}{l}{'+label+r' $\downarrow$} \\')
        for method,name in zip(methods,labels):
            reliability.append(' & '.join([name]+[cell(stats[b,method][metric+'_mean'], stats[b,method][metric+'_std'],best_mean(stats,b,method,metric,methods)) for b in BACKBONES])+r' \\')
        reliability.append(r'\midrule')
    common='All corrections use the same validation reservoir. Subscripts give sample SD over three seed averages of configurations. Bold means are best within each backbone and metric. DeepSTUQ uses native flow units; others use model-normalized units.'
    header=['Method']+[rf'\shortstack{{{LABELS[b]}\\({VENUES[b]})}}' for b in BACKBONES]
    eligible=['Method']+[rf'\shortstack{{{LABELS[b]}\\({stats[b,"Native"]["state_error_pp_n_configurations"]}/{stats[b,"Native"]["n_configurations"]})}}' for b in BACKBONES]
    return {
        'full_distribution.tex':table('Validation-fitted calibration methods: distribution scores on paired complete native test iterators. '+common,
            'tab:full-distribution',header,distribution[:-1],numeric_from=1),
        'full_reliability.tex':table('Validation-fitted calibration methods: state and event reliability on complete native test iterators. Parentheses give eligible configurations over all configurations. State errors use states with at least 128 coordinates in every seed; Brier scores retain all configurations and average threshold levels 0.8, 0.9, 0.95. '+common,
            'tab:full-reliability',eligible,reliability[:-1],numeric_from=1)}


def temporal_fitting(primary_directory):
    """Hold DC16 fixed and vary only the temporal fitting segment."""
    stats={(r['backbone'],r['method']):r for r in read(primary_directory/'partial_backbone_statistics.csv')}
    rows=[]
    for metric,label in [('CRPS','CRPS'),('QICE_percent','QICE (pp)'),
                         ('state_error_pp','State coverage error (pp)'),('BrierMean','Mean Brier score')]:
        rows.append(r'\multicolumn{8}{l}{'+label+r' $\downarrow$} \\')
        for method,name in [('TrainDC16','Train DC16'),('DC16','Val DC16')]:
            rows.append(' & '.join([name]+[cell(stats[b,method][metric+'_mean'],stats[b,method][metric+'_std'],best_mean(stats,b,method,metric,['TrainDC16','DC16'])) for b in BACKBONES])+r' \\')
        rows.append(r'\midrule')
    return table('Temporal fitting comparison for DC16 on identical complete native test forecasts. Subscripts give sample SD over three seed averages. State diagnostics and eligible configurations follow Table~\\ref{tab:full-reliability}; other metrics use all configurations. Brier scores average the three thresholds. CRPS uses flow units for DeepSTUQ and model-normalized units otherwise.',
        'tab:temporal-fitting',['Fitting segment']+[LABELS[b] for b in BACKBONES],rows[:-1],numeric_from=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--deepstuq', type=Path)
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--gaussian',type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_complete(args.primary, 249)
    if not args.primary_only:
        if args.deepstuq is None:
            parser.error('Final table generation requires the DeepSTUQ summary or explicit --primary-only')
        require_complete(args.deepstuq, 12)
    outputs = primary(args.primary)
    if args.gaussian is not None:
        outputs.update(complete_method_matrices(args.primary,args.gaussian))
    outputs['temporal_fitting.tex'] = temporal_fitting(args.primary)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        (args.output/filename).write_text(content)
    print(f'Rendered {len(outputs)} audited table files.')
