"""Render audited complete suites into paper tables, with measured sample SD to three decimals."""
import argparse
import csv
import json
from pathlib import Path

BACKBONES = ['K2VAE', 'PPM', 'TMDM', 'DiffusionTS', 'TimeDiff', 'NsDiff']
LABELS = {'K2VAE': r'$K^2$VAE', 'PPM':'PPM', 'TMDM':'TMDM', 'DiffusionTS':'Diffusion-TS',
          'TimeDiff':'TimeDiff', 'NsDiff':'NsDiff'}
VENUES = {'K2VAE':"ICML '25", 'PPM':"ICML '26", 'TMDM':"ICLR '24", 'DiffusionTS':"ICLR '24",
          'TimeDiff':"ICML '23", 'NsDiff':"ICML '25"}
METHODS = ['Native', 'TrainDC16', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']
HEADINGS = ['Native', 'Train DC16', 'Val DC16', 'Global DC', 'Global PIT', 'Cond. PIT']


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require_complete(directory, expected):
    audit = json.loads((directory/'audit.json').read_text())
    if not audit.get('all_tasks_complete') or audit['expected_tasks'] != expected or audit['audited_tasks'] != expected:
        raise ValueError(f'Refusing final table generation: {directory} has not passed the complete {expected}-task audit')


def cell(value, std):
    if value is None or value == '':
        return r'--'
    if std is None or std == '':
        raise ValueError('A reported mean requires its measured sample SD')
    return rf'${float(value):.3f}\pm{float(std):.3f}$'


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
            distribution.append(' & '.join([model, metric]+[cell(stats[b,m][key+'_mean'], stats[b,m][key+'_std']) for m in METHODS])+r' \\')
        distribution.append(r'\midrule')
        points.append(' & '.join([LABELS[b]]+[cell(stats[b,m]['MSE_mean'], stats[b,m]['MSE_std']) for m in ['Native', 'TrainDC16', 'DC16']])+r' \\')
        n = stats[b,'Native']['state_error_pp_n_configurations']; total = stats[b,'Native']['n_configurations']
        for i,(key,metric) in enumerate([('state_error_pp','State (pp)'), ('BrierMean','Brier')]):
            model = rf'{LABELS[b]} ({n}/{total})' if i == 0 else ''
            reliability.append(' & '.join([model, metric]+[cell(stats[b,m][key+'_mean'], stats[b,m][key+'_std']) for m in METHODS])+r' \\')
        reliability.append(r'\midrule')
        budget_methods = ['Budget2000','Budget10000','Budget50000','DC64','UnfilteredDC16']
        for i,(key,metric) in enumerate([('CRPS','CRPS'), ('QICE_percent','QICE (pp)')]):
            budgets.append(' & '.join([LABELS[b] if i == 0 else '', metric]+
                [cell(controls[b,m][key+'_mean'], controls[b,m][key+'_std']) for m in budget_methods])+r' \\')
        budgets.append(r'\midrule')
        resource = resources[b]
        keys = ['dc16_fit_seconds', 'dc16_calibrator_bytes', 'dc16_transform_us_per_coordinate',
            'native_generation_us_per_coordinate', 'dc16_transform_to_generation_ratio']
        factors = [1, 1/1024, 1, 1, 100]
        costs.append(' & '.join([LABELS[b]]+[cell(float(resource[k+'_mean'])*f,
            float(resource[k+'_std'])*f) for k,f in zip(keys,factors)])+r' \\')
    common = 'Entries report mean $\\pm$ sample standard deviation across three training seeds after equally averaging configurations within each seed.'
    return {
        'full_distribution.tex':table('Complete-test distribution scores. Validation-fitted methods share fitting coordinates and native test forecasts; Train DC16 uses the earlier training segment. '+common,
            'tab:full-distribution', ['Backbone','Metric']+HEADINGS, distribution[:-1]),
        'full_reliability.tex':table('Complete-test state and event reliability. Parentheses give configurations with eligible states in all three seeds over all configurations. State errors use that common subset; Brier scores retain all tasks and equally average threshold levels 0.8,0.9,0.95. '+common,
            'tab:full-reliability', ['Backbone (n)','Metric']+HEADINGS, reliability[:-1]),
        'reservoir_controls.tex':table('Calibration-budget and mechanism controls on shared uniformly retained test coordinates. Budgets are coordinate caps; DC64 and unfiltered DC16 use the full available validation reservoir. These scores are separate from complete-test scores. '+common,
            'tab:reservoir-controls', ['Backbone','Metric','DC16: 2k','DC16: 10k','DC16: 50k','DC64','Unfiltered DC16'], budgets[:-1]),
        'execution_cost.tex':table('Validation-fitted DC16 execution cost. Transform and generation are microseconds per scored coordinate; the ratio is CPU transform time divided by synchronized native GPU generation time. State size is KiB. '+common,
            'tab:execution-cost', ['Backbone','Fit (s)','State (KiB)',r'Transform ($\mu$s)',r'Native ($\mu$s)',r'Ratio (\%)'], costs, numeric_from=1),
        'point_mse.tex':table('Mean-forecast MSE on the complete native test iterators. '+common,
            'tab:point', ['Backbone','Native','Train DC16','Val DC16'], points, wide=False, numeric_from=1),
    }


def baseline(directory):
    stats = {(r['dataset'],r['method']):r for r in read(directory/'complete_dataset_statistics.csv')}
    methods = ['DeepSTUQ','DeepSTUQ+DC16','DeepSTUQ+GlobalDC','DeepSTUQ+PIT','NoTemperature']
    rows = []
    for dataset in ['PEMS03','PEMS04','PEMS07','PEMS08']:
        for i,(key,metric) in enumerate([('CRPS','CRPS'), ('QICE_percent','QICE (pp)')]):
            rows.append(' & '.join([dataset if i == 0 else '', metric]+
                [cell(stats[dataset,m][key+'_mean'], stats[dataset,m][key+'_std']) for m in methods])+r' \\')
        rows.append(r'\midrule')
    return table('DeepSTUQ on its original traffic tasks. Scores use flow units and 100 midpoint quantiles of its predictive Gaussian marginal. All methods share the full trained backbone and its test forecasts. Entries report mean $\\pm$ sample standard deviation across three training seeds. This suite is separate from the long-horizon benchmark.',
        'tab:deepstuq', ['Dataset','Metric',r"\shortstack{DeepSTUQ\\(ICDE '23)}",'+DC16','+Global DC','+PIT','No temperature'],rows[:-1])


def complete_method_matrices(primary_directory, gaussian_directory):
    """All eight methods share complete-test populations; compare them directly."""
    require_complete(gaussian_directory,237)
    stats={(r['backbone'],r['method']):r for r in read(primary_directory/'partial_backbone_statistics.csv')}
    stats.update({(r['backbone'],r['method']):r for r in read(gaussian_directory/'backbone_statistics.csv')})
    methods=METHODS+['GaussianVariance2P','GaussianAffine4P']
    labels=HEADINGS+['Gaussian2P','Gaussian4P']
    for b in BACKBONES:
        for key in ['n_configurations','state_error_pp_n_configurations']:
            if len({stats[b,m][key] for m in methods})!=1:
                raise ValueError('Gaussian and primary summary populations differ')
    distribution=[];reliability=[]
    for metric,label in [('CRPS','CRPS'),('QICE_percent','QICE (pp)')]:
        distribution.append(r'\multicolumn{7}{l}{'+label+r'} \\')
        for method,name in zip(methods,labels):
            distribution.append(' & '.join([name]+[cell(stats[b,method][metric+'_mean'], stats[b,method][metric+'_std']) for b in BACKBONES])+r' \\')
        distribution.append(r'\midrule')
    for metric,label in [('state_error_pp','State coverage error (pp)'),('BrierMean','Mean Brier score')]:
        reliability.append(r'\multicolumn{7}{l}{'+label+r'} \\')
        for method,name in zip(methods,labels):
            reliability.append(' & '.join([name]+[cell(stats[b,method][metric+'_mean'], stats[b,method][metric+'_std']) for b in BACKBONES])+r' \\')
        reliability.append(r'\midrule')
    common='Entries average configurations within each training seed, then report mean $\\pm$ sample SD over three seeds. Except Train DC16, all corrections receive the same validation reservoir.'
    header=['Method']+[rf'\shortstack{{{LABELS[b]}\\({VENUES[b]})}}' for b in BACKBONES]
    eligible=['Method']+[rf'\shortstack{{{LABELS[b]}\\({stats[b,"Native"]["state_error_pp_n_configurations"]}/{stats[b,"Native"]["n_configurations"]})}}' for b in BACKBONES]
    return {
        'full_distribution.tex':table('Distribution scores on paired complete native test iterators. Each backbone retains its own fixed dataset--horizon configuration set. '+common,
            'tab:full-distribution',header,distribution[:-1],numeric_from=1),
        'full_reliability.tex':table('State and event reliability on complete native test iterators. Parentheses give configurations with eligible states in all seeds over all configurations. State errors use that common subset; Brier scores retain all configurations and equally average threshold levels 0.8, 0.9, 0.95. '+common,
            'tab:full-reliability',eligible,reliability[:-1],numeric_from=1)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--deepstuq', type=Path)
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--gaussian',type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_complete(args.primary, 237)
    if not args.primary_only:
        if args.deepstuq is None:
            parser.error('Final table generation requires the DeepSTUQ summary or explicit --primary-only')
        require_complete(args.deepstuq, 12)
    outputs = primary(args.primary)
    if args.gaussian is not None:
        outputs.update(complete_method_matrices(args.primary,args.gaussian))
    if not args.primary_only:
        outputs['deepstuq.tex'] = baseline(args.deepstuq)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        (args.output/filename).write_text(content)
    print(f'Rendered {len(outputs)} audited table files.')
