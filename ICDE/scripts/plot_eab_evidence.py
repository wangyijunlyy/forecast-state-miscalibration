"""Rebuild main EAB diagnostic figures from integer counts and measured arrays."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

PAPER = Path(__file__).resolve().parents[1]
WORKSPACE = PAPER.parent
sys.path.insert(0, str(WORKSPACE/'experiments/icde_reliability'))
from aggregation import audit_metric, backbone_statistics, write_csv
from render_tables import BACKBONES, LABELS, cell, table

RESULTS = Path(os.environ.get('ICDE_ANALYSIS_RESULTS',str(WORKSPACE/'experiments/icde_reliability/results')))
SOURCE = PAPER/'figures/source_data'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10,
    'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8, 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
BLUE = '#0072B2'; ORANGE = '#D55E00'; GRAY = '#666666'


def export(fig, name):
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(PAPER/'figures'/f'{name}.{extension}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def diagnosis():
    rows = []; groups = []; cases = {}
    examples = ['K2VAE_ettm1_h192_seed1', 'PPM_ETTh1_h192_seed1']
    for path in sorted((RESULTS/'runs').glob('*/validation.json')):
        value = json.loads(path.read_text())
        if not value['full_validation_split'] or not value['validation_complete']:
            raise ValueError('Incomplete validation diagnostic')
        task = json.loads((path.parent/'results.json').read_text())['task']
        metric = value['diagnostics']['Native']; audit_metric(metric)
        state = metric['state_coverage']; counts = np.asarray(state['counts']); hits = np.asarray(state['covered'])[:,1]
        eligible = counts >= 128; coverage = hits/np.maximum(counts,1)
        n = int(counts[eligible].sum()); pooled = hits[eligible].sum()/n if n else None
        row = {k:task[k] for k in ['id','backbone','dataset','horizon','seed']}
        row.update(method='Native', pooled_median_error_pp=None if not n else 100*abs(pooled-.5),
            state_median_error_pp=None if not n else 100*np.sum(abs(hits[eligible]-.5*counts[eligible]))/n,
            state_median_range_pp=None if not n else 100*np.ptp(coverage[eligible]))
        rows.append(row)
        for index in range(16):
            groups.append(dict(id=task['id'], mean_group=index//4+1, scale_group=index%4+1,
                coordinates=int(counts[index]), hits=int(hits[index]), eligible=bool(eligible[index])))
        if task['id'] in examples:
            cases[task['id']] = dict(counts=counts.tolist(), hits=hits.tolist(),
                pooled_percent=float(100*pooled), minimum_percent=float(100*coverage[eligible].min()),
                maximum_percent=float(100*coverage[eligible].max()), coordinates=n)
    keys = ['pooled_median_error_pp','state_median_error_pp','state_median_range_pp']
    stats = backbone_statistics(rows, keys)
    write_csv(SOURCE/'validation_native_task_diagnostics.csv', rows)
    write_csv(SOURCE/'validation_native_state_counts.csv', groups)
    write_csv(SOURCE/'validation_native_backbone_diagnostics.csv', stats)
    (SOURCE/'state_cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    by_model = {r['backbone']:r for r in stats}; table_rows = []
    for backbone in BACKBONES:
        r = by_model[backbone]; n = r['pooled_median_error_pp_n_configurations']
        table_rows.append(' & '.join([LABELS[backbone],str(n)]+[cell(r[k+'_mean'], r[k+'_std']) for k in keys])+r' \\')
    (PAPER/'tables/problem_diagnostics.tex').write_text(table(
        'Native median-coverage diagnostics on complete validation iterators. Errors and ranges are percentage points. Each configuration has eligible states in all three seeds; both errors use identical states with at least 128 coordinates. Subscripts give sample SD over three seed averages of configurations.',
        'tab:diagnosis', ['Backbone','Configs',r'\shortstack{Pooled\\error $\downarrow$}',r'\shortstack{State\\error $\downarrow$}',r'\shortstack{State\\range $\downarrow$}'],table_rows,wide=False).replace(r'\tabcolsep}{4pt}',r'\tabcolsep}{2pt}'))
    fig, axes = plt.subplots(1, 2, figsize=(7.15,2.75), layout='constrained')
    cmap = LinearSegmentedColormap.from_list('coverage', [BLUE,'#FFFFFF',ORANGE])
    for i,(identity,ax) in enumerate(zip(examples,axes)):
        case = cases[identity]; counts = np.array(case['counts']).reshape(4,4)
        coverage = 100*np.array(case['hits']).reshape(4,4)/np.maximum(counts,1)
        image = ax.imshow(np.ma.masked_where(counts<128,coverage),origin='lower',cmap=cmap,
            norm=TwoSlopeNorm(vmin=20,vcenter=50,vmax=80))
        for m,s in np.ndindex(4,4):
            count_label = (f'{counts[m,s]/1000:.0f}k' if counts[m,s]>=100000 else
                           f'{counts[m,s]/1000:.1f}k' if counts[m,s]>=1000 else str(counts[m,s]))
            ax.text(s,m,f'{coverage[m,s]:.1f}\nn={count_label}' if counts[m,s]>=128 else 'n<128',
                ha='center',va='center',fontsize=7.1)
        label = r'$K^2$VAE, ETTm1' if i == 0 else 'PPM, ETTh1'
        ax.set_title(f'({chr(97+i)}) {label}\nPooled median coverage: {case["pooled_percent"]:.3f}%',loc='left')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),
            xlabel='Predicted scale group (low to high)', ylabel='Predicted mean group (low to high)')
    bar = fig.colorbar(image,ax=axes,shrink=.8,ticks=[20,35,50,65,80],pad=.025)
    bar.set_label('Observed median coverage (%)')
    export(fig,'conditional_miscalibration')
    print(json.dumps(dict(cases=cases,summary=stats),indent=2))


def amplification():
    record = json.loads((SOURCE/'source_case.json').read_text())
    with np.load(SOURCE/'source_case.npz') as data:
        scale,z = data['scale'],data['residual']; near = data['raw_scale']<=1.01e-6
        fig,axes = plt.subplots(1,2,figsize=(7.15,2.55),layout='constrained')
        ax = axes[0]
        ax.scatter(scale[~near],np.maximum(abs(z[~near]),1e-8),s=2,c=GRAY,alpha=.22,rasterized=True,label='Ordinary sources')
        ax.scatter(scale[near],np.maximum(abs(z[near]),1e-8),s=7,c=ORANGE,alpha=.7,marker='x',label='Near-floor sources')
        ax.set(xscale='log',yscale='log',xlabel='Fitting predictive scale',ylabel='Absolute standardized residual')
        ax.axvline(1.01e-6,color=ORANGE,ls='--',lw=.8)
        ax.set_title('(a) Validation residual sources',loc='left'); ax.legend(fontsize=7,loc='lower left')
        ax = axes[1]
        for key,label,color,style in [('native','Native',GRAY,':'),('unfiltered','Unfiltered DC16',ORANGE,'--'),('filtered','Source-filtered DC16',BLUE,'-')]:
            ax.plot(100*data['probabilities'],data[key],label=label,color=color,ls=style,lw=1.3)
        ax.axhline(record['target'],color='black',ls='-.',lw=.8,label='Observed target')
        ax.set_yscale('symlog',linthresh=1)
        ax.set(xlabel='Midpoint probability (%)',ylabel='Predictive value (normalized units)',
            ylim=(-3e6,30),yticks=[-1e6,-1e4,-1e2,0,10],
            yticklabels=[r'$-10^6$',r'$-10^4$',r'$-10^2$','0','10'])
        ax.set_title('(b) Same test forecast coordinate',loc='left'); ax.legend(fontsize=7,loc='lower right')
        export(fig,'residual_outlier_mechanism')


if __name__ == '__main__':
    diagnosis(); amplification()
