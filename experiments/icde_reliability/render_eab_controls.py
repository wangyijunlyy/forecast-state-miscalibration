"""Render independently audited EAB intervention and implementation controls."""
import argparse
from pathlib import Path
from render_tables import BACKBONES, LABELS, read, require_complete, cell, table, comparison_cells


def render(root, output, gaussian=False):
    state = root/'state_controls/summary'; compiled = root/'compiled/summary'
    require_complete(state,249); require_complete(compiled,249)
    rows = {(r['backbone'],r['method']):r for r in read(state/'backbone_statistics.csv')}
    values = []
    for b in BACKBONES:
        for i,(metric,label) in enumerate([('CRPS','CRPS'),('QICE_percent','QICE (pp)')]):
            values.append(' & '.join([LABELS[b] if i == 0 else '', label+r' $\downarrow$']+
                comparison_cells([rows[b,m] for m in ['Native','GlobalDC','ShuffledMean','DC16']],metric))+r' \\')
        values.append(r'\midrule')
    (output/'state_assignment.tex').write_text(table(
        'Validation-fit state-assignment controls on identical uniformly retained test coordinates. The shuffled column averages five permutations per task, preserving source validity, boundaries, counts and fallback eligibility. Subscripts give sample SD over three seed averages of configurations. DeepSTUQ CRPS uses flow units; other backbones use model-normalized units.',
        'tab:state-assignment',['Backbone','Metric','Native','Global template','Shuffled states','Genuine states'],values[:-1]))
    rows = {r['backbone']:r for r in read(compiled/'backbone_statistics.csv')}; values = []
    keys = ['compile_ms','compiled_kib','state_size_ratio','decode_speedup','direct_rows_query_speedup','cached_summary_query_speedup']
    for b in BACKBONES:
        values.append(' & '.join([LABELS[b]]+[cell(rows[b][k+'_mean'], rows[b][k+'_std']) for k in keys])+r' \\')
    (output/'compiled_execution.tex').write_text(table(
        'Exact DC16 compilation on the 249 retained test cohorts. Decode and query columns are speedup factors over reference reconstruction (and three exceedance queries for query columns). Query summaries reuse native mean/scale; rows include their computation. Each task uses the median of three randomized paired repetitions; size ratios and speedups are formed per task before three-seed aggregation. Subscripts give sample SD.',
        'tab:compiled', ['Backbone',r'\shortstack{Compile $\downarrow$\\(ms)}',r'\shortstack{State $\downarrow$\\(KiB)}',r'\shortstack{Size ratio $\uparrow$\\($\times$)}',r'\shortstack{Decode $\uparrow$\\($\times$)}',r'\shortstack{Queries: rows $\uparrow$\\($\times$)}',r'\shortstack{Queries: summaries $\uparrow$\\($\times$)}'],values,numeric_from=1))
    if gaussian:
        directory=root/'gaussian/summary';require_complete(directory,249)
        parametric={(r['backbone'],r['method']):r for r in read(directory/'backbone_statistics.csv')}
        empirical={(r['backbone'],r['method']):r for r in read(root/'summary/partial_backbone_statistics.csv')}
        values=[]
        for b in BACKBONES:
            for i,(metric,label) in enumerate([('CRPS','CRPS'),('QICE_percent','QICE (pp)')]):
                cells=[empirical[b,'Native'],parametric[b,'GaussianVariance2P'],
                    parametric[b,'GaussianAffine4P'],empirical[b,'DC16']]
                values.append(' & '.join([LABELS[b] if i==0 else '',label+r' $\downarrow$']+
                    comparison_cells(cells,metric))+r' \\')
            values.append(r'\midrule')
        (output/'gaussian_controls.tex').write_text(table(
            'Parametric and empirical calibration on complete native test iterators. All corrections use identical validation coordinates and paired native forecasts. Gaussian2P fits variance; Gaussian4P additionally fits affine location. Subscripts give sample SD over three seed averages of configurations.',
            'tab:gaussian',['Backbone','Metric','Native','Gaussian2P','Gaussian4P','Val DC16'],values[:-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gaussian',action='store_true');args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); render(args.results,args.output,args.gaussian)
