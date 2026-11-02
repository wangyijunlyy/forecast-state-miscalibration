"""Render audited granularity/budget/source controls in one paper table."""
import argparse
from pathlib import Path
from render_tables import BACKBONES, LABELS, read, require_complete, comparison_cells
from granularity import method_name


def render(results, primary, output):
    require_complete(results, 249)
    require_complete(primary, 249)
    stats = {(r['backbone'], r['method']):r for r in read(results/'backbone_statistics.csv')}
    old = {(r['backbone'], r['method']):r for r in read(primary/'partial_reservoir_control_statistics.csv')}

    def score_record(b, bins, g=1, budget=50000):
        return stats[b, method_name(bins, g, budget)]

    lines = [r'\begin{table*}[t]',
        r'\caption{State granularity, horizon information, and calibration budget: CRPS ($\downarrow$) on shared retained test coordinates for all 249 tasks, with validation fitting. Panel (a) uses up to 50,000 fitting coordinates. Panel (b) varies the DC16 budget or adds horizon groups; unfiltered DC16 uses the full budget. Subscripts give sample SD over three seed averages of configurations. Bold marks the best unrounded mean per backbone in each panel. DeepSTUQ uses flow units; others use model-normalized units.}\label{tab:granularity}\label{tab:reservoir-controls}',
        r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
        r'\multicolumn{9}{l}{(a) Mean--scale states $b^2$, with one horizon group} \\',
        r'Backbone & 1 & 4 & 9 & 16 & 36 & 64 & 144 & 256 \\', r'\midrule']
    for b in BACKBONES:
        lines.append(' & '.join([LABELS[b]]+comparison_cells([score_record(b, bins) for bins in [1, 2, 3, 4, 6, 8, 12, 16]],'CRPS'))+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\medskip',
        r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'\multicolumn{7}{l}{(b) DC16 at smaller budgets; $4\times4\times g$ states at the full budget; source ablation} \\',
        r'Backbone & DC16: 2k & DC16: 10k & $g=2$ (32) & $g=4$ (64) & $g=8$ (128) & Unfiltered DC16 \\',
        r'\midrule']
    for b in BACKBONES:
        unfiltered = old[b, 'UnfilteredDC16']
        records = [score_record(b, 4, budget=n) for n in [2000, 10000]]
        records += [score_record(b, 4, g=g) for g in [2, 4, 8]]
        values = comparison_cells(records+[unfiltered],'CRPS')
        lines.append(' & '.join([LABELS[b]]+values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    output.write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); render(args.results, args.primary, args.output)
