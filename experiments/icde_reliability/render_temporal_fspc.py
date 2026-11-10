"""Compare training- and validation-fitted FSPC, including paired Native ratios."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import backbone_statistics, write_csv
from aggregate_fsdc36 import metric_row
from aggregate_train_fspc import template_record
from mean_anchor_campaign import validate_tasks
from portable_dc import CompiledCalibrator
from render_tables import BACKBONES, LABELS, table, comparison_cells
from train_fspc import validate_record

METHODS = ['TrainFSPC','DC36ControlledMean']
NAMES = dict(TrainFSPC='Train FSPC',DC36ControlledMean='Val FSPC')
METRICS = [('MSE','MSE'),('CRPS','CRPS'),('QICE_percent','QICE (pp)'),
           ('state_error_pp','State coverage error (pp)'),('BrierMean','Mean Brier score')]


def native_normalized_statistics(rows):
    """Match configurations before forming backbone/run ratios and equal weights."""
    scores = {(r['id'],r['method']):r for r in rows}
    configurations = defaultdict(list)
    for row in rows:
        if row['method']=='Native':
            configurations[row['backbone'],row['dataset'],row['horizon']].append(row)
    records = {method:dict(method=method,n_backbones=7,n_training_seeds=3) for method in METHODS}
    paired = {}
    for metric,_ in METRICS:
        eligible_count = 0
        for backbone in BACKBONES:
            eligible = [items for key,items in configurations.items() if key[0]==backbone and
                all(scores[item['id'],method][metric] is not None
                    for item in items for method in ['Native']+METHODS)]
            assert eligible and all({item['seed'] for item in items}=={1,2,3} for items in eligible)
            eligible_count += len(eligible)
            for seed in (1,2,3):
                identities = [next(item['id'] for item in items if item['seed']==seed) for items in eligible]
                native = np.mean([scores[identity,'Native'][metric] for identity in identities])
                assert native>0
                for method in METHODS:
                    value = float(np.mean([scores[identity,method][metric] for identity in identities])/native)
                    key = backbone,method,seed
                    paired.setdefault(key,dict(backbone=backbone,method=method,seed=seed))[metric] = value
        for method in METHODS:
            runs = np.mean([[paired[b,method,seed][metric] for b in BACKBONES] for seed in (1,2,3)],axis=1)
            records[method].update({metric+'_mean':float(np.mean(runs)),
                metric+'_std':float(np.std(runs,ddof=1)),metric+'_n_configurations':eligible_count})
    return list(records.values()),list(paired.values())


def render(source, mean_root, train_root, paper):
    tasks = json.loads((train_root/'manifest.json').read_text())
    validate_tasks(tasks,json.loads((source/'manifest.json').read_text()))
    validate_tasks(tasks,json.loads((mean_root/'manifest.json').read_text()))
    audit = json.loads((train_root/'analysis/audit.json').read_text())
    assert audit['all_passed'] and audit['audited_tasks']==249
    proof = json.loads((train_root/'analysis/source_selection_audit.json').read_text())
    assert proof['all_passed'] and proof['tasks']==249 and proof['manifest']==tasks
    rows = []
    for task in tasks:
        folder = train_root/'runs'/task['id']
        value = json.loads((folder/'results.json').read_text()); validate_record(value,task)
        original = json.loads((source/'runs'/task['id']/'results.json').read_text())
        fspc = json.loads((mean_root/'runs'/task['id']/'results.json').read_text())
        assert original['task']==fspc['task']==task
        assert value['fit']==proof['fits'][task['id']]
        assert template_record(CompiledCalibrator.load(folder/'compiled.npz'))==proof['templates'][task['id']]
        for name in ['TrainDC36','DC36']:
            assert value['methods'][name]==original['methods'][name]
        assert value['methods']['Native']==original['methods']['Native']
        assert value['methods']['DC36ControlledMean']==fspc['methods']['DC36ControlledMean']
        for method in ['Native']+METHODS:
            rows.append(metric_row(task,method,value['methods'][method]))
    statistics = backbone_statistics(rows,[m for m,_ in METRICS])
    stats = {(r['backbone'],r['method']):r for r in statistics}
    ratios,paired = native_normalized_statistics(rows)
    ratio_stats = {r['method']:r for r in ratios}
    content = []
    for metric,label in METRICS:
        content.append(r'\multicolumn{9}{l}{'+label+r' $\downarrow$} \\')
        for index,method in enumerate(METHODS):
            values = [NAMES[method]]+[comparison_cells([stats[b,m] for m in METHODS],metric)[index]
                                        for b in BACKBONES]
            values.append(comparison_cells([ratio_stats[m] for m in METHODS],metric)[index])
            content.append((r'\rowcolor{black!8} ' if method=='DC36ControlledMean' else '')+
                           ' & '.join(values)+r' \\')
        content.append(r'\midrule')
    caption = ('Training- versus validation-fitted FSPC on identical complete test forecasts '
        '(83 configurations, three runs). Both select mean strength on validation. '
        'Avg. ratio divides each backbone/run configuration mean by its paired Native mean, '
        'then equally averages seven backbones. Native has ratio 1; lower is better. '
        'Subscripts: sample SD across three run averages; bold: lower unrounded mean within each column. '
        'State coverage uses 76 configurations eligible in all three runs; other metrics use all 83. '
        'Brier averages three thresholds. DeepSTUQ uses original traffic-flow units (MSE squared); '
        'the other six use model-normalized units. Ratios are dimensionless.')
    (paper/'tables/temporal_fitting.tex').write_text(table(caption,'tab:temporal-fitting',
        ['Fitting split']+[LABELS[b] for b in BACKBONES]+['Avg. ratio'],content[:-1],numeric_from=1)
        .replace(r'\centering\small',r'\centering\footnotesize'))
    data = paper/'table_data'; data.mkdir(exist_ok=True)
    write_csv(data/'temporal_fitting_raw_task_metrics.csv',rows)
    write_csv(data/'temporal_fitting_backbone_statistics.csv',statistics)
    write_csv(data/'temporal_fitting_native_ratios.csv',ratios)
    write_csv(data/'temporal_fitting_backbone_run_ratios.csv',paired)
    record = dict(all_passed=True,tasks=249,configurations=83,methods=METHODS,
        displayed_mean_sd_pairs=80,metrics=[m for m,_ in METRICS],
        native_normalized_equal_backbone_summary=True,native_reference_ratio=1,
        eligible_configurations={m:ratios[0][m+'_n_configurations'] for m,_ in METRICS},
        full_test_split=True,source_proof_bound_to_current_records=True,
        existing_fspc_scores_reused=True,experiments_rerun=False)
    (paper/'submission/fspc_temporal_render.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ['source','mean-root','train-root','paper']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); render(args.source,args.mean_root,args.train_root,args.paper)
