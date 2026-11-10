"""Render FSPC and its mean-control ablations from audited complete-test records."""
import argparse
import csv
import json
import math
from pathlib import Path
from aggregation import backbone_statistics, configuration_statistics
from aggregate_fsdc36 import metric_row
from aggregate_mean_anchor import METRICS
from aggregate_mean_anchor_all import normalized_ratios
from mean_anchor_campaign import validate_tasks, validate_record, source_proof_matches
from render_tables import BACKBONES, LABELS, VENUES, cell, table, comparison_cells
from render_supplement import DATASETS, ORDER

METHODS=['Native','DC36ControlledMean','GlobalDC','GlobalPIT','ConditionalPIT','DC36']
NAMES=dict(Native='Native', DC36ControlledMean='FSPC', DC36='w/o Mean Control',
           DC36MeanPreserving=r'FSPC ($\lambda=0$)', GlobalDC='Global DC',
           GlobalPIT='Global PIT', ConditionalPIT='Cond. PIT')


def render(source, mean_root, paper, train_root=None):
    tasks=json.loads((mean_root/'manifest.json').read_text())
    validate_tasks(tasks,json.loads((source/'manifest.json').read_text()))
    audit=json.loads((mean_root/'analysis/audit.json').read_text())
    assert audit['all_passed'] and audit['audited_tasks']==249
    rows=[];cohort_rows=[];fits={}
    for task in tasks:
        old=json.loads((source/'runs'/task['id']/'results.json').read_text())
        new=json.loads((mean_root/'runs'/task['id']/'results.json').read_text());validate_record(new,task)
        assert old['task']==task
        fits[task['id']]=new['fit']
        for method in ['Native','DC36','GlobalDC','GlobalPIT','ConditionalPIT']:
            rows.append(metric_row(task,method,old['methods'][method]))
        for method in ['DC36MeanPreserving','DC36ControlledMean']:
            rows.append(metric_row(task,method,new['methods'][method]))
        cohort=json.loads((mean_root/'runs'/task['id']/'cohort.json').read_text())
        source_cohort=json.loads((source/'cohort/runs'/task['id']/'results.json').read_text())
        assert cohort['task']==source_cohort['task']==task
        assert cohort['coordinates']==source_cohort['coordinates']
        for method,reference in [('Native',source_cohort['native']),
                                 ('DC36',source_cohort['methods']['b6_h1_n50000'])]:
            for metric in ['CRPS','QICE_percent']:
                assert math.isclose(cohort['methods'][method][metric],reference[metric],rel_tol=0,abs_tol=1e-10)
        metrics={m:cohort['methods'][m] for m in ['Native','DC36','DC36ControlledMean']}
        metrics['UnfilteredDC36']=source_cohort['methods']['UnfilteredDC36']
        for method,metric in metrics.items():
            assert metric['coordinate_count']==cohort['coordinates']
            assert metric['state_coverage']['counts']==source_cohort['native']['state_coverage']['counts']
            cohort_rows.append(metric_row(task,method,metric))
    proof=json.loads((mean_root/'analysis/source_selection_audit.json').read_text())
    assert source_proof_matches(proof,tasks,fits)
    stats={(r['backbone'],r['method']):r for r in backbone_statistics(rows,METRICS)}
    configs={(r['backbone'],r['dataset'],r['horizon'],r['method']):r
             for r in configuration_statistics(rows,METRICS)}
    ratios={r['method']:r for r in normalized_ratios(rows)[0]}
    output=paper/'tables'
    main_methods=['Native','DC36ControlledMean']
    comparison_methods=['DC36ControlledMean','GlobalDC','GlobalPIT','ConditionalPIT']
    component_methods=['DC36ControlledMean','DC36','UnfilteredDC36']
    component_names={**NAMES,'UnfilteredDC36':'w/o Mean/Filter'}
    cohort_stats={(r['backbone'],r['method']):r for r in backbone_statistics(cohort_rows,METRICS)}
    display_metrics=[('MSE','MSE'),('CRPS','CRPS'),('QICE_percent','QICE (pp)'),
                     ('state_error_pp','State coverage error (pp)'),('BrierMean','Mean Brier score')]
    header=['Backbone','Method',r'MSE $\downarrow$',r'CRPS $\downarrow$',
            r'QICE (pp) $\downarrow$',r'\shortstack{State coverage\\error (pp) $\downarrow$}',
            r'\shortstack{Mean Brier\\score $\downarrow$}']
    common=('Subscripts: three-run sample SD; bold: best unrounded backbone-wise mean. '
            'State errors use common eligible configurations. DeepSTUQ uses original traffic-flow units '
            '(MSE squared); others use model-normalized units. QICE/state errors: percentage points; '
            'Brier: dimensionless.')
    def metric_table(records,methods,names,caption,label):
        content=[]
        for backbone in BACKBONES:
            for index,method in enumerate(methods):
                values=[LABELS[backbone] if index==0 else '',names[method]]
                for metric,_ in display_metrics:
                    row=records[backbone,method]
                    optimum=min(float(records[backbone,m][metric+'_mean']) for m in methods)
                    best=math.isclose(float(row[metric+'_mean']),optimum,rel_tol=0,abs_tol=1e-12)
                    factor=1e6 if abs(float(row[metric+'_mean']))>=1e6 else 1.
                    value=cell(float(row[metric+'_mean'])/factor,float(row[metric+'_std'])/factor,best)
                    if factor>1:value+=r'\,\ensuremath{\times10^{6}}'
                    values.append(value)
                content.append((r'\rowcolor{black!8} ' if method=='DC36ControlledMean' else '')+
                               ' & '.join(values)+r' \\')
            content.append(r'\midrule')
        rendered=table(caption+' '+common,label,header,content[:-1],numeric_from=2).replace(
            r'\tabcolsep}{4pt}',r'\tabcolsep}{3pt}')
        return rendered.replace(r'\centering\small',r'\centering\footnotesize') if len(methods)>2 else rendered
    (output/'full_distribution.tex').write_text(metric_table(stats,main_methods,NAMES,
        'Overall effectiveness of FSPC against Native on identical complete test forecasts (249 tasks).',
        'tab:full-distribution'))
    (output/'calibration_comparison.tex').write_text(metric_table(stats,comparison_methods,NAMES,
        'Other calibration methods on paired complete test forecasts (249 tasks). '
        'FSPC uses validation templates and mean strength with training biases; '
        'Global DC, Global PIT and conditional PIT use validation fitting.',
        'tab:calibration-comparison'))
    (output/'component_ablation.tex').write_text(metric_table(cohort_stats,component_methods,component_names,
        'Component ablation on identical retained test coordinates (249 tasks, 36 states, 50k fitting cap). '
        'w/o Mean Control removes mean adjustment; w/o Mean/Filter additionally removes source filtering. '
        'Scientific multipliers apply to both mean and SD.',
        'tab:component-ablation'))
    (output/'full_reliability.tex').unlink(missing_ok=True)
    # Retain the exact metric populations and full-precision summaries behind the three tables.
    table_data=paper/'table_data';table_data.mkdir(exist_ok=True)
    from aggregation import write_csv
    for filename,records,methods in [('main_five_metrics',stats,main_methods),
            ('calibration_five_metrics',stats,comparison_methods),
            ('ablation_five_metrics',cohort_stats,component_methods)]:
        exported_stats=[]
        for backbone in BACKBONES:
            for method in methods:
                row=records[backbone,method]
                for metric,_ in display_metrics:
                    exported_stats.append(dict(backbone=backbone,method=method,
                        method_name=component_names[method],metric=metric,
                        configurations=row[metric+'_n_configurations'],mean=row[metric+'_mean'],
                        sample_sd=row[metric+'_std']))
        write_csv(table_data/(filename+'.csv'),exported_stats)
    write_csv(table_data/'calibration_internal_raw_task_metrics.csv',
        [r for r in rows if r['method'] in ['Native','DC36ControlledMean','GlobalDC','GlobalPIT','ConditionalPIT']])
    merged_audit=None
    if (table_data/'external_calibration_raw_task_metrics.csv').exists():
        import importlib.util
        spec=importlib.util.spec_from_file_location('calibration_comparison',paper/'scripts/prepare_calibration_comparison.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        merged_audit=module.build(paper)
    # Preserve the original seven tables and add the full method as a distinct column.
    exported=[];supp_methods=['Native','DC36ControlledMean','DC36','GlobalDC','GlobalPIT','ConditionalPIT']
    supp_header=r'Dataset & $H$ & Metric & Native & FSPC & \shortstack{w/o Mean\\Control} & \shortstack{Global\\DC} & \shortstack{Global\\PIT} & \shortstack{Cond.\\PIT} \\'
    for backbone in BACKBONES:
        identities=sorted({(t['dataset'],t['horizon']) for t in tasks if t['backbone']==backbone},
                          key=lambda x:(ORDER.index(DATASETS.get(x[0],x[0])),x[1]))
        units='original traffic-flow units' if backbone=='DeepSTUQ' else 'model-normalized units'
        caption=LABELS[backbone]+': all '+str(len(identities))+' complete-test configurations. FSPC uses validation-fitted templates, training biases and a validation-selected mean strength; w/o Mean Control omits mean control. CRPS uses '+units+'; QICE is in percentage points. Subscripts: sample SD over three training runs; bold: lowest unrounded mean.'
        lines=[r'\begin{longtable}{lllrrrrrr}',r'\caption{'+caption+r'}\label{tab:supp-'+backbone.lower()+r'}\\',
            r'\toprule',supp_header,r'\midrule',r'\endfirsthead',r'\multicolumn{9}{l}{\tablename~\thetable\ (continued)} \\',
            r'\toprule',supp_header,r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{9}{r}{Continued on next page} \\',
            r'\endfoot',r'\bottomrule',r'\endlastfoot']
        previous=None
        for dataset,horizon in identities:
            if previous is not None and previous!=dataset:lines.append(r'\midrule')
            for i,(metric,label) in enumerate([('CRPS','CRPS'),('QICE_percent','QICE')]):
                records=[configs[backbone,dataset,horizon,m] for m in supp_methods]
                left=[DATASETS.get(dataset,dataset),str(horizon),label+r' $\downarrow$'] if i==0 else ['','',label+r' $\downarrow$']
                lines.append(' & '.join(left+comparison_cells(records,metric))+(r' \\*' if i==0 else r' \\'))
                for method,record in zip(supp_methods,records):
                    exported.append(dict(backbone=backbone,dataset=dataset,horizon=horizon,method=method,
                        method_name=NAMES[method],metric=metric,mean=record[metric+'_mean'],sample_sd=record[metric+'_std'],training_seeds=3))
            previous=dataset
        lines += [r'\end{longtable}','']
        (paper/'supplementary_tables'/(backbone.lower()+'.tex')).write_text('\n'.join(lines))
    with (paper/'supplementary_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    details=[]
    panels=[(['MSE','MAE','RMSE','MedianMAE','CRPS','QICE_percent'],
             ['MSE','MAE','RMSE','Median MAE','CRPS','QICE (pp)']),
            (['state_error_pp','pooled_error_pp','Brier80','Brier90','Brier95','BrierMean'],
             ['State (pp)','Pooled (pp)',r'$\mathrm{BS}_{.80}$',r'$\mathrm{BS}_{.90}$',r'$\mathrm{BS}_{.95}$','Mean BS'])]
    for backbone in BACKBONES:
        identities=sorted({(t['dataset'],t['horizon']) for t in tasks if t['backbone']==backbone},
                          key=lambda x:(ORDER.index(DATASETS.get(x[0],x[0])),x[1]))
        for index,(metrics,labels) in enumerate(panels):
            suffix='point' if index==0 else 'reliability'
            caption=LABELS[backbone]+': full-method FSPC '+('point and distribution' if index==0 else 'coverage and threshold-event')+' scores for all '+str(len(identities))+' configurations. Subscripts give sample SD over three training runs; all metrics are minimized. '
            caption+=('MSE uses squared scoring units; the other error scores use scoring units. QICE is in percentage points. DeepSTUQ retains native units; other models use normalized units.' if index==0 else 'State and pooled coverage errors use the common diagnostic states with at least 128 coordinates. -- denotes no eligible states in at least one run. Brier thresholds are validation quantiles 0.8, 0.9, and 0.95; Mean BS averages these three scores.')
            head=' & '.join(['Dataset','$H$']+labels)+r' \\'
            lines=[r'\begin{longtable}{llrrrrrr}',r'\caption{'+caption+r'}\label{tab:supp-detail-'+backbone.lower()+'-'+suffix+r'}\\',
                r'\toprule',head,r'\midrule',r'\endfirsthead',r'\multicolumn{8}{l}{\tablename~\thetable\ (continued)} \\',
                r'\toprule',head,r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{8}{r}{Continued on next page} \\',
                r'\endfoot',r'\bottomrule',r'\endlastfoot']
            previous=None
            for dataset,horizon in identities:
                if previous is not None and previous!=dataset:lines.append(r'\addlinespace[2pt]')
                row=configs[backbone,dataset,horizon,'DC36ControlledMean']
                lines.append(' & '.join([DATASETS.get(dataset,dataset),str(horizon)]+[
                    cell(row[k+'_mean'],row[k+'_std']) for k in metrics])+r' \\')
                for metric in metrics:
                    details.append(dict(backbone=backbone,dataset=dataset,horizon=horizon,method='FSPC',
                        method_record_key='DC36ControlledMean',metric=metric,mean=row[metric+'_mean'],
                        sample_sd=row[metric+'_std'],training_seeds=3))
                previous=dataset
            lines += [r'\end{longtable}','']
            (paper/'supplementary_tables'/('fspc_'+backbone.lower()+'_'+suffix+'.tex')).write_text('\n'.join(lines))
    with (paper/'fspc_detailed_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    record=dict(all_passed=True,tasks=249,configurations=83,main_method='FSPC',
        main_method_record_key='DC36ControlledMean',mean_control_ablation='DC36',
        mean_preserving_ablation='DC36MeanPreserving',default_states=36,
        supplementary_mean_sd_pairs=len(exported),full_method_detailed_metric_rows=len(details),
        full_method_detailed_tables=14,original_measurements_unchanged=True)
    record.update(main_comparison_methods=main_methods,calibration_comparison_methods=comparison_methods,component_ablation_methods=component_methods,
                  component_ablation_evaluation='identical retained test coordinates; paired Native normalization',
                  component_ablation_mean_sd_pairs=105,component_ablation_metrics=[m for m,_ in display_metrics],
                  main_mean_sd_pairs=70,calibration_comparison_mean_sd_pairs=140,
                  component_ablation_bold_best=True)
    if merged_audit:
        record.update(calibration_comparison_methods=merged_audit['methods'],
            calibration_comparison_mean_sd_pairs=merged_audit['main_mean_sd_pairs'],
            calibration_comparison_configurations=merged_audit['configurations'])
    from render_temporal_fspc import render as render_temporal
    render_temporal(source,mean_root,train_root or mean_root.parent/'train_fspc',paper)
    record.update(temporal_fitting_methods=['Train FSPC','Val FSPC'],
                  temporal_fitting_mean_sd_pairs=80,temporal_fitting_native_ratios=True,
                  separate_point_table_removed=True)
    (paper/'submission/fspc_table_render.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mean-root',type=Path,required=True);p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--train-root',type=Path)
    a=p.parse_args();render(a.source,a.mean_root,a.paper,a.train_root)
