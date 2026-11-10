"""Independently reconstruct printed FSPC table values from frozen task metrics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import numpy as np

BACKBONES=['K2VAE','PPM','TMDM','DiffusionTS','TimeDiff','NsDiff','DeepSTUQ']
METHODS=['Native','DC36ControlledMean','GlobalDC','GlobalPIT','ConditionalPIT','DC36']
PATTERN=r'\\(?:beststat|stat)\{([^}]+)\}\{([^}]+)\}(?:\\,\\ensuremath\{\\times10\^\{(-?\d+)\}\})?'


def audit(source,mean_root,paper,train_root=None):
    train_root=train_root or mean_root.parent/'train_fspc'
    tasks=json.loads((mean_root/'manifest.json').read_text());assert len(tasks)==249
    values={};groups=defaultdict(list);cohort_values={};cohort_groups=defaultdict(list)
    granularity_values={};granularity_groups=defaultdict(list)
    fspc_sweep='FSPC CRPS' in (paper/'tables/granularity.tex').read_text()
    if fspc_sweep:
        sweep_root=mean_root.parent/'fspc_granularity'
        proof=json.loads((sweep_root/'source_selection_audit.json').read_text())
        assert proof['all_passed'] and proof['tasks']==249 and proof['fits']==3237
    for task in tasks:
        old=json.loads((source/'runs'/task['id']/'results.json').read_text())
        new=json.loads((mean_root/'runs'/task['id']/'results.json').read_text())
        assert old['task']==new['task']==task
        np.testing.assert_allclose(new['methods']['DC36MeanPreserving']['MSE'],old['methods']['Native']['MSE'],rtol=1e-10,atol=1e-9)
        scores={m:old['methods'][m] for m in ['Native','DC36','GlobalDC','GlobalPIT','ConditionalPIT']}
        scores.update({m:new['methods'][m] for m in ['DC36ControlledMean','DC36MeanPreserving']})
        train=json.loads((train_root/'runs'/task['id']/'results.json').read_text())
        assert train['task']==task and train['full_test_split'] and not train['baseline_metrics_recomputed']
        assert train['methods']['TrainDC36']==old['methods']['TrainDC36']
        assert train['methods']['DC36ControlledMean']==new['methods']['DC36ControlledMean']
        scores.update(TrainDC36=old['methods']['TrainDC36'],TrainFSPC=train['methods']['TrainFSPC'])
        for method,metric in scores.items():
            key=task['backbone'],task['dataset'],task['horizon'],method
            point=dict(CRPS=metric['CRPS'],QICE_percent=metric['QICE_percent'],MSE=metric['MSE'],
                MAE=metric['MAE'],RMSE=metric['RMSE'],MedianMAE=metric['MedianMAE'],
                state_error_pp=None if metric['state_coverage']['state_error'] is None else
                    100*metric['state_coverage']['state_error'],
                pooled_error_pp=None if metric['state_coverage']['pooled_error'] is None else
                    100*metric['state_coverage']['pooled_error'],
                Brier80=metric['events']['brier'][0],Brier90=metric['events']['brier'][1],
                Brier95=metric['events']['brier'][2],BrierMean=sum(metric['events']['brier'])/3)
            values[key,task['seed']]=point;groups[key].append(task['seed'])
        cohort=json.loads((mean_root/'runs'/task['id']/'cohort.json').read_text())
        raw_cohort=json.loads((source/'cohort/runs'/task['id']/'results.json').read_text())
        assert cohort['task']==raw_cohort['task']==task
        assert cohort['coordinates']==raw_cohort['coordinates']
        for method,reference in [('Native',raw_cohort['native']),
                                 ('DC36',raw_cohort['methods']['b6_h1_n50000'])]:
            np.testing.assert_allclose([cohort['methods'][method][k] for k in ['CRPS','QICE_percent']],
                                       [reference[k] for k in ['CRPS','QICE_percent']],rtol=0,atol=1e-10)
        scores={m:cohort['methods'][m] for m in ['Native','DC36','DC36ControlledMean']}
        scores.update(raw_cohort['methods'])
        for method,metric in scores.items():
            assert metric['coordinate_count']==cohort['coordinates']
            assert metric['state_coverage']['counts']==raw_cohort['native']['state_coverage']['counts']
            key=task['backbone'],task['dataset'],task['horizon'],method
            cohort_values[key,task['seed']]={k:metric[k] for k in ['MSE','CRPS','QICE_percent']}
            cohort_values[key,task['seed']].update(
                state_error_pp=None if metric['state_coverage']['state_error'] is None else
                    100*metric['state_coverage']['state_error'],
                BrierMean=sum(metric['events']['brier'])/3)
            cohort_groups[key].append(task['seed'])
        if fspc_sweep:
            sweep=json.loads((sweep_root/'runs'/task['id']/'results.json').read_text())
            assert sweep['task']==task and sweep['coordinates']==cohort['coordinates']
            for method,metric in dict(sweep['methods'],Native=sweep['native']).items():
                key=task['backbone'],task['dataset'],task['horizon'],method
                granularity_values[key,task['seed']]={k:metric[k] for k in ['CRPS','QICE_percent']}
                granularity_groups[key].append(task['seed'])
    assert len(groups)==83*9 and all(sorted(s)==[1,2,3] for s in groups.values())
    assert all(sorted(s)==[1,2,3] for s in cohort_groups.values())
    def runs(backbone,method,metric,cohort=False):
        if cohort=='fspc':records,identities_by_method=granularity_values,granularity_groups
        else:records,identities_by_method=(cohort_values,cohort_groups) if cohort else (values,groups)
        identities=[key for key in identities_by_method if key[0]==backbone and key[3]==method
                    and all(records[key,s][metric] is not None for s in [1,2,3])]
        return np.array([np.mean([records[key,s][metric] for key in identities]) for s in [1,2,3]])
    def pair(x):return float(np.mean(x)),float(np.std(x,ddof=1))
    def ratio(method,metric,cohort=False):return pair(np.mean([runs(b,method,metric,cohort)/runs(b,'Native',metric,cohort) for b in BACKBONES],axis=0))
    five_metrics=['MSE','CRPS','QICE_percent','state_error_pp','BrierMean']
    merged_comparison=(paper/'table_data/calibration_comparison_raw_task_metrics.csv').exists()
    layouts=[('full_distribution',['Native','DC36ControlledMean'],False),
             ('calibration_comparison',['DC36ControlledMean','GlobalDC','GlobalPIT','ConditionalPIT'],False),
             ('component_ablation',['DC36ControlledMean','DC36','UnfilteredDC36'],True)]
    if merged_comparison: layouts=[entry for entry in layouts if entry[0]!='calibration_comparison']
    expected={name:[] for name,_,_ in layouts}
    expected.update(granularity=[],temporal_fitting=[])
    for name,methods,cohort in layouts:
        expected[name]=[pair(runs(b,method,metric,cohort))
            for b in BACKBONES for method in methods for metric in five_metrics]
    if merged_comparison:
        raw=list(csv.DictReader((paper/'table_data/calibration_comparison_raw_task_metrics.csv').open()))
        external={(r['id'],r['method']):r for r in csv.DictReader((paper/'table_data/external_calibration_raw_task_metrics.csv').open())}
        merged_methods=['FSPC','GlobalDC','GlobalPIT','ConditionalPIT','EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']
        merged_displayed_methods=json.loads((paper/'submission/calibration_comparison_audit.json').read_text()).get('displayed_methods',merged_methods)
        assert merged_displayed_methods in (merged_methods,['Native']+merged_methods)
        combined={};merged_groups=defaultdict(set)
        for row in raw:
            b,d,h,seed,method=row['backbone'],row['dataset'],int(row['horizon']),int(row['seed']),row['method']
            key=b,d,h,method
            if method in ['Native','FSPC','GlobalDC','GlobalPIT','ConditionalPIT']:
                ref=values[(b,d,h,'DC36ControlledMean' if method=='FSPC' else method),seed]
            else:
                ref=external[row['id'],method]
            point={}
            for metric in five_metrics:
                actual=None if row[metric] in ('','None') else float(row[metric])
                wanted=None if ref[metric] in (None,'','None') else float(ref[metric])
                assert (actual is None)==(wanted is None)
                if actual is not None:assert abs(actual-wanted)<=1e-10
                point[metric]=actual
            assert (key,seed) not in combined
            combined[key,seed]=point;merged_groups[key].add(seed)
        matched_configs={k[:3] for k in merged_groups}
        assert all(seeds=={1,2,3} for seeds in merged_groups.values())
        assert len(raw)==len(matched_configs)*3*(len(merged_methods)+1)
        expected['calibration_comparison']=[]
        for metric in five_metrics:
            for method in merged_displayed_methods:
                for b in BACKBONES:
                    eligible=[c for c in matched_configs if c[0]==b and all(
                        combined[c+(m,),seed][metric] is not None for m in ['Native']+merged_methods for seed in (1,2,3))]
                    av=[np.mean([combined[c+(method,),seed][metric] for c in eligible]) for seed in (1,2,3)]
                    expected['calibration_comparison'].append(pair(av))
    for metric in five_metrics:
        for method in ['TrainFSPC','DC36ControlledMean']:
            expected['temporal_fitting'] += [pair(runs(b,method,metric)) for b in BACKBONES]
            expected['temporal_fitting'].append(ratio(method,metric))
    assert not (paper/'tables/point_mse.tex').exists()
    assert 'tab:point' not in (paper/'sections/primary_results.tex').read_text()
    point_changes=[]
    for b,d,h,m in groups:
        if m=='DC36ControlledMean':
            candidate=np.mean([values[(b,d,h,m),s]['MSE'] for s in [1,2,3]])
            native=np.mean([values[(b,d,h,'Native'),s]['MSE'] for s in [1,2,3]])
            point_changes.append(candidate-native)
    point_changes=np.asarray(point_changes)
    point_wins=dict(improved=int((point_changes< -1e-10).sum()),worsened=int((point_changes>1e-10).sum()),
                    tied=int((abs(point_changes)<=1e-10).sum()))
    assert point_wins==dict(improved=58,worsened=15,tied=10)
    assert all(np.mean(runs(b,'DC36ControlledMean','MSE'))<np.mean(runs(b,'Native','MSE')) for b in BACKBONES)
    for methods in [[f'b{b}_h1_n50000' for b in [1,2,3,4,6,8,12,16]],
                    ['b6_h1_n2000','b6_h1_n10000','b6_h2_n50000','b6_h4_n50000','b6_h8_n50000']]:
        population='fspc' if fspc_sweep else True
        for b in BACKBONES:expected['granularity'] += [pair(runs(b,m,'CRPS',population)) for m in methods]
        expected['granularity'] += [ratio(m,'CRPS',population) for m in methods]
    best={}
    for name,methods,_ in layouts:
        means=np.array([p[0] for p in expected[name]]).reshape(len(BACKBONES),len(methods),len(five_metrics))
        best[name]=np.isclose(means,np.min(means,axis=1,keepdims=True),rtol=0,atol=1e-12).ravel().tolist()
    if merged_comparison:
        means=np.array([p[0] for p in expected['calibration_comparison']]).reshape(5,len(merged_displayed_methods),7)
        best['calibration_comparison']=np.isclose(means,np.min(means,axis=1,keepdims=True),rtol=0,atol=1e-12).ravel().tolist()
    means=np.array([p[0] for p in expected['temporal_fitting']]).reshape(5,2,8)
    best['temporal_fitting']=np.isclose(means,np.min(means,axis=1,keepdims=True),rtol=0,atol=1e-12).ravel().tolist()
    best['granularity']=[]
    for pairs,size in [(expected['granularity'][:64],8),(expected['granularity'][64:],5)]:
        means=np.array([p[0] for p in pairs]).reshape(8,size)
        best['granularity']+=np.isclose(means,np.min(means,axis=1,keepdims=True),rtol=0,atol=1e-12).ravel().tolist()
    def verify_printed(path,pairs,flags=None):
        actual=[(float(m),float(s),10.**int(e or 0)) for m,s,e in re.findall(PATTERN,path.read_text())]
        assert len(actual)==len(pairs),(str(path),len(actual),len(pairs))
        for i,((mean,sd,scale),wanted) in enumerate(zip(actual,pairs)):
            assert np.all(abs(np.array([mean,sd])-np.array(wanted)/scale)<=.000500001),(str(path),i,(mean,sd,scale),wanted)
        if flags is not None:
            kinds=re.findall(r'\\(beststat|stat)\{',path.read_text())
            assert [k=='beststat' for k in kinds]==flags,('Bold best values differ',str(path))
    for name,pairs in expected.items():verify_printed(paper/'tables'/(name+'.tex'),pairs,best[name])
    assert 'FSDC' not in (paper/'tables/full_distribution.tex').read_text()
    assert 'Unfiltered' not in (paper/'tables/granularity.tex').read_text()
    exported=list(csv.DictReader((paper/'supplementary_results.csv').open()));assert len(exported)==996
    per_backbone=defaultdict(list)
    for row in exported:
        key=row['backbone'],row['dataset'],int(row['horizon']),row['method'];metric=row['metric']
        wanted=pair([values[key,s][metric] for s in [1,2,3]])
        np.testing.assert_allclose([float(row['mean']),float(row['sample_sd'])],wanted,rtol=1e-10,atol=1e-12)
        per_backbone[row['backbone']].append(wanted)
    for b,pairs in per_backbone.items():verify_printed(paper/'supplementary_tables'/(b.lower()+'.tex'),pairs)
    details=list(csv.DictReader((paper/'fspc_detailed_results.csv').open()));assert len(details)==83*12
    detailed_pairs=defaultdict(list);missing=0;seen=set()
    point_metrics={'MSE','MAE','RMSE','MedianMAE','CRPS','QICE_percent'}
    for row in details:
        key=row['backbone'],row['dataset'],int(row['horizon']),'DC36ControlledMean';metric=row['metric']
        assert row['method']=='FSPC' and row['method_record_key']=='DC36ControlledMean'
        assert (key,metric) not in seen;seen.add((key,metric))
        observations=[values[key,s][metric] for s in [1,2,3]]
        if any(v is None for v in observations):
            assert not row['mean'] and not row['sample_sd'];missing+=1;continue
        wanted=pair(observations)
        np.testing.assert_allclose([float(row['mean']),float(row['sample_sd'])],wanted,rtol=1e-10,atol=1e-12)
        detailed_pairs[row['backbone'],'point' if metric in point_metrics else 'reliability'].append(wanted)
    for (backbone,kind),pairs in detailed_pairs.items():
        verify_printed(paper/'supplementary_tables'/('fspc_'+backbone.lower()+'_'+kind+'.tex'),pairs)
    record=dict(all_passed=True,source_tasks=249,configurations=83,
        main_mean_sd_pairs={k:len(v) for k,v in expected.items()},supplementary_mean_sd_pairs=len(exported),
        independent_raw_record_reconstruction=True,maximum_printing_rounding_error=.000500001,
        main_method='FSPC',main_record_key='DC36ControlledMean')
    record.update(full_method_detailed_metric_rows=len(details),full_method_detailed_tables=len(detailed_pairs),
        full_method_printed_mean_sd_pairs=sum(map(len,detailed_pairs.values())),
        full_method_undefined_coverage_rows=missing)
    record.update(component_ablation_paired_tasks=249,best_mean_marks_verified=True,component_ablation_bold_best=True,
                  component_ablation_evaluation='same retained test coordinates for FSPC and both ablations',
                  component_ablation_metrics=five_metrics,main_metrics=five_metrics,calibration_comparison_metrics=five_metrics,
                  scientific_multipliers_verified=True)
    record.update(granularity_method='FSPC' if fspc_sweep else 'w/o Mean Control',
                  granularity_parameter_evaluations=3237 if fspc_sweep else None)
    temporal_text=(paper/'tables/temporal_fitting.tex').read_text()
    assert 'Train w/o Mean' not in temporal_text and 'Val w/o Mean' not in temporal_text
    assert 'Avg. ratio' in temporal_text and 'Val FSPC' in temporal_text
    ratio_rows={r['method']:r for r in csv.DictReader((paper/'table_data/temporal_fitting_native_ratios.csv').open())}
    for method in ['TrainFSPC','DC36ControlledMean']:
        for metric in five_metrics:
            wanted=ratio(method,metric)
            np.testing.assert_allclose([float(ratio_rows[method][metric+'_mean']),
                float(ratio_rows[method][metric+'_std'])],wanted,rtol=1e-10,atol=1e-12)
    record.update(temporal_fitting_methods=['Train FSPC','Val FSPC'],
                  temporal_fitting_complete_tasks=249,temporal_fitting_mean_sd_pairs=80,
                  temporal_fitting_metrics=five_metrics,temporal_fitting_native_ratios_verified=True)
    record.update(separate_point_table_removed=True,mean_preserving_native_mse_parity_tasks=249,
                  full_test_point_accuracy_in_text=point_wins)
    (paper/'submission/fspc_numeric_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mean-root',type=Path,required=True);p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--train-root',type=Path)
    a=p.parse_args();audit(a.source,a.mean_root,a.paper,a.train_root)
