"""Audit canonical 36-state records and build seed-aware numerical summaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, backbone_statistics, configuration_statistics, write_csv, close, flatten
from portable_dc import CompiledCalibrator

METHODS={'Native','TrainDC36','DC36','GlobalDC','GlobalPIT','ConditionalPIT'}
METRICS=['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE','state_error_pp','pooled_error_pp',
         'Brier80','Brier90','Brier95','BrierMean','transform_seconds']
PARTITIONS=[(b,1) for b in [1,2,3,4,6,8,12,16]]+[(6,g) for g in [2,4,8]]
BUDGETS=[2000,10000,50000]


def method_name(b,g,n):return f'b{b}_h{g}_n{n}'


def identify(value, task, version):
    if value['version']!=version or value['task']!=task:
        raise ValueError('Task/version mismatch')


def verify_pair(metric,native,states=36):
    audit_metric(metric)
    if len(metric['state_coverage']['counts'])!=states or metric['coordinate_count']!=native['coordinate_count']:
        raise ValueError('Wrong diagnostic dimension or population')
    if metric['state_coverage']['counts']!=native['state_coverage']['counts']:
        raise ValueError('Diagnostic populations differ')
    close(metric['events']['observed_frequencies'],native['events']['observed_frequencies'],'Event targets differ')


def metric_row(task,method,metric):
    row={k:task[k] for k in ['id','backbone','dataset','horizon','seed']}
    state=metric['state_coverage'];event=metric['events']
    row.update(method=method,coordinate_count=metric['coordinate_count'],
        eligible_coordinates=state['eligible_coordinates'],
        state_error_pp=None if state['state_error'] is None else 100*state['state_error'],
        pooled_error_pp=None if state['pooled_error'] is None else 100*state['pooled_error'])
    row.update({k:metric[k] for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE']})
    row.update(zip(['Brier80','Brier90','Brier95'],event['brier']))
    row['BrierMean']=float(np.mean(event['brier']))
    return row


def save_summary(directory,rows,metrics,expected,completed,errors,**extra):
    directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory/'raw_task_metrics.csv',rows)
    write_csv(directory/'configuration_statistics.csv',configuration_statistics(rows,metrics))
    write_csv(directory/'backbone_statistics.csv',backbone_statistics(rows,metrics))
    missing=sorted(set(expected)-set(completed))
    record=dict(expected_tasks=len(expected),audited_tasks=len(completed),
        all_tasks_complete=not missing and not errors and len(expected)==249,
        missing_tasks=missing,errors=errors,default_states=36,**extra)
    (directory/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def cohort(root,output,expected):
    rows=[];controls=[];native_rows=[];complete=[];errors=[]
    names={method_name(b,g,n) for b,g in PARTITIONS for n in BUDGETS}|{'UnfilteredDC36'}
    for identity,task in expected.items():
        path=root/'cohort/runs'/identity/'results.json'
        if not path.exists():continue
        try:
            value=json.loads(path.read_text());identify(value,task,'fsdc36-cohort-v1-20261103')
            if value['default_states']!=36 or set(value['methods'])!=names or set(value['resources'])!=names:
                raise ValueError('Incomplete 36-based cohort controls')
            if set(value['historical_parity_checks'])!={method_name(b,1,n) for b in [1,2,3,4,6,8,12,16] for n in BUDGETS}:
                raise ValueError('Missing historical grain parity')
            native=value['native'];verify_pair(native,native)
            if value['coordinates']!=native['coordinate_count'] or value['coordinates']>50000:
                raise ValueError('Wrong retained test population')
            native_rows.append(metric_row(task,'Native',native))
            for name,metric in value['methods'].items():
                verify_pair(metric,native);meta=value['resources'][name]
                b=meta['mean_bins'];g=meta['horizon_bins'];budget=meta['requested_budget']
                counts=np.array(meta['source_counts'])
                if meta['scale_bins']!=b or len(counts)!=b*b*g or counts.sum()!=meta['retained_sources'] or np.any(counts<0):
                    raise ValueError('Source population mismatch')
                if int((counts>=128).sum())!=meta['eligible_local_pools']:
                    raise ValueError('Local source threshold changed')
                n=min(budget,value['available_calibration_coordinates'])
                if meta['calibration_coordinates']!=n or meta['retained_sources']+meta['excluded_sources']!=n:
                    raise ValueError('Calibration budget mismatch')
                close(meta['fallback_percent'],100*meta['fallback_test_coordinates']/value['coordinates'],'Fallback mismatch')
                row=metric_row(task,name,metric)
                row.update(mean_bins=b,horizon_bins=g,nominal_states=b*b*g,requested_budget=budget,actual_budget=n,
                    fallback_percent=meta['fallback_percent'],retained_sources=meta['retained_sources'],
                    eligible_local_pools=meta['eligible_local_pools'],
                    crps_change_vs_dc36_percent=100*(metric['CRPS']/value['methods'][method_name(6,1,budget)]['CRPS']-1))
                rows.append(row)
            mapping={method_name(6,1,n):f'Budget{n}' for n in BUDGETS}
            mapping.update(b6_h4_n50000='DC144',UnfilteredDC36='UnfilteredDC36')
            for source,name in mapping.items():
                metric=value['methods'][source];row=metric_row(task,name,metric);row.update(value['resources'][source]);controls.append(row)
            complete.append(identity)
        except Exception as error:errors.append(dict(task=identity,error=repr(error)))
    metrics=[k for k in METRICS if k!='transform_seconds']+['fallback_percent','retained_sources','eligible_local_pools','crps_change_vs_dc36_percent']
    report=save_summary(output/'granularity/summary',rows,metrics,expected,complete,errors,
        fits_per_task=34,audited_fits=len(rows),evaluation='paired retained test cohort; 6x6 horizon/budget/source basis')
    directory=output/'summary';directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory/'raw_reservoir_controls.csv',controls)
    write_csv(directory/'partial_reservoir_control_statistics.csv',backbone_statistics(controls,['CRPS','QICE_percent']))
    write_csv(output/'granularity/summary/native_task_metrics.csv',native_rows)
    save_summary(output/'cohort_native/summary',native_rows,[k for k in METRICS if k!='transform_seconds'],expected,complete,errors,evaluation='paired retained Native cohort')
    save_summary(output/'cohort_controls/summary',controls,['CRPS','QICE_percent'],expected,complete,errors,evaluation='paired retained budget/source controls only')
    write_csv(output/'cohort_controls/summary/raw_reservoir_controls.csv',controls)
    write_csv(output/'cohort_controls/summary/partial_reservoir_control_statistics.csv',backbone_statistics(controls,['CRPS','QICE_percent']))

    configs=configuration_statistics(rows,metrics)
    by_config=defaultdict(dict)
    for row in configs:by_config[row['backbone'],row['dataset'],row['horizon']][row['method']]=row
    comparisons=[]
    for name in sorted(names):
        ds=[values[name]['CRPS_mean']-values['b6_h1_n50000']['CRPS_mean'] for values in by_config.values() if name in values and 'b6_h1_n50000' in values]
        comparisons.append(dict(method=name,configurations=len(ds),lower_than_dc36=sum(d<0 for d in ds),
            higher_than_dc36=sum(d>0 for d in ds),tied_with_dc36=sum(d==0 for d in ds)))
    write_csv(output/'granularity/summary/configuration_comparisons.csv',comparisons)
    return report


def primary(root,output,expected):
    rows=[];gaussian=[];gaussian_complete=[];resources=[];complete=[];errors=[];counts=defaultdict(dict)
    for identity,task in expected.items():
        path=root/'runs'/identity/'results.json'
        if not path.exists():continue
        try:
            value=json.loads(path.read_text());identify(value,task,'icde-fsdc36-v1-20261103')
            training=json.loads((root/'train_calibration/runs'/identity/'fit_metadata.json').read_text())
            identify(training,task,value['version'])
            if (training['mean_bins'],training['scale_bins'])!=(6,6) or training['fit_split']!='train' or not training['regenerated_training_reservoir_matches_original'] or set(training['models'])!={'DC36','DC144'}:
                raise ValueError('Training fitting configuration or frozen-reservoir parity differs')
            if (value['mean_bins'],value['scale_bins'],value['horizon_bins'])!=(6,6,1) or not value['full_test_split'] or not value['unchanged_method_parity']:
                raise ValueError('Wrong primary method or incomplete test')
            if set(value['methods'])!=METHODS or set(value['gaussian_methods']) not in [set(), {'GaussianVariance2P','GaussianAffine4P'}]:
                raise ValueError('Missing primary/Gaussian method')
            native=value['methods']['Native']
            if set(value['training_horizon_methods'])!={'Native','TrainDC36','TrainDC144'}:
                raise ValueError('Missing complete-test training/horizon comparison')
            for name,metric in {**value['methods'],**value['gaussian_methods'],**value['training_horizon_methods']}.items():
                verify_pair(metric,native)
            validation=value['validation']
            if validation['version']!=value['version'] or not validation['validation_complete'] or not validation['full_validation_split']:
                raise ValueError('Incomplete validation')
            for metric in validation['diagnostics'].values():
                audit_metric(metric)
                if len(metric['state_coverage']['counts'])!=36:raise ValueError('Wrong validation states')
            for name,metric in value['methods'].items():
                row=metric_row(task,name,metric);row['transform_seconds']=value['generation_or_transform_seconds'][name];rows.append(row)
            for name,metric in value['gaussian_methods'].items():gaussian.append(metric_row(task,name,metric))
            if value['gaussian_methods']:gaussian_complete.append(identity)
            n=native['coordinate_count'];meta=validation['models']['DC36']
            state_size=value.get('default_calibrator_bytes')
            if state_size is None:
                state_size=json.loads((root/'compiled_deployment/runs'/identity/'results.json').read_text())['original_pickle_bytes']
            row={k:task[k] for k in ['id','backbone','dataset','horizon','seed']}
            generation=value['generation_or_transform_seconds']['Native'];transform=value['generation_or_transform_seconds']['DC36']
            row.update(method='DC36',native_generation_us_per_coordinate=1e6*generation/n,
                dc36_transform_us_per_coordinate=1e6*transform/n,dc36_transform_to_generation_ratio=transform/generation,
                dc36_fit_seconds=meta['fit_seconds'],dc36_calibrator_bytes=state_size,
                source_excluded_percent=100*meta['excluded_sources']/meta['calibration_coordinates'],
                local_residual_pools=meta['local_pools']);resources.append(row)
            counts[task['backbone'],task['dataset'],task['horizon']][task['seed']]=value['training_horizon_methods']
            complete.append(identity)
        except Exception as error:errors.append(dict(task=identity,error=repr(error)))
    report=save_summary(output/'summary',rows,METRICS,expected,complete,errors,evaluation='complete native test iterator; 36-state calibration and diagnostics')
    directory=output/'summary'
    write_csv(directory/'complete_configuration_statistics.csv',configuration_statistics(rows,METRICS))
    write_csv(directory/'partial_backbone_statistics.csv',backbone_statistics(rows,METRICS))
    write_csv(directory/'raw_resource_metrics.csv',resources)
    keys=['native_generation_us_per_coordinate','dc36_transform_us_per_coordinate','dc36_transform_to_generation_ratio',
          'dc36_fit_seconds','dc36_calibrator_bytes','source_excluded_percent','local_residual_pools']
    write_csv(directory/'partial_resource_statistics.csv',backbone_statistics(resources,keys))
    save_summary(output/'gaussian/summary',gaussian,[k for k in METRICS if k!='transform_seconds'],expected,gaussian_complete,errors,evaluation='optional extra records; excluded from current five-method manuscript')
    win_counts=defaultdict(lambda:[0,0,0,0])
    for (b,d,h),seeds in counts.items():
        if set(seeds)!={1,2,3}:continue
        averages={(m,k):np.mean([v[m][k] for v in seeds.values()]) for m in ['Native','TrainDC36','TrainDC144'] for k in ['CRPS','QICE_percent']}
        n=win_counts[b];n[0]+=1;n[1]+=int(averages['TrainDC36','CRPS']<averages['Native','CRPS']);n[2]+=int(averages['TrainDC36','QICE_percent']<averages['Native','QICE_percent']);n[3]+=int(averages['TrainDC36','CRPS']<averages['TrainDC144','CRPS'])
    (output/'configuration_wins.json').write_text(json.dumps(dict(tasks=len(complete),counts=dict(win_counts)),indent=2)+'\n')
    return report


def controls(root,output,expected):
    rows=[];benchmarks=[];completed=[];errors=[]
    for identity,task in expected.items():
        path=root/'state_controls/runs'/identity/'results.json';bench=root/'compiled_deployment/runs'/identity/'results.json'
        if not path.exists() or not bench.exists():continue
        try:
            value=json.loads(path.read_text());runtime=json.loads(bench.read_text())
            identify(value,task,'fsdc36-assignment-compiled-v1-20261103');identify(runtime,task,value['version'])
            if value['default_states']!=36 or runtime['default_states']!=36:raise ValueError('Legacy controls passed as36')
            if set(value['methods'])!={'Native','DC36','GlobalDC'}|{f'Shuffled{i}' for i in range(5)}:raise ValueError('Missing assignment intervention')
            if not all(value[k] for k in ['fixed_source_mask','identical_global_pool','identical_state_counts','genuine_matches_existing_control']):raise ValueError('Shuffle changed controlled factors')
            genuine=CompiledCalibrator.load(path.parent/'genuine.npz');compiled=CompiledCalibrator.load(bench.parent/'compiled.npz')
            for key in ['mean_edges','scale_edges','quantiles','state_to_template']:
                np.testing.assert_array_equal(getattr(genuine,key),getattr(compiled,key))
            if compiled.state_to_template.shape!=(36,):raise ValueError('Compiled mapping is not36 states')
            if not runtime['exact_reconstructed_values'] or not runtime['exact_event_probabilities'] or runtime['maximum_value_difference']!=0.:raise ValueError('Compiled parity failed')
            cohort_record=json.loads((root/'cohort/runs'/identity/'results.json').read_text())
            for name,reference in [('Native',cohort_record['native']),('DC36',cohort_record['methods']['b6_h1_n50000'])]:
                for key in ['CRPS','QICE_percent','MSE']:
                    close(value['methods'][name][key],reference[key],'Cohort/state paired scores differ: '+name)
            task_rows=[]
            for name,metric in value['methods'].items():
                verify_pair(metric,value['methods']['Native']);r=metric_row(task,name,metric)
                r['crps_reduction_percent']=100*(1-metric['CRPS']/value['methods']['Native']['CRPS']);task_rows.append(r)
            average={k:task[k] for k in ['id','backbone','dataset','horizon','seed']};average['method']='ShuffledMean'
            for key in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE','state_error_pp','pooled_error_pp','Brier80','Brier90','Brier95','BrierMean','crps_reduction_percent','coordinate_count','eligible_coordinates']:
                values=[r[key] for r in task_rows if r['method'].startswith('Shuffled')]
                average[key]=None if any(v is None for v in values) else float(np.mean(values))
            task_rows.append(average);rows.extend(task_rows)
            for i in range(5):
                shuffled=CompiledCalibrator.load(path.parent/f'shuffled{i}.npz')
                for key in ['mean_edges','scale_edges','state_to_template']:np.testing.assert_array_equal(getattr(genuine,key),getattr(shuffled,key))
            medians={k:float(np.median(v)) for k,v in runtime['times'].items()}
            if any(len(v)!=3 or min(v)<=0 for v in runtime['times'].values()):raise ValueError('Wrong timing repeats')
            for name,v in medians.items():close(v,runtime['median_seconds'][name],'Timing median mismatch')
            r={k:task[k] for k in ['id','backbone','dataset','horizon','seed']}
            r.update(method='CompiledDC36',compile_ms=1000*runtime['compilation_seconds'],compiled_kib=runtime['compiled_npz_bytes']/1024,
                state_size_ratio=runtime['original_pickle_bytes']/runtime['compiled_npz_bytes'],
                decode_speedup=medians['reference_decode']/medians['compiled_decode'],
                direct_rows_query_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_rows'],
                cached_summary_query_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_summaries'])
            benchmarks.append(r);completed.append(identity)
        except Exception as error:errors.append(dict(task=identity,error=repr(error)))
    save_summary(output/'state_controls/summary',rows,[k for k in METRICS if k!='transform_seconds']+['crps_reduction_percent'],expected,completed,errors)
    keys=['compile_ms','compiled_kib','state_size_ratio','decode_speedup','direct_rows_query_speedup','cached_summary_query_speedup']
    return save_summary(output/'compiled/summary',benchmarks,keys,expected,completed,errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort-only',action='store_true');a=p.parse_args()
    expected={t['id']:t for t in json.loads((a.root/'manifest.json').read_text())}
    reports={'cohort':cohort(a.root,a.output,expected)}
    if not a.cohort_only:
        reports.update(primary=primary(a.root,a.output,expected),controls=controls(a.root,a.output,expected))
    print(json.dumps(reports,indent=2))
    if not all(r['all_tasks_complete'] for r in reports.values()):raise SystemExit(1)
