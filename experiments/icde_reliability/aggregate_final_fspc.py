"""Audit the complete final-FSPC experiment and aggregate paired evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric,backbone_statistics,configuration_statistics,write_csv
from aggregate_fsdc36 import metric_row,verify_pair
from final_fspc import VERSION,ARMS,CompiledFSPC
from fspc_unified import UnifiedMean

METRICS=['MSE','MAE','RMSE','MedianMAE','CRPS','QICE_percent','state_error_pp','pooled_error_pp',
         'Brier80','Brier90','Brier95','BrierMean']
FIVE=['MSE','CRPS','QICE_percent','state_error_pp','BrierMean']
OPERATIONS=['reference_decode','compiled_decode','reference_decode_and_queries','compiled_decode_and_queries',
            'direct_queries_from_rows','direct_queries_from_summaries']


def load(path):return json.loads(Path(path).read_text())


def average_ratios(rows):
    lookup={(r['id'],r['method']):r for r in rows};configs=defaultdict(list)
    for r in rows:
        if r['method']=='Native':configs[r['backbone'],r['dataset'],r['horizon']].append(r)
    output={m:dict(method=m) for m in ARMS}
    for k in FIVE:
        values={m:{s:[] for s in [1,2,3]} for m in ARMS};count=0
        for b in sorted({r['backbone'] for r in rows}):
            eligible=[group for identity,group in configs.items() if identity[0]==b and all(lookup[r['id'],m][k] is not None for r in group for m in ARMS)]
            assert eligible;count+=len(eligible)
            for s in [1,2,3]:
                ids=[r['id'] for group in eligible for r in group if r['seed']==s]
                denominator=float(np.mean([lookup[i,'Native'][k] for i in ids]));assert denominator>0
                for m in ARMS:values[m][s].append(float(np.mean([lookup[i,m][k] for i in ids]))/denominator)
        for m in ARMS:
            v=[np.mean(values[m][s]) for s in [1,2,3]]
            output[m].update({k+'_mean':float(np.mean(v)),k+'_std':float(np.std(v,ddof=1)),k+'_n_configurations':count})
    return list(output.values())


def aggregate(root,aux):
    tasks=load(root/'manifest.json');expected={t['id']:t for t in tasks};quality=[];cost=[];contrasts=[];complete=[];errors=[]
    environments=[];max_fit_difference=0.;max_weight_difference=0.
    for identity,task in expected.items():
        folder=root/'runs'/identity
        if not (folder/'results.json').exists():continue
        try:
            r=load(folder/'results.json');assert r['version']==VERSION and r['task']==task
            assert r['full_test_split'] is False and r['backbone_retrained'] is False and r['native_inference_repeated'] is False
            assert set(r['methods'])==set(ARMS) and r['samples']==100 and r['repeats']==3
            parity=r['parity']
            for k in ['exact_reconstructed_values','exact_event_probabilities','exact_boundary_tie_probabilities',
                      'serialized_reload_verified','original_cohort_native_and_fspc_scores_reproduced']:assert parity[k] is True
            assert parity['maximum_value_difference']==0
            old=load(aux/'components/runs'/identity/'results.json');assert r['coordinates']==old['coordinates']<=50000
            for m in ARMS:verify_pair(r['methods'][m],r['methods']['Native'])
            for m,reference in [('Native',old['native']),('FSPC',old['methods']['FSPCMean'])]:
                a=metric_row(task,m,r['methods'][m]);b=metric_row(task,m,reference)
                for k in METRICS:
                    if b[k] is None:assert a[k] is None
                    else:np.testing.assert_allclose(a[k],b[k],atol=1e-9,rtol=1e-10)
            np.testing.assert_allclose(r['methods']['Native']['MSE'],r['methods']['ShapeOnly']['MSE'],atol=1e-8,rtol=1e-10)
            np.testing.assert_allclose(r['methods']['MeanOnly']['MSE'],r['methods']['FSPC']['MSE'],atol=1e-8,rtol=1e-10)
            frozen=load(folder/'parameters.json');assert frozen==load(aux/'components/runs'/identity/'parameters.json')
            model=UnifiedMean.from_record(frozen);compiled=CompiledFSPC.load(folder/'compiled.npz')
            assert compiled.cal.horizon_length==task['horizon'] and compiled.cal.state_to_template.shape==(36,)
            assert compiled.bias.shape==compiled.state_shift.shape==(36,)
            assert compiled.cal.quantiles.shape[1]==100 and 1<=len(compiled.cal.quantiles)<=36
            assert np.isfinite(compiled.cal.quantiles).all() and (np.diff(compiled.cal.quantiles,axis=1)>=0).all()
            assert (folder/'compiled.npz').stat().st_size==r['compiled_npz_bytes']
            with np.load(folder/'parity_probe.npz') as d:
                rows,h,thresholds,reference=[d[k] for k in ['rows','horizon','thresholds','reference']]
                np.testing.assert_array_equal(model.predict(rows,h),reference)
                np.testing.assert_array_equal(compiled.transform_rows(rows,h),reference)
                for th in [thresholds,reference[:,[0,49,99]]]:
                    expected_prob=np.stack([(reference>th[:,j,None]).mean(1) for j in range(3)],axis=1)
                    np.testing.assert_array_equal(compiled.probabilities_rows(rows,h,th),expected_prob)
            fit=r['fit'];assert fit['full_protocol'] and fit['includes_mean_and_blend_selection']
            assert fit['excludes_native_generation_and_file_io']
            max_fit_difference=max(max_fit_difference,fit['replay_maximum_prediction_difference'])
            for check in fit.get('replay_checks',[]):
                assert check['prediction_difference']<=check['prediction_bound']
                assert check['selection_crps_difference']<=1e-10 and check['reference_crps_difference']<=1e-10
                assert check['selected_mse']<=check['reference_mse']+1e-10*max(1.,check['reference_mse'])
                max_weight_difference=max(max_weight_difference,check['weight_difference'])
            all_times=dict(r['times'],complete_fit=fit['wall_seconds'],compilation=r['compilation_seconds'])
            medians={}
            for k,v in all_times.items():
                a=np.asarray(v);assert a.shape==(3,) and np.isfinite(a).all() and (a>0).all();medians[k]=float(np.median(a))
                if k in OPERATIONS:assert medians[k]==r['median_seconds'][k]
            assert set(r['times'])==set(OPERATIONS) and medians['complete_fit']==fit['median_seconds']
            assert medians['compilation']==r['median_compilation_seconds']
            env=r['environment'];assert env['workers']==4 and env['torch_cpu_threads']==2 and len(env['cpu_affinity'])==2
            assert all(v=='2' for v in env['thread_limit_environment'].values());environments.append(env)
            pair={m:metric_row(task,m,r['methods'][m]) for m in ARMS};quality.extend(pair.values())
            for contrast,left,right in [('MeanGivenNative','MeanOnly','Native'),('ShapeGivenNative','ShapeOnly','Native'),
                                        ('MeanGivenShape','FSPC','ShapeOnly'),('ShapeGivenMean','FSPC','MeanOnly')]:
                v={k:task[k] for k in ['id','backbone','dataset','horizon','seed']};v['method']=contrast
                for k in FIVE:v[k]=None if pair[left][k] is None else pair[left][k]-pair[right][k]
                contrasts.append(v)
            v={k:task[k] for k in ['id','backbone','dataset','horizon','seed']};v['method']='Interaction'
            for k in FIVE:
                v[k]=None if pair['Native'][k] is None else pair['FSPC'][k]-pair['MeanOnly'][k]-pair['ShapeOnly'][k]+pair['Native'][k]
            contrasts.append(v)
            v={k:task[k] for k in ['id','backbone','dataset','horizon','seed']};v['method']='FullFSPC'
            v.update(coordinates=r['coordinates'],fit_seconds=medians['complete_fit'],compile_ms=1e3*medians['compilation'],
                reference_kib=r['reference_pickle_bytes']/1024,compiled_kib=r['compiled_npz_bytes']/1024,
                decode_speedup=medians['reference_decode']/medians['compiled_decode'],
                query_rows_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_rows'],
                query_summaries_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_summaries'])
            v.update({k+'_us':1e6*medians[k]/r['coordinates'] for k in OPERATIONS});cost.append(v);complete.append(identity)
        except Exception as error:errors.append(dict(task=identity,error=repr(error)))
    output=root/'analysis';output.mkdir(exist_ok=True)
    missing=sorted(set(expected)-set(complete));passed=not missing and not errors and len(tasks)==249
    if quality:
        write_csv(output/'factorial_raw_task_metrics.csv',quality)
        write_csv(output/'factorial_configuration_statistics.csv',configuration_statistics(quality,METRICS))
        write_csv(output/'factorial_backbone_statistics.csv',backbone_statistics(quality,METRICS))
        write_csv(output/'factorial_native_ratios.csv',average_ratios(quality))
        write_csv(output/'factorial_paired_contrasts.csv',contrasts)
        write_csv(output/'factorial_contrast_backbone_statistics.csv',backbone_statistics(contrasts,FIVE))
        write_csv(output/'full_cost_raw_task_metrics.csv',cost)
        keys=[k for k in cost[0] if k not in ['id','backbone','dataset','horizon','seed','method','coordinates']]
        write_csv(output/'full_cost_backbone_statistics.csv',backbone_statistics(cost,keys))
    audit=dict(all_passed=passed,expected_tasks=249,audited_tasks=len(complete),missing_tasks=missing,errors=errors,
        version=VERSION,factorial_arms=list(ARMS),factorial_metric_rows=len(quality),complete_fit_repetitions=3*len(complete),
        exact_full_fspc_reconstructions=len(complete),exact_historical_event_queries=len(complete),exact_boundary_tie_queries=len(complete),
        saved_state_reload_probes=len(complete),independent_existing_cohort_score_checks=2*len(complete),
        maximum_fit_replay_prediction_difference=max_fit_difference,maximum_blend_weight_replay_difference=max_weight_difference,
        fit_optimizer_xatol=1e-7,fit_replay_checks='historical selection CRPS, MSE feasibility and parameter/prediction bounds',compiled_value_tolerance=0,
        all_methods_same_coordinates=True,no_backbone_training=True,no_native_sampling=True,
        author_full_test_overlays_applied=False,timing_workers=4,cpu_threads_per_worker=2,
        hosts=sorted({e['host'] for e in environments}),evaluation='retained cohorts; complete cached-calibration fit; full frozen FSPC deployment')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['missing_tasks','errors']},indent=2))
    if errors:print(json.dumps(errors,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--aux',type=Path,required=True)
    a=p.parse_args();aggregate(a.root,a.aux)
