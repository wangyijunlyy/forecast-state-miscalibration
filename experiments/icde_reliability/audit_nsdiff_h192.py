"""Audit corrected forecast lengths, complete native populations and fitting identities."""
import argparse,json
from pathlib import Path
import numpy as np
from aggregation import audit_run,audit_metric,close
from aggregate_gaussian import audit as audit_gaussian


def audit(root,original_manifest,allow_pending=False):
    tasks=json.loads((root/'manifest.json').read_text());old={t['id']:t for t in json.loads(original_manifest.read_text())}
    assert len(tasks)==9 and {(t['dataset'],t['seed']) for t in tasks}=={(d,s) for d in ['ETTm1','ETTm2','ExchangeRate'] for s in [1,2,3]}
    expected={t['id']:t for t in tasks};errors=[];missing=[];complete=[]
    for t in tasks:
        identity=t['id'];out=root/'runs'/identity
        if not (out/'results.json').exists():missing.append(identity);continue
        try:
            original=old[t['replaces_task']]
            assert original['horizon']==24 and t['horizon']==192 and t['run_root']!=original['run_root']
            for key in ['model_args','training_args']:
                target=dict(original[key],pred_len=192);assert t[key]==target
            training=json.loads((Path(t['run_root'])/'training.json').read_text());assert training['training_complete']
            assert training['task_id']==identity and training['horizon']==192 and training['seed']==t['seed']
            assert 1<=training['metadata']['epochs_completed']<=50 and 'w168h1s192' in training['checkpoint']
            history=json.loads((Path(t['run_root'])/'training_history.json').read_text())
            assert [h['epoch'] for h in history]==list(range(1,training['metadata']['epochs_completed']+1))
            if identity=='NsDiff_ETTm2_h192_seed2':
                assert training['resumed_from_epoch'] in [13,18]
                assert training['metadata']['architecture_loss_optimizer_and_hyperparameters_unchanged']
                precision=json.loads((root/'precision_audit.json').read_text())
                assert precision['all_gradients_finite'] and precision['attention_float64_fallbacks']>0
                invalid=[h['epoch'] for h in history if h['validation'].get('CRPS') is None]
                assert invalid==training['metadata']['invalid_validation_candidates']==[19,20,21]
                best=min((h for h in history if h['validation'].get('CRPS') is not None),key=lambda h:h['validation']['CRPS'])
                assert best['epoch']==11 and training['metadata']['epochs_completed']-best['epoch']==t['patience']
            result=json.loads((out/'results.json').read_text());validation=json.loads((out/'validation.json').read_text());control=json.loads((out/'controls.json').read_text());audit_run(result,control,validation)
            assert result['task']==t and result['native_output_shape'][:2]==[100,192]
            channels=result['native_output_shape'][2]
            assert result['methods']['Native']['coordinate_count']==result['test_windows']*192*channels
            for key in ['CRPS','MSE','QICE_percent']:close(result['methods']['Native'][key],result['native_reference'][key],'Independent native metric mismatch')
            gaussian_id,_=audit_gaussian(root/'gaussian_runs'/identity/'results.json',root/'runs',expected);assert gaussian_id==identity
            train_dir=root/'train_calibration/runs'/identity;fit=json.loads((train_dir/'fit_metadata.json').read_text())
            assert fit['prediction_seed']==t['seed']+10000 and fit['fit_split']=='train' and 0<fit['fit_batches']<=100
            assert fit['retained_points']==50000 and validation['calibration_fit_limit']==100
            with np.load(out/'calibration.npz') as data:
                retained = min(50000, validation['calibration_seen_coordinates'])
                assert retained == validation['retained_calibration_coordinates']
                assert data['rows'].shape==(retained,100) and data['thresholds'].shape==(channels,3)
                assert data['horizon'].min()==0 and data['horizon'].max()==191
            comparison=json.loads((out/'training_partition_comparison.json').read_text())
            assert comparison['full_test_split'] and comparison['test_windows']==result['test_windows']
            for name,metric in comparison['methods'].items():
                audit_metric(metric);assert metric['coordinate_count']==result['methods']['Native']['coordinate_count']
                if name in ['Native','TrainDC16']:
                    for key in ['CRPS','MSE','QICE_percent']:close(metric[key],result['methods'][name][key],'Training-partition parity mismatch')
            assert all((root/p/identity/'results.json').exists() for p in ['compiled_deployment/runs','state_controls/runs'])
            replay=json.loads((out/'precision_replay.json').read_text())
            assert replay['all_native_metrics_match'] and replay['full_native_test_iterator']
            assert max(abs(v) for v in replay['metric_differences'].values())<=1e-9
            complete.append(identity)
        except Exception as e:errors.append(dict(task=identity,error=repr(e)))
    record=dict(expected_tasks=9,audited_tasks=len(complete),missing_tasks=missing,errors=errors,all_tasks_complete=len(complete)==9 and not errors,forecast_length=192,prior_length=24,old_records_relabelled=False,only_prediction_length_and_derived_dimensions_changed_in_model_args=True,numerical_correction='equivalent attention float64 overflow fallback; finite epoch-boundary resume; invalid validation candidates count as no improvement under patience10',native_metric_reference='independent CommonMetrics accumulator and complete native test replay with the precision policy',model_weights_distributed=False)
    (root/'horizon_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if errors or (missing and not allow_pending):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original-manifest',type=Path,required=True);p.add_argument('--allow-pending',action='store_true');a=p.parse_args();audit(a.root,a.original_manifest,a.allow_pending)
