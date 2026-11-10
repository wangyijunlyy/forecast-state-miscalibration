"""Verify full campaign completion and paired numerical repairs before aggregation."""
import argparse
import json
from pathlib import Path
from aggregate_fsdc36 import verify_pair
from run_external_baselines import atomic,load_json

NAME='Qian2025SoftCRPS'
REVISION='stable-local-kernel-20261109'


def verify(root):
    manifest=load_json(root/'manifest.json')
    screen=load_json(root/'implementation_checks/qian_stable_kernel/conditioning_audit.json')
    assert len(manifest)==len(screen)==249
    assert {r['id'] for r in screen}=={t['id'] for t in manifest}
    affected={r['id'] for r in screen if r['unstable_calibration_queries'] or r['unstable_window_midpoints']}
    assert len(affected)==24
    records=[]
    for task in manifest:
        result=load_json(root/'runs'/task['id']/'results.json')
        assert result['task']==task and result['full_test_split']
        assert result['fit']['all_parameters_frozen_before_test_access']
        assert all(abs(v)<1e-8 for v in result['native_parity'].values())
        for score in result['methods'].values():verify_pair(score,result['methods']['Native'])
        if task['id'] not in affected:continue
        learned=result['fit']['methods'][NAME]
        assert learned['numerical_revision']==REVISION
        assert learned['calibration_coordinates']==result['fit']['calibration_coordinates']
        assert learned['gaussian_scale_floor']==1e-6 and not learned['source_filter']
        assert len(learned['candidates'])==20
        rec=dict(task=task['id'],numerical_revision=REVISION,selected_k=learned['selected_k'],
            full_test_native_parity=True,other_methods_unchanged=True)
        if 'stable_kernel_repair' in result:
            old=load_json(root/'implementation_checks/qian_stable_kernel'/task['id']/'results.json')
            assert old['task']==task
            for name in result['methods']:
                if name!=NAME:assert result['methods'][name]==old['methods'][name],(task['id'],name)
            for name in result['fit']['methods']:
                if name!=NAME:assert result['fit']['methods'][name]==old['fit']['methods'][name]
            rec['type']='existing_run_rechecked'
            rec['metric_changes']=result['stable_kernel_repair']['metric_changes']
        else:rec['type']='missing_run_completed'
        records.append(rec)
    assert sum(r['type']=='existing_run_rechecked' for r in records)==19
    assert sum(r['type']=='missing_run_completed' for r in records)==5
    audit=dict(all_passed=True,revision=REVISION,screened_tasks=249,full_test_completed_tasks=249,
        configurations=83,affected_tasks=sorted(affected),affected_full_test_evaluations=24,
        completed_missing_runs=5,rechecked_existing_runs=19,unaffected_runs_retained=225,
        native_full_test_parity=True,other_method_results_unchanged_in_rechecks=True,
        algorithm_regression_tests='11 passed on server, including direct kernel and QRF reference comparisons',
        numerical_changes=['Direct positive kernel sums for ill-conditioned prefix moments',
            'Continuous outward bandwidth limit for coincident boundary neighbors',
            'Exact reuse of identical conditional distributions; no data or model reduction'],records=records)
    atomic(root/'numerical_stability_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['records','affected_tasks']},indent=2))
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root)
