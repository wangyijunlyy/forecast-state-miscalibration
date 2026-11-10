"""Numerical and paired-evidence checks on completed real experiment records."""
import argparse
import json
from pathlib import Path
import numpy as np
from location_scale import complete_result


def audit_records(root):
    root=Path(root);jobs=json.loads((root/'manifest.json').read_text());checked=0
    missing=[]
    for job in jobs:
        p=root/job['mode']/'runs'/job['task']['id']/'results.json'
        if not p.exists():missing.append(job['task']['id']+':'+job['mode']);continue
        r=json.loads(p.read_text())
        if not complete_result(r,job['task'],job['mode'],job['external']):raise ValueError('Protocol mismatch: '+str(p))
        native=r['methods']['Native'];ref=r['methods']['FSPCRefit']
        for name,v in r['methods'].items():
            if v['coordinate_count']!=native['coordinate_count']:raise ValueError('Coordinate mismatch: '+str(p)+':'+name)
            np.testing.assert_array_equal(v['state_coverage']['counts'],native['state_coverage']['counts'])
            assert np.isfinite([v['MSE'],v['CRPS'],v['QICE_percent']]).all()
        np.testing.assert_allclose(r['methods']['FSPCScale']['MSE'],ref['MSE'],atol=1e-9,rtol=1e-10)
        for name,v in r['fit']['selection'].items():
            assert v['selected_mse']<=v['reference_mse']+1e-9,(p,name,'selectionMSE')
            assert v['selected_crps']<=v['reference_crps']+1e-9,(p,name,'selectionCRPS')
        assert r['fit']['test_targets_used'] is False
        if job['mode']=='original':assert r['native_parity']=={'Native':True,'FSPCRefit':True}
        checked+=1
    out=dict(complete_records_checked=checked,expected=len(jobs),all_complete=not missing,missing=missing,
        coordinate_and_state_parity=True,scale_only_mean_preservation=True,
        selection_mse_constraint=True,selection_crps_nonincrease=True,test_targets_used_for_fit=False)
    (root/'analysis').mkdir(exist_ok=True)
    (root/'analysis/integrity_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(audit_records(a.root))
