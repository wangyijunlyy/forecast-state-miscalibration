"""Independently reconstruct train/validation parameter provenance for Table V."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np


def summary(rows):
    values=np.asarray(rows,dtype=np.float64)
    return values.mean(1),values.std(1,ddof=1)


def audit_task(args):
    task,root,source,mean_root=args
    record=json.loads((root/'runs'/task['id']/'results.json').read_text())
    reference=json.loads((mean_root/'runs'/task['id']/'cohort.json').read_text())
    assert record['task']==reference['task']==task
    assert record['all_variants_fit_before_test_access'] and not record['neural_training_performed']
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:
        tm,ts=summary(d['rows']);ty=d['targets'];th=d['horizon']
    with np.load(source/'runs'/task['id']/'calibration.npz') as d:
        vm,vs=summary(d['rows']);vy=d['targets'];vh=d['horizon']
    order=np.random.default_rng(task['seed']+60000).permutation(len(vy))
    maximum=0.
    for name,meta in record['resources'].items():
        b=meta['mean_bins'];g=meta['horizon_bins'];budget=meta['requested_budget'];take=order[:budget]
        me=np.quantile(vm[take],np.linspace(0,1,b+1))[1:-1]
        se=np.quantile(np.log(vs[take].clip(1e-6)),np.linspace(0,1,b+1))[1:-1]
        def assign(mean,scale,h):
            return (np.minimum(h*g//task['horizon'],g-1)*b+np.searchsorted(me,mean,side='right'))*b+np.searchsorted(se,np.log(scale.clip(1e-6)),side='right')
        train_ids=assign(tm,ts,th);val_ids=assign(vm[take],vs[take],vh[take]);states=b*b*g
        counts=np.bincount(train_ids,minlength=states)
        sums=np.bincount(train_ids,weights=ty-tm,minlength=states)
        bias=np.where(counts>=128,sums/(counts+128),0.)
        np.testing.assert_array_equal(counts,meta['training_state_counts'])
        np.testing.assert_allclose(bias,meta['state_bias'],rtol=1e-11,atol=1e-11)
        maximum=max(maximum,float(np.max(abs(bias-meta['state_bias']))))
        shift=bias[val_ids];error=vy[take]-vm[take]
        numerator=float(np.dot(error,shift));denominator=float(np.dot(shift,shift))
        strength=float(np.clip(numerator/denominator,0,1)) if denominator else 0.
        np.testing.assert_allclose([strength,numerator,denominator],
            [meta['strength'],meta['selection']['numerator'],meta['selection']['denominator']],rtol=1e-11,atol=1e-10)
        assert meta['pseudo_count']==meta['min_bias_count']==128
        assert meta['training_coordinates']==len(ty) and meta['calibration_coordinates']==len(take)
        assert meta['bias_fit_split']=='train' and meta['selection_split']=='validation'
        assert not meta['test_targets_used'] and meta['template_centered'] and meta['output_members']==100
        valid=vs[take]>1.01e-6
        np.testing.assert_array_equal(np.bincount(val_ids[valid],minlength=states),meta['source_counts'])
        assert int(valid.sum())==meta['retained_sources'] and int((~valid).sum())==meta['excluded_sources']
        native=float(np.mean(error**2));selected=float(np.mean((error-strength*shift)**2))
        np.testing.assert_allclose([native,selected],[meta['selection']['native_mse'],meta['selection']['selected_mse']],rtol=1e-11,atol=1e-10)
        assert selected<=native+1e-10*max(1,native)
        score=record['methods'][name]
        assert score['coordinate_count']==record['coordinates']
        assert all(np.isfinite(score[k]) and score[k]>=0 for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE'])
        np.testing.assert_allclose(score['RMSE']**2,score['MSE'],rtol=1e-12,atol=1e-10)
    default=record['methods']['b6_h1_n50000'];old=reference['methods']['DC36ControlledMean']
    differences={k:abs(default[k]-old[k]) for k in ['CRPS','QICE_percent','MSE','MAE','RMSE']}
    for k in differences:np.testing.assert_allclose(default[k],old[k],rtol=1e-10,atol=1e-9)
    return dict(id=task['id'],fits=len(record['methods']),maximum_bias_difference=maximum,default_differences=differences)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--mean-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    tasks=json.loads((a.root/'manifest.json').read_text())
    assert len(tasks)==249 and tasks==json.loads((a.source/'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        checked=list(pool.map(audit_task,[(t,a.root,a.source,a.mean_root) for t in tasks]))
    record=dict(all_passed=True,tasks=249,fits=sum(r['fits'] for r in checked),
        source_parameters_independently_reconstructed=True,
        maximum_bias_difference=max(r['maximum_bias_difference'] for r in checked),
        maximum_default_differences={k:max(r['default_differences'][k] for r in checked) for k in checked[0]['default_differences']},
        records=checked)
    (a.root/'source_selection_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='records'}))


if __name__=='__main__':main()
