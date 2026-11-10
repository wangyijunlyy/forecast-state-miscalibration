"""Audit and aggregate the measured thirteen-setting FSPC parameter study."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import backbone_statistics,configuration_statistics,write_csv
from aggregate_mean_anchor_all import normalized_ratios
from fspc_granularity import VERSION,VARIANTS
from granularity import method_name
from mean_anchor_campaign import validate_tasks

METRICS=['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE']


def aggregate(root):
    tasks=json.loads((root/'manifest.json').read_text());validate_tasks(tasks)
    proof=json.loads((root/'source_selection_audit.json').read_text())
    assert proof['all_passed'] and proof['tasks']==249 and proof['fits']==3237
    assert {t['id'] for t in tasks}=={r['id'] for r in proof['records']}
    expected={method_name(*v) for v in VARIANTS};rows=[]
    for task in tasks:
        record=json.loads((root/'runs'/task['id']/'results.json').read_text())
        assert record['task']==task and record['version']==VERSION
        assert set(record['methods'])==set(record['resources'])==expected
        assert record['all_variants_fit_before_test_access'] and not record['neural_training_performed']
        for name,metric in dict(record['methods'],Native=record['native']).items():
            assert metric['coordinate_count']==record['coordinates']
            row={k:task[k] for k in ['id','backbone','dataset','horizon','seed']}
            row.update(method=name,coordinate_count=metric['coordinate_count'])
            row.update({k:metric[k] for k in METRICS})
            if name!='Native':
                meta=record['resources'][name]
                row.update(fallback_percent=meta['fallback_percent'],strength=meta['strength'])
            rows.append(row)
    out=root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'raw_task_metrics.csv',rows)
    write_csv(out/'backbone_statistics.csv',backbone_statistics(rows,METRICS+['fallback_percent','strength']))
    write_csv(out/'configuration_statistics.csv',configuration_statistics(rows,METRICS+['fallback_percent','strength']))
    ratios,_=normalized_ratios(rows,METRICS);write_csv(out/'average_ratio_statistics.csv',ratios)
    record=dict(all_passed=True,expected_tasks=249,audited_tasks=249,all_tasks_complete=True,
        fits=3237,default_states=36,method='FSPC',evaluation='shared retained test coordinates',
        validation_budgets=[2000,10000,50000],training_budget='fixed original training reservoir',
        independent_source_audit=True,neural_training_performed=False)
    (out/'audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    aggregate(p.parse_args().root)
