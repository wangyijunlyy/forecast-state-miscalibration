"""Check complete native test replay after adding the equivalent attention fallback."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import run_task as primary
from core import CommonMetrics
from reliability_metrics import assert_traversal


def run(task,root):
    out=root/'runs'/task['id'];reference=json.loads((out/'results.json').read_text());interface=primary.prepare(task,out)
    primary.seed(task['seed']+30000);acc=CommonMetrics();windows=steps=0
    for step,batch in enumerate(interface['test'],1):
        with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
        assert x.shape[1:3]==(100,192) and np.isfinite(x).all();acc.update(x,y);windows+=len(y);steps=step
    assert_traversal(interface['test'],steps,windows);assert windows==reference['test_windows'] and steps==reference['test_batches']
    scores=acc.result();deltas={k:scores[k]-reference['methods']['Native'][k] for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE']}
    assert max(abs(v) for v in deltas.values())<=1e-9,deltas
    record=dict(task=task['id'],forecast_length=192,full_native_test_iterator=True,test_windows=windows,test_batches=steps,all_native_metrics_match=True,metric_differences=deltas,attention_precision_policy='native float32 with equivalent float64 overflow fallback')
    primary.atomic(out/'precision_replay.json',record);print('PRECISION_REPLAY_COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);run(json.loads(a.task.read_text()),a.root)
