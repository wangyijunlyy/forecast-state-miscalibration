"""One real training-window forecast from a fresh checkpoint; no scoring."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from core import atomic
from interfaces import prepare

p=argparse.ArgumentParser()
p.add_argument('--backbone',required=True)
p.add_argument('--task-id',help='Exact seed1 manifest task for an additional interface-only check')
a=p.parse_args()
root=Path('/home/wyj/three_seed_reproduction_20261031')
tasks=json.loads((root/'manifest.json').read_text())
t=next(t for t in tasks if t['backbone']==a.backbone and t['seed']==1
       and (t['id']==a.task_id if a.task_id else t['dataset'].lower()=='etth1'))
record=json.loads((Path(t['run_root'])/'training.json').read_text())
checkpoint=Path(record['checkpoint']).resolve()
assert record['training_complete'] and record['seed']==1
assert checkpoint.is_file() and root/'runs' in checkpoint.parents
out=(root/'validation/checkpoint_interface_probe_highdim'/t['id'] if a.task_id
     else root/'validation/checkpoint_interface_probe'/a.backbone)
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
interface=prepare(t,out)
loaded=Path(interface['checkpoint']).resolve()
assert loaded==checkpoint or loaded==checkpoint.parent

def first_window(value):
    if torch.is_tensor(value):return value[:1] if value.ndim else value
    if isinstance(value,dict):return {k:first_window(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        if value and all(isinstance(v,str) for v in value):return value[:1]
        items=[first_window(v) for v in value]
        return tuple(items) if isinstance(value,tuple) else items
    return value

batch=first_window(next(iter(interface['train'])))
with torch.no_grad():
    samples,truth,origin=interface['predict'](batch,'calibration',1)
assert samples.shape==(1,100,t['horizon'],truth.shape[-1]),(samples.shape,truth.shape)
assert truth.shape==(1,t['horizon'],samples.shape[-1])
assert all(np.isfinite(v).all() for v in (samples,truth,origin))
torch.cuda.synchronize()
result=dict(implementation_probe_only=True,excluded_from_main_scores=True,
            task_id=t['id'],backbone=t['backbone'],seed=t['seed'],
            checkpoint=str(checkpoint),loaded_checkpoint_reference=str(loaded),
            forecast_split='train',forecast_windows=1,forecast_samples=100,
            prediction_shape=list(samples.shape),target_shape=list(truth.shape),
            finite=True,test_iterator_consumed=False,calibrator_fitted=False,
            scientific_metrics_computed=False,seconds=time.monotonic()-start,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
atomic(out/'probe.json',result)
print('CHECKPOINT_INTERFACE_PROBE',json.dumps(result),flush=True)
