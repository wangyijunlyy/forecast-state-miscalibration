"""Read a real first native test cache through the evaluation adapter."""
import argparse,json
from pathlib import Path
import torch
from run_task import prepare

p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
t=json.loads(a.task.read_text());a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
m,_,test,predict,_=prepare(t,a.output)
native,target,_,hit=predict(next(iter(test)),'test',1)
assert hit,'This check requires the historical test cache'
r=dict(task=t['id'],cache_hit=hit,sample_shape=list(native.shape),target_shape=list(target.shape),cache_dataset_label=m._dataset_label())
(a.output/'cache_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
