"""Full legacy native evaluation for the observed ECL historical discrepancy."""
import argparse,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from run_task import prepare
from metrics import ScoreAccumulator

p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
t=json.loads(a.task.read_text());a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
m,_,test,_,_=prepare(t,a.output)
from src.experiments.NsDiff import NsDiffForecast
m.diffusion_config.testing.minisample=1
loader=DataLoader(test.dataset,batch_size=t['model_args'].get('batch_size',32),shuffle=False,num_workers=0)
# Restore the saved native evaluation batch size, overridden only for the new run.
saved=json.loads((Path(t['checkpoint'])/'args.json').read_text())
loader=DataLoader(test.dataset,batch_size=saved['batch_size'],shuffle=False,num_workers=0)
acc=ScoreAccumulator();windows=0;start=time.monotonic()
for batch in loader:
 x,y,_,oy,xm,ym=batch
 with torch.no_grad():
  pred,truth=NsDiffForecast._process_val_batch(m,x.cuda().float(),y.cuda().float(),xm.cuda().float(),ym.cuda().float())
 acc.update(pred.permute(0,3,1,2).cpu().numpy(),truth.cpu().numpy());windows+=len(y)
assert windows==len(test.dataset)
r=dict(task=t['id'],legacy_native_metrics=acc.result(),test_windows=windows,
    batch_size=saved['batch_size'],minisample=1,total_seconds=time.monotonic()-start,
    frozen_checkpoint=t['checkpoint'],full_original_test_dataset=True)
(a.output/'native_interface_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
