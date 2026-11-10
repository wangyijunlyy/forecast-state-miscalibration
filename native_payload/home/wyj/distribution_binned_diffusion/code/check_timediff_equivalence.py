"""Compare original sequential and batched native forecasts with identical noise."""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import torch
from run_task import prepare
from timediff_batching import sample_timediff_batched


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    t=json.loads(a.task.read_text());a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,fit,_,_,_=prepare(t,a.output)
    batch=next(iter(fit));x,y,_,_,xm,ym=batch
    x=x.cuda().float();y=y.cuda().float();xm=xm.cuda().float();ym=ym.cuda().float()
    count=4
    noise=torch.randn(count,len(x),model.dataset.num_features,t['horizon'],device='cuda:0')
    index=iter(range(count))
    start=time.monotonic()
    with torch.no_grad(),patch('torch.randn',side_effect=lambda *args,**kwargs:noise[next(index)]):
        original=model._sample_timediff_uncached(x,y,xm,ym,count,count)
    torch.cuda.synchronize();original_seconds=time.monotonic()-start
    repeated_noise=noise.transpose(0,1).reshape(len(x)*count,model.dataset.num_features,t['horizon'])
    start=time.monotonic()
    with torch.no_grad(),patch('torch.randn',side_effect=lambda *args,**kwargs:repeated_noise):
        batched=sample_timediff_batched(model.model,x,xm,y,ym,count,count)
    torch.cuda.synchronize();batched_seconds=time.monotonic()-start
    torch.testing.assert_close(batched,original,rtol=1e-4,atol=1e-5)
    record=dict(task=t['id'],same_initial_noise=True,trajectories=count,
        maximum_absolute_output_difference=float((batched-original).abs().max()),
        original_seconds=original_seconds,batched_seconds=batched_seconds,
        uses_original_native_forward=True)
    (a.output/'equivalence.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
