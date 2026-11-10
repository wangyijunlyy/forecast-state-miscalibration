"""One real-window implementation check, excluded from benchmark scores."""
import json,os,sys,time
from pathlib import Path
from dataclasses import fields
import torch
from torch.utils.data import DataLoader
from core import atomic
from validation_runtime import input_gradient_only,DTS_VALIDATION_VERSION

root=Path('/home/wyj/three_seed_reproduction_20261031');tid='DiffusionTS_ETTh1_h192_seed1'
t=json.loads((root/'tasks'/(tid+'.json')).read_text());arc=root/'archive/diffusionts_validation_v1'/tid
old=json.loads((arc/'training.json').read_text());ck=arc/Path(old['checkpoint']).relative_to(Path(t['run_root']))
os.chdir(t['repo']);sys.path.insert(0,t['repo'])
from src.experiments.DiffusionTS import DiffusionTSForecast
out=root/'validation/diffusionts_input_grad_probe';out.mkdir(parents=True,exist_ok=True)
allowed={f.name for f in fields(DiffusionTSForecast)};args={k:v for k,v in t['training_args'].items() if k in allowed}
args.update(device='cuda:0',num_worker=0,save_dir=str(out/'setup'),data_path=str(Path(t['repo'])/'data'))
torch.set_num_threads(2);m=DiffusionTSForecast(**args);m._init_metrics=lambda:None;m._setup_run(1)
m.model.load_state_dict(torch.load(ck,map_location='cuda:0',weights_only=True));m.model.eval()
loader=DataLoader(m.val_loader.dataset,batch_size=1,shuffle=False,drop_last=False,num_workers=0)
x,y,ox,oy,xm,ym=next(iter(loader));x=x.to('cuda:0').float();y=y.to('cuda:0').float();xm=xm.to('cuda:0').float();ym=ym.to('cuda:0').float()
before={n:p.detach().cpu().clone() for n,p in m.model.named_parameters()};flags=[p.requires_grad for p in m.model.parameters()]
start=time.monotonic();torch.cuda.reset_peak_memory_stats()
with input_gradient_only(m.model),torch.no_grad():
    pred,true=m._process_val_batch(x,y,xm,ym)
    assert torch.isfinite(pred).all() and pred.shape==(1,192,7,100)
    assert all(p.grad is None for p in m.model.parameters())
assert flags==[p.requires_grad for p in m.model.parameters()]
assert all(torch.equal(before[n],p.detach().cpu()) for n,p in m.model.named_parameters())
record=dict(implementation_probe_only=True,excluded_from_main_scores=True,protocol=DTS_VALIDATION_VERSION,
    real_dataset='ETTh1',real_validation_windows=1,prediction_shape=list(pred.shape),finite=True,network_parameter_gradients=False,
    model_parameters_unchanged=True,training_gradient_flags_restored=True,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start)
atomic(out/'probe.json',record);print('REAL_VALIDATION_PROBE',record,flush=True)
