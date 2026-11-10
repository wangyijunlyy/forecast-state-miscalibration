"""Replay the failed native epoch and locate its first nonfinite gradient."""
import json,os,sys,traceback
from dataclasses import fields
from pathlib import Path
import torch
from core import atomic

root=Path('/home/wyj/three_seed_reproduction_20261031')
t=json.loads((root/'tasks/TMDM_Traffic_h192_seed2.json').read_text())
out=root/'validation/tmdm_nonfinite_probe'
out.mkdir(parents=True,exist_ok=True)
os.chdir(t['repo']);sys.path.insert(0,t['repo'])
from src.experiments.TMDM import TMDMForecast
from torch_timeseries.utils.reproduce import reproducible

args={k:v for k,v in t['training_args'].items() if k in {f.name for f in fields(TMDMForecast)}}
args.update(device='cuda:0',num_worker=0,save_dir=str(out/'setup'),
            data_path=str(Path(t['repo'])/'data'))
torch.set_num_threads(2)
m=TMDMForecast(**args);m._init_metrics=lambda:None;m._setup_run(2)
saved=next((Path(t['run_root'])/'training').rglob('run_checkpoint.pth'))
state=torch.load(saved,map_location='cuda:0',weights_only=False)
m.model.load_state_dict(state['model']);m.cond_pred_model.load_state_dict(state['cond_pred_model'])
m.model_optim.load_state_dict(state['optimizer']);m.current_epoch=state['current_epoch']
stable='--stable-epoch' in sys.argv
if stable:
    from tmdm_numerics import enable_stable_tmdm_attention
    enable_stable_tmdm_attention(m.cond_pred_model)
named=[('diffusion.'+n,p) for n,p in m.model.named_parameters()]+[
       ('conditional.'+n,p) for n,p in m.cond_pred_model.named_parameters()]
assert all(torch.isfinite(p).all() for _,p in named)
observations={}
def observe(name):
    def hook(module,inputs,value):
        observations[name]=dict(min=float(value.detach().min()),max=float(value.detach().max()),
                                exp_zero=int((value.detach().exp()==0).sum()),
                                exp_nonfinite=int((~torch.isfinite(value.detach().exp())).sum()))
    return hook
m.cond_pred_model.z_logvar.register_forward_hook(observe('posterior_logvar'))
m.cond_pred_model.tau_learner.register_forward_hook(observe('log_tau'))
m.model.train();m.cond_pred_model.train()
reproducible(t['seed']+m.current_epoch)
for step,batch in enumerate(m.train_loader,1):
    x,y,ox,oy,xm,ym=batch
    x,y,xm,ym=(v.to(m.device).float() for v in (x,y,xm,ym))
    assert all(torch.isfinite(v).all() for v in (x,y,xm,ym))
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state()
    loss=m._process_train_batch(x,y,xm,ym)
    loss.backward()
    try:
        torch.nn.utils.clip_grad_norm_([p for _,p in named],float('inf'),error_if_nonfinite=True)
        bad=[]
    except RuntimeError:
        bad=[n for n,p in named if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad or not torch.isfinite(loss):
        record=dict(implementation_probe_only=True,excluded_from_main_scores=True,
            epoch=m.current_epoch,step=step,loss=float(loss.detach()),nonfinite_gradient_parameters=bad,
            observations=dict(observations),checkpoint=str(saved))
        torch.save(dict(model=m.model.state_dict(),conditional=m.cond_pred_model.state_dict(),
            optimizer=m.model_optim.state_dict(),x=x,y=y,xm=xm,ym=ym,
            rng_cpu=rng_cpu,rng_cuda=rng_cuda),out/'failing_step.pth')
        m.model_optim.zero_grad();torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
        try:
            with torch.autograd.detect_anomaly():
                m._process_train_batch(x,y,xm,ym).backward()
        except RuntimeError as e:record['anomaly']=str(e);traceback.print_exc()
        atomic(out/'trace.json',record)
        print('FIRST_NONFINITE',json.dumps(record),flush=True)
        break
    m.model_optim.step();m.model_optim.zero_grad()
    if step%50==0:print('REPLAY',step,float(loss),observations,flush=True)
    if not stable and step>=400:raise RuntimeError('Did not reproduce within400 batches')
else:
    assert stable
    from torch.utils.data import DataLoader
    m.model.eval();m.cond_pred_model.eval();windows=0
    with torch.no_grad():
        for x,y,ox,oy,xm,ym in DataLoader(m.val_loader.dataset,batch_size=1,shuffle=False,num_workers=0):
            x,y,xm,ym=(v.to(m.device).float() for v in (x,y,xm,ym))
            samples,true=m._process_val_batch(x,y,xm,ym)
            assert torch.isfinite(samples).all() and torch.isfinite(true).all()
            windows+=len(y)
    record=dict(implementation_probe_only=True,excluded_from_main_scores=True,
                native_epoch_completed=m.current_epoch,optimizer_steps=step,
                full_validation_windows=windows,finite=True,
                attention_fallbacks=sum(getattr(mod,'_overflow_fallbacks',0) for mod in m.cond_pred_model.modules()))
    atomic(out/'stable_epoch.json',record);print('STABLE_FULL_EPOCH',record,flush=True)
