"""Inspect the exact first failing native TMDM step, without training outputs."""
import json,os,sys
from dataclasses import fields
from pathlib import Path
import torch
from core import atomic
root=Path('/home/wyj/three_seed_reproduction_20261031')
out=root/'validation/tmdm_nonfinite_probe'
t=json.loads((root/'tasks/TMDM_Traffic_h192_seed2.json').read_text())
os.chdir(t['repo']);sys.path.insert(0,t['repo'])
from src.experiments.TMDM import TMDMForecast
args={k:v for k,v in t['training_args'].items() if k in {f.name for f in fields(TMDMForecast)}}
args.update(device='cuda:0',num_worker=0,save_dir=str(out/'attention_setup'),data_path=str(Path(t['repo'])/'data'))
torch.set_num_threads(2)
m=TMDMForecast(**args);m._init_metrics=lambda:None;m._setup_run(2)
s=torch.load(out/'failing_step.pth',map_location='cuda:0',weights_only=False)
m.model.load_state_dict(s['model']);m.cond_pred_model.load_state_dict(s['conditional'])
m.model_optim.load_state_dict(s['optimizer'])
m.model.train();m.cond_pred_model.train()
assert all(torch.isfinite(p).all() for p in m.model.parameters())
assert all(torch.isfinite(p).all() for p in m.cond_pred_model.parameters())

def hook(name):
    def inspect(module,inputs,kwargs):
        q,k,v,mask=inputs[:4]
        tau=kwargs.get('tau');delta=kwargs.get('delta')
        raw=torch.einsum('blhe,bshe->bhls',q,k)
        logits=raw*(tau.unsqueeze(1).unsqueeze(1) if tau is not None else 1.)
        logits=logits+(delta.unsqueeze(1).unsqueeze(1) if delta is not None else 0.)
        if not torch.isfinite(logits).all():
            wide=torch.einsum('blhe,bshe->bhls',q.double(),k.double())
            wide=wide*(tau.double().unsqueeze(1).unsqueeze(1) if tau is not None else 1.)
            wide=wide+(delta.double().unsqueeze(1).unsqueeze(1) if delta is not None else 0.)
            record=dict(implementation_probe_only=True,excluded_from_main_scores=True,module=name,
                queries_finite=bool(torch.isfinite(q).all()),keys_finite=bool(torch.isfinite(k).all()),
                values_finite=bool(torch.isfinite(v).all()),raw_dot_finite=bool(torch.isfinite(raw).all()),
                raw_dot_abs_max=float(raw.abs().max()),tau_max=float(tau.max()),
                float32_nonfinite_logits=int((~torch.isfinite(logits)).sum()),
                float64_logits_finite=bool(torch.isfinite(wide).all()),float64_logits_abs_max=float(wide.abs().max()))
            atomic(out/'attention_trace.json',record);print('ATTENTION_OVERFLOW',record,flush=True)
            raise RuntimeError('Identified first attention-logit overflow')
    return inspect
stable='--stable' in sys.argv
if stable:
    from tmdm_numerics import enable_stable_tmdm_attention
    enable_stable_tmdm_attention(m.cond_pred_model)
else:
    for n,mod in m.cond_pred_model.named_modules():
        if type(mod).__name__=='DSAttention':mod.register_forward_pre_hook(hook(n),with_kwargs=True)
torch.set_rng_state(s['rng_cpu'].cpu());torch.cuda.set_rng_state(s['rng_cuda'].cpu())
loss=m._process_train_batch(s['x'],s['y'],s['xm'],s['ym'])
if stable:
    loss.backward()
    bad=[n for n,p in list(m.model.named_parameters())+list(m.cond_pred_model.named_parameters())
         if p.grad is not None and not torch.isfinite(p.grad).all()]
    record=dict(implementation_probe_only=True,excluded_from_main_scores=True,
                loss=float(loss),loss_finite=bool(torch.isfinite(loss)),nonfinite_gradients=bad,
                attention_fallbacks=sum(getattr(mod,'_overflow_fallbacks',0) for mod in m.cond_pred_model.modules()))
    assert record['loss_finite'] and not bad,record
    m.model_optim.step()
    record['parameters_after_step_finite']=all(bool(torch.isfinite(p).all()) for p in list(m.model.parameters())+list(m.cond_pred_model.parameters()))
    assert record['parameters_after_step_finite']
    atomic(out/'stable_step.json',record);print('STABLE_STEP',record,flush=True)
