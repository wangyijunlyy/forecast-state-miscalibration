"""Measure native train/validation memory in a disposable model instance."""
import json, os, sys, time
from dataclasses import fields
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from core import atomic
from validation_runtime import input_gradient_only, DTS_VALIDATION_VERSION

root = Path('/home/wyj/three_seed_reproduction_20261031')
dataset = sys.argv[1] if len(sys.argv)>1 else 'ExchangeRate'
tid = f'DiffusionTS_{dataset}_h192_seed1'
t = json.loads((root/'tasks'/f'{tid}.json').read_text())
out = root/'validation'/f'diffusionts_resource_probe_{dataset}'
out.mkdir(parents=True, exist_ok=True)
os.chdir(t['repo'])
sys.path.insert(0, t['repo'])
from src.experiments.DiffusionTS import DiffusionTSForecast

allowed = {f.name for f in fields(DiffusionTSForecast)}
args = {k:v for k,v in t['training_args'].items() if k in allowed}
args.update(device='cuda:0', num_worker=0, save_dir=str(out/'setup'),
            data_path=str(Path(t['repo'])/'data'))
native_chunk = args.get('sample_chunk_size',20)
# Bound the native validation chunks, which are at most20.
args['sample_chunk_size'] = 20
torch.set_num_threads(2)
m = DiffusionTSForecast(**args)
m._init_metrics = lambda: None
m._setup_run(1)
native_batch = t['training_batch_size']
m.train_loader = DataLoader(Subset(m.train_loader.dataset, range(native_batch*3)),
                           batch_size=native_batch, shuffle=False, num_workers=0)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.monotonic()
loss = m._train()
torch.cuda.synchronize()
assert len(loss) == 3 and all(torch.isfinite(torch.tensor(v)) for v in loss)
train_seconds = time.monotonic()-start
train_peak = torch.cuda.max_memory_allocated()
v = DataLoader(m.val_loader.dataset, batch_size=1, shuffle=False, num_workers=0)
x,y,ox,oy,xm,ym = next(iter(v))
x,y,xm,ym = (z.to(m.device).float() for z in (x,y,xm,ym))
m.model.eval()
with input_gradient_only(m.model), torch.no_grad():
    pred,true = m._process_val_batch(x,y,xm,ym)
assert pred.shape == (1,t['horizon'],m.dataset.num_features,100)
assert torch.isfinite(pred).all()
torch.cuda.synchronize()
record = dict(resource_probe_only=True, excluded_from_main_scores=True,
              dataset=t['dataset'], task_reference=tid, protocol=DTS_VALIDATION_VERSION,
              model_instance='fresh disposable; no benchmark checkpoint loaded or saved',
              training_batch_size=native_batch, native_optimizer_steps=3,
              validation_window_batch_size=1, validation_samples=100,
              validation_sample_chunk_size=20, reference_native_sample_chunk_size=native_chunk,
              features=m.dataset.num_features, context=t['training_args']['windows'], horizon=t['horizon'],
              train_peak_allocated_bytes=train_peak,
              total_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              total_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              train_seconds=train_seconds, total_seconds=time.monotonic()-start,
              finite=True, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
atomic(out/'probe.json',record)
print('RESOURCE_PROBE',json.dumps(record),flush=True)
