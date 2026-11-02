"""Validate relocated model/data identities before any replay computation."""
import argparse
import json
import sys
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
args=parser.parse_args();bundle=args.bundle.resolve();home=bundle/'payload/home/wyj'
root=home/'icde_reliability_20261102'
tasks=json.loads((root/'manifest.json').read_text());checked=[]
for task in tasks:
    native=Path(task['run_root']);training=json.loads((native/'training.json').read_text())
    checkpoint=Path(training['checkpoint'])
    if not checkpoint.is_file() or home not in checkpoint.parents or not training['training_complete']:
        raise ValueError('Checkpoint did not relocate: '+task['id'])
    if not Path(task['repo']).is_dir() or home not in Path(task['repo']).parents:
        raise ValueError('Source did not relocate')
    if task['backbone']=='K2VAE':
        data=Path(task['data_root'])
    elif task['backbone']=='PPM':
        data=Path(task['model_args']['root_path'])/task['model_args']['data_path']
    else:
        data=Path(task.get('resolved_data_root',str(Path(task['repo'])/'data')))
    if not data.exists():
        raise ValueError('Missing native dataset: '+str(data))
    checked.append(task['id'])
sys.path.insert(0,str(root/'code'))
import run_task
import interfaces
import torch_timeseries
sys.path.insert(0,str(home/'SPARC/K2VAE_dacu_k2vae_ltsf'))
import probts
for module in [run_task,interfaces,torch_timeseries,probts]:
    if home not in Path(module.__file__).parents:
        raise ValueError('Import escaped artifact: '+module.__file__)
record=dict(checkpoints_checked=len(checked),all_native_data_paths_present=True,
    module_paths={m.__name__:m.__file__ for m in [run_task,interfaces,torch_timeseries]},
    python_executable=sys.executable,python_version=sys.version,
    scope='Path and import preflight only; dependency installation and full native replay are audited separately')
(bundle/'PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
