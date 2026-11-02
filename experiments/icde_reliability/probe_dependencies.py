"""Record installed distributions imported by actual relocated native interfaces."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--backbone',required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
bundle=args.bundle.resolve();home=bundle/'payload/home/wyj';root=home/'icde_reliability_20261102'
sys.path.insert(0,str(home/'miniconda3/envs/k2vae/lib/python3.10/site-packages'))
sys.path.insert(0,str(root/'code'))
import run_task
import train_task
tasks=json.loads((root/'manifest.json').read_text())
if args.backbone=='DeepSTUQ':
    import deepstuq_task
else:
    task=next(t for t in tasks if t['backbone']==args.backbone);sys.path.insert(0,task['repo'])
    if args.backbone=='K2VAE':
        import probts
        path=home/'K2VAE/run.py';sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('native_training_cli',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    elif args.backbone=='PPM':
        path=Path(task['repo'])/'scripts/evaluate_ppm_posthoc_baselines.py'
        spec=importlib.util.spec_from_file_location('native_ppm',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:
        name={'NsDiff':'NsDiff','TMDM':'TMDM','TimeDiff':'TimeDiff','DiffusionTS':'DiffusionTS'}[args.backbone]
        importlib.import_module('src.experiments.'+name+'_SPARC_PIT')
mapping=importlib.metadata.packages_distributions();requirements={}
normalize=lambda value:re.sub(r'[-_.]','',value).lower()
available={normalize(d.metadata['Name']):d for d in importlib.metadata.distributions() if d.metadata['Name']}
for name in list(sys.modules):
    for distribution in mapping.get(name.split('.')[0],[]):
        if distribution.lower().replace('_','-')!='torch-timeseries':
            requirements[distribution]=importlib.metadata.version(distribution)
    canonical=normalize(name.split('.')[0])
    if canonical in available and canonical!='torchtimeseries':
        distribution=available[canonical]
        requirements[distribution.metadata['Name']]=importlib.metadata.version(distribution.metadata['Name'])
# Conda metadata can omit top_level/RECORD information used by the module map.
# Pin numerical, model and training dependencies explicitly as well.
for distribution in ['numpy','scipy','pandas','matplotlib','scikit-learn','einops','torchdiffeq',
    'tensorboard','tensorboardX','docstring-parser','packaging','opt-einsum','jsonargparse','protobuf']:
    requirements[distribution]=importlib.metadata.version(distribution)
out=args.output or bundle/('dependency_probes_clean' if Path(sys.prefix).name=='icde_eab_clean_env' else 'dependency_probes')
out.mkdir(exist_ok=True)
(out/(args.backbone+'.json')).write_text(json.dumps(requirements,indent=2)+'\n')
print(args.backbone,len(requirements),'imported installed distributions',flush=True)
