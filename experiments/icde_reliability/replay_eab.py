"""Single-task replay entry points for a relocated EAB bundle."""
import argparse
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--stage',choices=['quality','traincal','train','deepstuq'],required=True)
parser.add_argument('--task');parser.add_argument('--dataset',choices=['PEMS03','PEMS04','PEMS07','PEMS08'])
parser.add_argument('--seed',type=int,choices=[1,2,3]);parser.add_argument('--from-scratch',action='store_true')
parser.add_argument('--output-root',type=Path,help='Fresh output parent for backbone or traffic retraining')
parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
bundle=args.bundle.resolve();home=bundle/'payload/home/wyj';root=home/'icde_reliability_20261102'
if not (bundle/'RELOCATION.json').exists():
    raise ValueError('Relocate the pristine bundle before replay')
os.environ.update(ICDE_RUN_ROOT=str(root),CUDA_VISIBLE_DEVICES=str(args.gpu),
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
runtime=home/'miniconda3/envs/k2vae/lib/python3.10/site-packages'
os.environ['PYTHONPATH']=str(runtime)+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0,str(runtime));sys.path.insert(0,str(root/'code'))
if args.stage=='deepstuq':
    if args.dataset is None or args.seed is None:
        parser.error('DeepSTUQ requires --dataset and --seed')
    import deepstuq_task as native
    standalone=args.output_root.resolve() if args.output_root else root/'standalone_traffic'
    data=standalone/'deepstuq/data';data.parent.mkdir(parents=True,exist_ok=True)
    if not data.exists():
        data.symlink_to(root/'deepstuq/data',target_is_directory=True)
    identity=f'{args.dataset}_seed{args.seed}';out=standalone/'deepstuq/runs'/identity;out.mkdir(parents=True,exist_ok=True)
    if not args.from_scratch:
        original=root/'deepstuq/runs'/identity
        for name in ['model.pt','pretrained.pt','training.json','training_history.json','temperature.json']:
            if not (out/name).exists():
                shutil.copy2(original/name,out/name)
    elif (out/'training.json').exists():
        raise ValueError('From-scratch training needs a fresh standalone output directory')
    # This command runs independently of the original campaign scheduler.
    # It retains the native training/calibration/evaluation code and settings.
    native.ROOT=standalone;native.wait_for_primary=lambda:None
    native.torch.set_num_threads(2);native.run(args.dataset,args.seed)
    raise SystemExit(0)
if args.task is None:
    parser.error('This stage requires --task ID')
task=next(t for t in json.loads((root/'manifest.json').read_text()) if t['id']==args.task)
directory=root/'replay_tasks';directory.mkdir(exist_ok=True)
if args.stage=='train':
    output=args.output_root.resolve() if args.output_root else root/'retrained'
    task=dict(task,run_root=str(output/task['id']))
taskfile=directory/(task['id']+'.json');taskfile.write_text(json.dumps(task,indent=2)+'\n')
if args.stage in ['quality','train']:
    script=root/'code/run_task.py' if args.stage=='quality' else home/'three_seed_reproduction_20261031/code/train_task.py'
    subprocess.run([sys.executable,str(script),'--task',str(taskfile)],check=True,env=os.environ)
else:
    path=home/'source_filtered_reproduction/code/evaluate.py'
    spec=importlib.util.spec_from_file_location('training_residual_replay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=root/'regenerated_training_calibrators'/task['id'];out.mkdir(parents=True,exist_ok=True)
    interface=module.prepare(task,out);models,metadata=module.fit(task,out,interface)
    with (home/'source_filtered_reproduction/runs'/task['id']/'calibrators.pkl').open('rb') as stream:
        expected=pickle.load(stream)
    equality={name:module.equal(model,expected[name]) for name,model in models.items()}
    (out/'replay_parity.json').write_text(json.dumps(equality,indent=2)+'\n')
    if not all(equality.values()):
        raise ValueError('Regenerated training residual pools differ from supplied states')
    print('REGENERATED_TRAINING_POOL_PARITY',task['id'],equality)
