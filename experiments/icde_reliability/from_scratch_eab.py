"""Train native checkpoints locally and refit calibrators without downloaded weights."""
import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import types
from dataclasses import fields
from pathlib import Path


def load_initializer(repo, name):
    path = repo/'src/experiments'/('pretrain_'+name.lower()+'.py')
    # Preserve the native class; omit the stray notebook magic in the author file.
    source = '\n'.join(line for line in path.read_text().splitlines() if line.strip() != '%ln')
    identity = 'artifact_native_pretrain_'+name
    module = types.ModuleType(identity); module.__file__ = str(path); sys.modules[identity] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return getattr(module, name+'Forecast')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stage', choices=['all', 'shared', 'train', 'fit', 'quality'], default='all')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); bundle = args.bundle.resolve(); home = bundle/'payload/home/wyj'
    if not (bundle/'RELOCATION.json').is_file():
        raise ValueError('Restore upstream sources and run relocate_eab.py before training')
    root = home/'icde_reliability_20261102'
    os.environ.update(ICDE_RUN_ROOT=str(root), CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    runtime = home/'miniconda3/envs/k2vae/lib/python3.10/site-packages'
    os.environ['PYTHONPATH'] = str(runtime)+os.pathsep+os.environ.get('PYTHONPATH', '')
    sys.path[:0] = [str(runtime), str(root/'code')]
    task = next(t for t in json.loads((root/'manifest.json').read_text()) if t['id'] == args.task)
    if home not in Path(task['run_root']).parents:
        raise ValueError('Training outputs must remain inside the relocated artifact')
    if not Path(task['repo']).is_dir() or not (home/'three_seed_reproduction_20261031/code/train_task.py').is_file():
        raise FileNotFoundError('Restore native source files before training')
    data = Path(task['data_root']) if task['backbone']=='K2VAE' else (Path(task['model_args']['root_path'])/task['model_args']['data_path'] if task['backbone']=='PPM' else Path(task.get('resolved_data_root',str(Path(task['repo'])/'data'))))
    if not data.exists():
        raise FileNotFoundError('Extract the data-only archive before training: '+str(data))
    import torch
    torch.set_num_threads(2)
    if task.get('shared_original_pretrain'):
        repo = Path(task['repo']); os.chdir(repo); sys.path.insert(0, str(repo))
        for name in ['F', 'G']:
            settings = json.loads((bundle/'provenance/shared_initializers'/(name+'.json')).read_text())
            cls = load_initializer(repo, name)
            if set(settings)-{f.name for f in fields(cls)}:
                raise ValueError('Initializer arguments do not match the native class')
            settings.update(device='cuda:0', num_worker=0, data_path=str(repo/'data'), save_dir=str(repo/'results'))
            model = cls(**settings)
            output = repo/'results/runs'/name/settings['dataset_type']/f'w{settings["windows"]}h1s{settings["pred_len"]}'/'1/best_model.pth'
            print('NATIVE_SHARED_INITIALIZER', name, 'seed=1', 'epochs='+str(settings['epochs']), flush=True)
            if not args.check_only and args.stage in ['all', 'shared'] and not output.is_file():
                model.run(seed=1)
                if not output.is_file():
                    raise FileNotFoundError(output)
    if args.check_only:
        print('CONFIGURATION_CHECK_PASSED', task['id']); return
    if args.stage == 'shared':
        return
    task_dir = root/'replay_tasks'; task_dir.mkdir(exist_ok=True)
    task_file = task_dir/(task['id']+'.json'); task_file.write_text(json.dumps(task, indent=2)+'\n')
    if args.stage in ['all', 'train']:
        subprocess.run([sys.executable, str(home/'three_seed_reproduction_20261031/code/train_task.py'), '--task', str(task_file)], check=True)
    if args.stage in ['all', 'fit']:
        import run_task as evaluation
        from reservoir import CalibrationReservoir
        output = evaluation.SOURCE/'runs'/task['id']; output.mkdir(parents=True, exist_ok=True)
        interface = evaluation.prepare(task, output)
        evaluation.seed(task['seed']+10000); pool = CalibrationReservoir(50000, task['seed'])
        with torch.no_grad():
            for step, batch in enumerate(interface['train'], 1):
                if step > interface['fit_limit']:
                    break
                samples, target, _ = interface['predict'](batch, 'calibration', step)
                pool.update(samples, target)
        rows, target, horizon, _ = pool.arrays(); models = {}; metadata = {}
        for name, bins in [('DC16', 1), ('DC64', 4)]:
            models[name], metadata[name] = evaluation.fit_dc(rows, target, horizon, task['horizon'], n_horizon=bins)
        with (output/'calibrators.pkl').open('wb') as stream:
            pickle.dump(models, stream)
        (output/'fit_metadata.json').write_text(json.dumps(dict(fit_split='train', freshly_trained_checkpoint=True, models=metadata), indent=2)+'\n')
        print('FRESH_TRAINING_CALIBRATORS', task['id'], flush=True)
    if args.stage in ['all', 'quality']:
        subprocess.run([sys.executable, str(root/'code/run_task.py'), '--task', str(task_file)], check=True)


if __name__ == '__main__':
    main()
