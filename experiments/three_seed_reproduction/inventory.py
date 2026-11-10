"""Freeze all original task configurations before training."""
import copy,json,re
from pathlib import Path
ROOT=Path('/home/wyj/three_seed_reproduction_20261031')

def main():
    base=json.loads(Path('/home/wyj/distribution_binned_table1/results/manifest.json').read_text())
    diffusion=json.loads(Path('/home/wyj/distribution_binned_diffusion/results/manifest.json').read_text())
    base=[t for t in base if t['backbone']=='K2VAE' or t['paper_seed']==1]+diffusion
    assert len(base)==79
    tasks=[]
    for source in base:
        for seed in (1,2,3):
            t=copy.deepcopy(source);t['historical_checkpoint']=t.pop('checkpoint')
            t.update(id=f'{t["backbone"]}_{t["dataset"]}_h{t["horizon"]}_seed{seed}',seed=seed,backbone_seed=seed,sampling_seed=seed,paper_seed=seed)
            t['run_root']=str(ROOT/'runs'/t['id']);t['test_cache_root']=''
            if t['backbone']=='K2VAE':
                t['train_repo']='/home/wyj/K2VAE';t['training_config']=f'config/ltsf/{t["dataset"]}/k2vae.yaml'
                # Per-task effective batch sizes differ from the default64;
                # retain the actual historical checkpoint configuration.
                history=json.loads((ROOT/'historical_k2_checkpoint_training_config.json').read_text())
                original=next(r for r in history if r['dataset']==t['dataset'] and r['horizon']==t['horizon'])
                t['training_batch_size']=original['batch_size'];t['max_epochs']=20;t['patience']=5
                t['historical_training_batch_size']=original['batch_size']
                t['historical_validation_batch_size']=original['test_batch_size']
                t['training_config_source']='actual historical checkpoint datamodule hyperparameters'
            elif t['backbone']=='PPM':
                name={'ETTm1':'ettm1','ETTm2':'ettm2'}.get(t['dataset'],t['dataset'])
                script=Path(t['repo'])/'scripts/MLP'/f'{name}.sh';text=script.read_text()
                def option(key,default):
                    m=re.search(r'--'+key+r'\s+([\w.]+)',text);return float(m[1]) if m and '.' in m[1] else int(m[1]) if m and m[1].isdigit() else default
                t['training_script']=str(script);t['training_batch_size']=option('batch_size',64)
                t['learning_rate']=option('learning_rate',.0001);t['max_epochs']=option('train_epochs',30);t['patience']=5
                t['training_test_batch_size']=option('test_batch_size',8)
            else:
                # Read saved training batch size, not manifest's evaluation batch1.
                t['training_args']=json.loads((Path(t['historical_checkpoint'])/'args.json').read_text())
                t['training_batch_size']=t['training_args']['batch_size'];t['max_epochs']=t['training_args']['epochs'];t['patience']=t['training_args']['patience']
                if t['backbone']=='NsDiff' and t['training_args'].get('load_pretrain'):
                    t['shared_original_pretrain']=[str(Path(t['repo'])/'results/runs'/k/t['dataset']/f'w{t["training_args"]["windows"]}h1s{t["horizon"]}'/'1/best_model.pth') for k in ('F','G')]
                    assert all(Path(p).exists() for p in t['shared_original_pretrain'])
            tasks.append(t)
    assert len(tasks)==len({t['id'] for t in tasks})==237
    # Interleave backbones; first8 are normal full training jobs, not smoke runs.
    order={'K2VAE':0,'PPM':1,'TMDM':2,'DiffusionTS':3,'TimeDiff':4,'NsDiff':5}
    tasks.sort(key=lambda t:(t['dataset'] not in ('etth1','ETTh1'),t['seed'],order[t['backbone']],t['dataset'],t['horizon']))
    ROOT.mkdir(exist_ok=True);(ROOT/'tasks').mkdir(exist_ok=True)
    (ROOT/'manifest.json').write_text(json.dumps(tasks,indent=2))
    for t in tasks:(ROOT/'tasks'/f'{t["id"]}.json').write_text(json.dumps(t,indent=2))
    print('FROZEN',len(tasks),'tasks')
if __name__=='__main__':main()
