"""Inventory retained native checkpoints by documented configuration, not scores."""
import csv
import json
import re
from pathlib import Path


DATASETS=['ETTh1','ETTh2','ETTm1','ETTm2','Electricity','ExchangeRate','ILI','SolarEnergy','Traffic']
ROOT=Path('/home/wyj/SPARC')


def manifest():
    tasks=[]
    refs=list(csv.DictReader((Path(__file__).parent/'reference/nsdiff_sparc_pit_paper_matched_results.csv').open()))
    ns={r['dataset']:r for r in refs}
    archived=ROOT/'_archive/diffusionts_before_20260716_20260716_112921/NsDiff/results/DiffusionTS'
    for b,repo,family in [('NsDiff','NsDiff','NsDiff4'),('TMDM','NsDiff','TMDM'),
        ('DiffusionTS','NsDiff_DiffusionTS_SPARC_PIT_20260716','DiffusionTS'),
        ('TimeDiff','NsDiff_TimeDiff_SPARC_PIT_Fresh','TimeDiff')]:
        root=ROOT/repo
        for ds in DATASETS:
            candidates=[];selection='retained seed1 checkpoint'
            base=root/'results/runs'/family/ds
            if b=='NsDiff':
                candidates=list(base.glob('*/'+ns[ds]['baseline_checkpoint']))
                selection='paper-matched seed1 checkpoint identifier'
            else:
                for p in base.glob('*/*/args.json'):
                    args=json.loads(p.read_text())
                    if b=='TMDM' and (args.get('beta_end')!=.5 or args.get('diffusion_steps')!=100):continue
                    if not any((p.parent/name).exists() for name in ['model.pth','best_model.pth']):continue
                    candidates.append(p.parent)
            if b=='DiffusionTS' and not candidates:
                candidates=[p.parent for p in (archived/ds).glob('*/*/args.json') if (p.parent/'best_model.pth').exists()]
                selection='retained archived seed1 checkpoint; same saved model configuration'
            if b=='TimeDiff':
                log=root/'logs'/f'timeDiff_sparc_{ds}_gpu0.log'
                logs=list((root/'logs').glob(f'timeDiff_sparc_{ds}_gpu*.log'))
                chosen=None;bs=None
                for log in logs:
                    text=log.read_text(errors='replace')
                    matches=re.findall(r'sparc_start[^\n]*batch_size=(\d+)[^\n]*checkpoint=(\S+)',text)
                    for batch,ck in matches:
                        path=(root/ck).resolve()
                        if path in [p.resolve() for p in candidates]:chosen=path;bs=int(batch)
                if chosen is not None:candidates=[chosen];selection='checkpoint bound to retained managed SPARC-PIT run log'
            assert len(candidates)==1,(b,ds,[str(p) for p in candidates])
            ck=candidates[0];args=json.loads((ck/'args.json').read_text())
            cache_root=''
            if b=='TimeDiff':
                assert bs is not None
                args['batch_size']=bs
                cache=root/'results/sparc_pit'/f'TimeDiff_{ds}_w168h1s{args["pred_len"]}_seed1_managed/prediction_cache'/f'{ds}_w168h1s{args["pred_len"]}_seed1'
                assert cache.exists(),cache
                cache_root=str(cache)
            else:
                # Evaluation batch size affects memory; preserve the full
                # original fast/non-overlapping test dataset, without limits.
                args['batch_size']=1
            tasks.append(dict(id=f'{b}_{ds}_h{args["pred_len"]}_seed1',backbone=b,dataset=ds,
                horizon=args['pred_len'],seed=1,backbone_seed=1,repo=str(root),checkpoint=str(ck),
                checkpoint_selection=selection,model_args=args,test_cache_root=cache_root,
                sampling_chunk_size=100 if ds not in ['Electricity','SolarEnergy','Traffic'] else (20 if b in ['TimeDiff','DiffusionTS'] else 10),
                fit_windows=100,fit_batch_size=1,fit_split='train',num_samples=100))
            if b in ['NsDiff','TMDM'] and ds=='SolarEnergy':
                tasks[-1]['resolved_data_root']=str(ROOT/'NsDiff_DiffusionTS_SPARC_PIT_20260716/data')
            if b=='TimeDiff' and ds in ['Electricity','SolarEnergy','Traffic','ExchangeRate']:
                tasks[-1]['native_sampling_execution']='batched_original_forward_tf32_disabled'
    return tasks


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=manifest();a.root.mkdir(parents=True,exist_ok=True);(a.root/'tasks').mkdir(exist_ok=True)
    (a.root/'manifest.json').write_text(json.dumps(rows,indent=2))
    for t in rows:(a.root/'tasks'/f'{t["id"]}.json').write_text(json.dumps(t,indent=2))
    print('INVENTORY',len(rows),'retained task checkpoints')
