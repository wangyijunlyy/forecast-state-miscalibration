"""Assemble a private, relocatable replay staging tree on the experiment server."""
import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path('/home/wyj/icde_reliability_20261102')
OLD = Path('/home/wyj/three_seed_reproduction_20261031')
SF = Path('/home/wyj/source_filtered_reproduction')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--stage',action='store_true',help='Allow pending baseline/control experiments; never declares a release complete')
    args = parser.parse_args(); destination = args.destination.resolve()
    if destination in (ROOT,OLD,SF) or ROOT in destination.parents:
        raise ValueError('Artifact staging must be separate from live experiment outputs')
    destination.mkdir(parents=True,exist_ok=True); payload = destination/'payload'
    if (destination/'RELOCATION.json').exists():
        raise ValueError('Refresh the pristine staging tree, not an already relocated replay copy')
    provenance = json.loads((ROOT/'provenance/source_provenance.json').read_text())
    mapping = {r['alias']:Path(r['source_path']) for r in provenance}; rewritable = set(); assets = {}
    previous_assets=json.loads((destination/'asset_inventory.json').read_text()) if (destination/'asset_inventory.json').exists() else {}

    def target(path):
        return payload/str(path).lstrip('/')

    def copy_text(path):
        out = target(path); out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,out); rewritable.add(str(out.relative_to(destination)))

    def asset(path):
        out = target(path); out.parent.mkdir(parents=True,exist_ok=True)
        # Some native dataset loaders rewrite extracted files on construction.
        # Keep staging independent of every live input inode.
        if not out.exists() or out.stat().st_ino == path.stat().st_ino or out.stat().st_size != path.stat().st_size:
            for attempt in range(3):
                before=path.stat(); temporary=out.with_name(out.name+'.asset-copy')
                shutil.copy2(path,temporary); after=path.stat()
                if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                    temporary.replace(out); break
                temporary.unlink();time.sleep(2)
            else:
                raise ValueError('Input changed repeatedly during asset copy: '+str(path))
        assets[str(out.relative_to(destination))] = dict(bytes=out.stat().st_size)

    # Source files are copies so relocation can never alter the live source.
    with tarfile.open(ROOT/'provenance/server_source_snapshot.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if parts[0] not in mapping or '..' in parts:
                raise ValueError('Unexpected source snapshot member')
            original = mapping[parts[0]].joinpath(*parts[1:]); out = target(original)
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(archive.extractfile(member).read())
            rewritable.add(str(out.relative_to(destination)))
    tasks = json.loads((ROOT/'manifest.json').read_text())
    copy_text(ROOT/'manifest.json')
    copy_text(OLD/'manifest.json')
    for task in tasks:
        native = Path(task['run_root']); training = json.loads((native/'training.json').read_text())
        if not training['training_complete']:
            raise ValueError('Missing native checkpoint: '+task['id'])
        checkpoint = Path(training['checkpoint']); asset(checkpoint)
        # The frozen diffusion interfaces load their conditional predictors too.
        if task['backbone'] in ['NsDiff','TMDM','DiffusionTS','TimeDiff']:
            for path in checkpoint.parent.iterdir():
                if path.name == 'args.json':
                    copy_text(path)
                elif path.name in ['cond_pred_model.pth','cond_pred_model_g.pth']:
                    asset(path)
        for name in ['training.json','training_history.json','task.json','fit_metadata.json','results.json']:
            if (native/name).is_file():
                copy_text(native/name)
        if (native/'calibrators.pkl').is_file():
            asset(native/'calibrators.pkl')
        for original in task.get('shared_original_pretrain',[]):
            asset(Path(original))
        for name in ['results.json','fit_metadata.json']:
            copy_text(SF/'runs'/task['id']/name)
        asset(SF/'runs'/task['id']/'calibrators.pkl')
    # Dereference dataset links, preserving each loader's expected layout.
    data_roots = [Path('/home/wyj/K2VAE/datasets'),Path('/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf/datasets'),Path('/home/wyj/SPARC/PPM/dataset'),
        Path('/home/wyj/SPARC/NsDiff/data'),Path('/home/wyj/SPARC/NsDiff_DiffusionTS_SPARC_PIT_20260716/data'),
        Path('/home/wyj/SPARC/NsDiff_TimeDiff_SPARC_PIT_Fresh/data'),ROOT/'deepstuq/data']
    for directory in data_roots:
        for current,dirs,files in os.walk(directory,followlinks=True):
            dirs[:] = [d for d in dirs if d not in ['.git','__pycache__']]
            if Path(current)==directory and directory!=ROOT/'deepstuq/data':
                needed={'ETT-small','ETTh1','ETTh2','ETTm1','ETTm2','ExchangeRate','exchange_rate',
                    'ILI','illness','electricity','Electricity','traffic','Traffic','weather','solar_AL'}
                dirs[:]=[d for d in dirs if d in needed]
            for filename in files:
                path = Path(current)/filename
                if path.is_file() and (directory != ROOT/'deepstuq/data' or path.suffix == '.npz'):
                    asset(path)
    complete_deep = []
    for run in sorted((ROOT/'deepstuq/runs').iterdir()):
        if not (run/'results.json').exists():
            continue
        complete_deep.append(run.name)
        for path in run.iterdir():
            if path.suffix == '.json':
                copy_text(path)
            elif path.name in ['model.pt','pretrained.pt','calibrators.pkl']:
                asset(path)
    gaussian_count = sum((ROOT/'gaussian_runs'/t['id']/'results.json').exists() for t in tasks)
    if not args.stage and (len(complete_deep) != 12 or gaussian_count != 237):
        raise ValueError('Release requires all12 native traffic runs and all237 Gaussian control tasks')
    # Original quality outputs remain reference-only; active replay runs start empty.
    references = destination/'reference_records'
    for group, source_group in [('runs','runs'),('gaussian_runs','gaussian_runs'),
                                ('compiled/runs','compiled_deployment/runs'),('state_controls/runs','state_controls/runs'),('deepstuq/runs','deepstuq/runs')]:
        source = ROOT/source_group
        for run in source.iterdir():
            if not run.is_dir():
                continue
            out = references/group/run.name; out.mkdir(parents=True,exist_ok=True)
            for path in run.iterdir():
                if path.suffix == '.json' or (group in ['compiled/runs','state_controls/runs'] and path.suffix == '.npz'):
                    shutil.copy2(path,out/path.name)
    shutil.copytree(ROOT/'provenance',destination/'provenance',dirs_exist_ok=True)
    (destination/'relocatable_text_files.json').write_text(json.dumps(sorted(rewritable),indent=2)+'\n')
    pruned=[]
    for relative in sorted(set(previous_assets)-set(assets)):
        path=destination/relative
        if payload not in path.parents:
            raise ValueError('Old asset inventory escaped the owned staging payload')
        path.unlink(missing_ok=True);pruned.append(relative)
    (destination/'asset_inventory.json').write_text(json.dumps(assets,indent=2)+'\n')
    status = dict(stage_only=args.stage, primary_checkpoints=len(tasks), deepstuq_completed=len(complete_deep),
        gaussian_controls_completed=gaussian_count, asset_paths=len(assets),
        logical_asset_bytes=sum(v['bytes'] for v in assets.values()),
        removed_unused_or_replaced_asset_paths=pruned,
        evaluation_outputs='reference_records; payload primary/runs starts empty to force actual replay',
        public_url=None, release_complete=False)
    (destination/'STAGING_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
