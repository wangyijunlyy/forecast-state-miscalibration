"""Reviewer entry points; all generated files stay in the chosen workspace."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def relocate_text(text, home):
    # Relative bundle paths such as 'payload/home/wyj' are not server paths.
    return re.sub(r'(?<!payload)/home/wyj', lambda match: str(home), text)


def prepare(work):
    marker = work/'RELOCATION.json'
    if marker.exists():
        if json.loads(marker.read_text())['root'] != str(work):
            raise ValueError('Prepare a fresh workspace after moving it')
        print('Workspace already prepared:', work)
        return
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO/'native_payload', work/'payload', dirs_exist_ok=True)
    shutil.copytree(REPO/'upstream_source_reconstruction', work/'upstream_source_reconstruction', dirs_exist_ok=True)
    shutil.copytree(REPO/'provenance', work/'provenance', dirs_exist_ok=True)
    home = work/'payload/home/wyj'
    main = home/'icde_reliability_20261102'
    for source, target in [('icde_reliability', main/'code'),
            ('three_seed_reproduction', home/'three_seed_reproduction_20261031/code')]:
        shutil.copytree(REPO/'experiments'/source, target, dirs_exist_ok=True)
    tasks = (REPO/'configs/tasks.json').read_text()
    main.mkdir(parents=True, exist_ok=True)
    (main/'manifest.json').write_text(tasks)
    (home/'three_seed_reproduction_20261031/manifest.json').write_text(tasks)
    (main/'deepstuq/data').mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO/'experiments/icde_reliability/restore_upstream.py'),
        '--bundle', str(work)], check=True)
    # Rewrite source and configuration text only, before any data/checkpoints exist.
    for top in [work/'payload', work/'provenance']:
        for path in top.rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml', '.conf', '.toml'}:
                text = path.read_text()
                rewritten = relocate_text(text, home)
                if rewritten != text:
                    path.write_text(rewritten)
    marker.write_text(json.dumps(dict(root=str(work), source_repository=str(REPO)), indent=2)+'\n')
    print('Prepared native sources and 249 task configurations:', work)


def fetch_data(work):
    if not (work/'RELOCATION.json').exists():
        raise ValueError('Run prepare first')
    cache = work/'download_cache'
    cache.mkdir(exist_ok=True)
    archive = cache/'data-only.tar.gz'
    if not archive.exists():
        with archive.with_suffix('.tmp').open('wb') as stream:
            for part in range(10):
                name = f'data-only.tar.gz.part-{part:03d}'
                url = 'https://github.com/wangyijunlyy/forecast-state-miscalibration/releases/download/v1.0.0/'+name
                print('Downloading', name, flush=True)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, stream)
        archive.with_suffix('.tmp').replace(archive)
    # The data-only release contains native-loader layouts, no experiment records.
    with tarfile.open(archive) as tar:
        for member in tar:
            target = (work/member.name).resolve()
            if not target.is_relative_to(work) or member.issym() or member.islnk():
                raise ValueError('Unexpected data archive member: '+member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    print('Datasets extracted into', work/'payload')


def environment(work, gpu):
    home = work/'payload/home/wyj'
    main = home/'icde_reliability_20261102'
    paths = [main/'code', home/'three_seed_reproduction_20261031/code',
        home/'distribution_binned_diffusion/code',
        home/'miniconda3/envs/k2vae/lib/python3.10/site-packages',
        home/'SPARC/K2VAE_dacu_k2vae_ltsf']
    return dict(os.environ, ICDE_RUN_ROOT=str(main), CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=os.pathsep.join(map(str, paths))+os.pathsep+os.environ.get('PYTHONPATH', ''),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        WANDB_MODE='disabled', MPLBACKEND='Agg')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['list', 'prepare', 'fetch-data', 'train', 'evaluate', 'summarize'])
    p.add_argument('--workspace', type=Path, default=REPO/'work')
    p.add_argument('--task', help='Exact identity from configs/tasks.json')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--controls', action='store_true', help='Fit granularity, temporal and source/assignment controls')
    p.add_argument('--external', action='store_true', help='Fit EMOS, learned PIT and 1000-tree QRF')
    a = p.parse_args(); work = a.workspace.resolve()
    if a.stage == 'list':
        for t in json.loads((REPO/'configs/tasks.json').read_text()):
            print(t['id'])
        return
    if a.stage == 'prepare':
        prepare(work); return
    if a.stage == 'fetch-data':
        fetch_data(work); return
    if a.stage == 'summarize':
        subprocess.run([sys.executable, str(REPO/'scripts/summarize.py'),
            '--root', str(work/'reviewer_outputs')], check=True); return
    home = work/'payload/home/wyj'; code = home/'icde_reliability_20261102/code'
    if not (work/'RELOCATION.json').exists():
        p.error('Run prepare and fetch-data first')
    tasks = json.loads((home/'icde_reliability_20261102/manifest.json').read_text())
    matches = [t for t in tasks if t['id'] == a.task]
    if len(matches) != 1:
        p.error('Select one --task from the list stage')
    task = matches[0]; env = environment(work, a.gpu)
    if a.stage == 'train' and task['backbone'] != 'DeepSTUQ':
        command = [sys.executable, str(code/'from_scratch_eab.py'), '--bundle', str(work),
            '--task', a.task, '--gpu', str(a.gpu), '--stage', 'train']
    else:
        command = [sys.executable, str(code/'reviewer_native.py'), '--workspace', str(work),
            '--task', a.task, '--stage', a.stage]
        if a.controls: command.append('--controls')
        if a.external: command.append('--external')
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
