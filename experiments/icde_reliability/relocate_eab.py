"""Rebase the staged experiment without editing native data or model binaries."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(); parser.add_argument('--bundle',type=Path,required=True)
args = parser.parse_args(); root = args.bundle.resolve()
marker = root/'RELOCATION.json'
if marker.exists():
    previous = json.loads(marker.read_text())['root']
    if previous != str(root):
        raise ValueError('Move a pristine bundle, then relocate; already-relocated source embeds another root')
    print('Already relocated:',root)
else:
    changed = 0
    for relative in json.loads((root/'relocatable_text_files.json').read_text()):
        path = root/relative
        try:
            value = path.read_text()
        except UnicodeDecodeError:
            continue
        rewritten = value.replace('/home/wyj',str(root/'payload/home/wyj'))
        if rewritten != value:
            path.write_text(rewritten); changed += 1
    (root/'RELOCATION.json').write_text(json.dumps(dict(root=str(root),changed_text_files=changed),indent=2)+'\n')
    print('Rebased text files:',changed)
home = root/'payload/home/wyj'
env = [f'export ICDE_RUN_ROOT="{home}/icde_reliability_20261102"',
    f'export PYTHONPATH="{home}/miniconda3/envs/k2vae/lib/python3.10/site-packages:${{PYTHONPATH:-}}"',
    'export OMP_NUM_THREADS=2','export MKL_NUM_THREADS=2','export OPENBLAS_NUM_THREADS=1']
(root/'activate.sh').write_text('\n'.join(env)+'\n')
print('Source activate.sh before running experiment commands.')
