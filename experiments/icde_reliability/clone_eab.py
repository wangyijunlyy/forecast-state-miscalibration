"""Make a fresh replay copy while sharing only immutable asset/reference files."""
import argparse
import json
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--destination',type=Path,required=True); args = parser.parse_args()
source = args.source.resolve(); destination = args.destination.resolve()
if (source/'RELOCATION.json').exists() or destination.exists():
    raise ValueError('Use a pristine source and a new replay directory')
rewritable = set(json.loads((source/'relocatable_text_files.json').read_text()))
for directory, _, files in os.walk(source):
    relative = Path(directory).relative_to(source); (destination/relative).mkdir(parents=True,exist_ok=True)
    for filename in files:
        path = Path(directory)/filename; target = destination/relative/filename
        relative_file = str(path.relative_to(source))
        if relative_file in rewritable or any('/'+name+'/' in '/'+relative_file for name in ['data','dataset','datasets']):
            shutil.copy2(path,target)
        else:
            os.link(path,target)
print('Fresh replay tree:',destination)
