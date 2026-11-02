"""Restore public upstream source files and actual local changes before relocation."""
import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
bundle=args.bundle.resolve();metadata=bundle/'upstream_source_reconstruction'
if (bundle/'RELOCATION.json').exists():
    raise ValueError('Restore upstream sources before relocating the bundle')
records=json.loads((metadata/'source_reconstruction.json').read_text());cache=bundle/'download_cache';cache.mkdir(exist_ok=True)
for record in records:
    target=bundle/record['payload_path'];target.mkdir(parents=True,exist_ok=True)
    if any((target/f).exists() for f in record['all_files']):
        raise ValueError('Upstream destination is not empty: '+record['alias'])
    archive=cache/(record['public_revision']+'.zip')
    if not archive.exists():
        with urllib.request.urlopen(record['archive_url'],timeout=60) as response,archive.with_suffix('.tmp').open('wb') as output:
            shutil.copyfileobj(response,output)
        archive.with_suffix('.tmp').replace(archive)
    with zipfile.ZipFile(archive) as source:
        prefix=source.namelist()[0].split('/')[0]+'/'
        for filename in record['base_files']:
            out=target/filename;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(source.read(prefix+filename))
    patch=metadata/record['patch']
    if patch.stat().st_size:
        subprocess.run(['patch','--batch','-p1','-i',str(patch)],cwd=target,check=True)
    for filename in record['all_files']:
        out=target/filename
        if not out.exists():
            if filename not in record['empty_files']:
                raise FileNotFoundError('Patch did not restore '+record['alias']+'/'+filename)
            out.parent.mkdir(parents=True,exist_ok=True);out.touch()
    if sum((target/f).stat().st_size for f in record['all_files'])!=record['reconstructed_bytes']:
        raise ValueError('Reconstructed source byte count mismatch')
    print('Restored',record['alias'],record['source_files'],'source files')
