"""Reconstruct observed third-party sources from verified public bases and patches."""
import argparse
import difflib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SPECS={
    'nsdiff_tmdm':('wwy155/NsDiff','dd7b97ae0c47629986f72ba01effbbdc1d44be2b'),
    'diffusionts':('wwy155/NsDiff','dd7b97ae0c47629986f72ba01effbbdc1d44be2b'),
    'timediff':('wwy155/NsDiff','dd7b97ae0c47629986f72ba01effbbdc1d44be2b'),
    'deepstuq_author':('WeizhuQIAN/DeepSTUQ_Pytorch','4f2281531d07b48ba81aa1c519a90e65a71993e2')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);cache=args.output/'downloads';cache.mkdir(exist_ok=True)
    provenance=json.loads((args.provenance/'source_provenance.json').read_text())
    records=[]
    with tarfile.open(args.provenance/'server_source_snapshot.tar.gz') as observed:
        members={m.name:m for m in observed.getmembers() if m.isfile()}
        for alias,(repository,revision) in SPECS.items():
            original=next(r for r in provenance if r['alias']==alias)
            archive=cache/(revision+'.zip');url=f'https://codeload.github.com/{repository}/zip/{revision}'
            if not archive.exists():
                with urllib.request.urlopen(url,timeout=60) as response, archive.with_suffix('.tmp').open('wb') as output:
                    shutil.copyfileobj(response,output)
                archive.with_suffix('.tmp').replace(archive)
            selected=sorted(n for n in members if n.startswith(alias+'/'))
            expected={};parts=[];base_files=[];new_files=[]
            with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(prefix='eab-source-') as temporary:
                root=Path(temporary);prefix=source.namelist()[0].split('/')[0]+'/'
                for name in selected:
                    relative=name[len(alias)+1:];target=observed.extractfile(members[name]).read()
                    expected[relative]=target;member=prefix+relative
                    exists=member in source.namelist()
                    base=source.read(member) if exists else b''
                    if exists:
                        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base);base_files.append(relative)
                    else:
                        new_files.append(relative)
                    if base==target:
                        continue
                    difference=difflib.unified_diff(base.decode('utf-8').splitlines(keepends=True),
                        target.decode('utf-8').splitlines(keepends=True),
                        fromfile='a/'+relative if exists else '/dev/null',tofile='b/'+relative,n=3)
                    for line in difference:
                        parts.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
                patch=args.output/(alias+'.patch');patch.write_text(''.join(parts))
                if parts:
                    subprocess.run(['patch','--batch','-p1','-i',str(patch.resolve())],cwd=root,check=True,stdout=subprocess.DEVNULL)
                for relative,target in expected.items():
                    if not (root/relative).exists() and not target:
                        (root/relative).parent.mkdir(parents=True,exist_ok=True);(root/relative).touch()
                    if (root/relative).read_bytes()!=target:
                        raise ValueError('Source reconstruction mismatch: '+alias+'/'+relative)
                records.append(dict(alias=alias,public_repository=repository,public_revision=revision,archive_url=url,
                    observed_local_git_revision=original['git_revision'],payload_path='payload/'+original['source_path'].lstrip('/'),
                    base_files=base_files,new_files=new_files,all_files=sorted(expected),patch=patch.name,
                    empty_files=sorted(name for name,value in expected.items() if not value),
                    source_files=len(expected),reconstructed_bytes=sum(map(len,expected.values())),
                    exact_observed_source_reconstruction=True))
                print('SOURCE_RECONSTRUCTION_EXACT',alias,len(expected),len(''.join(parts)),flush=True)
    (args.output/'source_reconstruction.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
