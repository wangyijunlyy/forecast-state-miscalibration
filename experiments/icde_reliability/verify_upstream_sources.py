"""Verify publicly fetchable upstream revisions before preparing source patches."""
import argparse
import json
import os
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--provenance',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
records=json.loads(args.provenance.read_text());args.output.mkdir(parents=True,exist_ok=True);report=[]
for name in ['nsdiff_tmdm','diffusionts','deepstuq_author']:
    record=next(r for r in records if r['alias']==name);repo=args.output/name;repo.mkdir(exist_ok=True)
    subprocess.run(['git','init','--bare',str(repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    environment=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    url=record['git_origin'];revision=record['git_revision']
    if name=='deepstuq_author':
        url=url or 'https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch.git'
        revision=revision or '4f2281531d07b48ba81aa1c519a90e65a71993e2'
    result=subprocess.run(['git','-C',str(repo),'-c','protocol.version=2','fetch','--filter=blob:none',
        '--depth=1',url,revision],env=environment,capture_output=True,text=True,timeout=120)
    row=dict(alias=name,url=url,revision=revision,local_git_revision=record['git_revision'],public_fetch_succeeded=result.returncode==0,
        output=result.stderr[-2000:])
    if name=='nsdiff_tmdm' and result.returncode:
        base='dd7b97ae0c47629986f72ba01effbbdc1d44be2b'
        fallback=subprocess.run(['git','-C',str(repo),'-c','protocol.version=2','fetch','--filter=blob:none',
            '--depth=1',url,base],env=environment,capture_output=True,text=True,timeout=120)
        row.update(public_base_revision=base,public_base_fetch_succeeded=fallback.returncode==0,
            reconstruction='apply actual source differences to the publicly fetchable base; local revision remains recorded separately')
    report.append(row);(args.output/'public_revision_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['public_fetch_succeeded'],flush=True)
