"""Package immutable native assets separately from the final traffic models."""
import argparse
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--staging',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);parser.add_argument('--subset',choices=['base','traffic'],required=True)
args=parser.parse_args();staging=args.staging.resolve();args.output.mkdir(parents=True,exist_ok=True)
status=json.loads((staging/'STAGING_STATUS.json').read_text())
if args.subset=='traffic' and status['deepstuq_completed']!=12:
    raise ValueError('Traffic asset archive requires all12 completed native runs')
inventory=json.loads((staging/'asset_inventory.json').read_text())
paths=sorted(name for name in inventory if ('/deepstuq/runs/' in name)==(args.subset=='traffic'))
filelist=args.output/(args.subset+'.members');filelist.write_bytes(b'\0'.join(p.encode() for p in paths)+b'\0')
archive=args.output/(args.subset+'-assets.tar.gz')
with archive.with_suffix('.tmp').open('wb') as output:
    tar=subprocess.Popen(['tar','-C',str(staging),'--null','-T',str(filelist.resolve()),'-cf','-'],stdout=subprocess.PIPE)
    compressor=subprocess.Popen(['pigz','-p','6','-c'],stdin=tar.stdout,stdout=output)
    tar.stdout.close();compressed=compressor.wait();tarcode=tar.wait()
    if compressed or tarcode:
        raise RuntimeError('Asset archiving failed')
archive.with_suffix('.tmp').replace(archive)
record=dict(subset=args.subset,members=len(paths),uncompressed_bytes=sum(inventory[n]['bytes'] for n in paths),
    archive=archive.name,archive_bytes=archive.stat().st_size,public_upload=False)
(args.output/(args.subset+'-assets.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
