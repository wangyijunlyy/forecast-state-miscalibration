"""Prepare a reviewable public package only after all experiment audits pass."""
import argparse
import json
import os
import shutil
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--staging',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);parser.add_argument('--audits',type=Path,required=True)
parser.add_argument('--reconstruction',type=Path,required=True);args=parser.parse_args()
staging=args.staging.resolve();output=args.output.resolve()
if output.exists() or (staging/'RELOCATION.json').exists():
    raise ValueError('Use pristine staging and a new public candidate directory')
for suite,expected in [('primary',237),('gaussian',237),('compiled',237),('state_controls',237),('deepstuq',12)]:
    record=json.loads((args.audits/suite/'audit.json').read_text())
    if not record['all_tasks_complete'] or record['expected_tasks']!=expected or record['audited_tasks']!=expected:
        raise ValueError('Incomplete audited release input: '+suite)
status=json.loads((staging/'STAGING_STATUS.json').read_text())
if status['deepstuq_completed']!=12 or status['gaussian_controls_completed']!=237 or status['primary_checkpoints']!=237:
    raise ValueError('Staging assets have not been refreshed after final experiments')
rewritable=set(json.loads((staging/'relocatable_text_files.json').read_text()))
for directory,_,files in os.walk(staging):
    relative=Path(directory).relative_to(staging);(output/relative).mkdir(parents=True,exist_ok=True)
    for name in files:
        source=Path(directory)/name;target=output/relative/name
        if str(source.relative_to(staging)) in rewritable:
            shutil.copy2(source,target)
        else:
            os.link(source,target)
metadata=output/'upstream_source_reconstruction'
shutil.copytree(args.reconstruction,metadata,ignore=shutil.ignore_patterns('downloads'),dirs_exist_ok=True)
for record in json.loads((metadata/'source_reconstruction.json').read_text()):
    for name in record['all_files']:
        path=output/record['payload_path']/name
        if not path.is_file():
            raise FileNotFoundError('Staged source is missing before public reconstruction: '+str(path))
        path.unlink()
# The private full snapshot would duplicate upstream source outside the bootstrap.
(output/'provenance/server_source_snapshot.tar.gz').unlink(missing_ok=True)
shutil.copytree(args.audits,output/'audits',dirs_exist_ok=True)
record=dict(publication_authorized=False,uploaded=False,public_url=None,
    scientific_audits_complete=True,primary_tasks=237,gaussian_tasks=237,
    compiled_tasks=237,state_control_tasks=237,deepstuq_tasks=12,
    source_policy='fetch verified public bases and apply measured source changes before relocation',
    manuscript_pdf_included=False)
(output/'PUBLICATION_REVIEW.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
