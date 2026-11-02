"""Audit raw records and regenerate the EAB tables and quantitative figures."""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

CODE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=CODE.parents[1])
parser.add_argument('--records',type=Path);parser.add_argument('--manifest',type=Path)
parser.add_argument('--training-runs',type=Path,help='Complete training-fitted DC16/DC64 records for configuration-win counts')
parser.add_argument('--primary-only',action='store_true',help='Skip the original native traffic-training audit')
args=parser.parse_args();workspace=args.workspace.resolve();local=workspace/'experiments/icde_reliability/results'
records=args.records.resolve() if args.records else (workspace/'reference_records' if (workspace/'reference_records').exists() else local)
current_manifest=workspace/'experiments/icde_reliability/current_manifest.json'
manifest=args.manifest or (current_manifest if current_manifest.exists() else workspace/'experiments/source_filtered_reproduction/results/manifest.json')
training=args.training_runs or (workspace/'payload/home/wyj/source_filtered_reproduction/runs' if (workspace/'payload').exists() else workspace/'experiments/source_filtered_reproduction/results/runs')
summary=local if records==local else workspace/'analysis_summaries';paper=workspace/'ICDE'
if (records/'deepstuq_backbone/manifest.json').exists():
    from merge_deepstuq_records import prepare_records
    view = 'seven_backbone_records_h192' if (records/'nsdiff_h192/manifest.json').exists() else 'seven_backbone_records'
    records, manifest, training = prepare_records(records,manifest,training,workspace/view)
expected_tasks=len(json.loads(manifest.read_text()))
if expected_tasks != 249:
    raise ValueError('The current manuscript requires all249 tasks from seven backbones')
native_layout=(records/'gaussian/runs').exists()
gaussian=records/('gaussian/runs' if native_layout else 'gaussian_runs')
compiled=records/('compiled_deployment/runs' if (records/'compiled_deployment/runs').exists() else 'compiled/runs')
state=records/'state_controls/runs';traffic=records/'deepstuq/runs'


def execute(script,arguments):
    subprocess.run([sys.executable,str(CODE/script),*map(str,arguments)],check=True)


def require(directory,count):
    value=json.loads((directory/'audit.json').read_text())
    if not value['all_tasks_complete'] or value['expected_tasks']!=count or value['audited_tasks']!=count:
        raise ValueError('Cannot render a complete suite from partial records: '+str(directory))


def configuration_wins():
    from render_tables import BACKBONES, LABELS
    groups=defaultdict(dict);tasks=json.loads(manifest.read_text())
    for task in tasks:
        value=json.loads((training/task['id']/'results.json').read_text())
        if not value['full_test_split'] or value['task']['id']!=task['id']:
            raise ValueError('Incomplete training-fit record: '+task['id'])
        methods=value['common_normalized']
        primary=json.loads((records/'runs'/task['id']/'results.json').read_text())
        for old,new in [('Native','Native'),('DC16','TrainDC16')]:
            for metric in ['CRPS','QICE_percent']:
                if not math.isclose(methods[old][metric],primary['methods'][new][metric],rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Training-fit count input differs from paired primary suite')
        key=(task['backbone'],task['dataset'],task['horizon'])
        if task['seed'] in groups[key]:
            raise ValueError('Duplicate training seed')
        groups[key][task['seed']]=methods
    counts={b:[0,0,0,0] for b in BACKBONES}
    for (backbone,_,_),seeds in groups.items():
        if set(seeds)!={1,2,3}:
            raise ValueError('Configuration wins require all three training seeds')
        values={(m,k):mean(v[m][k] for v in seeds.values()) for m in ['Native','DC16','DC64'] for k in ['CRPS','QICE_percent']}
        row=counts[backbone];row[0]+=1
        row[1]+=values['DC16','CRPS']<values['Native','CRPS']
        row[2]+=values['DC16','QICE_percent']<values['Native','QICE_percent']
        row[3]+=values['DC16','CRPS']<values['DC64','CRPS']
    (paper/'tables/configuration_wins.tex').write_text('\n'.join(' & '.join([LABELS[b],*map(str,counts[b])])+r' \\' for b in BACKBONES)+'\n')
    (summary/'configuration_wins.json').write_text(json.dumps(dict(tasks=len(tasks),configurations=len(groups),counts=counts),indent=2)+'\n')


execute('aggregation.py',['--runs',records/'runs','--manifest',manifest,'--output',summary/'summary'])
execute('aggregate_gaussian.py',['--runs',gaussian,'--primary-runs',records/'runs','--manifest',manifest,'--output',summary/'gaussian/summary'])
execute('aggregate_compiled.py',['--runs',compiled,'--primary-runs',records/'runs','--manifest',manifest,'--output',summary/'compiled/summary'])
execute('aggregate_state_controls.py',['--runs',state,'--primary-runs',records/'runs','--compiled-runs',compiled,'--manifest',manifest,'--output',summary/'state_controls/summary'])
for directory in ['summary','gaussian/summary','compiled/summary','state_controls/summary']:
    require(summary/directory,expected_tasks)
arguments=['--primary',summary/'summary','--gaussian',summary/'gaussian/summary','--output',paper/'tables']
if args.primary_only:
    arguments+=['--primary-only']
else:
    execute('aggregate_deepstuq.py',['--runs',traffic,'--output',summary/'deepstuq/summary'])
    require(summary/'deepstuq/summary',12);arguments+=['--deepstuq',summary/'deepstuq/summary']
execute('render_tables.py',arguments)
configuration_wins()
execute('render_eab_controls.py',['--results',summary,'--output',paper/'tables'])
if (records/'granularity/manifest.json').exists():
    execute('aggregate_granularity.py',['--results',records/'granularity','--output',summary/'granularity/summary'])
    require(summary/'granularity/summary',expected_tasks)
    execute('render_granularity.py',['--results',summary/'granularity/summary',
        '--primary',summary/'summary','--output',paper/'tables/granularity.tex'])
environment=dict(os.environ,ICDE_ANALYSIS_RESULTS=str(records))
subprocess.run([sys.executable,str(paper/'scripts/plot_eab_evidence.py')],env=environment,check=True,stdout=subprocess.DEVNULL)
print('Rebuilt seven-backbone numerical tables and both quantitative figures. Native traffic-training audit included:',not args.primary_only)
