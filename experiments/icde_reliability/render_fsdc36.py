"""Render canonical DC36 evidence with the established manuscript table layouts.

Legacy layout column keys are translated only in temporary dictionaries.
Raw experiment records and exported evidence retain explicit DC36 identities.
"""
import argparse
import json
from pathlib import Path
import render_tables as tables
import render_eab_controls as eab
import render_supplement as supplement
import audit_supplement as audit_supplement
import render_granularity as granularity
from aggregation import write_csv

ALIASES={'DC36':'DC16','TrainDC36':'TrainDC16','DC144':'DC64',
         'UnfilteredDC36':'UnfilteredDC16','CompiledDC36':'CompiledDC16'}
original_read=tables.read


def layout_rows(path):
    rows=original_read(path);output=[]
    for row in rows:
        changed={k.replace('dc36_','dc16_'):v for k,v in row.items()}
        if 'method' in changed:changed['method']=ALIASES.get(changed['method'],changed['method'])
        output.append(changed)
    return output


def render(results,paper):
    for directory in ['summary','state_controls/summary','compiled/summary','granularity/summary']:
        tables.require_complete(results/directory,249)
        audit=json.loads((results/directory/'audit.json').read_text())
        if audit.get('default_states')!=36:raise ValueError('36-state render requires canonical36 evidence')
    tables.read=layout_rows;eab.read=layout_rows
    outputs=tables.primary(results/'summary')
    outputs.update(tables.complete_method_matrices(results/'summary',results/'gaussian/summary'))
    outputs['temporal_fitting.tex']=tables.temporal_fitting(results/'summary')
    for name,content in outputs.items():
        content=content.replace('DC64','the 144-state ablation without mean control')
        (paper/'tables'/name).write_text(content)
    eab.render(results,paper/'tables')
    counts=json.loads((results/'configuration_wins.json').read_text())['counts']
    (paper/'tables/configuration_wins.tex').write_text('\n'.join(' & '.join([tables.LABELS[b],*map(str,counts[b])])+r' \\' for b in tables.BACKBONES)+'\n')
    supplement.read=original_read
    supplement.render(results,paper,default_method='DC36')
    audit_supplement.audit(results,paper,default_method='DC36')
    granularity.read=original_read
    granularity.render(results/'granularity/summary',results/'summary',paper/'tables/granularity.tex',
                       native_directory=results/'state_controls/summary',default_bins=6)
    record=dict(default_states=36,all_primary_tables_from_new_complete_test_records=True,
        sensitivity_and_source_controls='6x6 with horizon groups 1/2/4/8',
        normalization='paired Native within backbone/run, equal backbone average',
        legacy_layout_aliases='temporary dictionaries only; raw and supplementary CSV identities remainDC36')
    (paper/'submission/fsdc36_table_render.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True);a=p.parse_args();render(a.results,a.paper)
