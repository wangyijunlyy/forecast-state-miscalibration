"""Independently verify every supplementary cell against raw seeded task scores."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean,stdev


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def rounded(value):
    return str(Decimal(f'{value:.12f}').quantize(Decimal('0.001'),rounding=ROUND_HALF_UP))


def audit(results,paper,default_method="DC16"):
    methods=['Native',default_method,'GlobalDC','GlobalPIT','ConditionalPIT']
    metrics=['CRPS','QICE_percent']
    raw=read(results/'summary/raw_task_metrics.csv')
    groups=defaultdict(dict)
    for row in raw:
        if row['method'] not in methods:
            continue
        for metric in metrics:
            key=(row['backbone'],row['dataset'],int(row['horizon']),row['method'],metric)
            seed=int(row['seed']);value=float(row[metric])
            if seed in groups[key] or not math.isfinite(value):
                raise ValueError('Duplicate or nonfinite raw evidence: '+str(key))
            groups[key][seed]=value
    expected={}
    for key,seeds in groups.items():
        if set(seeds)!={1,2,3}:
            raise ValueError('Missing training seed: '+str(key))
        expected[key]=(mean(seeds.values()),stdev(seeds.values()))
    exported=read(paper/'supplementary_results.csv')
    seen=set();by_backbone=defaultdict(list)
    for row in exported:
        key=(row['backbone'],row['dataset'],int(row['horizon']),row['method'],row['metric'])
        if key in seen or key not in expected or int(row['training_seeds'])!=3:
            raise ValueError('Duplicate or unsupported supplementary cell: '+str(key))
        seen.add(key);m,s=expected[key]
        if row.get('method_name') != ('FSDC' if row['method']==default_method else row['method']):
            raise ValueError('Supplementary method display name differs: '+str(key))
        if not all(math.isclose(float(row[field]),value,rel_tol=0,abs_tol=1e-10)
                   for field,value in [('mean',m),('sample_sd',s)]):
            raise ValueError('Supplementary CSV differs from raw seed recomputation: '+str(key))
        alternatives=[expected[(key[0],key[1],key[2],method,key[4])][0] for method in methods]
        kind='beststat' if math.isclose(m,min(alternatives),rel_tol=0,abs_tol=1e-12) else 'stat'
        by_backbone[key[0]].append((kind,rounded(m),rounded(s)))
    if seen!=set(expected):
        raise ValueError('The supplement does not cover every raw configuration/method/metric')
    configs={key[:3] for key in expected}
    if len(configs)!=83 or len(by_backbone)!=7:
        raise ValueError('Expected seven backbones and 83 configurations')
    details={}
    for backbone,cells in by_backbone.items():
        path=paper/'supplementary_tables'/(backbone.lower()+'.tex');source=path.read_text()
        actual=re.findall(r'\\(beststat|stat)\{(-?\d+\.\d{3})\}\{(\d+\.\d{3})\}',source)
        if actual!=cells or source.count(r'\begin{longtable}')!=1:
            raise ValueError('Supplementary table values, bold marks or table count differ: '+backbone)
        if '\\downarrow' not in source or '\\pm' in source:
            raise ValueError('Supplementary metric/SD formatting missing: '+backbone)
        details[backbone]=dict(configurations=sum(c[0]==backbone for c in configs),mean_sd_cells=len(cells))
    record=dict(all_passed=True,tables=len(details),configurations=len(configs),training_tasks=len(configs)*3,
                methods=methods,metrics=metrics,mean_sd_cells=len(expected),per_backbone=details,
                mean_sd_and_best_marks='independent raw three-seed recomputation')
    (paper/'submission').mkdir(exist_ok=True)
    (paper/'submission/supplement_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--paper',type=Path,required=True)
    parser.add_argument("--default-method",default="DC16")
    args=parser.parse_args();audit(args.results,args.paper,args.default_method)
