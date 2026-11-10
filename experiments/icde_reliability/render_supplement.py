"""Render one complete configuration-level comparison table per backbone."""
import argparse
import csv
from pathlib import Path
from render_tables import BACKBONES, LABELS, read, require_complete, comparison_cells

METHODS = ['Native','DC16','GlobalDC','GlobalPIT','ConditionalPIT']
METRICS = [('CRPS',r'CRPS $\downarrow$'),('QICE_percent',r'QICE $\downarrow$')]
DATASETS = {'etth1':'ETTh1','etth2':'ETTh2','ettm1':'ETTm1','ettm2':'ETTm2',
            'electricity_ltsf':'Electricity','ECL':'Electricity','exchange_ltsf':'Exchange',
            'ExchangeRate':'Exchange','weather_ltsf':'Weather','weather':'Weather',
            'traffic_ltsf':'Traffic','traffic':'Traffic','illness_ltsf':'ILI'}
ORDER = ['ETTh1','ETTh2','ETTm1','ETTm2','Electricity','Exchange','Weather','Traffic',
         'ILI','SolarEnergy','PEMS03','PEMS04','PEMS07','PEMS08']


def render(results, paper, default_method="DC16"):
    methods = [default_method if m == "DC16" else m for m in METHODS]
    require_complete(results/'summary',249)
    rows = read(results/'summary/complete_configuration_statistics.csv')
    stats = {}
    for row in rows:
        if row['method'] not in methods:
            continue
        key = row['backbone'],row['dataset'],int(row['horizon']),row['method']
        if key in stats or int(row['n_training_seeds']) != 3:
            raise ValueError('Duplicate configuration or incomplete three-run summary: '+str(key))
        stats[key] = row
    output = paper/'supplementary_tables'
    output.mkdir(exist_ok=True)
    header = (r'Dataset & $H$ & Metric & Native & \shortstack{w/o Mean Control} & '
              r'\shortstack{Global\\DC} & \shortstack{Global\\PIT} & '
              r'\shortstack{Cond.\\PIT} \\')
    exported = []
    for backbone in BACKBONES:
        configs = {(d,h) for b,d,h,m in stats if b==backbone and m=='Native'}
        configs = sorted(configs,key=lambda v:(ORDER.index(DATASETS.get(v[0],v[0])),v[1]))
        units = 'original traffic-flow units' if backbone=='DeepSTUQ' else 'model-normalized units'
        caption = (LABELS[backbone]+': all '+str(len(configs))+
                   ' dataset--horizon configurations on complete paired native test iterators. '
                   'All corrections use validation fitting. CRPS uses '+units+
                   '; QICE is in percentage points. Subscripts give sample SD over three training runs. '
                   'Bold marks the lowest unrounded mean in each row.')
        lines = [r'\begin{longtable}{lll'+'r'*len(METHODS)+'}',
                 r'\caption{'+caption+r'}\label{tab:supp-'+backbone.lower()+r'}\\',
                 r'\toprule',header,r'\midrule',r'\endfirsthead',
                 r'\multicolumn{8}{l}{\tablename~\thetable\ (continued)} \\',
                 r'\toprule',header,r'\midrule',r'\endhead',
                 r'\midrule',r'\multicolumn{8}{r}{Continued on next page} \\',r'\endfoot',
                 r'\bottomrule',r'\endlastfoot']
        previous_dataset = None
        for dataset,horizon in configs:
            if previous_dataset is not None and previous_dataset != dataset:
                lines.append(r'\midrule')
            for index,(metric,label) in enumerate(METRICS):
                records = [stats[backbone,dataset,horizon,m] for m in methods]
                left = [DATASETS.get(dataset,dataset),str(horizon),label] if index==0 else ['', '',label]
                lines.append(' & '.join(left+comparison_cells(records,metric))+(r' \\*' if index==0 else r' \\'))
                for method,record in zip(methods,records):
                    exported.append(dict(backbone=backbone,dataset=dataset,horizon=horizon,
                                         method=method,method_name=('w/o Mean Control' if method==default_method else method),
                                         metric=metric,mean=record[metric+'_mean'],
                                         sample_sd=record[metric+'_std'],training_seeds=3))
            previous_dataset = dataset
        lines += [r'\end{longtable}','']
        (output/(backbone.lower()+'.tex')).write_text('\n'.join(lines))
    with (paper/'supplementary_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    print('Rendered seven backbone tables and '+str(len(exported))+' mean/SD cells.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--paper',type=Path,required=True)
    parser.add_argument("--default-method",default="DC16")
    args=parser.parse_args();render(args.results,args.paper,args.default_method)
