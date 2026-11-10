"""Aggregate freshly generated task scores with three-run sample SDs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/icde_reliability'))
from aggregation import configuration_statistics, backbone_statistics, write_csv
from aggregate_fsdc36 import metric_row
from aggregate_final_fspc import average_ratios

METRICS = ['MSE', 'MAE', 'RMSE', 'MedianMAE', 'CRPS', 'QICE_percent',
    'state_error_pp', 'pooled_error_pp', 'Brier80', 'Brier90', 'Brier95', 'BrierMean']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root.resolve(); out = root/'analysis'; out.mkdir(exist_ok=True)
    for filename, suffix in [('results.json', 'full'), ('retained_results.json', 'retained')]:
        rows = []
        for path in sorted((root/'runs').glob('*/'+filename)):
            result = json.loads(path.read_text())
            task = result.get('task')
            if task is None:
                full = json.loads(path.with_name('results.json').read_text()); task = full['task']
            for method, value in result['methods'].items(): rows.append(metric_row(task, method, value))
        if not rows: continue
        write_csv(out/(suffix+'_raw_task_metrics.csv'), rows)
        # Existing aggregators enforce complete three-run configurations; no partial SDs.
        configs = configuration_statistics(rows, METRICS)
        stats = backbone_statistics(rows, METRICS)
        write_csv(out/(suffix+'_configuration_statistics.csv'), configs)
        write_csv(out/(suffix+'_backbone_statistics.csv'), stats)
        # This study's ratio routine explicitly handles the four factorial arms.
        if suffix == 'retained' and len({r['backbone'] for r in rows}) == 7:
            subset = [r for r in rows if r['method'] in ['Native', 'MeanOnly', 'ShapeOnly', 'FSPC']]
            write_csv(out/'factorial_native_ratios.csv', average_ratios(subset))
        print(suffix, len({r['id'] for r in rows}), 'tasks aggregated into', out)


if __name__ == '__main__':
    main()
