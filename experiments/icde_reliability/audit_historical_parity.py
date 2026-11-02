"""Compare current outputs to the previously completed source-filtered suite."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    maxima = {}; audited = []; failures = []
    for path in sorted(args.current.glob('*/results.json')):
        current = json.loads(path.read_text()); task = current['task']['id']
        old = json.loads((args.historical/task/'results.json').read_text())
        errors = []
        for current_name, previous_name in [('Native', 'Native'), ('TrainDC16', 'DC16')]:
            for metric in ['CRPS', 'MSE', 'MAE', 'RMSE', 'MedianMAE', 'QICE_percent']:
                previous = old['common_normalized'][previous_name][metric]
                delta = abs(current['methods'][current_name][metric]-previous)
                key = current_name+'/'+metric; maxima[key] = max(maxima.get(key, 0.), delta)
                if delta > 1e-9:
                    errors.append(dict(method=current_name, metric=metric, absolute_difference=delta))
        for key in ['test_batches', 'test_windows']:
            if current[key] != old[key]:
                errors.append(dict(field=key, current=current[key], historical=old[key]))
        if errors:
            failures.append(dict(task=task, errors=errors))
        else:
            audited.append(task)
    report = dict(audited_tasks=len(audited), maximum_absolute_differences=maxima,
                  failures=failures, task_ids=audited,
                  comparison='Native and reused training-fit DC16 versus the prior full-test suite')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'task_ids'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
