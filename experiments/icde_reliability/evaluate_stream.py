"""Score paired model outputs through a common NumPy batch-file interface.

Each NPZ batch contains `targets` [B,H,C], `Native` [B,100,H,C], and
one identically shaped array per postprocessor. Historical targets and state
boundaries must be supplied separately; this evaluator never fits them on test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from reliability_metrics import ReliabilityMetrics, TargetReservoir, state_ids


def evaluate(history, states, batches):
    cuts = json.loads(states.read_text())
    edges = []
    for key in ('mean_cuts', 'log_scale_cuts'):
        a = np.asarray(cuts[key], dtype=np.float64)
        if a.ndim != 1 or len(a) not in (3,5) or not np.isfinite(a).all() or np.any(np.diff(a) < 0):
            raise ValueError(f'Expected three or five finite ordered boundaries for {key}')
        edges.append(np.r_[-np.inf, a, np.inf])
    if len(edges[0]) != len(edges[1]):
        raise ValueError('Mean and scale partitions must use the same number of groups')
    n_states = (len(edges[0])-1)*(len(edges[1])-1)
    target_pool = TargetReservoir(4096, seed=50001)
    with np.load(history, allow_pickle=False) as data:
        targets = data['targets']
        if targets.ndim < 2:
            raise ValueError('Historical targets must retain the channel dimension')
        target_pool.update(targets)
    thresholds = target_pool.thresholds()
    accumulators = None; windows = 0; expected_shape = None
    for path in batches:
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)-{'targets'}
            if 'Native' not in names:
                raise ValueError('Every batch must include Native forecasts')
            if accumulators is None:
                accumulators = {name: ReliabilityMetrics(thresholds,n_states=n_states) for name in sorted(names)}
            elif names != set(accumulators):
                raise ValueError('Paired method set differs between batches')
            x = data['Native']; y = data['targets']
            if x.ndim != 4 or x.shape[1] != 100 or x.shape[0] <= 0:
                raise ValueError('Expected nonempty ensembles [B,100,H,C]')
            if y.shape != (x.shape[0], x.shape[2], x.shape[3]) or x.shape[3] != len(thresholds):
                raise ValueError('Forecast/target/historical-channel shape mismatch')
            if expected_shape is not None and x.shape[1:] != expected_shape:
                raise ValueError('Forecast horizon/channel dimensions changed')
            expected_shape = x.shape[1:]
            groups = state_ids(x, *edges)
            for name, accumulator in accumulators.items():
                values = data[name]
                if values.shape != x.shape:
                    raise ValueError('Paired forecast shape mismatch')
                accumulator.update(values, y, groups)
            windows += len(y)
    if accumulators is None:
        raise ValueError('No forecast batches supplied')
    return dict(evaluation='all supplied forecast batches', batch_files=len(batches),
                windows=windows, horizon=expected_shape[1], channels=expected_shape[2],
                state_source=str(states), threshold_source=str(history),
                threshold_rows_seen=target_pool.seen,
                threshold_rows_retained=len(target_pool.values),
                thresholds=thresholds.tolist(), methods={k:v.result() for k,v in accumulators.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('batches', type=Path, nargs='+')
    args = parser.parse_args()
    result = evaluate(args.history, args.states, args.batches)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
