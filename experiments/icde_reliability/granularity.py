"""Partition ablation using the unchanged full empirical residual operator."""
import numpy as np
from portable_dc import EmpiricalCalibrator, bounded, summaries

VERSION = 'granularity-v1-20261102'
PARTITIONS = [(b, 1) for b in [1, 2, 3, 4, 6, 8, 12, 16]] + [(4, g) for g in [2, 4, 8]]
BUDGETS = [2000, 10000, 50000]


def method_name(b, g, budget):
    return f'b{b}_h{g}_n{budget}'


def fit_partition(rows, targets, horizon, horizon_length, bins, horizon_bins, source_filter=True):
    mean, raw, scale = summaries(rows)
    if bins < 1 or horizon_bins < 1 or horizon_bins > horizon_length:
        raise ValueError('Invalid partition resolution')
    valid = raw > 1.01e-6 if source_filter else np.ones(len(raw), dtype=bool)
    if not valid.any():
        raise ValueError('No valid residual sources')
    residual = (np.asarray(targets)-mean)/scale

    def edges(values):
        out = np.quantile(values, np.linspace(0., 1., bins+1))
        out[0] = -np.inf; out[-1] = np.inf
        return out

    cal = EmpiricalCalibrator(edges(mean), edges(np.log(scale)), bounded(residual[valid]), {},
        int(horizon_length), int(horizon_bins), int(valid.sum()), int((~valid).sum()))
    keys = cal.keys(mean, scale, horizon)
    state_ids = (keys[:, 0]*bins+keys[:, 1])*bins+keys[:, 2]
    counts = np.bincount(state_ids[valid], minlength=bins*bins*horizon_bins)
    order = np.argsort(state_ids[valid], kind='stable')
    sorted_ids = state_ids[valid][order]; values = residual[valid][order]
    for state, start, count in zip(*np.unique(sorted_ids, return_index=True, return_counts=True)):
        if count >= 128:
            key = (int(state)//(bins*bins), int(state)//bins % bins, int(state) % bins)
            cal.local_pools[key] = bounded(values[start:start+count])
    metadata = dict(mean_bins=bins, scale_bins=bins, horizon_bins=horizon_bins,
        nominal_states=bins*bins*horizon_bins, calibration_coordinates=len(mean),
        retained_sources=int(valid.sum()), excluded_sources=int((~valid).sum()),
        eligible_local_pools=len(cal.local_pools), occupied_states=int((counts > 0).sum()),
        source_counts=counts.tolist())
    return cal, metadata


def reconstruct(cal, rows, horizon):
    mean, _, scale = summaries(rows)
    unique, inverse, counts = np.unique(cal.keys(mean, scale, horizon), axis=0,
                                        return_inverse=True, return_counts=True)
    grid = (np.arange(100)+.5)/100
    templates = np.stack([np.quantile(cal.local_pools.get(tuple(key), cal.global_pool), grid)
                          for key in unique])
    fallback = int(sum(n for key, n in zip(unique, counts) if tuple(key) not in cal.local_pools))
    return mean[:, None]+scale[:, None]*templates[inverse], fallback
