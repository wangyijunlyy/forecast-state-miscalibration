"""NumPy deployment of the actual source-filtered empirical calibration.

This module contains the complete empirical fitting and reconstruction rule.
It needs neither a model repository nor a pickle import path. Callers supply
the retained historical fitting rows and preserve their real horizon IDs.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def bounded(values):
    a = np.sort(np.asarray(values, dtype=np.float64))
    return a if len(a) <= 20000 else a[np.linspace(0, len(a)-1, 20000).astype(int)]


def edges(values, bins=4):
    a = np.quantile(values, np.linspace(0., 1., bins+1))
    a[0] = -np.inf; a[-1] = np.inf
    return a


def summaries(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 100 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Expected finite nonempty coordinate rows [N,100]')
    mean = rows.mean(1); raw = rows.std(1, ddof=1)
    return mean, raw, raw.clip(1e-6)


@dataclass
class EmpiricalCalibrator:
    mean_edges: np.ndarray
    scale_edges: np.ndarray
    global_pool: np.ndarray
    local_pools: dict
    horizon_length: int
    n_horizon_bins: int
    retained_sources: int
    excluded_sources: int

    def keys(self, mean, scale, horizon):
        h = np.asarray(horizon)
        if h.shape != mean.shape or np.any(h != h.astype(int)) or np.any(h < 0) or np.any(h >= self.horizon_length):
            raise ValueError('Horizon IDs must be valid coordinates of the original forecast')
        return np.stack([np.minimum(h*self.n_horizon_bins//self.horizon_length, self.n_horizon_bins-1),
            np.searchsorted(self.mean_edges[1:-1], mean, side='right'),
            np.searchsorted(self.scale_edges[1:-1], np.log(scale), side='right')], axis=1).astype(int)

    @classmethod
    def fit_rows(cls, rows, targets, horizon, horizon_length, n_horizon_bins=1, source_filter=True, summary_bins=4):
        mean, raw, scale = summaries(rows)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != mean.shape or not np.isfinite(targets).all():
            raise ValueError('Expected one finite target per fitting coordinate')
        if horizon_length <= 0 or n_horizon_bins not in (1,2,4,8) or summary_bins not in (4,6):
            raise ValueError('Expected a positive horizon, four or six summary groups, and 1/2/4/8 horizon groups')
        valid = raw > 1.01e-6 if source_filter else np.ones(len(mean), dtype=bool)
        if not valid.any():
            raise ValueError('No fitting residual has a noncollapsed source scale')
        residual = (targets-mean)/scale
        cal = cls(edges(mean,summary_bins), edges(np.log(scale),summary_bins), bounded(residual[valid]), {},
                  int(horizon_length), int(n_horizon_bins), int(valid.sum()), int((~valid).sum()))
        keys = cal.keys(mean, scale, horizon)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for i, key in enumerate(unique):
            take = (inverse == i)&valid
            if take.sum() >= 128:
                cal.local_pools[tuple(key)] = bounded(residual[take])
        return cal

    def transform_rows(self, rows, horizon):
        mean, _, scale = summaries(rows)
        keys = self.keys(mean, scale, horizon)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        result = np.empty((len(mean), 100), dtype=np.float64)
        grid = (np.arange(100)+.5)/100
        for i, key in enumerate(unique):
            take = inverse == i
            q = np.quantile(self.local_pools.get(tuple(key), self.global_pool), grid)
            result[take] = mean[take, None]+scale[take, None]*q
        return result

    def transform(self, samples):
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 4 or x.shape[2] != self.horizon_length:
            raise ValueError('Expected ensembles [B,100,H,C] with their fitted forecast horizon')
        b, s, h, c = x.shape
        rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
        horizon = np.broadcast_to(np.arange(h)[None, :, None], (b, h, c)).reshape(-1)
        result = self.transform_rows(rows, horizon)
        return result.reshape(b, h, c, s).transpose(0, 3, 1, 2)

    def save(self, path):
        keys = sorted(self.local_pools)
        arrays = [self.local_pools[k] for k in keys]
        offsets = np.r_[0, np.cumsum([len(a) for a in arrays])]
        np.savez_compressed(path, format_version=np.array(1), mean_edges=self.mean_edges,
            scale_edges=self.scale_edges, global_pool=self.global_pool,
            keys=np.array(keys, dtype=int).reshape(-1, 3), offsets=offsets,
            local_values=np.concatenate(arrays) if arrays else np.empty(0),
            horizon_length=np.array(self.horizon_length), n_horizon_bins=np.array(self.n_horizon_bins),
            retained_sources=np.array(self.retained_sources), excluded_sources=np.array(self.excluded_sources))

    def compile(self):
        return CompiledCalibrator.from_empirical(self)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['format_version']) != 1:
                raise ValueError('Unsupported empirical calibrator archive')
            pools = {tuple(k):data['local_values'][int(a):int(b)].copy()
                     for k,a,b in zip(data['keys'], data['offsets'][:-1], data['offsets'][1:])}
            return cls(data['mean_edges'].copy(), data['scale_edges'].copy(), data['global_pool'].copy(),
                       pools, int(data['horizon_length']), int(data['n_horizon_bins']),
                       int(data['retained_sources']), int(data['excluded_sources']))


@dataclass
class CompiledCalibrator:
    """Exact fixed-grid deployment and threshold queries for corrected marginals."""
    mean_edges: np.ndarray
    scale_edges: np.ndarray
    quantiles: np.ndarray
    state_to_template: np.ndarray
    horizon_length: int
    n_horizon_bins: int

    @classmethod
    def from_empirical(cls, cal):
        templates = []; mapping = []; used = {}
        grid = (np.arange(100)+.5)/100
        for h in range(cal.n_horizon_bins):
            for m in range(len(cal.mean_edges)-1):
                for s in range(len(cal.scale_edges)-1):
                    key = (h, m, s); identity = key if key in cal.local_pools else 'global'
                    if identity not in used:
                        used[identity] = len(templates)
                        templates.append(np.quantile(cal.local_pools.get(key, cal.global_pool), grid))
                    mapping.append(used[identity])
        return cls(cal.mean_edges.copy(), cal.scale_edges.copy(), np.stack(templates),
                   np.asarray(mapping, dtype=np.int16), cal.horizon_length, cal.n_horizon_bins)

    def template_ids(self, mean, scale, horizon):
        keys = EmpiricalCalibrator.keys(self, mean, scale, horizon)
        return self.state_to_template[(keys[:, 0]*(len(self.mean_edges)-1)+keys[:, 1])*(len(self.scale_edges)-1)+keys[:, 2]]

    def transform_rows(self, rows, horizon):
        mean, _, scale = summaries(rows)
        ids = self.template_ids(mean, scale, horizon)
        return mean[:, None]+scale[:, None]*self.quantiles[ids]

    def probabilities_summaries(self, mean, scale, horizon, thresholds):
        """Strict exceedance probabilities, with exact reconstructed-value ties.

        Inputs must use the same ensemble mean/ddof1/floor conventions as fitting.
        Binary search compares reconstructed values instead of dividing the
        threshold by scale, preserving floating-point equality at a member.
        """
        mean = np.asarray(mean, dtype=np.float64); scale = np.asarray(scale, dtype=np.float64)
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if thresholds.ndim == 1:
            thresholds = thresholds[:, None]
        if thresholds.shape[0] != len(mean) or mean.shape != scale.shape or np.any(scale <= 0):
            raise ValueError('Expected matching positive-scale summaries and per-coordinate thresholds')
        ids = self.template_ids(mean, scale, horizon)
        lo = np.zeros(thresholds.shape, dtype=int); hi = np.full(thresholds.shape, 100, dtype=int)
        while np.any(lo < hi):
            active = lo < hi; mid = (lo+hi)//2
            value = mean[:, None]+scale[:, None]*self.quantiles[ids[:, None], np.minimum(mid, 99)]
            right = value <= thresholds
            lo = np.where(active&right, mid+1, lo)
            hi = np.where(active&~right, mid, hi)
        return (100-lo)/100

    def probabilities_rows(self, rows, horizon, thresholds):
        mean, _, scale = summaries(rows)
        return self.probabilities_summaries(mean, scale, horizon, thresholds)

    def save(self, path):
        np.savez(path, compiled_format_version=np.array(1), mean_edges=self.mean_edges,
            scale_edges=self.scale_edges, quantiles=self.quantiles, state_to_template=self.state_to_template,
            horizon_length=np.array(self.horizon_length), n_horizon_bins=np.array(self.n_horizon_bins))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['compiled_format_version']) != 1:
                raise ValueError('Unsupported compiled calibrator')
            return cls(data['mean_edges'].copy(), data['scale_edges'].copy(), data['quantiles'].copy(),
                       data['state_to_template'].copy(), int(data['horizon_length']), int(data['n_horizon_bins']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit'); fit.add_argument('--coordinates', type=Path, required=True)
    fit.add_argument('--horizon-length', type=int, required=True)
    fit.add_argument('--horizon-bins', type=int, choices=[1,2,4,8], default=1)
    fit.add_argument('--summary-bins',type=int,choices=[4,6],default=6,
                     help='Groups per summary; default 6 gives the manuscript 36-state partition')
    fit.add_argument('--output', type=Path, required=True)
    apply = commands.add_parser('apply'); apply.add_argument('--calibrator', type=Path, required=True)
    apply.add_argument('--samples', type=Path, required=True); apply.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        with np.load(args.coordinates, allow_pickle=False) as data:
            cal = EmpiricalCalibrator.fit_rows(data['rows'], data['targets'], data['horizon'],
                args.horizon_length, args.horizon_bins,summary_bins=args.summary_bins)
        cal.save(args.output)
    else:
        cal = EmpiricalCalibrator.load(args.calibrator)
        with np.load(args.samples, allow_pickle=False) as data:
            samples = cal.transform(data['samples'])
        np.savez_compressed(args.output, samples=samples)
