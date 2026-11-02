"""Complete-stream marginal scores, common-state coverage, and event reliability."""
import numpy as np


def assert_traversal(loader, batches, windows):
    if batches <= 0 or windows <= 0:
        raise ValueError('Empty evaluation iterator')
    try:
        expected = len(loader)
    except TypeError:
        expected = None
    if expected is not None and batches != expected:
        raise ValueError('Incomplete evaluation iterator')


def row_quantiles(ordered, probabilities):
    at = np.asarray(probabilities)*(ordered.shape[1]-1)
    lo = np.floor(at).astype(int); hi = np.ceil(at).astype(int)
    w = at-lo
    return ordered[:, lo]*(1-w)+ordered[:, hi]*w


def state_ids(samples, mean_edges, scale_edges):
    x = np.asarray(samples, dtype=np.float64)
    mu = x.mean(1).reshape(-1)
    sigma = x.std(1, ddof=1).clip(1e-6).reshape(-1)
    return (4*np.searchsorted(mean_edges[1:-1], mu, side='right')
            + np.searchsorted(scale_edges[1:-1], np.log(sigma), side='right'))


class TargetReservoir:
    """Uniform horizon-coordinate rows, retaining all channels in each row."""
    def __init__(self, capacity=4096, seed=1):
        self.capacity = capacity; self.rng = np.random.default_rng(seed)
        self.values = None; self.priority = np.empty(0); self.seen = 0

    def update(self, target):
        a = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite historical target')
        p = self.rng.random(len(a)); self.seen += len(a)
        if len(a) > self.capacity:
            ix = np.argpartition(p, -self.capacity)[-self.capacity:]; a, p = a[ix], p[ix]
        if self.values is not None:
            a = np.concatenate([self.values, a]); p = np.concatenate([self.priority, p])
        if len(a) > self.capacity:
            ix = np.argpartition(p, -self.capacity)[-self.capacity:]; a, p = a[ix], p[ix]
        self.values, self.priority = a, p

    def thresholds(self):
        if self.values is None:
            raise ValueError('No historical targets')
        return np.quantile(self.values, [.8, .9, .95], axis=0).T


class ReliabilityMetrics:
    def __init__(self, thresholds, min_state_count=128):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if self.thresholds.ndim != 2 or self.thresholds.shape[1] != 3:
            raise ValueError('Expected per-channel thresholds [C,3]')
        self.min_state_count = min_state_count
        self.n = 0; self.crps = self.se = self.ae = self.medae = 0.
        self.qice_counts = np.zeros(10, dtype=np.int64)
        self.state_counts = np.zeros(16, dtype=np.int64)
        self.covered = np.zeros((16, 3), dtype=np.int64)
        self.brier = np.zeros(3); self.event_count = np.zeros(3, dtype=np.int64)
        self.probability_bin_counts = np.zeros((3, 10), dtype=np.int64)
        self.probability_sums = np.zeros((3, 10)); self.observation_sums = np.zeros((3, 10))
        self.state_brier_sums = np.zeros((16, 3))

    def update(self, samples, target, groups, ordered=False):
        x = np.asarray(samples, dtype=np.float64); y = np.asarray(target, dtype=np.float64)
        if x.ndim != 4 or x.shape[:1]+x.shape[2:] != y.shape:
            raise ValueError('Incompatible forecast and target shapes')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite forecasts or targets')
        rows = x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1]); y = y.reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        channels = np.arange(len(y)) % x.shape[-1]
        if len(groups) != len(y) or np.any(groups < 0) or np.any(groups >= 16):
            raise ValueError('Invalid state assignments')
        for start in range(0, len(y), 8192):
            end = start+8192; a = rows[start:end]; b = y[start:end]; g = groups[start:end]
            z = a if ordered else np.sort(a, axis=1)
            s = a.shape[1]; weights = (2*np.arange(1, s+1)-s-1)/(s*s)
            self.crps += float((np.abs(a-b[:, None]).mean(1)-z@weights).sum())
            e = a.mean(1)-b; self.se += float((e*e).sum()); self.ae += float(np.abs(e).sum())
            self.medae += float(np.abs(row_quantiles(z, [.5])[:, 0]-b).sum())
            q = row_quantiles(z, np.arange(11)/10)
            membership = np.clip((b[:, None] > q).sum(1), 1, 10)-1
            self.qice_counts += np.bincount(membership, minlength=10)
            self.state_counts += np.bincount(g, minlength=16)
            hits = b[:, None] <= row_quantiles(z, [.1, .5, .9])
            for j in range(3):
                self.covered[:, j] += np.bincount(g, weights=hits[:, j], minlength=16).astype(np.int64)
                threshold = self.thresholds[channels[start:end], j]
                p = (a > threshold[:, None]).mean(1); event = b > threshold
                errors = (p-event)**2
                self.brier[j] += errors.sum(); self.event_count[j] += event.sum()
                self.state_brier_sums[:, j] += np.bincount(g, weights=errors, minlength=16)
                bins = np.minimum((p*10).astype(int), 9)
                self.probability_bin_counts[j] += np.bincount(bins, minlength=10)
                self.probability_sums[j] += np.bincount(bins, weights=p, minlength=10)
                self.observation_sums[j] += np.bincount(bins, weights=event, minlength=10)
            self.n += len(b)

    def result(self):
        if not self.n:
            raise ValueError('Empty evaluation stream')
        eligible = self.state_counts >= self.min_state_count
        n = int(self.state_counts[eligible].sum())
        coverage = self.covered/np.maximum(self.state_counts[:, None], 1)
        levels = np.array([.1, .5, .9])
        if n:
            pooled = self.covered[eligible].sum(0)/n
            pooled_error = float(np.abs(pooled-levels).mean())
            state_error = float((np.abs(coverage[eligible]-levels)*self.state_counts[eligible, None]).sum()/(3*n))
        else:
            pooled = [None]*3; pooled_error = state_error = None
        return dict(CRPS=self.crps/self.n, QICE=float(np.abs(self.qice_counts/self.n-.1).mean()),
            QICE_percent=float(100*np.abs(self.qice_counts/self.n-.1).mean()),
            MSE=self.se/self.n, MAE=self.ae/self.n, RMSE=float(np.sqrt(self.se/self.n)),
            MedianMAE=self.medae/self.n, coordinate_count=self.n,
            state_coverage=dict(levels=levels.tolist(), counts=self.state_counts.tolist(),
                covered=self.covered.tolist(), coverage=coverage.tolist(), eligible_states=int(eligible.sum()),
                eligible_coordinates=n, pooled_coverage=np.asarray(pooled).tolist(),
                pooled_error=pooled_error, state_error=state_error),
            events=dict(levels=[.8, .9, .95], brier=(self.brier/self.n).tolist(),
                observed_frequencies=(self.event_count/self.n).tolist(),
                probability_bin_counts=self.probability_bin_counts.tolist(),
                probability_sums=self.probability_sums.tolist(), observation_sums=self.observation_sums.tolist(),
                state_brier_sums=self.state_brier_sums.tolist()))
