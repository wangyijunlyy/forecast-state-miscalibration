"""Lightweight post-hoc calibration baselines for sample forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from sparc_adapters.pit_warp import PITWarpCalibrator


Array = np.ndarray
BinKey = Tuple[int, ...]


def _validate(samples: Array, targets: Array) -> Tuple[Array, Array]:
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.ndim != 4:
        raise ValueError(f"Expected samples [B, S, H, C], got {samples.shape}")
    if targets.ndim != 3:
        raise ValueError(f"Expected targets [B, H, C], got {targets.shape}")
    if samples.shape[0] != targets.shape[0] or samples.shape[2:] != targets.shape[1:]:
        raise ValueError(f"Sample shape {samples.shape} is incompatible with target shape {targets.shape}")
    return samples, targets


def flatten_forecast(samples: Array, targets: Array) -> Tuple[Array, Array, Array, Array]:
    """Return coordinate rows [B*H*C,S], targets, horizon ids, and channel ids."""
    samples, targets = _validate(samples, targets)
    batch_size, sample_count, horizon, channels = samples.shape
    rows = samples.transpose(0, 2, 3, 1).reshape(-1, sample_count)
    flat_targets = targets.reshape(-1)
    horizon_ids = np.broadcast_to(
        np.arange(horizon, dtype=np.int64).reshape(1, horizon, 1),
        (batch_size, horizon, channels),
    ).reshape(-1)
    channel_ids = np.broadcast_to(
        np.arange(channels, dtype=np.int64).reshape(1, 1, channels),
        (batch_size, horizon, channels),
    ).reshape(-1)
    return rows, flat_targets, horizon_ids, channel_ids


def restore_forecast(rows: Array, shape: Tuple[int, int, int, int]) -> Array:
    batch_size, sample_count, horizon, channels = shape
    return rows.reshape(batch_size, horizon, channels, sample_count).transpose(0, 3, 1, 2)


def _quantile_edges(values: Array, n_bins: int) -> Array:
    if n_bins <= 1:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _digitize(values: Array, edges: Array) -> Array:
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)


def _position_bins(indices: Array, length: int, n_bins: int) -> Array:
    if n_bins <= 1 or length <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    return np.minimum(indices * n_bins // length, n_bins - 1).astype(np.int64)


def _row_keys(*parts: Array) -> Array:
    return np.stack(parts, axis=1)


def _indices_by_key(keys: Array) -> Iterable[Tuple[BinKey, Array]]:
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    for index, values in enumerate(unique):
        yield tuple(int(value) for value in values), np.flatnonzero(inverse == index)


def _warp_rows(rows: Array, calibrator: PITWarpCalibrator) -> Array:
    rows = np.asarray(rows, dtype=np.float64)
    sample_count = rows.shape[1]
    sorted_rows = np.sort(rows, axis=1)
    uniform_grid = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    source_grid = calibrator.inverse_transform_uniform(uniform_grid)
    source_index = np.clip(source_grid * (sample_count - 1), 0.0, sample_count - 1)
    low = np.floor(source_index).astype(np.int64)
    high = np.ceil(source_index).astype(np.int64)
    weight = source_index - low
    return sorted_rows[:, low] * (1.0 - weight) + sorted_rows[:, high] * weight


def _bounded_pool(values: Array, max_size: int) -> Array:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if values.size <= max_size:
        return values
    indices = np.linspace(0, values.size - 1, max_size).astype(np.int64)
    return values[indices]


@dataclass
class GlobalAffineCalibrator:
    """One shared location bias and scale multiplier fit by Gaussian NLL."""

    bias: float
    scale: float

    @classmethod
    def fit(cls, samples: Array, targets: Array) -> "GlobalAffineCalibrator":
        rows, flat_targets, _, _ = flatten_forecast(samples, targets)
        means = rows.mean(axis=1)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        residual = flat_targets - means
        inverse_variance = 1.0 / np.square(scales)
        bias = float(np.sum(residual * inverse_variance) / np.sum(inverse_variance))
        scale = float(np.sqrt(np.mean(np.square((residual - bias) / scales))))
        return cls(bias=bias, scale=float(np.clip(scale, 0.05, 20.0)))

    def transform(self, samples: Array) -> Array:
        samples = np.asarray(samples, dtype=np.float64)
        mean = samples.mean(axis=1, keepdims=True)
        return mean + self.bias + self.scale * (samples - mean)


@dataclass
class ConditionalPITCalibrator:
    """PIT maps conditioned on horizon, forecast scale, and channel groups."""

    global_warp: PITWarpCalibrator
    bin_warps: Mapping[BinKey, PITWarpCalibrator]
    scale_edges: Array
    horizon_length: int
    channel_count: int
    n_horizon_bins: int
    n_channel_bins: int

    @classmethod
    def fit(
        cls,
        samples: Array,
        targets: Array,
        n_horizon_bins: int = 4,
        n_scale_bins: int = 4,
        n_channel_bins: int = 4,
        min_bin_size: int = 128,
        n_knots: int = 41,
    ) -> "ConditionalPITCalibrator":
        rows, flat_targets, horizon_ids, channel_ids = flatten_forecast(samples, targets)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        pit = np.mean(rows <= flat_targets[:, None], axis=1)
        return cls.fit_summaries(
            pit,
            scales,
            horizon_ids,
            channel_ids,
            horizon_length=samples.shape[2],
            channel_count=samples.shape[3],
            n_horizon_bins=n_horizon_bins,
            n_scale_bins=n_scale_bins,
            n_channel_bins=n_channel_bins,
            min_bin_size=min_bin_size,
            n_knots=n_knots,
        )

    @classmethod
    def fit_summaries(
        cls,
        pit: Array,
        scales: Array,
        horizon_ids: Array,
        channel_ids: Array,
        horizon_length: int,
        channel_count: int,
        n_horizon_bins: int = 4,
        n_scale_bins: int = 4,
        n_channel_bins: int = 4,
        min_bin_size: int = 128,
        n_knots: int = 41,
    ) -> "ConditionalPITCalibrator":
        pit = np.asarray(pit, dtype=np.float64).reshape(-1)
        log_scales = np.log(np.asarray(scales, dtype=np.float64).reshape(-1).clip(1e-6))
        scale_edges = _quantile_edges(log_scales, n_scale_bins)
        keys = _row_keys(
            _position_bins(np.asarray(horizon_ids), horizon_length, n_horizon_bins),
            _digitize(log_scales, scale_edges),
            _position_bins(np.asarray(channel_ids), channel_count, n_channel_bins),
        )
        global_warp = PITWarpCalibrator.from_pit(pit, n_knots=n_knots)
        bin_warps: Dict[BinKey, PITWarpCalibrator] = {}
        for key, indices in _indices_by_key(keys):
            if indices.size >= min_bin_size:
                bin_warps[key] = PITWarpCalibrator.from_pit(pit[indices], n_knots=n_knots)
        return cls(
            global_warp=global_warp,
            bin_warps=bin_warps,
            scale_edges=scale_edges,
            horizon_length=int(horizon_length),
            channel_count=int(channel_count),
            n_horizon_bins=int(n_horizon_bins),
            n_channel_bins=int(n_channel_bins),
        )

    def transform(self, samples: Array) -> Array:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 4:
            raise ValueError(f"Expected samples [B, S, H, C], got {samples.shape}")
        dummy = np.zeros((samples.shape[0], samples.shape[2], samples.shape[3]))
        rows, _, horizon_ids, channel_ids = flatten_forecast(samples, dummy)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        keys = _row_keys(
            _position_bins(horizon_ids, self.horizon_length, self.n_horizon_bins),
            _digitize(np.log(scales), self.scale_edges),
            _position_bins(channel_ids, self.channel_count, self.n_channel_bins),
        )
        calibrated = np.empty_like(rows)
        for key, indices in _indices_by_key(keys):
            calibrated[indices] = _warp_rows(rows[indices], self.bin_warps.get(key, self.global_warp))
        return restore_forecast(calibrated, samples.shape)


@dataclass
class DistributionCalibrationCalibrator:
    """Binned approximation to distribution calibration using residual redraw."""

    global_residuals: Array
    bin_residuals: Mapping[BinKey, Array]
    mean_edges: Array
    scale_edges: Array
    horizon_length: int
    n_horizon_bins: int

    @classmethod
    def fit(
        cls,
        samples: Array,
        targets: Array,
        n_mean_bins: int = 4,
        n_scale_bins: int = 4,
        n_horizon_bins: int = 4,
        min_bin_size: int = 128,
        max_pool_size: int = 20000,
    ) -> "DistributionCalibrationCalibrator":
        rows, flat_targets, horizon_ids, _ = flatten_forecast(samples, targets)
        means = rows.mean(axis=1)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        residuals = (flat_targets - means) / scales
        return cls.fit_summaries(
            residuals,
            means,
            scales,
            horizon_ids,
            horizon_length=samples.shape[2],
            n_mean_bins=n_mean_bins,
            n_scale_bins=n_scale_bins,
            n_horizon_bins=n_horizon_bins,
            min_bin_size=min_bin_size,
            max_pool_size=max_pool_size,
        )

    @classmethod
    def fit_summaries(
        cls,
        residuals: Array,
        means: Array,
        scales: Array,
        horizon_ids: Array,
        horizon_length: int,
        n_mean_bins: int = 4,
        n_scale_bins: int = 4,
        n_horizon_bins: int = 4,
        min_bin_size: int = 128,
        max_pool_size: int = 20000,
    ) -> "DistributionCalibrationCalibrator":
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        means = np.asarray(means, dtype=np.float64).reshape(-1)
        log_scales = np.log(np.asarray(scales, dtype=np.float64).reshape(-1).clip(1e-6))
        mean_edges = _quantile_edges(means, n_mean_bins)
        scale_edges = _quantile_edges(log_scales, n_scale_bins)
        keys = _row_keys(
            _position_bins(np.asarray(horizon_ids), horizon_length, n_horizon_bins),
            _digitize(means, mean_edges),
            _digitize(log_scales, scale_edges),
        )
        pools: Dict[BinKey, Array] = {}
        for key, indices in _indices_by_key(keys):
            if indices.size >= min_bin_size:
                pools[key] = _bounded_pool(residuals[indices], max_pool_size)
        return cls(
            global_residuals=_bounded_pool(residuals, max_pool_size),
            bin_residuals=pools,
            mean_edges=mean_edges,
            scale_edges=scale_edges,
            horizon_length=int(horizon_length),
            n_horizon_bins=int(n_horizon_bins),
        )

    def transform(self, samples: Array) -> Array:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 4:
            raise ValueError(f"Expected samples [B, S, H, C], got {samples.shape}")
        dummy = np.zeros((samples.shape[0], samples.shape[2], samples.shape[3]))
        rows, _, horizon_ids, _ = flatten_forecast(samples, dummy)
        means = rows.mean(axis=1)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        keys = _row_keys(
            _position_bins(horizon_ids, self.horizon_length, self.n_horizon_bins),
            _digitize(means, self.mean_edges),
            _digitize(np.log(scales), self.scale_edges),
        )
        uniform_grid = (np.arange(rows.shape[1], dtype=np.float64) + 0.5) / rows.shape[1]
        calibrated = np.empty_like(rows)
        for key, indices in _indices_by_key(keys):
            pool = self.bin_residuals.get(key, self.global_residuals)
            residual_quantiles = np.quantile(pool, uniform_grid)
            calibrated[indices] = means[indices, None] + scales[indices, None] * residual_quantiles[None]
        return restore_forecast(calibrated, samples.shape)


@dataclass
class CQRCalibrator:
    """Global conformal corrections for central prediction intervals."""

    corrections: Mapping[float, float]

    @classmethod
    def fit(cls, samples: Array, targets: Array, levels: Sequence[float]) -> "CQRCalibrator":
        rows, flat_targets, _, _ = flatten_forecast(samples, targets)
        corrections: Dict[float, float] = {}
        for level in levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"Interval level must be in (0,1), got {level}")
            alpha = 1.0 - float(level)
            lower = np.quantile(rows, alpha / 2.0, axis=1)
            upper = np.quantile(rows, 1.0 - alpha / 2.0, axis=1)
            scores = np.maximum(lower - flat_targets, flat_targets - upper)
            finite_level = min(1.0, np.ceil((scores.size + 1) * level) / scores.size)
            corrections[float(level)] = float(np.quantile(scores, finite_level, method="higher"))
        return cls(corrections=corrections)


def interval_metrics(
    samples: Array,
    targets: Array,
    levels: Sequence[float],
    calibrator: Optional[CQRCalibrator] = None,
) -> Dict[str, float]:
    rows, flat_targets, _, _ = flatten_forecast(samples, targets)
    coverage_errors = []
    interval_scores = []
    widths = []
    for level in levels:
        alpha = 1.0 - float(level)
        lower = np.quantile(rows, alpha / 2.0, axis=1)
        upper = np.quantile(rows, 1.0 - alpha / 2.0, axis=1)
        if calibrator is not None:
            correction = float(calibrator.corrections[float(level)])
            lower = lower - correction
            upper = upper + correction
        coverage = np.mean((flat_targets >= lower) & (flat_targets <= upper))
        width = upper - lower
        score = width.copy()
        below = flat_targets < lower
        above = flat_targets > upper
        score[below] += 2.0 / alpha * (lower[below] - flat_targets[below])
        score[above] += 2.0 / alpha * (flat_targets[above] - upper[above])
        coverage_errors.append(abs(float(coverage) - float(level)))
        interval_scores.append(float(np.mean(score)))
        widths.append(float(np.mean(width)))
    return {
        "coverage_error": float(np.mean(coverage_errors)),
        "mean_interval_score": float(np.mean(interval_scores)),
        "mean_interval_width": float(np.mean(widths)),
    }
