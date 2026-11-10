"""Empirical PIT/rank warp for canonical [B, N, H, C] sample forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def qice_from_pit(pit: np.ndarray, n_bins: int = 10) -> float:
    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    pit = np.clip(pit, 0.0, 1.0 - 1e-12)
    counts, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, n_bins + 1))
    proportions = counts / max(1, pit.size)
    return float(np.mean(np.abs(proportions - 1.0 / n_bins)))


def compute_sample_pit(samples: np.ndarray, targets: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.ndim != 4:
        raise ValueError(f"Expected samples [B, N, H, C], got {samples.shape}")
    if targets.ndim != 3:
        raise ValueError(f"Expected targets [B, H, C], got {targets.shape}")
    if samples.shape[0] != targets.shape[0] or samples.shape[2:] != targets.shape[1:]:
        raise ValueError(f"Sample shape {samples.shape} is incompatible with target shape {targets.shape}")
    return np.mean(samples <= targets[:, None, :, :], axis=1).reshape(-1)


@dataclass
class PITWarpCalibrator:
    """Monotone empirical map h such that h(F(y)) is closer to uniform."""

    pit_knots: np.ndarray
    uniform_knots: np.ndarray
    strength: float = 1.0

    @classmethod
    def from_pit(cls, pit: np.ndarray, n_knots: int = 41, strength: float = 1.0) -> "PITWarpCalibrator":
        pit = np.asarray(pit, dtype=np.float64).reshape(-1)
        pit = np.clip(pit, 1e-6, 1.0 - 1e-6)
        uniform_knots = np.linspace(0.0, 1.0, n_knots)
        pit_knots = np.quantile(pit, uniform_knots)
        pit_knots[0] = 0.0
        pit_knots[-1] = 1.0
        pit_knots = np.maximum.accumulate(pit_knots)
        eps = 1e-6
        for i in range(1, len(pit_knots)):
            if pit_knots[i] <= pit_knots[i - 1]:
                pit_knots[i] = min(1.0, pit_knots[i - 1] + eps)
        pit_knots[-1] = 1.0
        return cls(pit_knots=pit_knots, uniform_knots=uniform_knots, strength=float(strength))

    def transform_pit(self, pit: np.ndarray) -> np.ndarray:
        pit = np.asarray(pit, dtype=np.float64)
        empirical = np.interp(pit, self.pit_knots, self.uniform_knots)
        warped = (1.0 - self.strength) * pit + self.strength * empirical
        return np.clip(warped, 0.0, 1.0)

    def inverse_transform_uniform(self, uniform: np.ndarray) -> np.ndarray:
        uniform = np.asarray(uniform, dtype=np.float64)
        if self.strength >= 1.0 - 1e-12:
            inverse = np.interp(uniform, self.uniform_knots, self.pit_knots)
        else:
            dense = np.linspace(0.0, 1.0, 4097)
            mapped = np.maximum.accumulate(self.transform_pit(dense))
            inverse = np.interp(uniform, mapped, dense)
        return np.clip(inverse, 0.0, 1.0)

    def warp_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 4:
            raise ValueError(f"Expected samples [B, N, H, C], got {samples.shape}")
        batch_size, sample_count, horizon, channels = samples.shape
        sorted_samples = np.sort(samples, axis=1)
        uniform_grid = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
        base_grid = self.inverse_transform_uniform(uniform_grid)
        source_index = np.clip(base_grid * (sample_count - 1), 0.0, sample_count - 1)
        low = np.floor(source_index).astype(np.int64)
        high = np.ceil(source_index).astype(np.int64)
        weight = (source_index - low).reshape(1, sample_count, 1, 1)

        low_values = np.take_along_axis(
            sorted_samples,
            np.broadcast_to(low.reshape(1, sample_count, 1, 1), (batch_size, sample_count, horizon, channels)),
            axis=1,
        )
        high_values = np.take_along_axis(
            sorted_samples,
            np.broadcast_to(high.reshape(1, sample_count, 1, 1), (batch_size, sample_count, horizon, channels)),
            axis=1,
        )
        return low_values * (1.0 - weight) + high_values * weight
