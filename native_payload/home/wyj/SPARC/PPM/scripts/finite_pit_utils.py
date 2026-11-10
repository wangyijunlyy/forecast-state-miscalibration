"""Tie-aware finite-ensemble PIT utilities for the server audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PIT_METHODS = ("empirical", "midrank", "randomized")


def ensemble_pit(
    samples: np.ndarray,
    targets: np.ndarray,
    method: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    samples = np.asarray(samples)
    targets = np.asarray(targets)
    if samples.ndim != 4:
        raise ValueError(f"Expected samples [B, S, H, C], got {samples.shape}")
    if targets.ndim != 3:
        raise ValueError(f"Expected targets [B, H, C], got {targets.shape}")
    if samples.shape[0] != targets.shape[0] or samples.shape[2:] != targets.shape[1:]:
        raise ValueError(f"Sample shape {samples.shape} is incompatible with target shape {targets.shape}")
    if method not in PIT_METHODS:
        raise ValueError(f"Unsupported PIT method: {method}")

    target = targets[:, None, :, :]
    less = np.count_nonzero(samples < target, axis=1)
    equal = np.count_nonzero(samples == target, axis=1)
    sample_count = samples.shape[1]
    if method == "empirical":
        values = (less + equal) / sample_count
    elif method == "midrank":
        values = (less + 0.5 * (equal + 1.0)) / (sample_count + 1.0)
    else:
        if rng is None:
            raise ValueError("randomized PIT requires an RNG")
        values = (less + rng.random(less.shape) * (equal + 1.0)) / (sample_count + 1.0)
    return values.reshape(-1).astype(np.float64)


def qice_from_values(values: np.ndarray, n_bins: int) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.clip(values, 0.0, 1.0 - 1e-12)
    counts, _ = np.histogram(values, bins=np.linspace(0.0, 1.0, n_bins + 1))
    proportions = counts / max(1, values.size)
    return float(np.mean(np.abs(proportions - 1.0 / n_bins)))


@dataclass
class RankPITAccumulator:
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        size = self.sample_count + 1
        self.counts = np.zeros((size, size), dtype=np.int64)
        self.total = 0
        self.tied = 0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        samples = np.asarray(samples)
        targets = np.asarray(targets)
        if samples.shape[1] != self.sample_count:
            raise ValueError(f"Expected {self.sample_count} samples, got {samples.shape[1]}")
        target = targets[:, None, :, :]
        less = np.count_nonzero(samples < target, axis=1).reshape(-1)
        equal = np.count_nonzero(samples == target, axis=1).reshape(-1)
        size = self.sample_count + 1
        encoded = less * size + equal
        self.counts += np.bincount(encoded, minlength=size * size).reshape(size, size)
        self.total += int(less.size)
        self.tied += int(np.count_nonzero(equal))

    def values(self, method: str, rng: np.random.Generator | None = None) -> np.ndarray:
        if method not in PIT_METHODS:
            raise ValueError(f"Unsupported PIT method: {method}")
        chunks = []
        for less, equal in zip(*np.nonzero(self.counts)):
            count = int(self.counts[less, equal])
            if method == "empirical":
                chunks.append(np.full(count, (less + equal) / self.sample_count))
            elif method == "midrank":
                chunks.append(np.full(count, (less + 0.5 * (equal + 1.0)) / (self.sample_count + 1.0)))
            else:
                if rng is None:
                    raise ValueError("randomized PIT requires an RNG")
                chunks.append((less + rng.random(count) * (equal + 1.0)) / (self.sample_count + 1.0))
        return np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0, dtype=np.float64)

    def qice(self, method: str, n_bins: int, rng: np.random.Generator | None = None) -> float:
        return qice_from_values(self.values(method, rng), n_bins)
