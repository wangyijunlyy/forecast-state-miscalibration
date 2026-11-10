"""Streaming PPM metrics for post-hoc sample calibration comparisons."""

from __future__ import annotations

import numpy as np

from sparc_ppm.ppm_metrics import calculate_batch_crps, compute_qice_bin_count


class PPMMetricAccumulator:
    def __init__(self) -> None:
        self.total_mse = 0.0
        self.total_mae = 0.0
        self.total_batches = 0
        self.sum_crps = 0.0
        self.bin_count = np.zeros(10, dtype=np.float64)
        self.qice_points = 0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        samples = np.asarray(samples)
        targets = np.asarray(targets)
        batch_size, sample_count, _horizon, _channels = samples.shape
        rows = samples.transpose(0, 2, 3, 1).reshape(-1, sample_count)
        flat_targets = targets.reshape(-1, 1)
        self.bin_count += compute_qice_bin_count(10, flat_targets, rows)
        self.qice_points += rows.shape[0]
        self.sum_crps += calculate_batch_crps(samples, targets)
        prediction = samples.mean(axis=1)
        self.total_mse += float(np.mean(np.square(prediction - targets))) * batch_size
        self.total_mae += float(np.mean(np.abs(prediction - targets))) * batch_size
        self.total_batches += batch_size

    def result(self) -> dict[str, float]:
        if self.total_batches == 0 or self.qice_points == 0:
            raise ValueError("Metric accumulator is empty")
        coverage = self.bin_count / self.qice_points
        return {
            "CRPS": self.sum_crps / self.total_batches,
            "QICE": 100.0 * float(np.mean(np.abs(0.1 - coverage))),
            "MSE": self.total_mse / self.total_batches,
            "MAE": self.total_mae / self.total_batches,
        }


class MeanMetricAccumulator:
    def __init__(self) -> None:
        self.weight = 0
        self.sums: dict[str, float] = {}

    def update(self, metrics: dict[str, float], weight: int) -> None:
        self.weight += int(weight)
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * weight

    def result(self) -> dict[str, float]:
        if self.weight == 0:
            raise ValueError("Metric accumulator is empty")
        return {key: value / self.weight for key, value in self.sums.items()}
