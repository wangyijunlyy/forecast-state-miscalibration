"""Unchanged priority-reservoir class from the original binned evaluator."""
from typing import Tuple
import numpy as np
from sparc_adapters.posthoc_baselines import flatten_forecast

class CalibrationReservoir:
    """Priority reservoir of forecast coordinates and their sample rows."""

    def __init__(self, max_points: int = 50000, seed: int = 1):
        if max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self._priorities = np.empty(0, dtype=np.float64)
        self._rows = None
        self._targets = np.empty(0, dtype=np.float64)
        self._horizon_ids = np.empty(0, dtype=np.int64)
        self._channel_ids = np.empty(0, dtype=np.int64)
        self.horizon_length = 0
        self.channel_count = 0
        self.seen_points = 0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        rows, flat_targets, horizon_ids, channel_ids = flatten_forecast(samples, targets)
        if self._rows is not None and rows.shape[1] != self._rows.shape[1]:
            raise ValueError(f"Sample count changed from {self._rows.shape[1]} to {rows.shape[1]}")
        self.horizon_length = samples.shape[2]
        self.channel_count = samples.shape[3]
        self.seen_points += rows.shape[0]
        priorities = self.rng.random(rows.shape[0])
        if rows.shape[0] > self.max_points:
            selected = np.argpartition(priorities, -self.max_points)[-self.max_points :]
            priorities = priorities[selected]
            rows = rows[selected]
            flat_targets = flat_targets[selected]
            horizon_ids = horizon_ids[selected]
            channel_ids = channel_ids[selected]
        if self._rows is None:
            combined_rows = rows
        else:
            combined_rows = np.concatenate([self._rows, rows], axis=0)
        combined_priorities = np.concatenate([self._priorities, priorities])
        combined_targets = np.concatenate([self._targets, flat_targets])
        combined_horizon = np.concatenate([self._horizon_ids, horizon_ids])
        combined_channel = np.concatenate([self._channel_ids, channel_ids])
        if combined_priorities.size > self.max_points:
            keep = np.argpartition(combined_priorities, -self.max_points)[-self.max_points :]
            combined_priorities = combined_priorities[keep]
            combined_rows = combined_rows[keep]
            combined_targets = combined_targets[keep]
            combined_horizon = combined_horizon[keep]
            combined_channel = combined_channel[keep]
        self._priorities = combined_priorities
        self._rows = combined_rows
        self._targets = combined_targets
        self._horizon_ids = combined_horizon
        self._channel_ids = combined_channel

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._rows is None or self._rows.shape[0] == 0:
            raise ValueError("Calibration reservoir is empty")
        return self._rows, self._targets, self._horizon_ids, self._channel_ids

