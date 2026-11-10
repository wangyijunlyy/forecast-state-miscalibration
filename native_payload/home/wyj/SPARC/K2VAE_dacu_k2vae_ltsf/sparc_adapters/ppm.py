"""SPARC adapter for PPM-style sample forecasters.

PPM's MLP implementation returns samples as [B, K, H, C] from forward(x, K),
which is already the canonical SPARC/K2VAE evaluator layout.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from .samples import normalize_sample_layout, summarize_canonical_samples


class PPMSPARCAdapter:
    """Thin adapter around a PPM model or compatible callable."""

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        input_getter: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        self.model = model
        self.input_getter = input_getter

    def _input(self, batch: Any) -> torch.Tensor:
        if self.input_getter is not None:
            return self.input_getter(batch)
        if isinstance(batch, dict):
            for key in ("x", "batch_x", "past_target", "past_target_cdf"):
                if key in batch:
                    return batch[key]
            raise KeyError("PPM batch dict must contain one of: x, batch_x, past_target, past_target_cdf")
        return batch

    def native_samples(self, batch: Any, num_samples: int) -> torch.Tensor:
        x = self._input(batch)
        samples = self.model(x, num_samples)
        return normalize_sample_layout(samples, "ppm")

    def sample_statistics(self, batch: Any, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return summarize_canonical_samples(self.native_samples(batch, num_samples))
