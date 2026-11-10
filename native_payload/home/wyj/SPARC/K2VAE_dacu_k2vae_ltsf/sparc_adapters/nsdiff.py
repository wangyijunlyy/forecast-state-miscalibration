"""SPARC adapter for NsDiff-style diffusion sample forecasters.

The NsDiff experiment code commonly stores samples as [B, H, C, S]. This
adapter converts that layout to the canonical [B, S, H, C] layout used by
K2VAE/SPARC metrics.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch

from .samples import normalize_sample_layout, summarize_canonical_samples


class NsDiffSPARCAdapter:
    """Thin adapter around an NsDiff sampling callable or model with sample()."""

    def __init__(self, sample_source: Callable[..., torch.Tensor] | Any):
        self.sample_source = sample_source

    def _sample(self, batch: Any, num_samples: int) -> torch.Tensor:
        if callable(self.sample_source):
            return self.sample_source(batch, num_samples)
        if hasattr(self.sample_source, "sample"):
            return self.sample_source.sample(batch, num_samples)
        raise TypeError("NsDiffSPARCAdapter requires a callable or an object with sample(batch, num_samples)")

    def native_samples(self, batch: Any, num_samples: int) -> torch.Tensor:
        samples = self._sample(batch, num_samples)
        return normalize_sample_layout(samples, "nsdiff")

    def sample_statistics(self, batch: Any, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return summarize_canonical_samples(self.native_samples(batch, num_samples))
