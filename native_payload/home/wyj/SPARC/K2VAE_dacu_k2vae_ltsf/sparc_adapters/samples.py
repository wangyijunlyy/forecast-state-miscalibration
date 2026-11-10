"""Shared sample utilities for model-independent SPARC adapters.

The canonical sample layout used here is [batch, samples, horizon, target_dim],
matching ProbTS/K2VAE evaluator expectations.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

SampleLayout = Literal["canonical", "bnhc", "ppm", "nsdiff", "bhcn", "bhnc"]


def normalize_sample_layout(samples: torch.Tensor, layout: SampleLayout = "canonical") -> torch.Tensor:
    """Convert backbone-native samples to [B, N, H, C]."""

    if samples.ndim != 4:
        raise ValueError(f"Expected a 4D sample tensor, got shape {tuple(samples.shape)}")

    if layout in {"canonical", "bnhc", "ppm"}:
        return samples
    if layout in {"nsdiff", "bhcn"}:
        return samples.permute(0, 3, 1, 2).contiguous()
    if layout == "bhnc":
        return samples.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"Unsupported sample layout: {layout}")


def summarize_canonical_samples(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sample mean and scale for canonical [B, N, H, C] samples."""

    if samples.ndim != 4:
        raise ValueError(f"Expected canonical samples as [B, N, H, C], got {tuple(samples.shape)}")
    unbiased = samples.shape[1] > 1
    mean = samples.mean(dim=1)
    scale = samples.std(dim=1, unbiased=unbiased).clamp_min(1e-6)
    return mean, scale


def sample_preserving_rescale(
    samples: torch.Tensor,
    refined_mean: torch.Tensor,
    calibrated_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply SPARC sample-preserving mean/scale calibration.

    This keeps each native backbone sample identity and relative residual shape,
    only shifting the center and rescaling residuals.
    """

    if samples.ndim != 4:
        raise ValueError(f"Expected canonical samples as [B, N, H, C], got {tuple(samples.shape)}")
    expected = samples.shape[:1] + samples.shape[2:]
    if refined_mean.shape != expected:
        raise ValueError(f"Expected refined_mean shape {tuple(expected)}, got {tuple(refined_mean.shape)}")
    if calibrated_scale.shape != expected:
        raise ValueError(f"Expected calibrated_scale shape {tuple(expected)}, got {tuple(calibrated_scale.shape)}")

    base_mean, base_scale = summarize_canonical_samples(samples)
    standardized = (samples - base_mean.unsqueeze(1)) / base_scale.unsqueeze(1)
    return refined_mean.unsqueeze(1) + standardized * calibrated_scale.unsqueeze(1)


def augmentation_residuals(
    base_samples: torch.Tensor,
    augmented_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute SPARC loc/scale residual features from two native sample sets."""

    base_mean, base_scale = summarize_canonical_samples(base_samples)
    aug_mean, aug_scale = summarize_canonical_samples(augmented_samples)
    loc_residual = torch.abs(base_mean - aug_mean)
    scale_residual = torch.abs(torch.log(base_scale.clamp_min(1e-6)) - torch.log(aug_scale.clamp_min(1e-6)))
    return loc_residual.detach(), scale_residual.detach()
