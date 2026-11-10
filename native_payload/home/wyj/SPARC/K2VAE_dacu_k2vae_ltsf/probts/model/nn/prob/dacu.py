import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DACULossBreakdown:
    base_loss: torch.Tensor
    uncertainty_loss: torch.Tensor
    consistency_loss: torch.Tensor
    total_loss: torch.Tensor


class TimeSeriesAugmenter(nn.Module):
    """Shape-preserving augmentations for augmentation-consistency training."""

    def __init__(
        self,
        jitter_std: float = 0.0,
        scaling_std: float = 0.0,
        time_mask_ratio: float = 0.0,
        channel_mask_ratio: float = 0.0,
        freq_dropout_ratio: float = 0.0,
    ):
        super().__init__()
        self.jitter_std = jitter_std
        self.scaling_std = scaling_std
        self.time_mask_ratio = time_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.freq_dropout_ratio = freq_dropout_ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, bool]]:
        if x.ndim != 3:
            raise ValueError(f"Expected a [B, T, C] tensor, got shape {tuple(x.shape)}")

        y = x.clone()
        meta = {
            "jitter": False,
            "scaling": False,
            "time_mask": False,
            "channel_mask": False,
            "freq_dropout": False,
        }

        if self.jitter_std > 0:
            y = y + torch.randn_like(y) * self.jitter_std
            meta["jitter"] = True

        if self.scaling_std > 0:
            scale = 1.0 + torch.randn(y.shape[0], 1, y.shape[2], device=y.device, dtype=y.dtype) * self.scaling_std
            y = y * scale.clamp_min(1e-3)
            meta["scaling"] = True

        if self.time_mask_ratio > 0:
            mask = torch.rand(y.shape[0], y.shape[1], 1, device=y.device) < self.time_mask_ratio
            fill = y.mean(dim=1, keepdim=True).detach()
            y = torch.where(mask, fill.expand_as(y), y)
            meta["time_mask"] = True

        if self.channel_mask_ratio > 0:
            mask = torch.rand(y.shape[0], 1, y.shape[2], device=y.device) < self.channel_mask_ratio
            fill = y.mean(dim=2, keepdim=True).detach()
            y = torch.where(mask, fill.expand_as(y), y)
            meta["channel_mask"] = True

        if self.freq_dropout_ratio > 0:
            y = self._frequency_dropout(y, self.freq_dropout_ratio)
            meta["freq_dropout"] = True

        return y, meta

    @staticmethod
    def _frequency_dropout(x: torch.Tensor, ratio: float) -> torch.Tensor:
        if x.shape[1] < 2:
            return x
        freq = torch.fft.rfft(x, dim=1)
        keep = torch.rand(freq.shape[0], freq.shape[1], freq.shape[2], device=x.device) >= ratio
        keep[:, 0, :] = True
        freq = freq * keep
        return torch.fft.irfft(freq, n=x.shape[1], dim=1).to(dtype=x.dtype)


class ReliabilityGate(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(target_dim * 2, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        reliability = torch.sigmoid(-log_var.detach())
        correction = self.proj(torch.cat([mean, reliability], dim=-1))
        return mean + correction * reliability


class UncertaintyCalibrator(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        hidden_dim: int = 64,
        use_reliability_gate: bool = True,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.log_var_bias = nn.Parameter(torch.zeros(prediction_length, target_dim))
        self.residual_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.gate = ReliabilityGate(target_dim) if use_reliability_gate else None

    def forward(self, mean: torch.Tensor, base_scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean.ndim != 3:
            raise ValueError(f"Expected mean as [B, H, C], got {tuple(mean.shape)}")
        bias = self.log_var_bias[: mean.shape[1], : mean.shape[2]].unsqueeze(0)
        signal = self.residual_encoder(mean.detach())
        log_var = bias + signal
        if base_scale is not None:
            log_var = log_var + 2.0 * torch.log(base_scale.clamp_min(1e-6))
        refined_mean = self.gate(mean, log_var) if self.gate is not None else mean
        return refined_mean, log_var.clamp(min=-12.0, max=12.0)

    def distribution(self, mean: torch.Tensor, base_scale: torch.Tensor = None):
        refined_mean, log_var = self(mean, base_scale)
        scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return torch.distributions.Normal(refined_mean, scale), log_var


def consistency_residual(
    prediction: torch.Tensor,
    augmented_prediction: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    residual = torch.abs(prediction - augmented_prediction)
    return residual.detach() if detach else residual


def uncertainty_nll(
    residual: torch.Tensor,
    log_var: torch.Tensor,
    likelihood: str = "laplace",
) -> torch.Tensor:
    if likelihood == "laplace":
        return math.sqrt(2.0) * torch.exp(-0.5 * log_var) * residual + 0.5 * log_var
    if likelihood == "gaussian":
        return 0.5 * (torch.exp(-log_var) * residual.pow(2) + log_var)
    raise ValueError(f"Unsupported likelihood '{likelihood}'. Use 'laplace' or 'gaussian'.")


def sample_from_normal(mean: torch.Tensor, log_var: torch.Tensor, num_samples: int) -> torch.Tensor:
    scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
    dist = torch.distributions.Normal(mean, scale)
    return dist.rsample(sample_shape=(num_samples,)).permute(1, 0, 2, 3)


def rescale_samples_by_calibration(
    base_samples: torch.Tensor,
    refined_mean: torch.Tensor,
    calibrated_scale: torch.Tensor,
) -> torch.Tensor:
    if base_samples.ndim != 4:
        raise ValueError(f"Expected base_samples as [B, N, H, C], got {tuple(base_samples.shape)}")
    if refined_mean.shape != base_samples.shape[:1] + base_samples.shape[2:]:
        raise ValueError(
            f"Expected refined_mean shape {tuple(base_samples.shape[:1] + base_samples.shape[2:])}, "
            f"got {tuple(refined_mean.shape)}"
        )
    if calibrated_scale.shape != refined_mean.shape:
        raise ValueError(f"Expected calibrated_scale shape {tuple(refined_mean.shape)}, got {tuple(calibrated_scale.shape)}")

    unbiased = base_samples.shape[1] > 1
    base_mean = base_samples.mean(dim=1, keepdim=True)
    base_scale = base_samples.std(dim=1, keepdim=True, unbiased=unbiased).clamp_min(1e-6)
    standardized = (base_samples - base_mean) / base_scale
    return refined_mean.unsqueeze(1) + standardized * calibrated_scale.unsqueeze(1)


def blend_log_variance(
    base_scale: torch.Tensor,
    calibrated_log_var: torch.Tensor,
    calibration_strength: float,
) -> torch.Tensor:
    if not 0.0 <= calibration_strength <= 1.0:
        raise ValueError(f"calibration_strength must be in [0, 1], got {calibration_strength}")
    base_log_var = 2.0 * torch.log(base_scale.clamp_min(1e-6))
    return (1.0 - calibration_strength) * base_log_var + calibration_strength * calibrated_log_var


def sparsification_error(
    targets: torch.Tensor,
    forecasts: torch.Tensor,
    uncertainty: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:
    error = torch.abs(forecasts - targets).reshape(targets.shape[0], -1)
    unc = uncertainty.reshape(targets.shape[0], -1)
    fractions = torch.linspace(0, 1, steps + 1, device=targets.device)
    curve = []
    oracle = []
    for frac in fractions:
        keep = max(1, int((1.0 - float(frac)) * error.shape[1]))
        unc_idx = torch.argsort(unc, dim=1)[:, :keep]
        err_idx = torch.argsort(error, dim=1)[:, :keep]
        curve.append(torch.gather(error, 1, unc_idx).mean(dim=1))
        oracle.append(torch.gather(error, 1, err_idx).mean(dim=1))
    curve = torch.stack(curve, dim=1)
    oracle = torch.stack(oracle, dim=1)
    return torch.trapz(torch.abs(curve - oracle), fractions, dim=1).mean()
