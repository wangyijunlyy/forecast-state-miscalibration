"""Sample-preserving SPARC calibration head for PPM samples."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def summarize_samples(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 4:
        raise ValueError(f"Expected [B, K, L, C] samples, got {tuple(samples.shape)}")
    mean = samples.mean(dim=1)
    scale = samples.std(dim=1, unbiased=samples.shape[1] > 1).clamp_min(1e-6)
    return mean, scale


def augmentation_residuals(base_samples: torch.Tensor, augmented_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    base_mean, base_scale = summarize_samples(base_samples)
    aug_mean, aug_scale = summarize_samples(augmented_samples)
    loc_residual = torch.abs(base_mean - aug_mean)
    scale_residual = torch.abs(torch.log(base_scale.clamp_min(1e-6)) - torch.log(aug_scale.clamp_min(1e-6)))
    return loc_residual.detach(), scale_residual.detach()


def sample_preserving_rescale(samples: torch.Tensor, refined_mean: torch.Tensor, calibrated_scale: torch.Tensor) -> torch.Tensor:
    base_mean, base_scale = summarize_samples(samples)
    standardized = (samples - base_mean.unsqueeze(1)) / base_scale.unsqueeze(1)
    return refined_mean.unsqueeze(1) + standardized * calibrated_scale.unsqueeze(1)


def augment_context(x: torch.Tensor, jitter_std: float = 0.01, scaling_std: float = 0.05, time_mask_ratio: float = 0.05) -> torch.Tensor:
    y = x.clone()
    if jitter_std > 0:
        y = y + torch.randn_like(y) * jitter_std
    if scaling_std > 0:
        scale = 1.0 + torch.randn(y.shape[0], 1, y.shape[2], device=y.device, dtype=y.dtype) * scaling_std
        y = y * scale.clamp_min(1e-3)
    if time_mask_ratio > 0:
        mask = torch.rand(y.shape[0], y.shape[1], 1, device=y.device) < time_mask_ratio
        fill = y.mean(dim=1, keepdim=True).detach()
        y = torch.where(mask, fill.expand_as(y), y)
    return y


class SampleSPARCCalibrator(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        hidden_dim: int = 128,
        horizon_embedding_dim: int = 8,
        mean_correction: bool = True,
        max_mean_delta: float = 0.05,
        max_log_scale_delta: float = 0.5,
        calibration_strength: float = 0.5,
        nll_weight: float = 0.5,
        quantile_weight: float = 0.5,
        pit_weight: float = 30.0,
        pit_temperature: float = 0.01,
        use_perturbation_features: bool = True,
        bounded_updates: bool = True,
        reconstruction_mode: str = "residual",
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.mean_correction = mean_correction
        self.max_mean_delta = max_mean_delta
        self.max_log_scale_delta = max_log_scale_delta
        self.calibration_strength = calibration_strength
        self.nll_weight = nll_weight
        self.quantile_weight = quantile_weight
        self.pit_weight = pit_weight
        self.pit_temperature = pit_temperature
        self.use_perturbation_features = use_perturbation_features
        self.bounded_updates = bounded_updates
        if reconstruction_mode not in {"residual", "gaussian"}:
            raise ValueError(f"Unknown reconstruction mode: {reconstruction_mode}")
        self.reconstruction_mode = reconstruction_mode

        self.horizon_embedding = nn.Embedding(prediction_length, horizon_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(4 + horizon_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.base_log_scale_bias = nn.Parameter(torch.zeros(prediction_length, target_dim))

    def _features(
        self,
        mean: torch.Tensor,
        base_scale: torch.Tensor,
        loc_residual: torch.Tensor,
        scale_residual: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, target_dim = mean.shape
        horizon_idx = torch.arange(horizon, device=mean.device)
        horizon_emb = self.horizon_embedding(horizon_idx).view(1, horizon, 1, -1)
        horizon_emb = horizon_emb.expand(batch_size, horizon, target_dim, -1)
        if not self.use_perturbation_features:
            loc_residual = torch.zeros_like(loc_residual)
            scale_residual = torch.zeros_like(scale_residual)
        scalar_features = torch.stack(
            [
                mean.detach(),
                torch.log(base_scale.clamp_min(1e-6)).detach(),
                loc_residual.detach(),
                scale_residual.detach(),
            ],
            dim=-1,
        )
        return torch.cat([scalar_features, horizon_emb], dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        base_scale: torch.Tensor,
        loc_residual: torch.Tensor,
        scale_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._features(mean, base_scale, loc_residual, scale_residual)
        raw = self.net(features)
        if self.bounded_updates:
            mean_delta = torch.tanh(raw[..., 0]) * self.max_mean_delta * base_scale.detach()
            log_scale_delta = torch.tanh(raw[..., 1]) * self.max_log_scale_delta
        else:
            mean_delta = raw[..., 0] * base_scale.detach()
            log_scale_delta = raw[..., 1]
        bias = self.base_log_scale_bias[: mean.shape[1], : mean.shape[2]].unsqueeze(0)
        refined_mean = mean + mean_delta if self.mean_correction else mean
        calibrated_log_var = 2.0 * (torch.log(base_scale.clamp_min(1e-6)) + bias + log_scale_delta)
        base_log_var = 2.0 * torch.log(base_scale.clamp_min(1e-6))
        log_var = (1.0 - self.calibration_strength) * base_log_var + self.calibration_strength * calibrated_log_var
        return refined_mean, log_var.clamp(min=-12.0, max=12.0)

    def distribution_from_samples(
        self,
        base_stats_samples: torch.Tensor,
        augmented_stats_samples: torch.Tensor,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean, base_scale = summarize_samples(base_stats_samples)
        loc_residual, scale_residual = augmentation_residuals(base_stats_samples, augmented_stats_samples)
        refined_mean, log_var = self(mean, base_scale, loc_residual, scale_residual)
        scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return torch.distributions.Normal(refined_mean, scale), log_var

    def calibrate_samples_with_stats(
        self,
        samples_to_calibrate: torch.Tensor,
        base_stats_samples: torch.Tensor,
        augmented_stats_samples: torch.Tensor,
    ) -> torch.Tensor:
        dist, _log_var = self.distribution_from_samples(base_stats_samples, augmented_stats_samples)
        if self.reconstruction_mode == "residual":
            return sample_preserving_rescale(samples_to_calibrate, dist.loc, dist.scale)
        sample_count = samples_to_calibrate.shape[1]
        return dist.rsample((sample_count,)).permute(1, 0, 2, 3)

    @staticmethod
    def _normal_quantile_loss(dist: torch.distributions.Normal, target: torch.Tensor) -> torch.Tensor:
        quantiles = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            device=target.device,
            dtype=target.dtype,
        )
        losses = []
        for q in quantiles:
            q_value = dist.icdf(q.expand_as(target))
            error = target - q_value
            losses.append(torch.maximum(q * error, (q - 1.0) * error))
        return torch.stack(losses, dim=0).mean()

    @staticmethod
    def _pit_uniform_loss(dist: torch.distributions.Normal, target: torch.Tensor, temperature: float) -> torch.Tensor:
        quantiles = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            device=target.device,
            dtype=target.dtype,
        )
        pit = dist.cdf(target).clamp(1e-6, 1.0 - 1e-6)
        temperature = max(float(temperature), 1e-4)
        coverages = [torch.sigmoid((q - pit) / temperature).mean() for q in quantiles]
        all_coverages = [torch.zeros((), device=target.device, dtype=target.dtype)] + coverages + [
            torch.ones((), device=target.device, dtype=target.dtype)
        ]
        proportions = torch.stack([all_coverages[i + 1] - all_coverages[i] for i in range(len(all_coverages) - 1)])
        return torch.mean(torch.abs(proportions - 1.0 / proportions.numel()))

    def loss(self, base_stats_samples: torch.Tensor, augmented_stats_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist, _log_var = self.distribution_from_samples(base_stats_samples, augmented_stats_samples)
        nll = -dist.log_prob(target).mean()
        qloss = self._normal_quantile_loss(dist, target)
        pit = self._pit_uniform_loss(dist, target, self.pit_temperature)
        return self.nll_weight * nll + self.quantile_weight * qloss + self.pit_weight * pit
