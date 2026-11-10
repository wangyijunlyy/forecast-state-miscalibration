"""Model-independent SPARC calibration head for sample-based forecasters."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .samples import augmentation_residuals, sample_preserving_rescale, summarize_canonical_samples


class SampleSPARCCalibrator(nn.Module):
    """Sample-Preserving Adaptive Residual Calibration for external backbones."""

    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        hidden_dim: int = 128,
        horizon_embedding_dim: int = 8,
        mean_correction: bool = True,
        max_mean_delta: float = 0.25,
        max_log_scale_delta: float = 1.0,
        calibration_strength: float = 0.8,
        nll_weight: float = 1.0,
        quantile_weight: float = 0.2,
        coverage_weight: float = 0.1,
        ranking_weight: float = 0.05,
        pit_weight: float = 0.0,
        pit_temperature: float = 0.02,
    ):
        super().__init__()
        if not 0.0 <= calibration_strength <= 1.0:
            raise ValueError("calibration_strength must be in [0, 1]")
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.mean_correction = mean_correction
        self.max_mean_delta = max_mean_delta
        self.max_log_scale_delta = max_log_scale_delta
        self.calibration_strength = calibration_strength
        self.nll_weight = nll_weight
        self.quantile_weight = quantile_weight
        self.coverage_weight = coverage_weight
        self.ranking_weight = ranking_weight
        self.pit_weight = pit_weight
        self.pit_temperature = pit_temperature

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
        if horizon > self.prediction_length or target_dim > self.target_dim:
            raise ValueError(
                f"Input horizon/target_dim {(horizon, target_dim)} exceeds calibrator "
                f"{(self.prediction_length, self.target_dim)}"
            )
        horizon_idx = torch.arange(horizon, device=mean.device)
        horizon_emb = self.horizon_embedding(horizon_idx).view(1, horizon, 1, -1)
        horizon_emb = horizon_emb.expand(batch_size, horizon, target_dim, -1)
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
        mean_delta = torch.tanh(raw[..., 0]) * self.max_mean_delta * base_scale.detach()
        log_scale_delta = torch.tanh(raw[..., 1]) * self.max_log_scale_delta
        bias = self.base_log_scale_bias[: mean.shape[1], : mean.shape[2]].unsqueeze(0)
        refined_mean = mean + mean_delta if self.mean_correction else mean
        calibrated_log_var = 2.0 * (torch.log(base_scale.clamp_min(1e-6)) + bias + log_scale_delta)
        base_log_var = 2.0 * torch.log(base_scale.clamp_min(1e-6))
        log_var = (1.0 - self.calibration_strength) * base_log_var + self.calibration_strength * calibrated_log_var
        return refined_mean, log_var.clamp(min=-12.0, max=12.0)

    def distribution(
        self,
        mean: torch.Tensor,
        base_scale: torch.Tensor,
        loc_residual: torch.Tensor,
        scale_residual: torch.Tensor,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        refined_mean, log_var = self(mean, base_scale, loc_residual, scale_residual)
        scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return torch.distributions.Normal(refined_mean, scale), log_var

    def calibrate_samples(self, base_samples: torch.Tensor, augmented_samples: torch.Tensor) -> torch.Tensor:
        mean, base_scale = summarize_canonical_samples(base_samples)
        loc_residual, scale_residual = augmentation_residuals(base_samples, augmented_samples)
        refined_mean, log_var = self(mean, base_scale, loc_residual, scale_residual)
        calibrated_scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return sample_preserving_rescale(base_samples, refined_mean, calibrated_scale)

    def params_from_stats_samples(
        self,
        base_stats_samples: torch.Tensor,
        augmented_stats_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate calibration parameters from a potentially larger stats sample set."""

        mean, base_scale = summarize_canonical_samples(base_stats_samples)
        loc_residual, scale_residual = augmentation_residuals(base_stats_samples, augmented_stats_samples)
        return self(mean, base_scale, loc_residual, scale_residual)

    def calibrate_samples_with_stats(
        self,
        samples_to_calibrate: torch.Tensor,
        base_stats_samples: torch.Tensor,
        augmented_stats_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate output samples using more stable mean/std statistics.

        `samples_to_calibrate` controls the number of returned samples. The stats
        tensors can use a larger sample count to reduce Monte Carlo noise in the
        SPARC features.
        """

        refined_mean, log_var = self.params_from_stats_samples(base_stats_samples, augmented_stats_samples)
        calibrated_scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return sample_preserving_rescale(samples_to_calibrate, refined_mean, calibrated_scale)

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
    def _soft_coverage_loss(dist: torch.distributions.Normal, target: torch.Tensor) -> torch.Tensor:
        lower = dist.icdf(torch.full_like(target, 0.05))
        upper = dist.icdf(torch.full_like(target, 0.95))
        temperature = dist.scale.detach().clamp_min(1e-3) * 0.1
        soft_inside = torch.sigmoid((target - lower) / temperature) * torch.sigmoid((upper - target) / temperature)
        return (soft_inside.mean() - 0.9).pow(2)

    @staticmethod
    def _pit_uniform_loss(
        dist: torch.distributions.Normal,
        target: torch.Tensor,
        temperature: float = 0.02,
    ) -> torch.Tensor:
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
        target_proportion = 1.0 / proportions.numel()
        return torch.mean(torch.abs(proportions - target_proportion))

    @staticmethod
    def _ranking_loss(target: torch.Tensor, refined_mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        error = torch.abs(target - refined_mean).detach()
        pred_unc = torch.exp(0.5 * log_var)
        error = (error - error.mean()) / error.std(unbiased=False).clamp_min(1e-6)
        pred_unc = (pred_unc - pred_unc.mean()) / pred_unc.std(unbiased=False).clamp_min(1e-6)
        return F.mse_loss(pred_unc, error)

    def loss(self, base_samples: torch.Tensor, augmented_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, base_scale = summarize_canonical_samples(base_samples)
        loc_residual, scale_residual = augmentation_residuals(base_samples, augmented_samples)
        dist, log_var = self.distribution(mean, base_scale, loc_residual, scale_residual)
        nll = -dist.log_prob(target).mean()
        qloss = self._normal_quantile_loss(dist, target)
        coverage = self._soft_coverage_loss(dist, target)
        ranking = self._ranking_loss(target, dist.loc, log_var)
        pit = self._pit_uniform_loss(dist, target, self.pit_temperature)
        return (
            self.nll_weight * nll
            + self.quantile_weight * qloss
            + self.coverage_weight * coverage
            + self.ranking_weight * ranking
            + self.pit_weight * pit
        )
