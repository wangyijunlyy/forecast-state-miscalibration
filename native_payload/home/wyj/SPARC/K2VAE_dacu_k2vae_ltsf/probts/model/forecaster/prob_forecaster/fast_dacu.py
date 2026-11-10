import copy
import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.forecaster import Forecaster
from probts.model.nn.prob.dacu import (
    TimeSeriesAugmenter,
    UncertaintyCalibrator,
    blend_log_variance,
    consistency_residual,
    rescale_samples_by_calibration,
    sample_from_normal,
    uncertainty_nll,
)


RECONSTRUCTION_MODES = {"residual", "gaussian"}


def _load_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _merge_backbone_kwargs(backbone_init_args: Optional[Dict[str, Any]], shared_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(shared_kwargs)
    merged.update(backbone_init_args or {})
    return merged


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


class FastK2VAEMixin:
    def _context(self, batch_data) -> torch.Tensor:
        return batch_data.past_target_cdf[:, -self.max_context_length :, :]

    def _clone_with_past(self, batch_data, past_target: torch.Tensor):
        augmented = copy.copy(batch_data)
        augmented.past_target_cdf = past_target
        return augmented

    def _with_augmented_context(self, batch_data, augmented_context: torch.Tensor) -> torch.Tensor:
        past_target = batch_data.past_target_cdf
        if past_target.shape[1] <= self.max_context_length:
            return augmented_context

        augmented_past = past_target.clone()
        augmented_past[:, -self.max_context_length :, :] = augmented_context
        return augmented_past

    def _distribution(self, batch_data) -> torch.distributions.Normal:
        _, _, post_dist = self.backbone.forward(self._context(batch_data))
        return torch.distributions.Normal(post_dist.loc, post_dist.scale.clamp_min(1e-6))

    @staticmethod
    def _normalize_samples(samples: torch.Tensor, batch_size: int, num_samples: int) -> torch.Tensor:
        if samples.shape[0] == batch_size:
            return samples
        if samples.shape[0] == num_samples and samples.shape[1] == batch_size:
            return samples.permute(1, 0, 2, 3)
        return samples.reshape(batch_size, num_samples, samples.shape[-2], samples.shape[-1])

    def _backbone_samples(self, batch_data, num_samples: int) -> torch.Tensor:
        samples = self.backbone.forecast(batch_data, num_samples)
        return self._normalize_samples(samples, batch_data.past_target_cdf.shape[0], num_samples)

    def _load_backbone_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        candidates = [
            _strip_prefix(state_dict, "forecaster.backbone."),
            _strip_prefix(state_dict, "forecaster."),
            state_dict,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            missing, unexpected = self.backbone.load_state_dict(candidate, strict=False)
            loaded = len(candidate) - len(unexpected)
            if loaded > 0 and len(missing) < len(self.backbone.state_dict()):
                return
        raise RuntimeError(f"Could not load backbone weights from {checkpoint_path}")

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _calibrated_samples(
        self,
        batch_data,
        num_samples: int,
        refined_mean: torch.Tensor,
        log_var: torch.Tensor,
        calibration_strength: float,
    ) -> torch.Tensor:
        base_samples = self._backbone_samples(batch_data, num_samples)
        base_scale = base_samples.std(dim=1, unbiased=num_samples > 1).clamp_min(1e-6)
        log_var = blend_log_variance(base_scale, log_var, calibration_strength)
        if getattr(self, "reconstruction_mode", "residual") == "gaussian":
            return sample_from_normal(refined_mean, log_var, num_samples)
        calibrated_scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return rescale_samples_by_calibration(base_samples, refined_mean, calibrated_scale)


class AugmentedFeatureCalibrator(nn.Module):
    """Feature-driven post-hoc calibrator using distribution and augmentation residuals."""

    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        hidden_dim: int = 128,
        horizon_embedding_dim: int = 8,
        mean_correction: bool = True,
        max_mean_delta: float = 0.25,
        max_log_scale_delta: float = 1.0,
        bounded_updates: bool = True,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.mean_correction = mean_correction
        self.max_mean_delta = max_mean_delta
        self.max_log_scale_delta = max_log_scale_delta
        self.bounded_updates = bounded_updates
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
        log_var = 2.0 * (torch.log(base_scale.clamp_min(1e-6)) + bias + log_scale_delta)
        return refined_mean, log_var.clamp(min=-12.0, max=12.0)

    def distribution(
        self,
        mean: torch.Tensor,
        base_scale: torch.Tensor,
        loc_residual: torch.Tensor,
        scale_residual: torch.Tensor,
    ):
        refined_mean, log_var = self(mean, base_scale, loc_residual, scale_residual)
        scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
        return torch.distributions.Normal(refined_mean, scale), log_var


class FastPosthocProbDACUCalibrator(FastK2VAEMixin, Forecaster):
    """Frozen-backbone DACU calibrator trained from fast K2VAE distribution statistics."""

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        backbone_checkpoint_path: Optional[str] = None,
        likelihood: str = "laplace",
        uncertainty_weight: float = 0.1,
        nll_weight: float = 1.0,
        calibration_hidden_dim: int = 64,
        calibration_strength: float = 0.2,
        jitter_std: float = 0.01,
        scaling_std: float = 0.05,
        time_mask_ratio: float = 0.05,
        channel_mask_ratio: float = 0.0,
        freq_dropout_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.backbone_checkpoint_path = backbone_checkpoint_path
        self.likelihood = likelihood
        self.uncertainty_weight = uncertainty_weight
        self.nll_weight = nll_weight
        self.calibration_strength = calibration_strength

        backbone_cls = _load_class(backbone_class_path)
        self.backbone = backbone_cls(**_merge_backbone_kwargs(self.backbone_init_args, kwargs))
        self._load_backbone_checkpoint(backbone_checkpoint_path)
        self._freeze_backbone()

        self.augmenter = TimeSeriesAugmenter(
            jitter_std=jitter_std,
            scaling_std=scaling_std,
            time_mask_ratio=time_mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            freq_dropout_ratio=freq_dropout_ratio,
        )
        self.calibrator = UncertaintyCalibrator(
            prediction_length=self.max_prediction_length,
            target_dim=self.target_dim,
            hidden_dim=calibration_hidden_dim,
            use_reliability_gate=False,
        )

    @property
    def name(self):
        return f"FastPosthocDACU-{self.backbone.__class__.__name__}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _frozen_stats(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dist = self._distribution(batch_data)
            augmented_context, _ = self.augmenter(self._context(batch_data))
            augmented_batch = self._clone_with_past(
                batch_data,
                self._with_augmented_context(batch_data, augmented_context),
            )
            aug_dist = self._distribution(augmented_batch)
            residual = consistency_residual(dist.loc, aug_dist.loc, detach=True)
        return dist.loc.detach(), dist.scale.detach(), residual

    def loss(self, batch_data):
        mean, base_scale, residual = self._frozen_stats(batch_data)
        refined_dist, log_var = self.calibrator.distribution(mean, base_scale)
        nll = -refined_dist.log_prob(batch_data.future_target_cdf).mean()
        unc = uncertainty_nll(residual, log_var, likelihood=self.likelihood).mean()
        return self.nll_weight * nll + self.uncertainty_weight * unc

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        with torch.no_grad():
            dist = self._distribution(batch_data)
            refined_mean, log_var = self.calibrator(dist.loc, dist.scale)
            return self._calibrated_samples(batch_data, sample_count, refined_mean, log_var, self.calibration_strength)


class SPARCForecaster(FastK2VAEMixin, Forecaster):
    """Sample-Preserving Adaptive Residual Calibration.

    SPARC freezes a pretrained probabilistic forecasting backbone, preserves its
    native sampling process, and learns constrained mean/residual-scale
    corrections as a post-hoc calibration wrapper.
    """

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        backbone_checkpoint_path: Optional[str] = None,
        likelihood: str = "laplace",
        uncertainty_weight: float = 0.05,
        nll_weight: float = 1.0,
        crps_weight: float = 0.2,
        coverage_weight: float = 0.1,
        ranking_weight: float = 0.05,
        pit_weight: float = 0.0,
        pit_temperature: float = 0.02,
        calibration_hidden_dim: int = 128,
        horizon_embedding_dim: int = 8,
        mean_correction: bool = True,
        max_mean_delta: float = 0.25,
        max_log_scale_delta: float = 1.0,
        bounded_updates: bool = True,
        calibration_strength: float = 0.8,
        reconstruction_mode: str = "residual",
        use_perturbation_features: bool = True,
        jitter_std: float = 0.01,
        scaling_std: float = 0.05,
        time_mask_ratio: float = 0.05,
        channel_mask_ratio: float = 0.0,
        freq_dropout_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.backbone_checkpoint_path = backbone_checkpoint_path
        self.likelihood = likelihood
        self.uncertainty_weight = uncertainty_weight
        self.nll_weight = nll_weight
        self.crps_weight = crps_weight
        self.coverage_weight = coverage_weight
        self.ranking_weight = ranking_weight
        self.pit_weight = pit_weight
        self.pit_temperature = pit_temperature
        self.calibration_strength = calibration_strength
        if reconstruction_mode not in RECONSTRUCTION_MODES:
            raise ValueError(
                f"reconstruction_mode must be one of {sorted(RECONSTRUCTION_MODES)}, "
                f"got {reconstruction_mode!r}"
            )
        self.reconstruction_mode = reconstruction_mode
        self.use_perturbation_features = use_perturbation_features

        backbone_cls = _load_class(backbone_class_path)
        self.backbone = backbone_cls(**_merge_backbone_kwargs(self.backbone_init_args, kwargs))
        self._load_backbone_checkpoint(backbone_checkpoint_path)
        self._freeze_backbone()

        self.augmenter = TimeSeriesAugmenter(
            jitter_std=jitter_std,
            scaling_std=scaling_std,
            time_mask_ratio=time_mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            freq_dropout_ratio=freq_dropout_ratio,
        )
        self.calibrator = AugmentedFeatureCalibrator(
            prediction_length=self.max_prediction_length,
            target_dim=self.target_dim,
            hidden_dim=calibration_hidden_dim,
            horizon_embedding_dim=horizon_embedding_dim,
            mean_correction=mean_correction,
            max_mean_delta=max_mean_delta,
            max_log_scale_delta=max_log_scale_delta,
            bounded_updates=bounded_updates,
        )

    @property
    def name(self):
        return f"SPARC-{self.backbone.__class__.__name__}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _frozen_features(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dist = self._distribution(batch_data)
            augmented_context, _ = self.augmenter(self._context(batch_data))
            augmented_batch = self._clone_with_past(
                batch_data,
                self._with_augmented_context(batch_data, augmented_context),
            )
            aug_dist = self._distribution(augmented_batch)
            loc_residual = torch.abs(dist.loc - aug_dist.loc)
            scale_residual = torch.abs(
                torch.log(dist.scale.clamp_min(1e-6)) - torch.log(aug_dist.scale.clamp_min(1e-6))
            )
            if not self.use_perturbation_features:
                loc_residual = torch.zeros_like(loc_residual)
                scale_residual = torch.zeros_like(scale_residual)
        return dist.loc.detach(), dist.scale.detach(), loc_residual.detach(), scale_residual.detach()

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
        lower_q = torch.full_like(target, 0.05)
        upper_q = torch.full_like(target, 0.95)
        lower = dist.icdf(lower_q)
        upper = dist.icdf(upper_q)
        temperature = dist.scale.detach().clamp_min(1e-3) * 0.1
        soft_inside = torch.sigmoid((target - lower) / temperature) * torch.sigmoid((upper - target) / temperature)
        return (soft_inside.mean() - 0.9).pow(2)

    @staticmethod
    def _ranking_loss(target: torch.Tensor, refined_mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        error = torch.abs(target - refined_mean).detach()
        pred_unc = torch.exp(0.5 * log_var)
        error = (error - error.mean()) / error.std(unbiased=False).clamp_min(1e-6)
        pred_unc = (pred_unc - pred_unc.mean()) / pred_unc.std(unbiased=False).clamp_min(1e-6)
        return F.mse_loss(pred_unc, error)

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
        return torch.mean(torch.abs(proportions - 1.0 / proportions.numel()))

    def loss(self, batch_data):
        mean, base_scale, loc_residual, scale_residual = self._frozen_features(batch_data)
        refined_dist, log_var = self.calibrator.distribution(mean, base_scale, loc_residual, scale_residual)
        target = batch_data.future_target_cdf
        nll = -refined_dist.log_prob(target).mean()
        unc = uncertainty_nll(loc_residual, log_var, likelihood=self.likelihood).mean()
        qloss = self._normal_quantile_loss(refined_dist, target)
        coverage = self._soft_coverage_loss(refined_dist, target)
        ranking = self._ranking_loss(target, refined_dist.loc, log_var)
        pit = self._pit_uniform_loss(refined_dist, target, self.pit_temperature)
        return (
            self.nll_weight * nll
            + self.uncertainty_weight * unc
            + self.crps_weight * qloss
            + self.coverage_weight * coverage
            + self.ranking_weight * ranking
            + self.pit_weight * pit
        )

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        with torch.no_grad():
            mean, base_scale, loc_residual, scale_residual = self._frozen_features(batch_data)
            refined_mean, log_var = self.calibrator(mean, base_scale, loc_residual, scale_residual)
            return self._calibrated_samples(batch_data, sample_count, refined_mean, log_var, self.calibration_strength)


class FastPosthocProbDACUv2(SPARCForecaster):
    """Backward-compatible alias for historical fast_posthoc_v2 configs."""


class AffineProbCalibrator(FastK2VAEMixin, Forecaster):
    """Frozen K2VAE post-hoc affine mean/scale calibrator."""

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        backbone_checkpoint_path: Optional[str] = None,
        mean_correction: bool = False,
        calibration_strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.backbone_checkpoint_path = backbone_checkpoint_path
        self.mean_correction = mean_correction
        self.calibration_strength = calibration_strength

        backbone_cls = _load_class(backbone_class_path)
        self.backbone = backbone_cls(**_merge_backbone_kwargs(self.backbone_init_args, kwargs))
        self._load_backbone_checkpoint(backbone_checkpoint_path)
        self._freeze_backbone()

        self.mean_bias = nn.Parameter(torch.zeros(self.max_prediction_length, self.target_dim))
        self.log_scale_bias = nn.Parameter(torch.zeros(self.max_prediction_length, self.target_dim))

    @property
    def name(self):
        return f"AffineCal-{self.backbone.__class__.__name__}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _calibrated_distribution(self, batch_data):
        with torch.no_grad():
            dist = self._distribution(batch_data)
        bias = self.mean_bias[: dist.loc.shape[1], : dist.loc.shape[2]].unsqueeze(0)
        log_scale_bias = self.log_scale_bias[: dist.loc.shape[1], : dist.loc.shape[2]].unsqueeze(0)
        loc = dist.loc.detach() + bias if self.mean_correction else dist.loc.detach()
        scale = dist.scale.detach() * torch.exp(log_scale_bias).clamp_min(1e-6)
        return torch.distributions.Normal(loc, scale.clamp_min(1e-6))

    def loss(self, batch_data):
        dist = self._calibrated_distribution(batch_data)
        return -dist.log_prob(batch_data.future_target_cdf).mean()

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        with torch.no_grad():
            dist = self._calibrated_distribution(batch_data)
            return self._calibrated_samples(
                batch_data,
                sample_count,
                dist.loc,
                2.0 * torch.log(dist.scale.clamp_min(1e-6)),
                self.calibration_strength,
            )


class GlobalAffineProbCalibrator(FastK2VAEMixin, Forecaster):
    """Frozen K2VAE calibrator with one global location and scale update."""

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        backbone_checkpoint_path: Optional[str] = None,
        mean_correction: bool = True,
        calibration_strength: float = 1.0,
        reconstruction_mode: str = "residual",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.backbone_checkpoint_path = backbone_checkpoint_path
        self.mean_correction = mean_correction
        self.calibration_strength = calibration_strength
        if reconstruction_mode not in RECONSTRUCTION_MODES:
            raise ValueError(
                f"reconstruction_mode must be one of {sorted(RECONSTRUCTION_MODES)}, "
                f"got {reconstruction_mode!r}"
            )
        self.reconstruction_mode = reconstruction_mode

        backbone_cls = _load_class(backbone_class_path)
        self.backbone = backbone_cls(**_merge_backbone_kwargs(self.backbone_init_args, kwargs))
        self._load_backbone_checkpoint(backbone_checkpoint_path)
        self._freeze_backbone()

        self.mean_bias = nn.Parameter(torch.zeros(()))
        self.log_scale_bias = nn.Parameter(torch.zeros(()))

    @property
    def name(self):
        return f"GlobalAffineCal-{self.backbone.__class__.__name__}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _calibrated_distribution(self, batch_data):
        with torch.no_grad():
            dist = self._distribution(batch_data)
        loc = dist.loc.detach() + self.mean_bias if self.mean_correction else dist.loc.detach()
        scale = dist.scale.detach() * torch.exp(self.log_scale_bias)
        return torch.distributions.Normal(loc, scale.clamp_min(1e-6))

    def loss(self, batch_data):
        dist = self._calibrated_distribution(batch_data)
        return -dist.log_prob(batch_data.future_target_cdf).mean()

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        with torch.no_grad():
            dist = self._calibrated_distribution(batch_data)
            return self._calibrated_samples(
                batch_data,
                sample_count,
                dist.loc,
                2.0 * torch.log(dist.scale.clamp_min(1e-6)),
                self.calibration_strength,
            )


class FastJointProbDACUForecaster(FastK2VAEMixin, Forecaster):
    """Jointly trained fast DACU wrapper that keeps K2VAE's forward distribution path."""

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        likelihood: str = "laplace",
        uncertainty_weight: float = 0.005,
        nll_weight: float = 0.0,
        consistency_weight: float = 0.0,
        calibration_hidden_dim: int = 64,
        calibration_strength: float = 0.2,
        jitter_std: float = 0.01,
        scaling_std: float = 0.05,
        time_mask_ratio: float = 0.05,
        channel_mask_ratio: float = 0.0,
        freq_dropout_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.likelihood = likelihood
        self.uncertainty_weight = uncertainty_weight
        self.nll_weight = nll_weight
        self.consistency_weight = consistency_weight
        self.calibration_strength = calibration_strength

        backbone_cls = _load_class(backbone_class_path)
        self.backbone = backbone_cls(**_merge_backbone_kwargs(self.backbone_init_args, kwargs))
        self.augmenter = TimeSeriesAugmenter(
            jitter_std=jitter_std,
            scaling_std=scaling_std,
            time_mask_ratio=time_mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            freq_dropout_ratio=freq_dropout_ratio,
        )
        self.calibrator = UncertaintyCalibrator(
            prediction_length=self.max_prediction_length,
            target_dim=self.target_dim,
            hidden_dim=calibration_hidden_dim,
            use_reliability_gate=False,
        )

    @property
    def name(self):
        return f"FastJointDACU-{self.backbone.__class__.__name__}"

    def loss(self, batch_data):
        base_loss = self.backbone.loss(batch_data)
        dist = self._distribution(batch_data)
        augmented_context, _ = self.augmenter(self._context(batch_data))
        augmented_batch = self._clone_with_past(batch_data, self._with_augmented_context(batch_data, augmented_context))
        aug_dist = self._distribution(augmented_batch)

        residual = consistency_residual(dist.loc, aug_dist.loc, detach=True)
        refined_dist, log_var = self.calibrator.distribution(dist.loc, dist.scale)
        unc = uncertainty_nll(residual, log_var, likelihood=self.likelihood).mean()
        nll = -refined_dist.log_prob(batch_data.future_target_cdf).mean()
        consistency = torch.abs(dist.loc - aug_dist.loc).mean()
        return base_loss + self.uncertainty_weight * unc + self.nll_weight * nll + self.consistency_weight * consistency

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        with torch.no_grad():
            dist = self._distribution(batch_data)
            refined_mean, log_var = self.calibrator(dist.loc, dist.scale)
            return self._calibrated_samples(batch_data, sample_count, refined_mean, log_var, self.calibration_strength)
