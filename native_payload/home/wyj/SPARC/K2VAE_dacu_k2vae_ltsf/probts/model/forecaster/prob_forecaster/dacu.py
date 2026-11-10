import copy
import importlib
from typing import Any, Dict, Optional, Tuple

import torch

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


FULL_ADAPTERS = {"k2vae", "dlinear", "patchtst", "itransformer", "voldy_vae"}
FORECAST_MODES = {"calibrated_normal", "backbone", "rescale"}


def _load_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _merge_backbone_kwargs(backbone_init_args: Optional[Dict[str, Any]], shared_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(shared_kwargs)
    merged.update(backbone_init_args or {})
    return merged


class DACUForecaster(Forecaster):
    """Decoupled augmentation-consistency uncertainty wrapper for ProbTS forecasters."""

    def __init__(
        self,
        backbone_class_path: str,
        backbone_init_args: Optional[Dict[str, Any]] = None,
        adapter_mode: str = "auto",
        likelihood: str = "laplace",
        uncertainty_weight: float = 0.005,
        consistency_weight: float = 0.02,
        calibration_hidden_dim: int = 64,
        use_reliability_gate: bool = False,
        consistency_samples: int = 8,
        forecast_mode: str = "rescale",
        calibration_strength: float = 0.2,
        jitter_std: float = 0.01,
        scaling_std: float = 0.05,
        time_mask_ratio: float = 0.05,
        channel_mask_ratio: float = 0.05,
        freq_dropout_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args or {}
        self.adapter_mode = adapter_mode
        self.likelihood = likelihood
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        self.consistency_samples = consistency_samples
        if forecast_mode not in FORECAST_MODES:
            raise ValueError(f"forecast_mode must be one of {sorted(FORECAST_MODES)}, got {forecast_mode!r}")
        self.forecast_mode = forecast_mode
        if not 0.0 <= calibration_strength <= 1.0:
            raise ValueError(f"calibration_strength must be in [0, 1], got {calibration_strength}")
        self.calibration_strength = calibration_strength

        backbone_cls = _load_class(backbone_class_path)
        backbone_kwargs = _merge_backbone_kwargs(self.backbone_init_args, kwargs)
        self.backbone = backbone_cls(**backbone_kwargs)
        self.backbone_name = self.backbone.__class__.__name__.lower()
        self.full_adapter = self._resolve_full_adapter()

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
            use_reliability_gate=use_reliability_gate,
        )

    @property
    def name(self):
        return f"DACU-{self.backbone.__class__.__name__}"

    def _resolve_full_adapter(self) -> bool:
        if self.adapter_mode == "full":
            return True
        if self.adapter_mode == "sample":
            return False
        if self.adapter_mode != "auto":
            raise ValueError("adapter_mode must be one of: auto, full, sample")
        return any(name in self.backbone_name for name in FULL_ADAPTERS)

    def _clone_with_past(self, batch_data, past_target: torch.Tensor):
        augmented = copy.copy(batch_data)
        augmented.past_target_cdf = past_target
        return augmented

    def _context(self, batch_data) -> torch.Tensor:
        return batch_data.past_target_cdf[:, -self.max_context_length :, :]

    def _with_augmented_context(self, batch_data, augmented_context: torch.Tensor) -> torch.Tensor:
        past_target = batch_data.past_target_cdf
        if past_target.shape[1] <= self.max_context_length:
            return augmented_context

        augmented_past = past_target.clone()
        augmented_past[:, -self.max_context_length :, :] = augmented_context
        return augmented_past

    def _distribution_from_full_adapter(self, batch_data):
        if "k2vae" in self.backbone_name:
            result = self.backbone.forward(self._context(batch_data))
            return result[-1]
        if "voldy_vae" in self.backbone_name:
            _, y_mu, _, _, _, y_sigma = self.backbone.forward(self._context(batch_data))
            return torch.distributions.Normal(y_mu, y_sigma.clamp_min(1e-6))

        inputs = self.backbone.get_inputs(batch_data, "encode")
        if hasattr(self.backbone, "enc_linear"):
            inputs = self.backbone.enc_linear(inputs)
        if hasattr(self.backbone, "encoder"):
            result = self.backbone.encoder(inputs)
        else:
            result = self.backbone(inputs)
        if hasattr(result, "loc") and hasattr(result, "scale"):
            return result
        raise RuntimeError(f"Backbone {self.backbone.__class__.__name__} did not return a distribution.")

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

    def _sample_adapter_stats(self, batch_data, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_count = num_samples or self.consistency_samples
        samples = self._backbone_samples(batch_data, sample_count)
        mean = samples.mean(dim=1)
        scale = samples.std(dim=1, unbiased=sample_count > 1).clamp_min(1e-6)
        return mean, scale

    def _predict_stats(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.full_adapter:
            dist = self._distribution_from_full_adapter(batch_data)
            return dist.loc, dist.scale.clamp_min(1e-6)
        return self._sample_adapter_stats(batch_data)

    def _dacu_terms(self, batch_data):
        mean, base_scale = self._predict_stats(batch_data)
        augmented_context, _ = self.augmenter(self._context(batch_data))
        augmented_past = self._with_augmented_context(batch_data, augmented_context)
        augmented_batch = self._clone_with_past(batch_data, augmented_past)
        augmented_mean, _ = self._predict_stats(augmented_batch)

        residual = consistency_residual(mean, augmented_mean, detach=True)
        refined_dist, log_var = self.calibrator.distribution(mean, base_scale)
        unc_loss = uncertainty_nll(residual, log_var, likelihood=self.likelihood).mean()
        consistency_loss = torch.abs(mean - augmented_mean).mean()
        return refined_dist, unc_loss, consistency_loss

    def loss(self, batch_data):
        base_loss = self.backbone.loss(batch_data)
        if self.uncertainty_weight == 0.0 and self.consistency_weight == 0.0:
            return base_loss

        _, unc_loss, consistency_loss = self._dacu_terms(batch_data)
        return base_loss + self.uncertainty_weight * unc_loss + self.consistency_weight * consistency_loss

    def forecast(self, batch_data, num_samples=None):
        sample_count = num_samples or 100
        if self.forecast_mode == "backbone":
            return self._backbone_samples(batch_data, sample_count)

        if self.forecast_mode == "rescale":
            base_samples = self._backbone_samples(batch_data, sample_count)
            base_mean = base_samples.mean(dim=1)
            base_scale = base_samples.std(dim=1, unbiased=sample_count > 1).clamp_min(1e-6)
            refined_mean, log_var = self.calibrator(base_mean, base_scale)
            log_var = blend_log_variance(base_scale, log_var, self.calibration_strength)
            calibrated_scale = torch.exp(0.5 * log_var).clamp_min(1e-6)
            return rescale_samples_by_calibration(base_samples, refined_mean, calibrated_scale)

        mean, base_scale = self._predict_stats(batch_data)
        refined_mean, log_var = self.calibrator(mean, base_scale)
        log_var = blend_log_variance(base_scale, log_var, self.calibration_strength)
        return sample_from_normal(refined_mean, log_var, sample_count)
