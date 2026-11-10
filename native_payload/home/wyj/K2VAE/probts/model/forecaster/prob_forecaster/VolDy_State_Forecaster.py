"""ProbTS wrapper for the state-aware VolDy-VAE."""

import torch

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.VolDy_State_VAE import VolDy_State_VAE


class ConvertedParams:
    def __init__(self, values):
        self.__dict__.update(values)


class VolDy_State_VAEModel(Forecaster):
    def __init__(self, d_model, d_ff, e_layers, dropout, activation, n_heads,
                 factor, patch_len, multistep, dynamic_dim, hidden_layers,
                 hidden_dim, weight_beta=0.01, vol_hidden_dim=None,
                 reconstruction_weight=1.0, sample_schedule=5,
                 future_loss_mode="joint", scale_backbone="state",
                 anchor_weight=0.0,
                 init_kalman="identity", init_koopman="both", **kwargs):
        super().__init__(**kwargs)
        config = ConvertedParams(kwargs)
        config.patch_len = patch_len
        config.multistep = multistep
        config.dynamic_dim = dynamic_dim
        config.hidden_layers = hidden_layers
        config.hidden_dim = hidden_dim
        config.vol_hidden_dim = vol_hidden_dim or dynamic_dim
        config.scale_backbone = scale_backbone
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        self.model = VolDy_State_VAE(config)
        self.weight_beta = weight_beta
        self.reconstruction_weight = reconstruction_weight
        if future_loss_mode not in ("joint", "marginal"):
            raise ValueError(f"Unsupported future_loss_mode: {future_loss_mode}")
        self.future_loss_mode = future_loss_mode
        if not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("anchor_weight must be between 0 and 1")
        if anchor_weight and scale_backbone != "legacy":
            raise ValueError("anchor_weight requires the legacy scale backbone")
        self.anchor_weight = anchor_weight
        self.joint_path_metrics = True

    def loss(self, batch_data, **kwargs):
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        y = batch_data.future_target_cdf
        out = self.model(x)
        rec = self.model.gaussian_nll(x, out["history_mean"], out["history_sigma"])
        if self.future_loss_mode == "joint":
            future = self.model.joint_nll(y, out)
        else:
            future = self.model.marginal_nll(y, out)
        if self.anchor_weight:
            base_future = self.model.gaussian_nll(
                y, out["future_mean"], out["base_future_sigma"]
            )
            future = (1.0 - self.anchor_weight) * future + self.anchor_weight * base_future
        kl = self.model.kl_divergence(out["latent_mean"], out["latent_logvar"])
        return future + self.reconstruction_weight * rec + self.weight_beta * kl

    def forecast(self, batch_data, num_samples=None):
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        with torch.no_grad():
            if num_samples is None:
                return self.model(x)["future_mean"].unsqueeze(1)
            return self.model.sample(x, num_samples)
