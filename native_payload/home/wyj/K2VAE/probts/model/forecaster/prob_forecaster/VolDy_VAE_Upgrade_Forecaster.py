"""ProbTS forecaster for the original-backbone VolDy-VAE upgrade."""

import torch
from probts.model.forecaster.prob_forecaster.VolDy_VAE import VolDy_VAEModel
from probts.model.nn.prob.k2VAE.VolDy_VAE_Upgrade import VolDy_VAE_Upgrade


class VolDy_VAE_UpgradeModel(VolDy_VAEModel):
    def __init__(self, anchor_weight=0.8, state_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.anchor_weight = anchor_weight
        config = self.model.config
        config.state_dim = state_dim if state_dim is not None else config.dynamic_dim
        self.model = VolDy_VAE_Upgrade(config)
        self.joint_path_metrics = True

    def forward(self, input):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        result = self.model(x)
        rec = self.model.gaussian_nll(x, result["x_mu"], result["x_sigma"])
        base = self.model.gaussian_nll(target, result["y_mu"], result["base_sigma"])
        marginal = self.model.marginal_nll(target, result)
        kl = self.model.kl_divergence(result["mu_z"], result["logvar_z"])
        return rec + self.anchor_weight * base + (1 - self.anchor_weight) * marginal + self.weight_beta * kl

    def forecast(self, batch_data, num_samples=None):
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        if num_samples is None:
            return self.model(x)["y_mu"].unsqueeze(1)
        return self.model.sample(x, num_samples)
