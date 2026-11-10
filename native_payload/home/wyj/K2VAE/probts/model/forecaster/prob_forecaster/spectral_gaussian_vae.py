import torch

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.spectral_gaussian_vae import SpectralGaussianVAE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class SpectralGaussianVAEModel(Forecaster):
    """Incremental frequency-aware GaussianVAE wrapper."""

    def __init__(
        self,
        d_model,
        d_ff,
        e_layers,
        dropout,
        activation,
        n_heads,
        factor,
        patch_len,
        multistep,
        dynamic_dim,
        hidden_layers,
        hidden_dim,
        weight_beta=0.001,
        spectral_cond_dim=64,
        sample_schedule=5,
        init_kalman="identity",
        init_koopman="both",
        **kwargs,
    ):
        super().__init__(**kwargs)

        config = ConvertedParams(kwargs)
        config.d_model = d_model
        config.d_ff = d_ff
        config.hidden_layers = hidden_layers
        config.dropout = dropout
        config.activation = activation
        config.e_layers = e_layers
        config.n_heads = n_heads
        config.factor = factor
        config.patch_len = patch_len
        config.multistep = multistep
        config.dynamic_dim = dynamic_dim
        config.hidden_dim = hidden_dim
        config.spectral_cond_dim = spectral_cond_dim
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        self.weight_beta = weight_beta
        self.model = SpectralGaussianVAE(config)

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        self.model.train()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        target = batch_data.future_target_cdf

        x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar, _, _ = self.forward(x)
        rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")
        pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")
        kld_loss = self.model.kl_divergence(mu_z, logvar_z)

        print(
            f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, kld_loss: {kld_loss:.4f}"
        )
        return rec_loss + pred_loss + self.weight_beta * kld_loss

    def forecast(self, batch_data, num_samples=None):
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                _, y_mu, _, _, _, _, _, _ = self.forward(x)
                outputs = y_mu.unsqueeze(1)
        return outputs
