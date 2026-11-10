"""
SpecVAE wrapper for ProbTS Forecaster.

Upgrades from LSG-VAE:
- Learnable spectral decomposition (STFT with learnable window + band boundaries)
- Band-wise variational encoding with cross-band attention
- Frequency-aware latent dynamics
- GRU-based volatility dynamics with cross-band uncertainty coupling
- Band-wise spectral ELBO loss
"""

import torch
import torch.nn.functional as F

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.SpecVAE import SpecVAE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class SpecVAEModel(Forecaster):
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
        n_bands=3,
        cross_attn_dk=32,
        vol_hidden_dim=64,
        alpha_init=0.1,
        weight_beta=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ---- Config assembly ----
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

        # SpecVAE-specific
        config.n_bands = n_bands
        config.cross_attn_dk = cross_attn_dk
        config.vol_hidden_dim = vol_hidden_dim
        config.alpha_init = alpha_init

        # ProbTS info
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length

        # KL weight
        self.weight_beta = weight_beta
        self.n_bands = n_bands

        # ---- Instantiate Model ----
        self.model = SpecVAE(config)

    def forward(self, input: torch.Tensor):
        """
        input: [B, L, C]
        Returns all SpecVAE outputs.
        """
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        """
        Spectral ELBO loss:
            L = sum_b [ NLL_rec^(b) + NLL_pred^(b) ] + beta * sum_b KL^(b)

        Each band has its own heteroscedastic Gaussian NLL.
        """
        self.model.train()

        # Get history and target
        x = batch_data.past_target_cdf[:, -self.context_length:, :]      # [B, L, C]
        target = batch_data.future_target_cdf                             # [B, pred_len, C]

        # Forward
        (x_mu, y_mu, x_sigma, y_sigma,
         band_mu_z, band_logvar_z, band_signals,
         band_x_mu, band_y_mu, band_x_sigma, band_y_sigma) = self.forward(x)

        # ---- Band-wise Reconstruction NLL ----
        # Decompose target into bands for band-wise loss
        # For past: use band_signals from encoder (already decomposed)
        rec_loss = self.model.band_nll_loss(
            band_signals, band_x_mu, band_x_sigma, reduction="mean"
        )

        # ---- Band-wise Prediction NLL ----
        # For future: decompose target through spectral module
        # Since we don't have ground-truth band decomposition of the future,
        # we use the aggregated prediction against the full target
        pred_loss = self.model.gaussian_nll_sigma(
            target, y_mu, y_sigma, reduction="mean"
        )

        # ---- KL Loss (sum across bands) ----
        kld_loss = self.model.kl_divergence(band_mu_z, band_logvar_z)

        print(
            f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, "
            f"kld_loss: {kld_loss:.4f}, alpha: {self.model.alpha.item():.4f}"
        )

        loss = rec_loss + pred_loss + self.weight_beta * kld_loss
        return loss

    def forecast(self, batch_data, num_samples=None):
        """
        Forecast future steps using SpecVAE.

        If num_samples is provided, performs generative forecasting
        (sampling latent + observation noise with spectral uncertainty).

        Returns:
            outputs: [B, num_samples, pred_len, C]
                or [B, 1, pred_len, C] when num_samples is None
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]

        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)  # [B, num_samples, pred_len, C]
            else:
                (x_mu, y_mu, x_sigma, y_sigma,
                 *_) = self.forward(x)
                outputs = y_mu.unsqueeze(1)  # [B, 1, pred_len, C]

        return outputs
