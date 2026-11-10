"""
* @description: FreqDecompGaussianVAE wrapper for ProbTS Forecaster
  Frequency-Decomposed VAE with explicit Location-Scale Decoupling
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.gaussian_vae import FreqDecompGaussianVAE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class FreqDecompVAEModel(Forecaster):
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
        freq_cutoff_ratio=0.1,
        sample_schedule=5,
        init_kalman="identity",
        init_koopman="both",
        **kwargs,
    ):
        """
        FreqDecompGaussianVAE wrapper constructor
        
        Args:
            freq_cutoff_ratio: Fraction of frequency bins to keep as "low frequency"
                               Default 0.1 means lowest 10% frequencies for trend
        """
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

        # Frequency decomposition parameter
        config.freq_cutoff_ratio = freq_cutoff_ratio

        # ProbTS info
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        # KL weight
        self.weight_beta = weight_beta

        # ---- Instantiate Model ----
        self.model = FreqDecompGaussianVAE(config)

    def forward(self, input: torch.Tensor):
        """
        Parameters:
            input: [B, L, C]
        Returns:
            x_mu:           [B, L, C]        (past mean, original scale)
            y_mu:           [B, pred_len, C] (future mean, original scale)
            mu_z_loc:       [B, F, H]        (location latent mean)
            logvar_z_loc:   [B, F, H]        (location latent logvar)
            mu_z_scale:     [B, F, H]        (scale latent mean)
            logvar_z_scale: [B, F, H]        (scale latent logvar)
            x_logvar:       [B, L, C]        (past log variance)
            y_logvar:       [B, pred_len, C] (future log variance)
        """
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        """
        loss = rec_nll + pred_nll + weight_beta * KL

        使用 Location-Scale 高斯:
            x_t ~ N(mu_t, sigma_t^2)
        
        频域分解：
            - 低频 -> Location Head (趋势/均值预测)
            - 高频 -> Scale Head (波动率/方差预测)
        """
        self.model.train()

        # Get history part
        # shape: [B, L, C]
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        # shape: [B, pred_len, C]
        target = batch_data.future_target_cdf

        # Forward
        (x_mu, y_mu, mu_z_loc, logvar_z_loc, 
         mu_z_scale, logvar_z_scale, x_logvar, y_logvar) = self.forward(x)

        # ---- Reconstruction NLL (Past) ----
        rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")

        # ---- Prediction NLL (Future) ----
        pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")

        # ---- KL Loss (both location and scale latents) ----
        kld_loss = self.model.kl_divergence(
            mu_z_loc, logvar_z_loc, mu_z_scale, logvar_z_scale
        )

        print(
            f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, kld_loss: {kld_loss:.4f}"
        )

        loss = rec_loss + pred_loss + self.weight_beta * kld_loss

        return loss

    def forecast(self, batch_data, num_samples=None):
        """
        Forecast future steps using Frequency-Decomposed Gaussian VAE.

        If num_samples is provided, it performs generative forecasting
        (sampling both latent and observation noise in Location-Scale model).

        Returns:
            outputs: [B, num_samples, pred_len, C]
                or [B, 1, pred_len, C] when num_samples is None (deterministic mean)
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                # Generative Forecasting (latent + observation noise)
                outputs = self.model.sample(x, num_samples)  # [B, num_samples, pred_len, C]
            else:
                # Deterministic Forecasting (Mean only)
                (x_mu, y_mu, _, _, _, _, _, _) = self.forward(x)
                outputs = y_mu.unsqueeze(1)  # [B, 1, pred_len, C]

        return outputs
