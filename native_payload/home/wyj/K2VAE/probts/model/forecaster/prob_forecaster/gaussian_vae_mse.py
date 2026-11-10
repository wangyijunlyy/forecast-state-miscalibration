"""
* @description: GaussianVAE wrapper for ProbTS Forecaster
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.gaussian_vae_mse import GaussianVAE_MSE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class GaussianVAE_MSE_Model(Forecaster):
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
        sample_schedule=5,
        init_kalman='identity',
        init_koopman='both',   
        **kwargs,
    ):
        """
        GaussianVAE wrapper constructor
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
        self.model = GaussianVAE_MSE(config)

    def forward(self, input: torch.Tensor):
        """
        Parameters:
            input: [B, L, C]
        Returns:
            x_rec:      [B, L, C]
            y_rec:      [B, pred_len, C]
            mu:         [B, F, H]
            logvar:     [B, F, H]
        """
        x_rec, y_rec, mu, logvar = self.model(input)
        return x_rec, y_rec, mu, logvar

    def loss(self, batch_data, **kwargs):
        """
        loss = rec_loss + pred_loss + weight_beta * KL
        """
        self.model.train()

        # Get history part
        x = batch_data.past_target_cdf[:, -self.context_length :, :]   # [B, L, C]
        target = batch_data.future_target_cdf                          # [B, pred_len, C]
        
        # Forward
        x_rec, y_rec, mu, logvar = self.forward(x)

        # ---- Reconstruction Loss (Past) ----
        rec_loss = F.mse_loss(x_rec, x)
        
        # ---- Prediction Loss (Future) ----
        pred_loss = F.mse_loss(y_rec, target)

        # ---- KL Loss ----
        kld_loss = self.model.kl_divergence(mu, logvar)
        
        print(f"rec_loss: {rec_loss:.4f}, pred_loss: {pred_loss:.4f}, kld_loss: {kld_loss:.4f}")

        loss = 0.2*rec_loss + pred_loss + self.weight_beta * kld_loss
        
        return loss

    def forecast(self, batch_data, num_samples=None):
        """
        Forecast future steps using Gaussian VAE.
        Returns prediction of the future repeated num_samples times.
        If num_samples is provided, it performs generative forecasting (sampling from posterior).
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        
        with torch.no_grad():
            if num_samples is not None:
                # Generative Forecasting
                outputs = self.model.sample(x, num_samples) # [B, num_samples, pred_len, C]
            else:
                # Deterministic Forecasting (Mean)
                _, y_rec, _, _ = self.forward(x)
                outputs = y_rec.unsqueeze(1) # [B, 1, pred_len, C]
                
        return outputs

