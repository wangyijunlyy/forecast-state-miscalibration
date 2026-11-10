from probts.model.forecaster.forecaster import Forecaster
import torch
import torch.nn.functional as F
from probts.model.nn.prob.k2VAE.QVAE_probf import QVAE_Probf_NN

class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

class QVAE_Probf_Forecaster(Forecaster):
    def __init__(
        self,
        n_qbits=10,
        n_blocks=1,
        bottleneck_qbits=4,
        kl_weight=1.0,
        learning_rate=1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Config assembly
        config = ConvertedParams(kwargs)
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.n_vars = self.input_size
        
        # Quantum Params
        config.n_qbits = n_qbits
        config.n_blocks = n_blocks
        config.bottleneck_qbits = bottleneck_qbits
        
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight  # β-VAE style KL weight
        
        self.model = QVAE_Probf_NN(config)

    def forward(self, x):
        return self.model(x)

    def gaussian_nll_loss(self, target, mu, logvar):
        """
        Gaussian Negative Log-Likelihood loss.
        NLL = 0.5 * (logvar + (target - mu)^2 / exp(logvar))
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (target - mu) ** 2 / var)
        return nll.mean()

    def loss(self, batch_data, **kwargs):
        """
        VAE ELBO loss with Gaussian NLL: Reconstruction + Prediction + KL Divergence.
        """
        # Get data
        x = batch_data.past_target_cdf[:, -self.context_length:, :]  # [B, L, C]
        y = batch_data.future_target_cdf  # [B, P, C]
        
        # Forward - now returns dual-head outputs
        x_rec_mu, x_rec_logvar, y_pred_mu, y_pred_logvar, mu_latent, logvar_latent, _ = self.forward(x)
        
        # Reconstruction and Prediction Losses (NLL instead of MSE)
        loss_rec = self.gaussian_nll_loss(x, x_rec_mu, x_rec_logvar)
        loss_pred = self.gaussian_nll_loss(y, y_pred_mu, y_pred_logvar)
        
        # KL Divergence for latent space: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
        
        # Total loss (ELBO)
        loss = loss_rec + loss_pred + self.kl_weight * kl_loss
        
        return loss

    def forecast(self, batch_data, num_samples=None):
        """
        Probabilistic forecasting with sampling from predicted Gaussian distribution.
        Returns samples from N(mu, var) for each prediction.
        """
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            if num_samples is None or num_samples == 1:
                # Single deterministic prediction (return mean)
                _, _, y_pred_mu, _, _, _, _ = self.forward(x)
                return y_pred_mu.unsqueeze(1)  # [B, 1, P, C]
            else:
                # Stochastic: sample from predicted Gaussian
                _, _, y_pred_mu, y_pred_logvar, _, _, _ = self.forward(x)
                std = torch.exp(0.5 * torch.clamp(y_pred_logvar, min=-10, max=10))
                
                # Generate num_samples samples
                preds = []
                for _ in range(num_samples):
                    eps = torch.randn_like(y_pred_mu)
                    sample = y_pred_mu + eps * std
                    preds.append(sample)
                
                return torch.stack(preds, dim=1)  # [B, N, P, C]
