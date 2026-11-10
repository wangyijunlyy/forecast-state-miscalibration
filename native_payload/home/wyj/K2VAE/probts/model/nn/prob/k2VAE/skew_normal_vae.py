"""
Skew-Normal VAE for Time Series Forecasting

Uses Azzalini Skew-Normal observation model:
    x_t ~ SkewNormal(μ_t, σ_t, α_t)

Latent space remains standard Gaussian N(0, I).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class SkewNormalVAE(nn.Module):
    """
    VAE with Gaussian Prior N(0, I) and Skew-Normal observation model.
    
    For each time step and variable:
        x_t ~ SkewNormal(μ_t, σ_t, α_t)
    
    Encoder:
        x -> patches -> encoder -> latent (mu_z, logvar_z)
    
    Decoder:
        z -> decoder -> [mu_out, logvar_out, alpha_out]
        - mu_out: RevIN denormalized (original scale)
        - logvar_out: clamped log variance
        - alpha_out: soft-clipped skewness via 5*tanh(·)
    """

    def __init__(self, config):
        super().__init__()

        # === Basic Parameters ===
        self.config = config
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep
        self.dynamic_dim = config.dynamic_dim  # latent dim H
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars             # input channels C

        # === Patching ===
        # number of input patches F
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.pred_len = config.pred_len
        # number of output patches S
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Future Projection ===
        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=self.freq * self.dynamic_dim // 2,
            hidden_layers=1,
        )
        
        # === Encoder ===
        # Encoder outputs 2 * dynamic_dim (mu_z and logvar_z)
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Decoder ===
        # Decoder outputs 3 * (patch_len * enc_in):
        #   - mu_out: mean
        #   - logvar_out: log variance
        #   - alpha_out: skewness parameter
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=3 * self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === Numerical Stability Constants ===
        self.logvar_min = -10.0
        self.logvar_max = 10.0
        self.alpha_scale = 5.0  # soft clipping: 5*tanh(alpha_raw)

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            mu_z:      [B, F, H]
            logvar_z:  [B, F, H]
            z:         [B, F, H] (sampled latent)
        """
        B, L, C = x.shape

        # RevIN Normalization (instance-wise)
        x_norm = self.revin(x, "norm")  # [B, L, C]

        # === Patch Padding ===
        if self.padding_len > 0:
            padded = torch.cat(
                [x_norm[:, L - self.padding_len:, :], x_norm],
                dim=1
            )  # [B, F*P, C]
        else:
            padded = x_norm

        # Split into F patches of length P
        patches = padded.chunk(self.freq, dim=1)   # freq * [B, P, C]
        patches = torch.stack(patches, dim=1)      # [B, F, P, C]
        patches = patches.reshape(B, self.freq, -1)  # [B, F, P*C]

        # === Encoder Output ===
        encoded = self.encoder(patches)            # [B, F, 2*H]
        mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1)  # [B, F, H], [B, F, H]

        # === Reparameterization ===
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z if self.training else mu_z  # [B, F, H]

        return mu_z, logvar_z, z

    # =============================================================
    #           DECODING (Skew-Normal)
    # =============================================================
    def decode(self, z: torch.Tensor):
        """
        z: [B, F, H] (Past latent)
        return:
            x_mu:       [B, L, C]        (past mean, denormed)
            y_mu:       [B, pred_len, C] (future mean, denormed)
            x_logvar:   [B, L, C]        (past log variance, clamped)
            y_logvar:   [B, pred_len, C] (future log variance, clamped)
            x_alpha:    [B, L, C]        (past skewness, soft-clipped)
            y_alpha:    [B, pred_len, C] (future skewness, soft-clipped)
        """
        B = z.shape[0]

        # ----- 1. Reconstruct Past (Skew-Normal) -----
        # Decoder outputs 3 * P*C
        x_out = self.decoder(z)                    # [B, F, 3*P*C]
        
        # Split into mu, logvar, alpha (each P*C)
        x_mu_raw, x_logvar_raw, x_alpha_raw = torch.chunk(x_out, 3, dim=-1)

        # Reshape to time dimension
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]

        x_logvar_raw = x_logvar_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_logvar_raw = x_logvar_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]

        x_alpha_raw = x_alpha_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_alpha_raw = x_alpha_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]

        # Apply RevIN denormalization to mean only
        x_mu = self.revin(x_mu_raw, "denorm")      # [B, L, C]
        
        # Clamp logvar and soft-clip alpha
        x_logvar = torch.clamp(x_logvar_raw, self.logvar_min, self.logvar_max)
        x_alpha = self.alpha_scale * torch.tanh(x_alpha_raw)

        # ----- 2. Predict Future (Skew-Normal) -----
        # Flatten past latent: [B, F, H] -> [B, F*H]
        z_flat = z.reshape(B, -1)
        # Project to future latent: [B, F*H] -> [B, S*H]
        z_future_flat = self.future_proj(z_flat)
        # Reshape to patches: [B, S*H] -> [B, S, H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)

        # Decode future latent
        y_out = self.decoder(z_future)             # [B, S, 3*P*C]
        y_mu_raw, y_logvar_raw, y_alpha_raw = torch.chunk(y_out, 3, dim=-1)

        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]

        y_logvar_raw = y_logvar_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_logvar_raw = y_logvar_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]

        y_alpha_raw = y_alpha_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_alpha_raw = y_alpha_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]

        # Apply RevIN denormalization to mean only
        y_mu = self.revin(y_mu_raw, "denorm")      # [B, pred_len, C]
        
        # Clamp logvar and soft-clip alpha
        y_logvar = torch.clamp(y_logvar_raw, self.logvar_min, self.logvar_max)
        y_alpha = self.alpha_scale * torch.tanh(y_alpha_raw)

        return x_mu, y_mu, x_logvar, y_logvar, x_alpha, y_alpha

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_mu:      [B, L, C]        (past mean, original scale)
            y_mu:      [B, pred_len, C] (future mean, original scale)
            mu_z:      [B, F, H]        (latent mean)
            logvar_z:  [B, F, H]        (latent logvar)
            x_logvar:  [B, L, C]        (past log variance)
            y_logvar:  [B, pred_len, C] (future log variance)
            x_alpha:   [B, L, C]        (past skewness)
            y_alpha:   [B, pred_len, C] (future skewness)
        """
        mu_z, logvar_z, z = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar, x_alpha, y_alpha = self.decode(z)
        return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar, x_alpha, y_alpha

    # =============================================================
    #                  GENERATIVE SAMPLING (Skew-Normal)
    # =============================================================
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Forecasting with Skew-Normal observation.

        Skew-Normal sampling method:
            δ = α / sqrt(1 + α²)
            u, v ~ N(0, 1) independent
            z = δ|u| + sqrt(1 - δ²)v
            y = μ + σz

        x: [B, L, C]
        return:
            samples: [B, num_samples, pred_len, C]
        """
        B, _, _ = x.shape

        # 1. Encode to get latent posterior parameters
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)

        sample_list = []
        for _ in range(num_samples):
            # 2. Sample from latent posterior
            eps_z = torch.randn_like(std_z)
            z_sample = mu_z + eps_z * std_z  # [B, F, H]

            # 3. Decode to get observation parameters
            _, y_mu, _, y_logvar, _, y_alpha = self.decode(z_sample)

            # 4. Sample from Skew-Normal(y_mu, y_sigma, y_alpha)
            y_sample = self._sample_skew_normal(y_mu, y_logvar, y_alpha)

            sample_list.append(y_sample)

        samples = torch.stack(sample_list, dim=1)  # [B, num_samples, pred_len, C]
        return samples

    def _sample_skew_normal(self, mu: torch.Tensor, logvar: torch.Tensor, alpha: torch.Tensor):
        """
        Sample from Skew-Normal distribution.
        
        Args:
            mu:     [B, T, C] mean
            logvar: [B, T, C] log variance
            alpha:  [B, T, C] skewness
        
        Returns:
            samples: [B, T, C]
        """
        sigma = torch.exp(0.5 * logvar)
        
        # Compute delta = alpha / sqrt(1 + alpha^2)
        delta = alpha / torch.sqrt(1.0 + alpha ** 2)
        
        # Sample u, v ~ N(0, 1)
        u = torch.randn_like(mu)
        v = torch.randn_like(mu)
        
        # z = delta * |u| + sqrt(1 - delta^2) * v
        z = delta * torch.abs(u) + torch.sqrt(1.0 - delta ** 2) * v
        
        # y = mu + sigma * z
        return mu + sigma * z

    # =============================================================
    #                    KL DIVERGENCE
    # =============================================================
    def kl_divergence(self, mu, logvar):
        """
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        mu:     [B, F, H]
        logvar: [B, F, H]
        return: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, F]
        return kl.mean()

    # =============================================================
    #                Skew-Normal NLL (Azzalini)
    # =============================================================
    @staticmethod
    def skew_normal_nll(x, mu, logvar, alpha, reduction: str = "mean"):
        """
        Negative log-likelihood for Azzalini Skew-Normal distribution.
        
        log p(x) = log(2) - log(σ) + log(φ(z)) + log(Φ(α·z))
        where z = (x - μ) / σ, φ is standard normal pdf, Φ is standard normal cdf.
        
        Expanded:
            log p(x) = log(2) - log(σ) - 0.5*log(2π) - 0.5*z² + log(Φ(α·z))
        
        Args:
            x, mu, logvar, alpha: [B, L, C] or [B, pred_len, C]
            reduction: "mean", "sum", or "none"
        
        Returns:
            NLL value (scalar for mean/sum, [B] for none)
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        sigma = torch.exp(0.5 * logvar)
        z = (x - mu) / sigma
        
        # Standard normal log pdf: log(φ(z)) = -0.5*log(2π) - 0.5*z²
        log_phi = -0.5 * math.log(2 * math.pi) - 0.5 * z ** 2
        
        # Standard normal log cdf: log(Φ(α·z))
        # Use numerically stable log_ndtr (log of normal cdf)
        alpha_z = alpha * z
        log_Phi = SkewNormalVAE._log_ndtr(alpha_z)
        
        # Log-likelihood: log(2) - log(σ) + log(φ(z)) + log(Φ(α·z))
        log_prob = math.log(2) - 0.5 * logvar + log_phi + log_Phi
        
        # NLL = -log_prob
        nll = -log_prob  # [B, L, C]
        
        # Sum over time and variables
        nll = nll.sum(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll  # [B]

    @staticmethod
    def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log of the standard normal CDF.
        
        log(Φ(x)) where Φ is the standard normal CDF.
        
        Uses different approximations for different ranges of x to maintain stability.
        """
        # For x >= 0: log(Φ(x)) = log(0.5 * (1 + erf(x / sqrt(2))))
        # For x < 0:  log(Φ(x)) = log(0.5 * erfc(-x / sqrt(2)))
        #             = log(0.5) + log(erfc(-x / sqrt(2)))
        
        sqrt2 = math.sqrt(2.0)
        
        # Use log1p for better numerical stability when x is large positive
        # Φ(x) = 0.5 * (1 + erf(x/√2)) = 0.5 + 0.5*erf(x/√2)
        # log(Φ(x)) ≈ log(1 - 0.5*erfc(x/√2)) for large x
        
        # Simple stable implementation using torch.special.log_ndtr if available
        # Fallback to manual implementation
        try:
            return torch.special.log_ndtr(x)
        except AttributeError:
            # Manual implementation for older PyTorch versions
            # For x >= 0: use log(0.5 * (1 + erf(x/sqrt(2))))
            # For x < 0: use log(0.5) + log(erfc(-x/sqrt(2)))
            
            pos_mask = x >= 0
            result = torch.empty_like(x)
            
            # Positive values
            if pos_mask.any():
                erf_val = torch.erf(x[pos_mask] / sqrt2)
                result[pos_mask] = torch.log(0.5 * (1.0 + erf_val))
            
            # Negative values (more stable computation)
            neg_mask = ~pos_mask
            if neg_mask.any():
                erfc_val = torch.erfc(-x[neg_mask] / sqrt2)
                result[neg_mask] = math.log(0.5) + torch.log(erfc_val)
            
            return result

    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction: str = "mean"):
        """
        Standard Gaussian NLL for comparison/debugging.
        When alpha=0, skew_normal_nll should equal this.
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        nll = 0.5 * (
            math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp()
        )
        nll = nll.sum(dim=[1, 2])

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll
