import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class VolatilityDynamics(nn.Module):
    """
    GRU-based Volatility Dynamics Module.

    Models the temporal coherence of volatility through a recurrent hidden state:
        h^vol_t = GRU_vol(h^vol_{t-1}, Z_t)
        sigma_t = Softplus(Linear(h^vol_t)) + xi

    This captures volatility clustering, regime persistence, and smooth
    transitions between uncertainty regimes.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, xi=1e-6):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()
        self.xi = xi

    def forward(self, z: torch.Tensor, h0: torch.Tensor = None):
        """
        z:  [B, T, D]   — latent states (patch-level)
        h0: [1, B, H]   — optional initial hidden state

        Returns:
            sigma: [B, T, output_dim]  — predicted scale (positive)
            h_n:   [1, B, H]           — final hidden state
        """
        gru_out, h_n = self.gru(z, h0)        # [B, T, H], [1, B, H]
        sigma = self.softplus(self.linear(gru_out)) + self.xi  # [B, T, output_dim]
        return sigma, h_n


class VolDy_VAE(nn.Module):
    """
    Volatility Dynamics Variational Autoencoder (VolDy-VAE)

    A variational autoencoder for probabilistic time series forecasting
    with temporally coherent uncertainty modeling.

    Architecture:
        Encoder:  x → patches → encoder MLP → (μ_z, logvar_z) → reparameterize → Z_past
        Dynamics: Z_past → flatten → MLP projection → Z_future
        Location Decoder: Z → MLP → μ_raw → RevIN^{-1} → μ
        Scale Decoder (Volatility Dynamics):
            Z_t → GRU_vol(h^vol_{t-1}, Z_t) → Softplus(Linear(h^vol_t)) + ξ → σ_t
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
        self.freq = math.ceil(self.input_len / self.patch_len)   # N: number of input patches
        self.pred_len = config.pred_len
        self.step = math.ceil(self.pred_len / self.patch_len)    # M: number of output patches
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Encoder ===
        # Maps each patch [P*C] → [2*H] (μ_z and logvar_z)
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Non-autoregressive Latent Dynamics ===
        # Projects flattened past latents [B, N*H] → future latents [B, M*H]
        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=self.freq * self.dynamic_dim // 2,
            hidden_layers=1,
        )

        # === Location Decoder (MLP backbone) ===
        # Maps latent [H] → location output [P*C]
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Scale Decoder (Volatility Dynamics via GRU) ===
        vol_hidden_dim = getattr(config, 'vol_hidden_dim', self.dynamic_dim)
        self.volatility_dynamics = VolatilityDynamics(
            input_dim=self.dynamic_dim,
            hidden_dim=vol_hidden_dim,
            output_dim=self.patch_len * self.enc_in,
        )

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        Returns:
            mu_z:      [B, N, H]
            logvar_z:  [B, N, H]
            z:         [B, N, H] (sampled latent)
        """
        B, L, C = x.shape

        # RevIN Normalization (instance-wise)
        x_norm = self.revin(x, "norm")  # [B, L, C]

        # === Patch Padding ===
        if self.padding_len > 0:
            padded = torch.cat(
                [x_norm[:, L - self.padding_len:, :], x_norm],
                dim=1
            )  # [B, N*P, C]
        else:
            padded = x_norm

        # Split into N patches of length P
        patches = padded.chunk(self.freq, dim=1)       # N * [B, P, C]
        patches = torch.stack(patches, dim=1)           # [B, N, P, C]
        patches = patches.reshape(B, self.freq, -1)     # [B, N, P*C]

        # === Encoder Output ===
        encoded = self.encoder(patches)                  # [B, N, 2*H]
        mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1) # [B, N, H] each

        # === Reparameterization ===
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z if self.training else mu_z  # [B, N, H]

        return mu_z, logvar_z, z

    # =============================================================
    #           DECODING (Location-Scale with Volatility Dynamics)
    # =============================================================
    def decode(self, z: torch.Tensor):
        """
        z: [B, N, H] (Past latent representations)
        Returns:
            x_mu:    [B, L, C]        (past mean, denormed)
            y_mu:    [B, pred_len, C]  (future mean, denormed)
            x_sigma: [B, L, C]        (past scale, positive)
            y_sigma: [B, pred_len, C]  (future scale, positive)
        """
        B = z.shape[0]

        # ----- 1. Reconstruct Past (Location Head) -----
        x_mu_raw = self.decoder(z)  # [B, N, P*C]
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]
        x_mu = self.revin(x_mu_raw, "denorm")  # [B, L, C]

        # ----- 2. Reconstruct Past (Scale Head — Volatility Dynamics) -----
        x_sigma_patches, h_vol = self.volatility_dynamics(z)  # [B, N, P*C], [1, B, H_vol]
        x_sigma_patches = x_sigma_patches.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_sigma = x_sigma_patches.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]

        # ----- 3. Predict Future (Latent Dynamics) -----
        z_flat = z.reshape(B, -1)                              # [B, N*H]
        z_future_flat = self.future_proj(z_flat)               # [B, M*H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)  # [B, M, H]

        # ----- 4. Decode Future (Location Head) -----
        y_mu_raw = self.decoder(z_future)  # [B, M, P*C]
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]
        y_mu = self.revin(y_mu_raw, "denorm")  # [B, pred_len, C]

        # ----- 5. Decode Future (Scale Head — Volatility Dynamics, continued) -----
        # Pass h_vol from past to maintain volatility continuity across past→future
        y_sigma_patches, _ = self.volatility_dynamics(z_future, h0=h_vol)  # [B, M, P*C]
        y_sigma_patches = y_sigma_patches.reshape(B, self.step, self.patch_len, self.enc_in)
        y_sigma = y_sigma_patches.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]

        return x_mu, y_mu, x_sigma, y_sigma

    def decode_future(self, z: torch.Tensor):
        """
        z: [B, N, H] (Past latent representations)
        Returns:
            y_mu:    [B, pred_len, C]  (future mean, denormed)
            y_sigma: [B, pred_len, C]  (future scale, positive)
        """
        B = z.shape[0]

        # Run past volatility only to carry the recurrent volatility state into
        # the future; past reconstruction itself is not needed during sampling.
        _, h_vol = self.volatility_dynamics(z)

        z_flat = z.reshape(B, -1)
        z_future_flat = self.future_proj(z_flat)
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)

        y_mu_raw = self.decoder(z_future)
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]
        y_mu = self.revin(y_mu_raw, "denorm")

        y_sigma_patches, _ = self.volatility_dynamics(z_future, h0=h_vol)
        y_sigma_patches = y_sigma_patches.reshape(B, self.step, self.patch_len, self.enc_in)
        y_sigma = y_sigma_patches.reshape(B, -1, self.enc_in)[:, :self.pred_len]

        return y_mu, y_sigma

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        Returns:
            x_mu:      [B, L, C]        (past mean, original scale)
            y_mu:      [B, pred_len, C]  (future mean, original scale)
            mu_z:      [B, N, H]         (latent mean)
            logvar_z:  [B, N, H]         (latent logvar)
            x_sigma:   [B, L, C]         (past scale, positive)
            y_sigma:   [B, pred_len, C]  (future scale, positive)
        """
        mu_z, logvar_z, z = self.encode(x)
        x_mu, y_mu, x_sigma, y_sigma = self.decode(z)
        return x_mu, y_mu, mu_z, logvar_z, x_sigma, y_sigma

    # =============================================================
    #                  GENERATIVE SAMPLING
    # =============================================================
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative forecasting with Location-Scale observation + Volatility Dynamics.

        x: [B, L, C]
        Returns:
            samples: [B, num_samples, pred_len, C]
        """
        B = x.shape[0]

        # Encode to obtain latent posterior parameters
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)

        # Vectorize sampling by folding the sample dimension into the batch.
        # RevIN stores per-item statistics from encode(x), so repeat them while
        # decoding the expanded [B * S, ...] latent batch, then restore them.
        eps_z = torch.randn(
            B,
            num_samples,
            self.freq,
            self.dynamic_dim,
            device=x.device,
            dtype=x.dtype,
        )
        z_sample = mu_z.unsqueeze(1) + eps_z * std_z.unsqueeze(1)
        z_sample = z_sample.reshape(B * num_samples, self.freq, self.dynamic_dim)

        revin_state = {}
        for name in ("mean", "stdev", "last"):
            if hasattr(self.revin, name):
                value = getattr(self.revin, name)
                revin_state[name] = value
                setattr(self.revin, name, value.repeat_interleave(num_samples, dim=0))

        try:
            y_mu, y_sigma = self.decode_future(z_sample)  # [B*S, pred_len, C]
        finally:
            for name, value in revin_state.items():
                setattr(self.revin, name, value)

        eps_y = torch.randn_like(y_mu)
        y_sample = y_mu + eps_y * y_sigma
        return y_sample.reshape(B, num_samples, self.pred_len, self.enc_in)

    # =============================================================
    #                    KL DIVERGENCE
    # =============================================================
    def kl_divergence(self, mu, logvar):
        """
        KL(N(mu, sigma^2) || N(0, I))

        mu:     [B, N, H]
        logvar: [B, N, H]
        Returns: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, N]
        return kl.mean()

    # =============================================================
    #             Gaussian NLL for Location-Scale
    # =============================================================
    @staticmethod
    def gaussian_nll(x, mu, sigma, reduction: str = "mean"):
        """
        Heteroscedastic Gaussian NLL matching Eq.(9) in the paper:
            L_NLL = 1/(TC) * sum_t,c [ log(sigma_{t,c}) + (x_{t,c} - mu_{t,c})^2 / (2 * sigma_{t,c}^2) ]

        x, mu, sigma: [B, T, C]  (sigma is positive, from Softplus + xi)

        reduction:
            "mean" -> average over batch
            "sum"  -> sum over batch
            "none" -> return [B] vector
        """
        # Clamp sigma for numerical stability
        sigma = torch.clamp(sigma, min=1e-6, max=1e3)

        T, C = x.shape[1], x.shape[2]
        nll = torch.log(sigma) + (x - mu) ** 2 / (2.0 * sigma ** 2)  # [B, T, C]
        # Average over time and variables (matching paper Eq.9)
        nll = nll.sum(dim=[1, 2]) / (T * C)  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll  # [B]
