import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class CrossBandAttention(nn.Module):
    """
    Lightweight cross-band attention for information sharing between frequency bands.

    Given B band latents, for each band b, attend to all other bands {b' != b}
    to capture cross-frequency uncertainty propagation (e.g., volatility spillover).
    """

    def __init__(self, d_model, d_k=32):
        super().__init__()
        self.d_k = d_k
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_k)
        self.out_proj = nn.Linear(d_k, d_model)

    def forward(self, z_b, z_others):
        """
        z_b:      [B, N, D]  — query band
        z_others: [B, num_other_bands, N, D] — other bands
        return:   [B, N, D]
        """
        B, N, D = z_b.shape
        num_others = z_others.shape[1]

        # Query from the target band
        Q = self.q_proj(z_b)  # [B, N, d_k]

        # Key/Value from all other bands, flattened across bands and patches
        z_ctx = z_others.reshape(B, num_others * N, D)  # [B, num_others*N, D]
        K = self.k_proj(z_ctx)  # [B, num_others*N, d_k]
        V = self.v_proj(z_ctx)  # [B, num_others*N, d_k]

        # Scaled dot-product attention
        attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # [B, N, num_others*N]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V)  # [B, N, d_k]

        return self.out_proj(out)  # [B, N, D]


class LearnableSpectralDecomposition(nn.Module):
    """
    Learnable Spectral Decomposition via STFT with learnable window function
    and learnable band boundaries.

    Decomposes input into B non-overlapping frequency bands using STFT,
    then reconstructs each band via iSTFT.
    """

    def __init__(self, n_bands, seq_len, n_vars, win_length=None, hop_length=None):
        super().__init__()
        self.n_bands = n_bands
        self.seq_len = seq_len
        self.n_vars = n_vars

        # STFT parameters
        self.win_length = win_length or seq_len // 4
        self.hop_length = hop_length or self.win_length // 2
        self.n_fft = self.win_length

        # Learnable window function (initialized as Hann window)
        hann = torch.hann_window(self.win_length)
        self.window = nn.Parameter(hann.clone())

        # Learnable band boundaries (B-1 internal boundaries)
        # frequency bins: n_fft // 2 + 1
        n_freq = self.n_fft // 2 + 1
        # Initialize boundaries uniformly
        init_boundaries = torch.linspace(0, n_freq, n_bands + 1)[1:-1].float()
        self.band_boundaries_raw = nn.Parameter(init_boundaries)
        self.n_freq = n_freq

    def _get_band_boundaries(self):
        """Return sorted band boundaries clamped to valid range."""
        boundaries = torch.sigmoid(self.band_boundaries_raw) * self.n_freq
        boundaries, _ = torch.sort(boundaries)
        # Prepend 0, append n_freq
        zeros = torch.zeros(1, device=boundaries.device)
        maxes = torch.full((1,), self.n_freq, device=boundaries.device)
        return torch.cat([zeros, boundaries, maxes])

    def forward(self, x):
        """
        x: [B, L, C] — normalized input
        return: list of B tensors, each [B, L, C]
        """
        B, L, C = x.shape
        bands = []

        # Get soft band boundaries
        boundaries = self._get_band_boundaries()  # [n_bands + 1]

        # Process each variable independently
        # Reshape to [B*C, L] for STFT
        x_flat = x.permute(0, 2, 1).reshape(B * C, L)

        # Apply STFT with learnable window
        window = self.window.to(x.device)
        # Ensure window is positive (soft constraint)
        window_pos = F.softplus(window)

        spec = torch.stft(
            x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window_pos,
            return_complex=True, center=True, pad_mode='reflect'
        )  # [B*C, n_freq, n_frames]

        n_freq, n_frames = spec.shape[1], spec.shape[2]

        for b in range(self.n_bands):
            # Create soft frequency mask using sigmoid transitions
            freq_idx = torch.arange(n_freq, device=x.device, dtype=torch.float32)
            # Smooth band boundaries using sigmoid (straight-through compatible)
            low = boundaries[b]
            high = boundaries[b + 1]
            sharpness = 10.0
            mask = torch.sigmoid(sharpness * (freq_idx - low)) * \
                   torch.sigmoid(sharpness * (high - freq_idx))  # [n_freq]
            mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, n_freq, 1]

            # Apply mask
            spec_band = spec * mask  # [B*C, n_freq, n_frames]

            # iSTFT to reconstruct band signal
            band_signal = torch.istft(
                spec_band, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=window_pos,
                length=L
            )  # [B*C, L]

            band_signal = band_signal.reshape(B, C, L).permute(0, 2, 1)  # [B, L, C]
            bands.append(band_signal)

        return bands  # list of B tensors, each [B, L, C]


class VolatilityGRU(nn.Module):
    """
    GRU-based volatility dynamics for modeling temporal coherence
    of uncertainty within each frequency band.

    Optionally incorporates cross-band uncertainty coupling.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_bands=1):
        super().__init__()
        # input: latent features; cross-band sigma from other bands
        cross_band_dim = (n_bands - 1) * output_dim if n_bands > 1 else 0
        self.input_dim = input_dim
        self.cross_band_dim = cross_band_dim
        self.gru = nn.GRU(
            input_size=input_dim + cross_band_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.xi = 1e-6  # numerical stability

    def forward(self, z, cross_sigma=None):
        """
        z: [B, N, D] — latent representations (patches)
        cross_sigma: [B, N, (n_bands-1)*out_dim] — sigma from other bands (optional)
        return: [B, N, out_dim] — sigma (positive)
        """
        if cross_sigma is not None:
            gru_input = torch.cat([z, cross_sigma], dim=-1)
        elif self.cross_band_dim > 0:
            zero_cross_sigma = z.new_zeros(z.shape[0], z.shape[1], self.cross_band_dim)
            gru_input = torch.cat([z, zero_cross_sigma], dim=-1)
        else:
            gru_input = z

        h, _ = self.gru(gru_input)  # [B, N, hidden_dim]
        sigma = F.softplus(self.proj(h)) + self.xi  # [B, N, out_dim], positive
        return sigma


class SpecVAE(nn.Module):
    """
    Spectral Variational Autoencoder (SpecVAE)

    Key upgrades from LSG-VAE:
    1. Learnable Spectral Decomposition: STFT-based decomposition into B frequency bands
    2. Band-wise Variational Encoding: Independent encoder per band
    3. Cross-Band Attention: Lightweight attention for cross-frequency uncertainty propagation
    4. Frequency-aware Latent Dynamics: Band-specific future projections
    5. Spectral Location-Scale Decoding: GRU-based volatility dynamics per band
    6. Spectral Synthesis: Band-wise aggregation with frequency-resolved uncertainty
    """

    def __init__(self, config):
        super().__init__()

        # === Basic Parameters ===
        self.config = config
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        self.dynamic_dim = config.dynamic_dim     # latent dim D
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars               # input channels C
        self.pred_len = config.pred_len
        self.n_bands = getattr(config, 'n_bands', 3)
        self.cross_attn_dk = getattr(config, 'cross_attn_dk', 32)
        self.vol_hidden_dim = getattr(config, 'vol_hidden_dim', 64)
        self.alpha_init = getattr(config, 'alpha_init', 0.1)

        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)   # N: number of input patches
        self.step = math.ceil(self.pred_len / self.patch_len)    # M: number of output patches
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === Learnable Spectral Decomposition ===
        self.spectral_decomp = LearnableSpectralDecomposition(
            n_bands=self.n_bands,
            seq_len=self.input_len,
            n_vars=self.enc_in,
            win_length=self.input_len // 4,
            hop_length=self.input_len // 8,
        )

        # === Band-wise Encoders ===
        self.encoders = nn.ModuleList([
            MLP(
                f_in=self.patch_len * self.enc_in,
                f_out=self.dynamic_dim * 2,  # mu_z and logvar_z
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
            )
            for _ in range(self.n_bands)
        ])

        # === Cross-Band Attention ===
        self.cross_attn = CrossBandAttention(self.dynamic_dim, self.cross_attn_dk)
        # Learnable gate alpha
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init))

        # === Band-wise Future Projections ===
        self.future_projs = nn.ModuleList([
            MLP(
                f_in=self.freq * self.dynamic_dim,
                f_out=self.step * self.dynamic_dim,
                activation="relu",
                hidden_dim=self.freq * self.dynamic_dim // 2,
                hidden_layers=1,
            )
            for _ in range(self.n_bands)
        ])

        # === Band-wise Decoders (Location Head) ===
        self.decoders_mu = nn.ModuleList([
            MLP(
                f_in=self.dynamic_dim,
                f_out=self.patch_len * self.enc_in,
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
            )
            for _ in range(self.n_bands)
        ])

        # === Band-wise Volatility GRU (Scale Head) ===
        self.vol_grus = nn.ModuleList([
            VolatilityGRU(
                input_dim=self.dynamic_dim,
                hidden_dim=self.vol_hidden_dim,
                output_dim=self.patch_len * self.enc_in,
                n_bands=self.n_bands,
            )
            for _ in range(self.n_bands)
        ])

    # =============================================================
    #                SPECTRAL DECOMPOSITION
    # =============================================================
    def spectral_decompose(self, x_norm):
        """
        x_norm: [B, L, C] — RevIN-normalized input
        return: list of B tensors, each [B, L, C]
        """
        return self.spectral_decomp(x_norm)

    # =============================================================
    #                    PATCHING
    # =============================================================
    def _patchify(self, x, num_patches):
        """
        x: [B, L, C]
        return: [B, num_patches, P*C]
        """
        B, L, C = x.shape
        P = self.patch_len
        total_len = num_patches * P

        if L < total_len:
            pad_len = total_len - L
            x = torch.cat([x[:, L - pad_len:, :], x], dim=1)

        patches = x[:, :total_len].reshape(B, num_patches, P, C)
        return patches.reshape(B, num_patches, P * C)

    # =============================================================
    #               BAND-WISE ENCODING
    # =============================================================
    def encode(self, x):
        """
        x: [B, L, C]
        return:
            band_mu_z:      list of B tensors [B, N, D]
            band_logvar_z:  list of B tensors [B, N, D]
            band_z:         list of B tensors [B, N, D] (after cross-band attention)
            band_signals:   list of B tensors [B, L, C] (decomposed band signals)
        """
        B, L, C = x.shape

        # 1. RevIN normalization
        x_norm = self.revin(x, "norm")  # [B, L, C]

        # 2. Spectral decomposition into B bands
        band_signals = self.spectral_decompose(x_norm)  # list of B x [B, L, C]

        # 3. Encode each band independently
        band_mu_z = []
        band_logvar_z = []
        band_z_raw = []

        for b in range(self.n_bands):
            patches = self._patchify(band_signals[b], self.freq)  # [B, N, P*C]
            encoded = self.encoders[b](patches)                    # [B, N, 2*D]
            mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1)      # [B, N, D] each

            # Reparameterization
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            z = mu_z + eps * std_z if self.training else mu_z      # [B, N, D]

            band_mu_z.append(mu_z)
            band_logvar_z.append(logvar_z)
            band_z_raw.append(z)

        # 4. Cross-band attention
        band_z = []
        for b in range(self.n_bands):
            others = [band_z_raw[b_] for b_ in range(self.n_bands) if b_ != b]
            others = torch.stack(others, dim=1)  # [B, n_bands-1, N, D]
            cross_out = self.cross_attn(band_z_raw[b], others)  # [B, N, D]
            z_hat = band_z_raw[b] + self.alpha * cross_out       # [B, N, D]
            band_z.append(z_hat)

        return band_mu_z, band_logvar_z, band_z, band_signals

    # =============================================================
    #             BAND-WISE DECODING (Location-Scale)
    # =============================================================
    def decode_band(self, z_past, band_idx):
        """
        Decode a single band: past latent -> past & future predictions.

        z_past: [B, N, D]
        return:
            x_mu_band:    [B, L, C] (past mean, normalized scale)
            y_mu_band:    [B, pred_len, C] (future mean, normalized scale)
            x_sigma_band: [B, L, C] (past sigma, positive)
            y_sigma_band: [B, pred_len, C] (future sigma, positive)
            z_future:     [B, M, D] (future latent)
        """
        B = z_past.shape[0]

        # --- Location Head (Past) ---
        x_mu_raw = self.decoders_mu[band_idx](z_past)  # [B, N, P*C]
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]

        # --- Future Projection ---
        z_flat = z_past.reshape(B, -1)                           # [B, N*D]
        z_future_flat = self.future_projs[band_idx](z_flat)      # [B, M*D]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)  # [B, M, D]

        # --- Location Head (Future) ---
        y_mu_raw = self.decoders_mu[band_idx](z_future)  # [B, M, P*C]
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]

        return x_mu_raw, y_mu_raw, z_future

    def decode(self, band_z):
        """
        Decode all bands and aggregate.

        band_z: list of B tensors [B, N, D]
        return:
            x_mu:       [B, L, C] (aggregated past mean, denormed)
            y_mu:       [B, pred_len, C] (aggregated future mean, denormed)
            x_sigma:    [B, L, C] (aggregated past sigma)
            y_sigma:    [B, pred_len, C] (aggregated future sigma)
            band_x_mu:  list of B tensors [B, L, C]
            band_y_mu:  list of B tensors [B, pred_len, C]
            band_x_sigma: list of B tensors [B, L, C]
            band_y_sigma: list of B tensors [B, pred_len, C]
        """
        B = band_z[0].shape[0]

        band_x_mu = []
        band_y_mu = []
        band_z_past_list = []
        band_z_future_list = []

        # Step 1: Get location outputs and future latents for all bands
        for b in range(self.n_bands):
            x_mu_raw, y_mu_raw, z_future = self.decode_band(band_z[b], b)
            band_x_mu.append(x_mu_raw)
            band_y_mu.append(y_mu_raw)
            band_z_past_list.append(band_z[b])
            band_z_future_list.append(z_future)

        # Step 2: Scale head with GRU volatility dynamics + cross-band coupling
        band_x_sigma = []
        band_y_sigma = []

        for b in range(self.n_bands):
            # Collect cross-band sigma for past (from other bands)
            # First pass: compute sigma without cross-band info to bootstrap
            pass

        # Bootstrap: compute initial sigma estimates without cross-band coupling
        init_x_sigma = []
        init_y_sigma = []
        for b in range(self.n_bands):
            x_s = self.vol_grus[b](band_z_past_list[b], cross_sigma=None)
            x_s = x_s.reshape(B, self.freq, self.patch_len, self.enc_in)
            x_s = x_s.reshape(B, -1, self.enc_in)[:, :self.input_len]
            init_x_sigma.append(x_s)

            y_s = self.vol_grus[b](band_z_future_list[b], cross_sigma=None)
            y_s = y_s.reshape(B, self.step, self.patch_len, self.enc_in)
            y_s = y_s.reshape(B, -1, self.enc_in)[:, :self.pred_len]
            init_y_sigma.append(y_s)

        # Second pass: with cross-band sigma coupling
        for b in range(self.n_bands):
            # Gather cross-band sigma from other bands (using init estimates)
            other_x_sigmas = [init_x_sigma[b_] for b_ in range(self.n_bands) if b_ != b]
            # We need to map these back to patch-level for GRU input
            # Re-patchify the sigma from other bands
            other_x_patches = []
            for s in other_x_sigmas:
                s_patched = self._patchify(s, self.freq)  # [B, N, P*C]
                other_x_patches.append(s_patched)
            cross_x = torch.cat(other_x_patches, dim=-1)  # [B, N, (n_bands-1)*P*C]

            x_sigma = self.vol_grus[b](band_z_past_list[b], cross_sigma=cross_x)
            x_sigma = x_sigma.reshape(B, self.freq, self.patch_len, self.enc_in)
            x_sigma = x_sigma.reshape(B, -1, self.enc_in)[:, :self.input_len]
            band_x_sigma.append(x_sigma)

            # Future sigma
            other_y_sigmas = [init_y_sigma[b_] for b_ in range(self.n_bands) if b_ != b]
            other_y_patches = []
            for s in other_y_sigmas:
                s_patched = self._patchify(s, self.step)  # [B, M, P*C]
                other_y_patches.append(s_patched)
            cross_y = torch.cat(other_y_patches, dim=-1)  # [B, M, (n_bands-1)*P*C]

            y_sigma = self.vol_grus[b](band_z_future_list[b], cross_sigma=cross_y)
            y_sigma = y_sigma.reshape(B, self.step, self.patch_len, self.enc_in)
            y_sigma = y_sigma.reshape(B, -1, self.enc_in)[:, :self.pred_len]
            band_y_sigma.append(y_sigma)

        # Step 3: Spectral synthesis — aggregate across bands
        # Mean: sum of band means, then RevIN denorm
        x_mu_sum = sum(band_x_mu)         # [B, L, C]
        y_mu_sum = sum(band_y_mu)         # [B, pred_len, C]
        x_mu = self.revin(x_mu_sum, "denorm")   # [B, L, C]
        y_mu = self.revin(y_mu_sum, "denorm")   # [B, pred_len, C]

        # Variance: sum of band variances (sigma^2), then take sqrt
        x_var = sum(s ** 2 for s in band_x_sigma)   # [B, L, C]
        y_var = sum(s ** 2 for s in band_y_sigma)   # [B, pred_len, C]
        x_sigma = torch.sqrt(x_var)                  # [B, L, C]
        y_sigma = torch.sqrt(y_var)                  # [B, pred_len, C]

        return (x_mu, y_mu, x_sigma, y_sigma,
                band_x_mu, band_y_mu, band_x_sigma, band_y_sigma)

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x):
        """
        x: [B, L, C]
        return:
            x_mu:      [B, L, C]         (aggregated past mean, original scale)
            y_mu:      [B, pred_len, C]  (aggregated future mean, original scale)
            x_sigma:   [B, L, C]         (aggregated past sigma)
            y_sigma:   [B, pred_len, C]  (aggregated future sigma)
            band_mu_z:     list of B x [B, N, D]
            band_logvar_z: list of B x [B, N, D]
            band_signals:  list of B x [B, L, C] (decomposed input bands)
            band_x_mu:     list of B x [B, L, C]
            band_y_mu:     list of B x [B, pred_len, C]
            band_x_sigma:  list of B x [B, L, C]
            band_y_sigma:  list of B x [B, pred_len, C]
        """
        band_mu_z, band_logvar_z, band_z, band_signals = self.encode(x)
        (x_mu, y_mu, x_sigma, y_sigma,
         band_x_mu, band_y_mu, band_x_sigma, band_y_sigma) = self.decode(band_z)

        return (x_mu, y_mu, x_sigma, y_sigma,
                band_mu_z, band_logvar_z, band_signals,
                band_x_mu, band_y_mu, band_x_sigma, band_y_sigma)

    # =============================================================
    #                  GENERATIVE SAMPLING
    # =============================================================
    def sample(self, x, num_samples=100):
        """
        Generative Forecasting with spectral uncertainty.

        x: [B, L, C]
        return: [B, num_samples, pred_len, C]
        """
        B = x.shape[0]

        # Encode
        band_mu_z, band_logvar_z, _, _ = self.encode(x)
        band_std_z = [torch.exp(0.5 * lv) for lv in band_logvar_z]

        sample_list = []
        for _ in range(num_samples):
            # Sample band-wise latents
            band_z_sample = []
            for b in range(self.n_bands):
                eps = torch.randn_like(band_std_z[b])
                z_b = band_mu_z[b] + eps * band_std_z[b]
                band_z_sample.append(z_b)

            # Apply cross-band attention on sampled latents
            band_z_hat = []
            for b in range(self.n_bands):
                others = [band_z_sample[b_] for b_ in range(self.n_bands) if b_ != b]
                others = torch.stack(others, dim=1)
                cross_out = self.cross_attn(band_z_sample[b], others)
                z_hat = band_z_sample[b] + self.alpha * cross_out
                band_z_hat.append(z_hat)

            # Decode
            (_, y_mu, _, y_sigma,
             _, _, _, _) = self.decode(band_z_hat)

            # Sample from observation distribution
            eps_y = torch.randn_like(y_mu)
            y_sample = y_mu + eps_y * y_sigma  # [B, pred_len, C]
            sample_list.append(y_sample)

        return torch.stack(sample_list, dim=1)  # [B, num_samples, pred_len, C]

    # =============================================================
    #                    KL DIVERGENCE
    # =============================================================
    def kl_divergence(self, band_mu_z, band_logvar_z):
        """
        Sum of KL divergences across all bands.
        KL(N(mu, sigma^2) || N(0, I))

        return: scalar
        """
        kl_total = 0.0
        for b in range(self.n_bands):
            mu = band_mu_z[b]        # [B, N, D]
            logvar = band_logvar_z[b] # [B, N, D]
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, N]
            kl_total = kl_total + kl.mean()
        return kl_total

    # =============================================================
    #           Band-wise Gaussian NLL (using sigma directly)
    # =============================================================
    @staticmethod
    def gaussian_nll_sigma(x, mu, sigma, reduction="mean"):
        """
        NLL for N(mu, sigma^2) where sigma is given directly (not log).

        x, mu, sigma: [B, T, C]
        NLL = 0.5 * (log(2π) + 2*log(sigma) + (x-mu)^2 / sigma^2)
        """
        sigma = torch.clamp(sigma, min=1e-6)
        nll = 0.5 * (
            math.log(2 * math.pi) + 2 * torch.log(sigma) + (x - mu) ** 2 / sigma ** 2
        )
        # Average over time and variables
        nll = nll.mean(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    # =============================================================
    #                Band-wise NLL Loss
    # =============================================================
    def band_nll_loss(self, band_targets, band_mu, band_sigma, reduction="mean"):
        """
        Compute sum of NLL losses across all frequency bands.

        band_targets: list of B x [B, T, C]
        band_mu:      list of B x [B, T, C]
        band_sigma:   list of B x [B, T, C]
        """
        total = 0.0
        for b in range(self.n_bands):
            total = total + self.gaussian_nll_sigma(
                band_targets[b], band_mu[b], band_sigma[b], reduction=reduction
            )
        return total
