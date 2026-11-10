import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN
# from probts.model.nn.prob.k2VAE.koopman import KPLayer, KPLayerApprox  # 如需 Koopman 可自己打开

class IdentityRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        初始化接收与原 RevIN 相同的参数，防止报错，
        但不初始化任何可学习参数。
        """
        super(IdentityRevIN, self).__init__()

    def forward(self, x, mode: str):
        """
        接收输入 x 和模式 mode，
        但不做任何处理，直接返回 x。
        """
        return x

class GaussianVAE(nn.Module):
    """
    Standard VAE with Gaussian Prior N(0, I)
    + Location-Scale observation model:
        For each time step and variable:
            x_t ~ N(mu_t, sigma_t^2)

    Encoder:
        x -> patches -> encoder -> latent mu_z, logvar_z

    Decoder:
        z -> decoder -> [mu_out, logvar_out]
        - mu_out 经过 RevIN 反归一化，得到输出均值 (原始尺度)
        - logvar_out 直接作为原始尺度上的 log variance 使用
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
        self.likelihood = getattr(config, "likelihood", "gaussian").lower()
        if self.likelihood in {"student", "student-t", "student_t", "t"}:
            self.likelihood = "student_t"
        elif self.likelihood != "gaussian":
            raise ValueError(f"Unsupported GaussianVAE likelihood: {self.likelihood}")
        self.student_t_df_mode = getattr(config, "student_t_df_mode", "global").lower()
        self.student_t_min_df = float(getattr(config, "student_t_min_df", 2.0))
        self.student_t_init_df = float(getattr(config, "student_t_init_df", 30.0))
        if self.student_t_df_mode not in {"global", "time"}:
            raise ValueError(
                "student_t_df_mode must be 'global' or 'time', "
                f"got {self.student_t_df_mode}"
            )

        # === Patching ===
        # number of input patches F
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.pred_len = config.pred_len
        # number of output patches S
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Future Projection ===
        # Project flattened past latents [B, F*H] to flattened future latents [B, S*H]
        # self.future_proj = nn.Linear(
        #     self.freq * self.dynamic_dim,
        #     self.step * self.dynamic_dim
        # )
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
        # Gaussian: decoder outputs [mu, logvar].
        # Student-t with time-varying df: decoder outputs [mu, logvar, df_raw].
        decoder_heads = 3 if (
            self.likelihood == "student_t" and self.student_t_df_mode == "time"
        ) else 2
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=decoder_heads * self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.decoder_f = MLP(
            f_in=self.dynamic_dim,
            f_out=decoder_heads * self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        # === Decoder for mean (mu) ===
        self.decoder_mu = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Decoder for log-variance (logvar) ===
        self.decoder_logvar = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        # === RevIN ===
        # self.revin = RevIN(self.enc_in)
        self.revin = IdentityRevIN(self.enc_in)

        if self.likelihood == "student_t" and self.student_t_df_mode == "global":
            init_offset = max(self.student_t_init_df - self.student_t_min_df, 1e-4)
            init_raw = math.log(math.expm1(init_offset))
            self.student_t_df_raw = nn.Parameter(torch.tensor(init_raw))


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
            # 前面补一段来自序列末尾的片段
            padded = torch.cat(
                [x_norm[:, L - self.padding_len:, :], x_norm],
                dim=1
            )  # [B, F*P, C]
        else:
            padded = x_norm

        # 切成 F 个 patch，每个长度 P
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
    #           DECODING (Location-Scale)
    # =============================================================
    def decode(self, z: torch.Tensor):
        """
        z: [B, F, H] (Past latent)
        return:
            x_mu:       [B, L, C]  (past mean, denormed)
            y_mu:       [B, pred_len, C] (future mean, denormed)
            x_logvar:   [B, L, C]  (past log variance, original scale)
            y_logvar:   [B, pred_len, C] (future log variance, original scale)
        """
        B = z.shape[0]

        # ----- 1. Reconstruct Past (Location-Scale) -----
        x_out = self.decoder(z)                    # [B, F, 2*P*C] 
        if self.likelihood == "student_t" and self.student_t_df_mode == "time":
            x_mu_raw, x_logvar, x_df_raw = torch.chunk(x_out, 3, dim=-1)
        else:
            x_mu_raw, x_logvar = torch.chunk(x_out, 2, dim=-1)  # [B, F, P*C] 各自
            x_df_raw = None
        # x_mu_raw = self.decoder_mu(z)
        # x_logvar = self.decoder_logvar(z)

        # reshape 成时间维度
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]

        x_logvar = x_logvar.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_logvar = x_logvar.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]

        # 仅对均值头做 RevIN 反归一化，使得 x_mu 与原始尺度的 x 对齐
        x_mu = self.revin(x_mu_raw, "denorm")      # [B, L, C]

        x_df = None
        if x_df_raw is not None:
            x_df_raw = x_df_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
            x_df_raw = x_df_raw.reshape(B, -1, self.enc_in)[:, : self.input_len]
            x_df = self._student_t_df_from_raw(x_df_raw)

        # ----- 2. Predict Future (Location-Scale) -----
        # Flatten past latent: [B, F, H] -> [B, F*H]
        z_flat = z.reshape(B, -1)                  # [B, F*H]
        # Project to future latent: [B, F*H] -> [B, S*H]
        z_future_flat = self.future_proj(z_flat)   # [B, S*H]
        # Reshape to patches: [B, S*H] -> [B, S, H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)  # [B, S, H]

        # 解码 future latent
        y_out = self.decoder(z_future)             # [B, S, 2*P*C]
        if self.likelihood == "student_t" and self.student_t_df_mode == "time":
            y_mu_raw, y_logvar, y_df_raw = torch.chunk(y_out, 3, dim=-1)
        else:
            y_mu_raw, y_logvar = torch.chunk(y_out, 2, dim=-1)  # [B, S, P*C]
            y_df_raw = None
        # y_mu_raw = self.decoder_mu(z_future)
        # y_logvar = self.decoder_logvar(z_future)
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]

        y_logvar = y_logvar.reshape(B, self.step, self.patch_len, self.enc_in)
        y_logvar = y_logvar.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]

        # 同样只对均值做 RevIN 反归一化
        y_mu = self.revin(y_mu_raw, "denorm")      # [B, pred_len, C]

        y_df = None
        if y_df_raw is not None:
            y_df_raw = y_df_raw.reshape(B, self.step, self.patch_len, self.enc_in)
            y_df_raw = y_df_raw.reshape(B, -1, self.enc_in)[:, : self.pred_len]
            y_df = self._student_t_df_from_raw(y_df_raw)

        if self.likelihood == "student_t":
            if self.student_t_df_mode == "global":
                df = self.student_t_df
                x_df = df.expand_as(x_mu)
                y_df = df.expand_as(y_mu)
            return x_mu, y_mu, x_logvar, y_logvar, x_df, y_df

        return x_mu, y_mu, x_logvar, y_logvar
       

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
            x_logvar:  [B, L, C]        (past log variance, original scale)
            y_logvar:  [B, pred_len, C] (future log variance, original scale)
        """
        mu_z, logvar_z, z = self.encode(x)
        decoded = self.decode(z)
        if self.likelihood == "student_t":
            x_mu, y_mu, x_logvar, y_logvar, x_df, y_df = decoded
            return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar, x_df, y_df

        x_mu, y_mu, x_logvar, y_logvar = decoded
        return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar

    # =============================================================
    #                  GENERATIVE SAMPLING
    # =============================================================
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Forecasting with Location-Scale observation.

        x: [B, L, C]
        return:
            samples: [B, num_samples, pred_len, C]
        """
        B, _, _ = x.shape

        # 1. Encode 得到 latent posterior 参数
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)

        sample_list = []
        for _ in range(num_samples):
            # 2. 从 latent posterior 采样
            eps_z = torch.randn_like(std_z)
            z_sample = mu_z + eps_z * std_z  # [B, F, H]

            # 3. 解码到输出的均值 & logvar
            decoded = self.decode(z_sample)
            if self.likelihood == "student_t":
                _, y_mu, _, y_logvar, _, y_df = decoded
            else:
                _, y_mu, _, y_logvar = decoded  # [B, pred_len, C]

            # 4. 再从观测分布中采样
            std_y = torch.exp(0.5 * y_logvar)
            if self.likelihood == "student_t":
                y_sample = torch.distributions.StudentT(
                    df=y_df, loc=y_mu, scale=std_y
                ).sample()
            else:
                eps_y = torch.randn_like(std_y)
                y_sample = y_mu + eps_y * std_y               # [B, pred_len, C]

            sample_list.append(y_sample)

        samples = torch.stack(sample_list, dim=1)         # [B, num_samples, pred_len, C]
        return samples

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
    #                Gaussian NLL for Location-Scale
    # =============================================================
    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction: str = "mean"):
        """
        x, mu, logvar: [B, L, C] 或 [B, pred_len, C]

        对应每个元素：
            NLL = 0.5 * (log(2π) + logvar + (x-mu)^2 / exp(logvar))

        reduction:
            "mean" -> 对 batch 取平均
            "sum"  -> 对 batch 求和
            "none" -> 返回 [B] 的向量
        """
        # 防止 logvar 过大/过小导致数值问题
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        nll = 0.5 * (
            math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp()
        )  # [B, L, C]
        # 对时间和变量求和
        nll = nll.sum(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll  # [B]

    @property
    def student_t_df(self):
        return self._student_t_df_from_raw(self.student_t_df_raw)

    def _student_t_df_from_raw(self, raw):
        return self.student_t_min_df + F.softplus(raw)

    @staticmethod
    def student_t_nll(x, mu, logvar, df, reduction: str = "mean"):
        """
        Student-t negative log-likelihood.

        x, mu, logvar, df: broadcastable to [B, L, C] or [B, pred_len, C]
        logvar stores log(sigma^2), where sigma is the Student-t scale.
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        df = torch.clamp(df, min=1e-4)
        sigma2 = logvar.exp()

        log_prob = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * (torch.log(df) + math.log(math.pi) + logvar)
            - ((df + 1.0) / 2.0)
            * torch.log1p((x - mu).pow(2) / (df * sigma2))
        )
        nll = -log_prob.sum(dim=[1, 2])

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll


class FreqDecompGaussianVAE(nn.Module):
    """
    Frequency-Decomposed Gaussian VAE with explicit Location-Scale Decoupling.
    
    Key Idea:
        - Low-frequency components (x_low) -> Location Head (trend/mean prediction)
        - High-frequency components (x_high) -> Scale Head (volatility/variance prediction)
    
    This design leverages the physical interpretation:
        - Trend (Location) is dominated by low-frequency signals
        - Volatility (Scale) energy is concentrated in high-frequency components
    
    Encoder:
        x -> FFT decomposition -> x_low, x_high
        x_low -> location_encoder -> latent for mu prediction (z_loc)
        x_high -> scale_encoder -> latent for logvar prediction (z_scale)
    
    Decoder (Shared):
        [z_loc; z_scale] -> shared_decoder -> [mu_out, logvar_out]
        - mu_out 经过 RevIN 反归一化
        - logvar_out 直接作为 log variance
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

        # === Frequency Decomposition Parameters ===
        # cutoff_ratio: fraction of frequency bins to keep as "low frequency"
        # e.g., 0.1 means keep lowest 10% of frequencies
        self.cutoff_ratio = getattr(config, "freq_cutoff_ratio", 0.1)

        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.pred_len = config.pred_len
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Future Projection (for both location and scale latents) ===
        # Projects concatenated [z_loc; z_scale] to future
        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim * 2,  # 2x because of concatenation
            f_out=self.step * self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.freq * self.dynamic_dim,
            hidden_layers=1,
        )

        # === Location Encoder (processes low-frequency input) ===
        # Outputs: mu_z_loc, logvar_z_loc for VAE latent
        self.location_encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Scale Encoder (processes high-frequency input) ===
        # Outputs: mu_z_scale, logvar_z_scale for VAE latent
        self.scale_encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Shared Decoder (outputs both mu and logvar) ===
        # Input: concatenated [z_loc; z_scale] -> 2 * dynamic_dim
        # Output: [mu_out; logvar_out] -> 2 * patch_len * enc_in
        self.decoder = MLP(
            f_in=self.dynamic_dim * 2,
            f_out=2 * self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === RevIN (for location head denormalization) ===
        self.revin = RevIN(self.enc_in)

    # =============================================================
    #               FREQUENCY DECOMPOSITION
    # =============================================================
    def freq_decompose(self, x: torch.Tensor):
        """
        Decompose input into low-frequency and high-frequency components using FFT.
        
        x: [B, L, C]
        return:
            x_low:  [B, L, C] - Low-frequency (trend) component
            x_high: [B, L, C] - High-frequency (volatility) component
        """
        B, L, C = x.shape
        
        # FFT along time dimension
        x_freq = torch.fft.rfft(x, dim=1)  # [B, L//2+1, C] complex
        
        freq_len = x_freq.shape[1]
        cutoff = max(1, int(freq_len * self.cutoff_ratio))
        
        # Create low-pass mask: keep lowest frequencies
        low_mask = torch.zeros(freq_len, device=x.device, dtype=x.dtype)
        low_mask[:cutoff] = 1.0
        
        # Create high-pass mask: keep higher frequencies
        high_mask = torch.ones(freq_len, device=x.device, dtype=x.dtype)
        high_mask[:cutoff] = 0.0

        # Apply masks
        x_low_freq = x_freq * low_mask.unsqueeze(0).unsqueeze(-1)
        x_high_freq = x_freq * high_mask.unsqueeze(0).unsqueeze(-1)

        # Inverse FFT to get time-domain signals
        x_low = torch.fft.irfft(x_low_freq, n=L, dim=1)   # [B, L, C]
        x_high = torch.fft.irfft(x_high_freq, n=L, dim=1) # [B, L, C]

        return x_low, x_high

    def _create_patches(self, x: torch.Tensor):
        """
        Create patches from input sequence.
        x: [B, L, C]
        return: [B, F, P*C]
        """
        B, L, C = x.shape
        
        if self.padding_len > 0:
            padded = torch.cat(
                [x[:, L - self.padding_len:, :], x],
                dim=1
            )
        else:
            padded = x
        
        patches = padded.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1)       # [B, F, P, C]
        patches = patches.reshape(B, self.freq, -1) # [B, F, P*C]
        
        return patches

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            mu_z_loc:      [B, F, H]  (latent mean for location)
            logvar_z_loc:  [B, F, H]  (latent logvar for location)
            z_loc:         [B, F, H]  (sampled location latent)
            mu_z_scale:    [B, F, H]  (latent mean for scale)
            logvar_z_scale:[B, F, H]  (latent logvar for scale)
            z_scale:       [B, F, H]  (sampled scale latent)
        """
        B, L, C = x.shape

        # RevIN Normalization
        x_norm = self.revin(x, "norm")  # [B, L, C]

        # === Frequency Decomposition ===
        x_low, x_high = self.freq_decompose(x_norm)  # [B, L, C] each

        # === Create Patches ===
        patches_low = self._create_patches(x_low)    # [B, F, P*C]
        patches_high = self._create_patches(x_high)  # [B, F, P*C]

        # === Location Encoder (low-frequency -> location latent) ===
        loc_encoded = self.location_encoder(patches_low)  # [B, F, 2*H]
        mu_z_loc, logvar_z_loc = torch.chunk(loc_encoded, 2, dim=-1)

        # === Scale Encoder (high-frequency -> scale latent) ===
        scale_encoded = self.scale_encoder(patches_high)  # [B, F, 2*H]
        mu_z_scale, logvar_z_scale = torch.chunk(scale_encoded, 2, dim=-1)

        # === Reparameterization ===
        std_z_loc = torch.exp(0.5 * logvar_z_loc)
        eps_loc = torch.randn_like(std_z_loc)
        z_loc = mu_z_loc + eps_loc * std_z_loc if self.training else mu_z_loc

        std_z_scale = torch.exp(0.5 * logvar_z_scale)
        eps_scale = torch.randn_like(std_z_scale)
        z_scale = mu_z_scale + eps_scale * std_z_scale if self.training else mu_z_scale

        return mu_z_loc, logvar_z_loc, z_loc, mu_z_scale, logvar_z_scale, z_scale

    # =============================================================
    #           DECODING (Location-Scale)
    # =============================================================
    def decode(self, z_loc: torch.Tensor, z_scale: torch.Tensor):
        """
        z_loc:   [B, F, H] (location latent from low-freq)
        z_scale: [B, F, H] (scale latent from high-freq)
        return:
            x_mu:       [B, L, C]  (past mean, denormed)
            y_mu:       [B, pred_len, C] (future mean, denormed)
            x_logvar:   [B, L, C]  (past log variance)
            y_logvar:   [B, pred_len, C] (future log variance)
        """
        B = z_loc.shape[0]

        # === Concatenate z_loc and z_scale ===
        z_combined = torch.cat([z_loc, z_scale], dim=-1)  # [B, F, 2*H]

        # ----- 1. Reconstruct Past (shared decoder) -----
        x_out = self.decoder(z_combined)  # [B, F, 2*P*C]
        x_mu_raw, x_logvar = torch.chunk(x_out, 2, dim=-1)  # [B, F, P*C] each

        # Reshape mu
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]
        x_mu = self.revin(x_mu_raw, "denorm")  # [B, L, C]

        # Reshape logvar
        x_logvar = x_logvar.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_logvar = x_logvar.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]

        # ----- 2. Predict Future (shared decoder) -----
        # Flatten and project to future
        z_combined_flat = z_combined.reshape(B, -1)  # [B, F*2*H]
        z_future_flat = self.future_proj(z_combined_flat)  # [B, S*2*H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim * 2)  # [B, S, 2*H]

        # Decode future
        y_out = self.decoder(z_future)  # [B, S, 2*P*C]
        y_mu_raw, y_logvar = torch.chunk(y_out, 2, dim=-1)  # [B, S, P*C] each

        # Reshape future mu
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]
        y_mu = self.revin(y_mu_raw, "denorm")  # [B, pred_len, C]

        # Reshape future logvar
        y_logvar = y_logvar.reshape(B, self.step, self.patch_len, self.enc_in)
        y_logvar = y_logvar.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]

        return x_mu, y_mu, x_logvar, y_logvar

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_mu:           [B, L, C]        (past mean, original scale)
            y_mu:           [B, pred_len, C] (future mean, original scale)
            mu_z_loc:       [B, F, H]        (location latent mean)
            logvar_z_loc:   [B, F, H]        (location latent logvar)
            mu_z_scale:     [B, F, H]        (scale latent mean)
            logvar_z_scale: [B, F, H]        (scale latent logvar)
            x_logvar:       [B, L, C]        (past log variance)
            y_logvar:       [B, pred_len, C] (future log variance)
        """
        mu_z_loc, logvar_z_loc, z_loc, mu_z_scale, logvar_z_scale, z_scale = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar = self.decode(z_loc, z_scale)
        return x_mu, y_mu, mu_z_loc, logvar_z_loc, mu_z_scale, logvar_z_scale, x_logvar, y_logvar

    # =============================================================
    #                  GENERATIVE SAMPLING
    # =============================================================
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Forecasting with Frequency-Decomposed Location-Scale observation.

        x: [B, L, C]
        return:
            samples: [B, num_samples, pred_len, C]
        """
        B, _, _ = x.shape

        # 1. Encode
        mu_z_loc, logvar_z_loc, _, mu_z_scale, logvar_z_scale, _ = self.encode(x)
        std_z_loc = torch.exp(0.5 * logvar_z_loc)
        std_z_scale = torch.exp(0.5 * logvar_z_scale)

        sample_list = []
        for _ in range(num_samples):
            # 2. Sample from location latent posterior
            eps_loc = torch.randn_like(std_z_loc)
            z_loc_sample = mu_z_loc + eps_loc * std_z_loc

            # 3. Sample from scale latent posterior
            eps_scale = torch.randn_like(std_z_scale)
            z_scale_sample = mu_z_scale + eps_scale * std_z_scale

            # 4. Decode to get mu and logvar
            _, y_mu, _, y_logvar = self.decode(z_loc_sample, z_scale_sample)

            # 5. Sample from observation distribution
            std_y = torch.exp(0.5 * y_logvar)
            eps_y = torch.randn_like(std_y)
            y_sample = y_mu + eps_y * std_y

            sample_list.append(y_sample)

        samples = torch.stack(sample_list, dim=1)  # [B, num_samples, pred_len, C]
        return samples

    # =============================================================
    #                    KL DIVERGENCE
    # =============================================================
    def kl_divergence(self, mu_loc, logvar_loc, mu_scale, logvar_scale):
        """
        Combined KL divergence for both location and scale latents.
        
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        kl_loc = -0.5 * torch.sum(
            1 + logvar_loc - mu_loc.pow(2) - logvar_loc.exp(), dim=-1
        )  # [B, F]
        
        kl_scale = -0.5 * torch.sum(
            1 + logvar_scale - mu_scale.pow(2) - logvar_scale.exp(), dim=-1
        )  # [B, F]
        
        return kl_loc.mean() + kl_scale.mean()

    # =============================================================
    #                Gaussian NLL for Location-Scale
    # =============================================================
    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction: str = "mean"):
        """
        x, mu, logvar: [B, L, C] 或 [B, pred_len, C]
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        nll = 0.5 * (
            math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp()
        )
        nll = nll.sum(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll
