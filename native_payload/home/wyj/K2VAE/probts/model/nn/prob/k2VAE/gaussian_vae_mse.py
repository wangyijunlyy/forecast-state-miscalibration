import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN
from probts.model.nn.prob.k2VAE.koopman import KPLayer, KPLayerApprox, MLP
class GaussianVAE_MSE(nn.Module):
    """
    Standard VAE with Gaussian Prior N(0, I)
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
        self.freq = math.ceil(self.input_len / self.patch_len)   # number of input patches F
        self.pred_len = config.pred_len
        self.step = math.ceil(self.pred_len / self.patch_len)    # number of output patches S
        self.padding_len = self.patch_len * self.freq - self.input_len
        
        # === Future Projection ===
        # Project flattened past latents [B, F*H] to flattened future latents [B, S*H]
        self.future_proj = nn.Linear(self.freq * self.dynamic_dim, self.step * self.dynamic_dim)

        # === Encoder ===
        # Encoder outputs 2 * dynamic_dim (mu and logvar)
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2, 
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        
        # === Decoder ===
        self.decoder_mu = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        # self.koopman = KPLayerApprox(config.dynamic_dim * 2, config.init_koopman) if self.multistep else KPLayer(
            # config.dynamic_dim * 2, config.init_koopman)
        # === RevIN ===
        self.revin = RevIN(self.enc_in)

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            mu:      [B, F, H]
            logvar:  [B, F, H]
            z:       [B, F, H] (sampled latent)
        """
        B, L, C = x.shape

        # RevIN Normalization
        x_norm = self.revin(x, "norm")

        # === Patch Padding ===
        padded = torch.cat([x_norm[:, L - self.padding_len :, :], x_norm], dim=1)
        patches = padded.chunk(self.freq, dim=1)                # freq * [B, P, C]
        patches = torch.stack(patches, dim=1).reshape(B, self.freq, -1)  # [B, F, P*C]

        # === Encoder Output ===
        encoded = self.encoder(patches)                        # [B, F, 2*H]
        # encoded, Y = self.koopman(encoded, self.step)  # B F H, B S H
        mu, logvar = torch.chunk(encoded, 2, dim=-1)           # [B, F, H], [B, F, H]
        
        # === Reparameterization ===
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std if self.training else mu            # [B, F, H]

        return mu, logvar, z

    # =============================================================
    #           DECODING
    # =============================================================
    def decode(self, z: torch.Tensor):
        """
        z: [B, F, H] (Past latent)
        return:
            x_rec:     [B, L, C] (Past reconstruction)
            y_rec:     [B, pred_len, C] (Future prediction)
        """
        B = z.shape[0]
        
        # ----- 1. Reconstruct Past (x_rec) -----
        x_rec_mu = self.decoder_mu(z)  # [B, F, P*C]
        x_rec_mu = x_rec_mu.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_mu = x_rec_mu.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]
        x_rec = self.revin(x_rec_mu, "denorm")

        # ----- 2. Predict Future (y_rec) -----
        # Flatten past latent: [B, F, H] -> [B, F*H]
        z_flat = z.reshape(B, -1)
        # Project to future latent: [B, F*H] -> [B, S*H]
        z_future_flat = self.future_proj(z_flat) 
        # z_future_flat = z_flat
        # Reshape to patches: [B, S*H] -> [B, S, H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)

        # Decode Future
        y_rec_mu = self.decoder_mu(z_future)  # [B, S, P*C]
        y_rec_mu = y_rec_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        y_rec_mu = y_rec_mu.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]
        y_rec = self.revin(y_rec_mu, "denorm")
        
        return x_rec, y_rec

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_rec:   [B, L, C]
            y_rec:   [B, pred_len, C]
            mu:      [B, F, H]
            logvar:  [B, F, H]
        """
        mu, logvar, z = self.encode(x)
        x_rec, y_rec = self.decode(z)
        return x_rec, y_rec, mu, logvar

    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Forecasting: Sample from posterior q(z|x) multiple times.
        x: [B, L, C]
        return:
            samples: [B, num_samples, pred_len, C]
        """
        B, L, C = x.shape
        
        # 1. Encode to get posterior params
        mu, logvar, _ = self.encode(x)
        std = torch.exp(0.5 * logvar)
        
        sample_list = []
        for _ in range(num_samples):
            # 2. Sample z from posterior
            eps = torch.randn_like(std)
            z_sample = mu + eps * std
            
            # 3. Decode
            _, y_rec_sample = self.decode(z_sample)
            sample_list.append(y_rec_sample)
            
        # 4. Stack
        samples = torch.stack(sample_list, dim=1) # [B, num_samples, pred_len, C]
        return samples

    def kl_divergence(self, mu, logvar):
        """
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        # sum over last dim (H), mean over batch and patches? 
        # Usually sum over all latent dims, then mean over batch.
        # mu: [B, F, H]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) # [B, F]
        return kl.mean()


