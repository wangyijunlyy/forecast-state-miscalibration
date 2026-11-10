import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange

# ==== Reuse components from k2VAE ====
from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN
import kaiwu as kw
import os
# ==== Reuse components from QVAE ====
from probts.utils.qvae_dist_util import MixtureGeneric
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer


# kw.license.init(os.getenv("kaiwu_USER_ID"), os.getenv("kaiwu_SDK_CODE"))


class BoltzmannVAE(nn.Module):
    """
    Pure VAE with Boltzmann Machine Prior (No Kalman, No Koopman)
    """

    def __init__(self, config):
        super().__init__()

        # === Basic Parameters ===
        self.config = config
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        
        self.dynamic_dim = config.dynamic_dim  # latent dim H
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars             # input channels C

        # === Patching ===
        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)   # number of input patches F
        self.pred_len = config.pred_len
        self.step = math.ceil(self.pred_len / self.patch_len)    # number of output patches S
        self.padding_len = self.patch_len * self.freq - self.input_len
        
        # === Future Projection ===
        # Project flattened past latents [B, F*H] to flattened future latents [B, S*H]
        self.future_proj = nn.Linear(self.freq * self.dynamic_dim, self.step * self.dynamic_dim)

        # === Encoder & Decoder ===
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.decoder_mu = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        # We might not need sigma decoder if we only do reconstruction with MSE, 
        # but keeping it for consistency if we want to model distribution later or use it in loss.
        # For simple MSE reconstruction, we often just use mu. 
        # However, kQVAE uses it for future prediction distribution. 
        # For pure reconstruction, we can just output mu.
        # Let's keep it simple and just output mu for reconstruction.

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === QVAE-related ===
        self.smoothing_beta = getattr(config, 'dist_beta', 10.0)

        # === Boltzmann Machine Prior (RBM) ===
        total_latent_vars = self.freq * self.dynamic_dim
        visible_dim = getattr(config, "bm_visible_dim", None)
        hidden_dim = getattr(config, "bm_hidden_dim", None)

        if visible_dim is None and hidden_dim is None:
            visible_dim = total_latent_vars // 2
            hidden_dim = total_latent_vars - visible_dim
        elif visible_dim is None:
            hidden_dim = int(hidden_dim)
            visible_dim = total_latent_vars - hidden_dim
        elif hidden_dim is None:
            visible_dim = int(visible_dim)
            hidden_dim = total_latent_vars - visible_dim

        visible_dim = int(visible_dim)
        hidden_dim = int(hidden_dim)

        if visible_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"Invalid RBM dims: bm_visible_dim={visible_dim}, bm_hidden_dim={hidden_dim}."
            )

        if visible_dim + hidden_dim != total_latent_vars:
            raise ValueError(
                f"RBM dims mismatch: bm_visible_dim + bm_hidden_dim = {visible_dim + hidden_dim}, "
                f"but latent dim is freq * dynamic_dim = {total_latent_vars}."
            )

        self.bm = RestrictedBoltzmannMachine(
            num_visible=visible_dim,
            num_hidden=hidden_dim,
            h_range=[-1, 1],
            j_range=[-1, 1],
        )
        self.bm_sampler = SimulatedAnnealingOptimizer(alpha=0.95)
        self.num_var1 = visible_dim

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            zeta:         [B, F, H] (sampled continuous latent)
            posterior:    MixtureGeneric posterior q(z, ζ|x)
        """
        B, L, C = x.shape

        # RevIN Normalization
        x_norm = self.revin(x, "norm")

        # === Patch Padding ===
        padded = torch.cat([x_norm[:, L - self.padding_len :, :], x_norm], dim=1)
        patches = padded.chunk(self.freq, dim=1)                # freq * [B, P, C]
        patches = torch.stack(patches, dim=1).reshape(B, self.freq, -1)  # [B, F, P*C]

        # === Encoder Output logits ===
        q_logits = self.encoder(patches)                        # [B, F, H]
        q_logits_flat = q_logits.reshape(q_logits.size(0), -1)
        
        # === QVAE Posterior ===
        posterior = MixtureGeneric(q_logits_flat, self.smoothing_beta)
        zeta = posterior.reparameterize(self.training)          # [B, F, H]
        
        # Reshape zeta back to [B, F, H] if needed (MixtureGeneric might flatten it? No, it usually keeps batch)
        # In kQVAE: zeta = posterior.reparameterize(self.training) -> [B, F, H] because q_logits_flat was [B, F*H]?
        # Wait, in kQVAE:
        # q_logits_flat = q_logits.reshape(q_logits.size(0), -1) -> [B, F*H]
        # posterior = MixtureGeneric(q_logits_flat, ...)
        # zeta = posterior.reparameterize(...) -> [B, F*H]
        # zeta_koopman = zeta.reshape(B, self.freq, self.dynamic_dim) -> [B, F, H]
        
        zeta = zeta.reshape(B, self.freq, self.dynamic_dim)

        return zeta, posterior,q_logits

    # =============================================================
    #           DECODING
    # =============================================================
    def decode(self, zeta: torch.Tensor, q_logits: torch.Tensor):
        """
        zeta: [B, F, H] (Past latent)
        return:
            x_rec:     [B, L, C] (Past reconstruction)
            y_rec:     [B, pred_len, C] (Future prediction)
        """
        B = zeta.shape[0]
        
        # ----- 1. Reconstruct Past (x_rec) -----
        x_rec_mu = self.decoder_mu(zeta)  # [B, F, P*C]
        x_rec_mu = x_rec_mu.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_mu = x_rec_mu.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]
        x_rec = self.revin(x_rec_mu, "denorm")

        # ----- 2. Predict Future (y_rec) -----
        # Flatten past latent: [B, F, H] -> [B, F*H]
        zeta_flat = zeta.reshape(B, -1)
        # Project to future latent: [B, F*H] -> [B, S*H]
        zeta_future_flat = self.future_proj(zeta_flat)
        # Reshape to patches: [B, S*H] -> [B, S, H]
        zeta_future = zeta_future_flat.reshape(B, self.step, self.dynamic_dim)

        # Decode Future
        y_rec_mu = self.decoder_mu(zeta_future)  # [B, S, P*C]
        y_rec_mu = y_rec_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        y_rec_mu = y_rec_mu.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]
        y_rec = self.revin(y_rec_mu, "denorm")
        
        return x_rec, y_rec

    # =============================================================
    #             BM Related: cross-entropy + KL(q‖p_BM)
    # =============================================================
    def _maybe_move_bm(self, device: torch.device):
        if self.bm is None:
            return
        if hasattr(self.bm, "to"):
            self.bm = self.bm.to(device)

    def _cross_entropy(
        self, logit_q: torch.Tensor, log_ratio: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-entropy term for the overlap distribution proposed in DVAE++"""
        # Split logit_q into two parts
        if self.bm.num_nodes != logit_q.shape[1]:
            raise ValueError(
                f"The number of variables in the Boltzmann machine {self.bm.num_nodes}"
                f" does not match the shape of logit_q {logit_q.shape[1]}."
            )
        logit_q1 = logit_q[:, : self.num_var1]
        logit_q2 = logit_q[:, self.num_var1 :]

        # Compute probabilities
        q1 = torch.sigmoid(logit_q1)
        q2 = torch.sigmoid(logit_q2)
        log_ratio1 = log_ratio[:, : self.num_var1]
        q1_pert = torch.sigmoid(logit_q1 + log_ratio1)

        # Compute cross-entropy
        cross_entropy = - torch.matmul(
            torch.cat([q1, q2], dim=-1), self.bm.linear_bias
        ) - torch.sum(
            torch.matmul(q1_pert, self.bm.quadratic_coef) * q2, dim=1, keepdim=True
        )
        cross_entropy = cross_entropy.squeeze(dim=1)
        s_neg = self.bm.sample(self.bm_sampler)
        cross_entropy = cross_entropy - self.bm(s_neg).mean()
       
        return cross_entropy

    def _kl_dist_from(self, posterior, zeta: torch.Tensor):
        """
        KL(q(z, ζ | x) || p_BM(z)) ≈ cross_entropy - entropy
        """
        # ----- 1) entropy -----
        entropy = 0
        entropy += torch.sum(posterior.entropy(), dim=1)
        
        # ----- 2) cross-entropy -----
        # zeta needs to be flattened to match posterior's expectation if posterior was created from flattened logits
        # In encode: q_logits_flat [B, F*H] -> posterior
        # zeta was reshaped to [B, F, H]
        # posterior.log_ratio expects input matching its distribution dim?
        # Let's check kQVAE.
        # kQVAE: log_ratio = posterior.log_ratio(zeta) 
        # In kQVAE encode: zeta = posterior.reparameterize(...) -> [B, F*H] (before reshape)
        # But then it passes `zeta` (which is [B, F, H] in kQVAE forward return?)
        # Wait, in kQVAE forward: `x_rec_latent, pred_latent, posterior, zeta, prior_dist = self.encode(x)`
        # In encode: `zeta = posterior.reparameterize(...)` (This is [B, F*H]?)
        # `zeta_koopman = zeta.reshape(B, self.freq, self.dynamic_dim)`
        # Returns `zeta` (the flattened one? or the reshaped one?)
        # In kQVAE encode returns `zeta` which is `zeta` (the flattened one if not overwritten).
        # Actually in kQVAE code:
        # `zeta = posterior.reparameterize(self.training)`
        # `zeta_koopman = zeta.reshape(...)`
        # `return ..., zeta, ...`
        # So `zeta` returned is [B, F*H].
        
        # However, in my encode I did:
        # `zeta = zeta.reshape(B, self.freq, self.dynamic_dim)`
        # So my returned `zeta` is [B, F, H].
        
        # `posterior.log_ratio(zeta)`: if posterior is over F*H dims, zeta should be [B, F*H].
        # So I should flatten zeta before passing to log_ratio if I return reshaped zeta.
        
        zeta_flat = zeta.reshape(zeta.shape[0], -1)
        log_ratio = posterior.log_ratio(zeta_flat)      # [B, F*H]
      
        cross_ent = self._cross_entropy(posterior.logit_mu, log_ratio)   # [B]

        # ----- 3) KL -----
        kl = cross_ent - entropy                    # [B]

        return kl

    def kl_divergence(self, posterior, zeta: torch.Tensor):
        """
        Return average KL over batch
        """
        kl = self._kl_dist_from(posterior, zeta)  # [B]
        return kl.mean()

    def wd_loss(self):
        # Weight decay loss
        w_weight_decay = 0.01 * torch.sum(self.bm.quadratic_coef**2)
        b_weight_decay = 0.005 * torch.sum(self.bm.linear_bias**2)
        wd_loss = w_weight_decay + b_weight_decay
        return wd_loss
    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_rec:     [B, L, C]
            y_rec:     [B, pred_len, C]
            posterior: MixtureGeneric
            zeta:      [B, F, H]
        """
        zeta, posterior,q_logits = self.encode(x)
        x_rec, y_rec = self.decode(zeta,q_logits)
        return x_rec, y_rec, posterior, zeta

    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Reconstruction: Sample from posterior q(z|x) multiple times.
        x: [B, L, C]
        return:
            samples: [B, num_samples, L, C]
        """
        B, L, C = x.shape
        
        # 1. Encode to get posterior
        # We don't need the single zeta sample from encode, just the posterior
        _, posterior,p = self.encode(x+torch.randn_like(x) )
        
        sample_list = []
        for _ in range(num_samples):
            # 2. Sample zeta from posterior
            # posterior.reparameterize returns [B, F*H] (based on MixtureGeneric)
            # We need to reshape it to [B, F, H] for decode
            zeta_sample = posterior.reparameterize(is_training=True)
            zeta_sample = zeta_sample.reshape(B, self.freq, self.dynamic_dim)
            
            # 3. Decode
            _, y_rec_sample = self.decode(zeta_sample,p) # Unpack tuple, get y_rec
            sample_list.append(y_rec_sample)
            
        # 4. Stack
        samples = torch.stack(sample_list, dim=1) # [B, num_samples, pred_len, C]
        return samples

        # sample from encoder embedding
        # future from encoder embedding
    #。energy diffusion / transformer
