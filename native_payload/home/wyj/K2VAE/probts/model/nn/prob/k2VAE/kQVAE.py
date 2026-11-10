import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange

# ==== 来自你的 k2VAE ====
from probts.model.nn.prob.k2VAE.koopman import KPLayer, KPLayerApprox, MLP
from probts.model.nn.prob.k2VAE.kalman import KalmanFilter
from probts.model.nn.prob.k2VAE.auxiliarynet import Transformer as AuxiliaryNet
from probts.model.nn.prob.k2VAE.RevIN import RevIN
import kaiwu as kw
import os
# ==== 来自你的 QVAE ====
from probts.utils.qvae_dist_util import MixtureGeneric
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer


# print("kaiwu User ID:", os.getenv("kaiwu_USER_ID"), "kaiwu SDK Code:", os.getenv("kaiwu_SDK_CODE"))
kw.license.init(os.getenv("kaiwu_USER_ID"), os.getenv("kaiwu_SDK_CODE"))


class kQVAE(nn.Module):
    """
    Koopman + Kalman + QVAE latent 的时间序列模型
    """

    def __init__(self, config):
        super().__init__()

        # === 基础参数 ===
        self.config = config
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep

        self.dynamic_dim = config.dynamic_dim  # latent dim H
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars             # 输入通道数 C

        # === patch 分割 ===
        self.freq = math.ceil(self.input_len / self.patch_len)   # 输入 patch 数 F
        self.step = math.ceil(self.pred_len / self.patch_len)    # 输出 patch 数 S
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Koopman operator ===
        self.koopman = (
            KPLayerApprox(config.dynamic_dim, config.init_koopman)
            if self.multistep else
            KPLayer(config.dynamic_dim, config.init_koopman)
        )

        # === Kalman Filter ===
        self.kalman = KalmanFilter(
            state_dim=self.dynamic_dim,
            init=config.init_kalman,
        )

        # === Auxiliary network（Transformer）===
        self.auxiliary = AuxiliaryNet(self.config)

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
        self.decoder_sigma = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === QVAE-related ===
        self.smoothing_beta = 10.0

        # === Boltzmann Machine 先验（RBM）===
        visible_dim = 512
        hidden_dim = 512
        self.bm = RestrictedBoltzmannMachine(
            num_visible=visible_dim,
            num_hidden=hidden_dim,
            h_range=[-1, 1],
            j_range=[-1, 1],
        )
        self.bm_sampler = SimulatedAnnealingOptimizer(alpha=0.95,size_limit=64)
        self.num_var1 = visible_dim
    # =============================================================
    #               ENCODING  (带 QVAE 后验)
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_rec_latent: [B, F, H]   Koopman 对训练段 latent 的重构
            pred_latent:  [B, S, H]   未来的 latent（Kalman 均值）
            posterior:    MixtureGeneric 后验 q(z, ζ|x)
            zeta:         后验采样的连续 latent，形状 [B, F, H]
            prior_dist:   Kalman 给出的未来 latent 高斯分布
        """
        B, L, C = x.shape

        # RevIN 标准化
        x_norm = self.revin(x, "norm")

        # === patch padding ===
        padded = torch.cat([x_norm[:, L - self.padding_len :, :], x_norm], dim=1)
        patches = padded.chunk(self.freq, dim=1)                # freq 个 [B, P, C]
        patches = torch.stack(patches, dim=1).reshape(B, self.freq, -1)  # [B, F, P*C]

        # === Encoder 输出 logits（Bernoulli logit）===
        q_logits = self.encoder(patches)                        # [B, F, H]
        q_logits_flat = q_logits.reshape(q_logits.size(0), -1)
        # === QVAE posterior ===
        posterior = MixtureGeneric(q_logits_flat, self.smoothing_beta)
        zeta = posterior.reparameterize(self.training)          # [B, F, H]
        zeta_koopman = q_logits
        # zeta_koopman = zeta.reshape(B, self.freq, self.dynamic_dim)     # [B, F, H]
        # === Koopman 演化 latent ===
        x_rec_latent, Y = self.koopman(zeta_koopman, self.step)         # [B,F,H], [B,S,H]

        # === Residual dynamics ===
        latent_residual = zeta_koopman - x_rec_latent                   # [B,F,H]
        residual = self.auxiliary(latent_residual)              # [B,F,H] 或相似形状

        # === Kalman predict ===
        # 这里假设 KalmanFilter 接口和你的 k2VAE 一样：
        #   in:  last_state, residual, Y
        #   out: pred_latent_mean, prior_dist (Normal)
        pred_latent, prior_dist = self.kalman(
            zeta_koopman[:, -1, :],   # 最后一个 latent 状态
            residual,
            Y,
        )
        pred_latent = pred_latent + residual  # 叠加 residual 修正

        return x_rec_latent, pred_latent, posterior, zeta, prior_dist

    # =============================================================
    #           DECODING : latent → Gaussian distribution
    # =============================================================
    def decode(self, x_rec_latent: torch.Tensor, x_pred_latent: torch.Tensor):
        """
        x_rec_latent: [B, F, H]
        x_pred_latent: [B, S, H]
        return:
            x_rec:    [B, L, C]
            post_dist: Normal([B, pred_len, C])
        """
        B = x_rec_latent.shape[0]

        # ----- reconstruct input -----
        x_rec_mu = self.decoder_mu(x_rec_latent)  # [B, F, P*C]
        x_rec_mu = x_rec_mu.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_mu = x_rec_mu.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]

        # ----- predict future -----
        pred_mu = self.decoder_mu(x_pred_latent)      # [B, S, P*C]
        pred_sigma = self.decoder_sigma(x_pred_latent)

        pred_mu = pred_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        pred_mu = pred_mu.reshape(B, -1, self.enc_in)[:, : self.pred_len]      # [B, pred_len, C]

        pred_sigma = pred_sigma.reshape(B, self.step, self.patch_len, self.enc_in)
        pred_sigma = pred_sigma.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]

        # RevIN inverse
        x_rec = self.revin(x_rec_mu, "denorm")
        pred_mu = self.revin(pred_mu, "denorm")
        pred_sigma = self.revin(pred_sigma, "denorm")

        post_dist = Normal(pred_mu, F.softplus(pred_sigma) + 1e-6)
        return x_rec, post_dist

    # =============================================================
    #             BM 相关：cross-entropy + KL(q‖p_BM)
    # =============================================================
    def _maybe_move_bm(self, device: torch.device):
        if self.bm is None:
            return
        if hasattr(self.bm, "to"):
            self.bm = self.bm.to(device)

    def _cross_entropy(
        self, logit_q: torch.Tensor, log_ratio: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-entropy term for the overlap distribution proposed in DVAE++

        Args:
            logit_q (torch.Tensor): Log-odds of Bernoulli distribution defined for each variable
            log_ratio (torch.Tensor): Log(r(ζ|z=1)/r(ζ|z=0)) for each ζ

        Returns:
            torch.Tensor: Cross-entropy tensor for each ζ
        """
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
        posterior.entropy(): [B, F, H]
        posterior.log_ratio(zeta): [B, F, H]
        """

        # ----- 1) entropy -----
        # 原始 QVAE 对所有 latent sum，这里 latent 在 H 维
        # 然后必须对 F 做 mean，使其与 cross-entropy 对齐
        entropy = 0
        logit_q = 0
        log_ratio = 0
       
        # print("posterior:",posterior.entropy().shape)
        entropy += torch.sum(posterior.entropy(), dim=1)
        # ----- 2) cross-entropy -----
        log_ratio = posterior.log_ratio(zeta)      # [B, F, H]
      
        cross_ent = self._cross_entropy(posterior.logit_mu, log_ratio)   # [B]

        # ----- 3) KL -----
        kl = cross_ent - entropy                    # [B]

        return kl

    def kl_divergence(self, posterior, zeta: torch.Tensor):
        """
        返回 batch 上的平均 KL
        """
        kl = self._kl_dist_from(posterior, zeta)  # [B]
        return kl.mean()

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_rec:     [B, L, C]       历史重构
            posterior: MixtureGeneric  QVAE 后验
            post_dist: Normal          未来预测分布
            zeta:      [B, F, H]       连续 latent 样本
        """
        x_rec_latent, pred_latent, posterior, zeta, prior_dist = self.encode(x)
       
        # 从 Kalman prior_dist 里采样 latent 噪声（类似 k2VAE）
        pred_latent_noisy = pred_latent + prior_dist.rsample()

        x_rec, post_dist = self.decode(x_rec_latent, pred_latent_noisy)
        return x_rec, posterior, post_dist, zeta

    # =============================================================
    #                     SAMPLING
    # =============================================================
    @torch.no_grad()
    # def sample(self, x: torch.Tensor, num_samples: int):
    #     """
    #     x: [B, L, C]
    #     return: [num_samples, B, pred_len, C]
    #     """
    #     x_rec_latent, pred_latent, posterior, zeta, prior_dist = self.encode(x)

    #     samples = []
    #     for _ in range(num_samples):
    #         latent = pred_latent + prior_dist.rsample()
    #         _, dist = self.decode(x_rec_latent, latent)
    #         samples.append(dist.rsample())   # [B, pred_len, C]

    #     return torch.stack(samples, dim=0)   # [N, B, pred_len, C]
    def sample(self, x, num_samples):
        x_rec, pred, posterior, zeta, prior_dist = self.encode(x)
        dist_list = []
        sample_list = []
        for i in range(self.config.sample_schedule):
            t = pred + torch.randn_like(pred).to(x.device)
            _, post_dist = self.decode(x_rec, t)
            dist_list.append(post_dist)

        cnt = 0
        k = num_samples // self.config.sample_schedule
        for id, dist in enumerate(dist_list):
            sample_num = k if id < self.config.sample_schedule - 1 else num_samples - cnt
            samples = dist.rsample(sample_shape=(sample_num,))
            sample_list.append(samples)
            cnt += sample_num

        if num_samples == self.config.sample_schedule:
            sample_list = [dist.loc.unsqueeze(0) for dist in dist_list]

        return torch.concat(sample_list, dim=0)