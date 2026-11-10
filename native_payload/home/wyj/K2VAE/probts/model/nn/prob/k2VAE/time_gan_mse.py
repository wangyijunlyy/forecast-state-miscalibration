"""
* @description: 普通版 TimeGAN (MSE Loss) - 神经网络实现
*
* 特点:
* - Generator 直接输出预测值 y_hat (单头)
* - 辅助损失使用 MSE
* - 判别器看到的是确定性的点估计
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange

from probts.model.nn.prob.k2VAE.RevIN import RevIN
from probts.model.nn.prob.k2VAE.koopman import MLP


class TemporalBlock(nn.Module):
    """
    通用的 LSTM 编码器块
    将输入序列编码为固定维度的隐状态
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            out: [batch, hidden_dim] - 最后一个时间步的隐状态
        """
        out, (h, c) = self.lstm(x)
        return out[:, -1, :]  # 取最后一个时间步


class Discriminator(nn.Module):
    """
    判别器：判断输入的序列是真实的还是生成的
    使用 MLP 处理展平后的序列
    """

    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        flat_dim = input_dim * seq_len
        self.model = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            validity: [batch, 1] - 真实性概率
        """
        flat = x.reshape(x.size(0), -1)
        validity = self.model(flat)
        return validity


class GeneratorMSE(nn.Module):
    """
    普通版生成器: 直接输出预测值 y_hat

    结构:
        x -> RevIN(norm) -> Patches -> LSTM Encoder -> MLP Decoder -> y_hat -> RevIN(denorm)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === 基本参数 ===
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.enc_in = config.n_vars  # 输入通道数 C
        self.hidden_dim = config.hidden_dim
        self.dynamic_dim = config.dynamic_dim  # LSTM隐层维度

        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)  # 输入patch数 F
        self.step = math.ceil(self.pred_len / self.patch_len)  # 输出patch数 S
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === Encoder (LSTM) ===
        # 输入: [B, F, patch_len * enc_in], 输出: [B, dynamic_dim]
        self.encoder = TemporalBlock(
            input_dim=self.patch_len * self.enc_in,
            hidden_dim=self.dynamic_dim,
            num_layers=2,
        )

        # === Decoder (MLP) - 双头输出 ===
        # 将编码后的隐状态映射到未来预测的 mu 和 sigma
        self.decoder_mu = MLP(
            f_in=self.dynamic_dim,
            f_out=self.pred_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
        self.decoder_sigma = MLP(
            f_in=self.dynamic_dim,
            f_out=self.pred_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=config.hidden_layers,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入序列切分为patches

        Args:
            x: [B, L, C] 归一化后的输入
        Returns:
            patches: [B, F, patch_len * C]
        """
        B, L, C = x.shape

        # Padding: 前面补一段来自序列末尾的片段
        if self.padding_len > 0:
            padded = torch.cat([x[:, L - self.padding_len :, :], x], dim=1)
        else:
            padded = x

        # 切成 F 个 patch
        patches = padded.chunk(self.freq, dim=1)  # freq * [B, P, C]
        patches = torch.stack(patches, dim=1)  # [B, F, P, C]
        patches = patches.reshape(B, self.freq, -1)  # [B, F, P*C]

        return patches

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: [B, L, C] 历史序列
        Returns:
            mu: [B, pred_len, C] 预测均值
            sigma: [B, pred_len, C] 预测标准差 (正数)
        """
        B = x.shape[0]

        # 1. RevIN 归一化
        x_norm = self.revin(x, "norm")

        # 2. Patchify
        patches = self._patchify(x_norm)  # [B, F, P*C]

        # 3. LSTM 编码
        encoded = self.encoder(patches)  # [B, dynamic_dim]

        # 4. MLP 解码 (双头)
        mu_raw = self.decoder_mu(encoded)  # [B, pred_len * C]
        mu_raw = mu_raw.reshape(B, self.pred_len, self.enc_in)  # [B, pred_len, C]
        
        sigma_raw = self.decoder_sigma(encoded)  # [B, pred_len * C]
        sigma_raw = sigma_raw.reshape(B, self.pred_len, self.enc_in)  # [B, pred_len, C]

        # 5. RevIN 反归一化
        mu = self.revin(mu_raw, "denorm")
        
        # sigma 只用 stdev 缩放，不加 mean
        sigma = F.softplus(sigma_raw) + 1e-6  # 确保正数
        sigma = sigma * self.revin.stdev  # 用 RevIN 的 std 缩放

        return mu, sigma

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        重建历史序列 (用于额外的重建损失)

        Args:
            x: [B, L, C]
        Returns:
            x_rec: [B, L, C]
        """
        B = x.shape[0]

        x_norm = self.revin(x, "norm")
        patches = self._patchify(x_norm)
        encoded = self.encoder(patches)

        # 使用单独的重建头
        if not hasattr(self, "rec_decoder"):
            self.rec_decoder = MLP(
                f_in=self.dynamic_dim,
                f_out=self.input_len * self.enc_in,
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.config.hidden_layers,
            ).to(x.device)

        x_rec_raw = self.rec_decoder(encoded)
        x_rec_raw = x_rec_raw.reshape(B, self.input_len, self.enc_in)
        x_rec = self.revin(x_rec_raw, "denorm")

        return x_rec


class TimeGANMSE(nn.Module):
    """
    完整的 TimeGAN (MSE版本)

    包含:
    - Generator: 生成未来序列
    - Discriminator: 判别真假序列
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Generator
        self.generator = GeneratorMSE(config)

        # Discriminator
        self.discriminator = Discriminator(
            input_dim=config.n_vars,
            hidden_dim=config.hidden_dim,
            seq_len=config.pred_len,
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播 (返回分布对象，类似 PatchTST)

        Args:
            x: [B, L, C]
        Returns:
            dist: Normal 分布对象
        """
        mu, sigma = self.generator(x)
        return Normal(loc=mu, scale=sigma)

    def discriminate(self, y: torch.Tensor) -> torch.Tensor:
        """
        判别序列真假

        Args:
            y: [B, pred_len, C]
        Returns:
            validity: [B, 1]
        """
        return self.discriminator(y)

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """
        从高斯分布中采样 (类似 PatchTST)

        Args:
            x: [B, L, C]
            num_samples: 采样数量
        Returns:
            samples: [B, num_samples, pred_len, C]
        """
        dist = self.forward(x)  # Normal 分布
        samples = dist.rsample(sample_shape=torch.Size([num_samples]))  # [num_samples, B, pred_len, C]
        samples = samples.permute(1, 0, 2, 3)  # [B, num_samples, pred_len, C]
        return samples
