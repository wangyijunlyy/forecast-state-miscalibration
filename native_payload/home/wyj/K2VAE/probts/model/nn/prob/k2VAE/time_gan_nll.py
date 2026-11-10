"""
* @description: 改进版 TimeGAN (Gaussian NLL Loss) - 神经网络实现
*
* 特点:
* - Generator 双头输出: mu (均值) 和 logvar (对数方差)
* - 使用重参数化技巧采样，让梯度可以通过采样过程回传
* - 辅助损失使用 Gaussian NLL，而非 MSE
* - 判别器看到的是从分布中采样的样本
*
* 优势:
* - 更强的抗噪能力: 可以通过增大 sigma 来"包裹"异常值
* - 更高的生成多样性: 采样机制引入随机性
* - 不确定性量化: 可以捕捉预测的不确定性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from probts.model.nn.prob.k2VAE.RevIN import RevIN
from probts.model.nn.prob.k2VAE.koopman import MLP


class TemporalBlock(nn.Module):
    """
    通用的 LSTM 编码器块
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
            out: [batch, hidden_dim]
        """
        out, (h, c) = self.lstm(x)
        return out[:, -1, :]


class Discriminator(nn.Module):
    """
    判别器：判断输入的序列是真实的还是生成的
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
            validity: [batch, 1]
        """
        flat = x.reshape(x.size(0), -1)
        validity = self.model(flat)
        return validity


class GeneratorLS(nn.Module):
    """
    改进版生成器: Location-Scale Generator (双头输出)

    结构:
        x -> RevIN(norm) -> Patches -> LSTM Encoder -> Shared Decoder
                                                     -> mu_head -> mu
                                                     -> logvar_head -> logvar

    输出分布参数 (mu, logvar)，通过重参数化技巧采样
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === 基本参数 ===
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.enc_in = config.n_vars
        self.hidden_dim = config.hidden_dim
        self.dynamic_dim = config.dynamic_dim

        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

        # === Encoder (LSTM) ===
        self.encoder = TemporalBlock(
            input_dim=self.patch_len * self.enc_in,
            hidden_dim=self.dynamic_dim,
            num_layers=2,
        )

        # === 共享解码层 ===
        self.shared_decoder = nn.Sequential(
            nn.Linear(self.dynamic_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # === 双头输出 ===
        # Location Head (Mean)
        self.mu_head = nn.Linear(self.hidden_dim, self.pred_len * self.enc_in)

        # Scale Head (Log Variance)
        self.logvar_head = nn.Linear(self.hidden_dim, self.pred_len * self.enc_in)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """将输入序列切分为patches"""
        B, L, C = x.shape

        if self.padding_len > 0:
            padded = torch.cat([x[:, L - self.padding_len :, :], x], dim=1)
        else:
            padded = x

        patches = padded.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1)
        patches = patches.reshape(B, self.freq, -1)

        return patches

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: [B, L, C]
        Returns:
            mu: [B, pred_len, C] - 预测均值 (原始尺度)
            logvar: [B, pred_len, C] - 预测对数方差
        """
        B = x.shape[0]

        # 1. RevIN 归一化
        x_norm = self.revin(x, "norm")

        # 2. Patchify
        patches = self._patchify(x_norm)  # [B, F, P*C]

        # 3. LSTM 编码
        encoded = self.encoder(patches)  # [B, dynamic_dim]

        # 4. 共享解码
        decoded = self.shared_decoder(encoded)  # [B, hidden_dim]

        # 5. 双头输出
        # 均值头
        mu_raw = self.mu_head(decoded)  # [B, pred_len * C]
        mu_raw = mu_raw.reshape(B, self.pred_len, self.enc_in)
        mu = self.revin(mu_raw, "denorm")  # 反归一化到原始尺度

        # 对数方差头 (不需要反归一化，直接作为 log variance)
        logvar = self.logvar_head(decoded)  # [B, pred_len * C]
        logvar = logvar.reshape(B, self.pred_len, self.enc_in)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # 数值稳定性

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧: 让梯度可以通过采样过程回传

        z = mu + eps * std, where eps ~ N(0, I)

        Args:
            mu: [B, pred_len, C]
            logvar: [B, pred_len, C]
        Returns:
            sampled: [B, pred_len, C]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_distribution_params(self, x: torch.Tensor):
        """
        获取分布参数 (用于NLL损失计算)

        Returns:
            mu, logvar: 都是 [B, pred_len, C]
        """
        return self.forward(x)


class TimeGANNLL(nn.Module):
    """
    完整的 TimeGAN (NLL版本 / Location-Scale版本)

    包含:
    - Generator: 双头生成器，输出分布参数
    - Discriminator: 判别真假序列
    - gaussian_nll: Gaussian NLL 损失函数
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Generator (双头)
        self.generator = GeneratorLS(config)

        # Discriminator
        self.discriminator = Discriminator(
            input_dim=config.n_vars,
            hidden_dim=config.hidden_dim,
            seq_len=config.pred_len,
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: [B, L, C]
        Returns:
            mu: [B, pred_len, C]
            logvar: [B, pred_len, C]
        """
        return self.generator(x)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成采样 (使用重参数化技巧)

        Args:
            x: [B, L, C]
        Returns:
            gen_future: [B, pred_len, C] - 从分布中采样的预测
        """
        mu, logvar = self.generator(x)
        gen_future = self.generator.reparameterize(mu, logvar)
        return gen_future

    def discriminate(self, y: torch.Tensor) -> torch.Tensor:
        """判别序列真假"""
        return self.discriminator(y)

    @staticmethod
    def gaussian_nll(
        y: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Gaussian Negative Log-Likelihood

        NLL = 0.5 * (log(2π) + logvar + (y - mu)^2 / exp(logvar))

        这迫使 Generator 在"拿不准"的时候增大 logvar，而不是乱猜均值

        Args:
            y: [B, pred_len, C] - 真实目标
            mu: [B, pred_len, C] - 预测均值
            logvar: [B, pred_len, C] - 预测对数方差
            reduction: "mean", "sum", or "none"
        Returns:
            nll: scalar or [B]
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        nll = 0.5 * (
            math.log(2 * math.pi)
            + logvar
            + (y - mu) ** 2 / torch.exp(logvar)
        )  # [B, pred_len, C]

        # 对时间和变量维度求和
        nll = nll.sum(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """
        生成多个多样化样本

        由于使用了 Location-Scale 模型，每次采样都会得到不同的结果

        Args:
            x: [B, L, C]
            num_samples: 采样数量
        Returns:
            samples: [B, num_samples, pred_len, C]
        """
        mu, logvar = self.generator(x)
        std = torch.exp(0.5 * logvar)

        sample_list = []
        for _ in range(num_samples):
            eps = torch.randn_like(std)
            sample = mu + eps * std  # 重参数化采样
            sample_list.append(sample)

        samples = torch.stack(sample_list, dim=1)  # [B, num_samples, pred_len, C]
        return samples

    def get_mean_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取确定性均值预测

        Args:
            x: [B, L, C]
        Returns:
            mu: [B, pred_len, C]
        """
        mu, _ = self.generator(x)
        return mu
