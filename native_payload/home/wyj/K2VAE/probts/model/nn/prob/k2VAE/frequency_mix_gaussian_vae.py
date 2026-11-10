import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.gaussian_vae import GaussianVAE


class FrequencyMixProjection(nn.Module):
    """
    MixLinear-inspired latent projection.

    It maps past latent patches [B, F, H] to future latent patches [B, S, H]
    with two branches:
      - a time-domain square segment mixer from MixLinear
      - a frequency-domain mixer over segment indexes
    """

    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        hidden_dim: int,
        period_len: int = 2,
        lpf: int = 2,
        alpha: float = 0.5,
        use_local_conv: bool = True,
    ):
        super().__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.hidden_dim = hidden_dim
        self.period_len = max(1, min(int(period_len), input_steps))
        self.alpha = float(alpha)

        self.seg_num_x = math.ceil(self.input_steps / self.period_len)
        self.seg_num_y = math.ceil(self.output_steps / self.period_len)
        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seg_num_x))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.seg_num_y))
        self.square_seg_num_x = self.sqrt_seg_num_x ** 2
        self.square_seg_num_y = self.sqrt_seg_num_y ** 2
        self.padded_input_steps = self.seg_num_x * self.period_len

        self.lpf = max(1, min(int(lpf), self.seg_num_x))

        self.time_linear_1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.time_linear_2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        self.freq_linear_1 = nn.Linear(self.lpf, 2, bias=False).to(torch.cfloat)
        self.freq_linear_2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

        self.local_conv = None
        if use_local_conv:
            kernel_size = self.period_len + 1
            self.local_conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=self.period_len // 2,
                padding_mode="zeros",
                bias=False,
            )

    def _pad_time(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        if z.size(1) >= target_len:
            return z[:, :target_len, :]
        pad_len = target_len - z.size(1)
        pad = z[:, -1:, :].expand(z.size(0), pad_len, z.size(2))
        return torch.cat([z, pad], dim=1)

    def _time_branch(self, segments: torch.Tensor) -> torch.Tensor:
        bsz, hidden_dim, period_len, seg_num = segments.shape
        padded = F.pad(segments, (0, self.square_seg_num_x - seg_num))
        grid = padded.reshape(
            bsz,
            hidden_dim,
            period_len,
            self.sqrt_seg_num_x,
            self.sqrt_seg_num_x,
        )
        mixed = self.time_linear_1(grid).permute(0, 1, 2, 4, 3)
        mixed = self.time_linear_2(mixed).permute(0, 1, 2, 4, 3)
        return mixed.reshape(bsz, hidden_dim, period_len, self.square_seg_num_y)

    def _freq_branch(self, segments: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(segments, dim=-1)[..., : self.lpf]
        mixed = self.freq_linear_1(spectrum)
        mixed = self.freq_linear_2(mixed)
        return torch.fft.ifft(mixed, dim=-1).real

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        bsz = z_flat.size(0)
        z = z_flat.reshape(bsz, self.input_steps, self.hidden_dim)

        seq_mean = z.mean(dim=1, keepdim=True)
        z = z - seq_mean

        if self.local_conv is not None:
            conv_in = z.permute(0, 2, 1).reshape(-1, 1, self.input_steps)
            conv_out = self.local_conv(conv_in)
            if conv_out.size(-1) < self.input_steps:
                conv_out = F.pad(conv_out, (0, self.input_steps - conv_out.size(-1)))
            conv_out = conv_out[..., : self.input_steps]
            z = z + conv_out.reshape(bsz, self.hidden_dim, self.input_steps).permute(0, 2, 1)

        z = self._pad_time(z, self.padded_input_steps)
        segments = z.permute(0, 2, 1).reshape(
            bsz, self.hidden_dim, self.seg_num_x, self.period_len
        )
        segments = segments.permute(0, 1, 3, 2)

        time_out = self._time_branch(segments)[..., : self.seg_num_y]
        freq_out = self._freq_branch(segments)
        mixed = self.alpha * time_out + (1.0 - self.alpha) * freq_out

        mixed = mixed.permute(0, 1, 3, 2).reshape(bsz, self.hidden_dim, -1)
        mixed = mixed.permute(0, 2, 1)[:, : self.output_steps, :]
        mixed = mixed + seq_mean
        return mixed.reshape(bsz, self.output_steps * self.hidden_dim)


class FrequencyMixGaussianVAE(GaussianVAE):
    """GaussianVAE with a MixLinear-style frequency-aware latent future projector."""

    def __init__(self, config):
        super().__init__(config)
        self.future_proj = FrequencyMixProjection(
            input_steps=self.freq,
            output_steps=self.step,
            hidden_dim=self.dynamic_dim,
            period_len=getattr(config, "mix_period_len", 2),
            lpf=getattr(config, "mix_lpf", 2),
            alpha=getattr(config, "mix_alpha", 0.5),
            use_local_conv=getattr(config, "mix_use_local_conv", True),
        )
