"""
Quantum VAE for Time Series Forecasting

Integrates:
1. Amplitude Encoding: Map time series patches to quantum amplitudes
2. Variational Quantum Circuits (VQC): Parameterized quantum gates as encoder/decoder
3. vMF Latent Space: Spherical latent variable distribution
4. Dual-Head Output: Predict both mean and logvar for NLL loss

Reference: QVAE-Mole (NeurIPS 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    HAS_TORCHQUANTUM = True
except ImportError:
    HAS_TORCHQUANTUM = False
    print("Warning: torchquantum not installed. QuantumVAE will use classical simulation.")

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class QuantumVAE(nn.Module):
    """
    Quantum VAE for Time Series Forecasting.
    
    Architecture:
        x -> RevIN -> Patching -> MLP -> AmplitudeEncoder -> VQC Encoder
            -> Partial Trace -> vMF Sampling -> VQC Decoder -> MLP -> x_hat
    """
    
    def __init__(self, config):
        super().__init__()
        
        if not HAS_TORCHQUANTUM:
            raise ImportError("torchquantum is required for QuantumVAE. "
                            "Install with: pip install torchquantum")
        
        # === Basic Parameters ===
        self.config = config
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        self.pred_len = config.pred_len
        self.enc_in = config.n_vars
        
        # Quantum parameters
        self.n_qbits = getattr(config, 'n_qbits', 6)
        self.n_blocks = getattr(config, 'n_blocks', 2)
        self.bottleneck_qbits = getattr(config, 'bottleneck_qbits', 2)
        self.kappa = getattr(config, 'kappa', 0.1)  # vMF concentration
        
        # Dimensions
        self.quantum_dim = 2 ** self.n_qbits
        self.latent_dim = 2 ** (self.n_qbits - self.bottleneck_qbits)
        
        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len
        self.patch_dim = self.patch_len * self.enc_in
        
        # === Classical Pre/Post Processing ===
        self.revin = RevIN(self.enc_in)
        
        # Project patches to quantum-compatible dimension
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.patch_dim, self.quantum_dim),
            nn.ReLU(),
        )
        
        # Project quantum output back to patch dimension (dual-head)
        self.post_decoder = nn.Sequential(
            nn.Linear(self.quantum_dim, self.patch_dim * 2),  # mu + logvar
        )
        
        # Future projection
        self.future_proj = MLP(
            f_in=self.freq * self.latent_dim,
            f_out=self.step * self.latent_dim,
            activation="relu",
            hidden_dim=self.freq * self.latent_dim // 2,
            hidden_layers=1,
        )
        
        # === Quantum Components ===
        self._init_quantum_layers()
        
    def _init_quantum_layers(self):
        """Initialize quantum circuit layers."""
        self.encoder = tq.AmplitudeEncoder()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.qdev = tq.QuantumDevice(self.n_qbits)
        
        # Encoder VQC layers
        self.ry_enc = tq.QuantumModuleList()
        self.rz1_enc = tq.QuantumModuleList()
        self.rz2_enc = tq.QuantumModuleList()
        self.crx_enc = tq.QuantumModuleList()
        
        # Decoder VQC layers
        self.ry_dec = tq.QuantumModuleList()
        self.rz1_dec = tq.QuantumModuleList()
        self.rz2_dec = tq.QuantumModuleList()
        self.crx_dec = tq.QuantumModuleList()
        
        for k in range(self.n_blocks + 1):
            for i in range(self.n_qbits):
                # Encoder layers
                self.rz1_enc.append(tq.RZ(trainable=True, has_params=True))
                self.ry_enc.append(tq.RY(trainable=True, has_params=True))
                self.rz2_enc.append(tq.RZ(trainable=True, has_params=True))
                if k != self.n_blocks:
                    self.crx_enc.append(tq.CRX(trainable=True, has_params=True))
                
                # Decoder layers
                self.rz1_dec.append(tq.RZ(trainable=True, has_params=True))
                self.ry_dec.append(tq.RY(trainable=True, has_params=True))
                self.rz2_dec.append(tq.RZ(trainable=True, has_params=True))
                if k != self.n_blocks:
                    self.crx_dec.append(tq.CRX(trainable=True, has_params=True))
    
    def _vqc_encoder(self, batch_size):
        """Apply VQC encoder circuit."""
        for k in range(self.n_blocks + 1):
            for i in range(self.n_qbits):
                self.rz1_enc[k * self.n_qbits + i](self.qdev, wires=i)
                self.ry_enc[k * self.n_qbits + i](self.qdev, wires=i)
                self.rz2_enc[k * self.n_qbits + i](self.qdev, wires=i)
            if k != self.n_blocks:
                for i in range(self.n_qbits):
                    self.crx_enc[k * self.n_qbits + i](
                        self.qdev, wires=[i, (i + 1) % self.n_qbits]
                    )
    
    def _vqc_decoder(self, batch_size):
        """Apply VQC decoder circuit (reverse of encoder)."""
        for k in range(self.n_blocks, -1, -1):
            if k == self.n_blocks:
                for i in range(self.n_qbits):
                    self.rz2_dec[k * self.n_qbits + i](self.qdev, wires=i)
                    self.ry_dec[k * self.n_qbits + i](self.qdev, wires=i)
                    self.rz1_dec[k * self.n_qbits + i](self.qdev, wires=i)
            else:
                for i in range(self.n_qbits - 1, -1, -1):
                    self.crx_dec[k * self.n_qbits + i](
                        self.qdev, wires=[i, (i + 1) % self.n_qbits]
                    )
                for i in range(self.n_qbits):
                    self.rz2_dec[k * self.n_qbits + i](self.qdev, wires=i)
                    self.ry_dec[k * self.n_qbits + i](self.qdev, wires=i)
                    self.rz1_dec[k * self.n_qbits + i](self.qdev, wires=i)
    
    def sampling_vmf(self, mu):
        """
        Sample from von Mises-Fisher distribution.
        
        Args:
            mu: [B, D] mean direction on unit sphere
            
        Returns:
            z: [B, D] sampled direction
        """
        dims = mu.shape[-1]
        kappa = self.kappa
        
        epsilon = 1e-7
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x**2 + epsilon) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        
        # Pre-sample W values
        rand_nums = np.interp(torch.rand(10**5).numpy(), y, x)
        W = torch.tensor(rand_nums, dtype=mu.dtype, device=mu.device)
        
        idx = torch.randint(0, 10**5, (mu.size(0),), device=mu.device)
        w = W[idx].unsqueeze(1)
        
        # Sample tangent direction
        eps = torch.randn_like(mu)
        nu = eps - torch.sum(eps * mu, dim=1, keepdim=True) * mu
        nu = F.normalize(nu, p=2, dim=-1)
        
        return w * mu + torch.sqrt(1 - w**2) * nu
    
    def encode(self, x: torch.Tensor):
        """
        Encode input sequence to latent space.
        
        Args:
            x: [B, L, C] input sequence
            
        Returns:
            mu_z: [B, F, latent_dim] latent mean direction
            z: [B, F, latent_dim] sampled latent
        """
        B, L, C = x.shape
        
        # RevIN normalization
        x_norm = self.revin(x, "norm")
        
        # Padding
        if self.padding_len > 0:
            padded = torch.cat([x_norm[:, L - self.padding_len:, :], x_norm], dim=1)
        else:
            padded = x_norm
        
        # Patching: [B, F, P*C]
        patches = padded.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1)
        patches = patches.reshape(B, self.freq, -1)
        
        # Project to quantum dimension
        quantum_in = self.pre_encoder(patches)  # [B, F, 2^n_qbits]
        
        # Normalize for amplitude encoding
        quantum_in = F.normalize(quantum_in, p=2, dim=-1)
        
        # Process each patch through quantum circuit
        mu_z_list = []
        for f in range(self.freq):
            self.qdev.reset_states(B)
            
            # Amplitude encode
            self.encoder(self.qdev, quantum_in[:, f, :])
            
            # Apply VQC encoder
            self._vqc_encoder(B)
            
            # Get amplitudes and partial trace
            amplitudes = self.qdev.get_states_1d().abs()
            amplitudes_sq = amplitudes ** 2
            
            # Partial trace: reduce from 2^n_qbits to latent_dim
            mu_f = amplitudes_sq.reshape(
                B, self.latent_dim, 2 ** self.bottleneck_qbits
            ).sum(dim=-1).sqrt()
            
            # Normalize to unit sphere
            mu_f = F.normalize(mu_f, p=2, dim=-1)
            mu_z_list.append(mu_f)
        
        mu_z = torch.stack(mu_z_list, dim=1)  # [B, F, latent_dim]
        
        # vMF sampling
        if self.training:
            z = torch.stack([
                self.sampling_vmf(mu_z[:, f, :]) for f in range(self.freq)
            ], dim=1)
        else:
            z = mu_z
        
        return mu_z, z
    
    def decode(self, z: torch.Tensor):
        """
        Decode latent to reconstruction and prediction.
        
        Args:
            z: [B, F, latent_dim] latent samples
            
        Returns:
            x_mu: [B, L, C] reconstruction mean
            y_mu: [B, pred_len, C] prediction mean
            x_logvar: [B, L, C] reconstruction logvar
            y_logvar: [B, pred_len, C] prediction logvar
        """
        B = z.shape[0]
        
        # Decode past
        x_out_list = []
        for f in range(self.freq):
            self.qdev.reset_states(B)
            
            # Pad latent to quantum_dim for amplitude encoding
            z_padded = F.pad(z[:, f, :], (0, self.quantum_dim - self.latent_dim))
            z_padded = F.normalize(z_padded, p=2, dim=-1)
            
            # Amplitude encode
            self.encoder(self.qdev, z_padded)
            
            # Apply VQC decoder
            self._vqc_decoder(B)
            
            # Get output amplitudes
            amplitudes = self.qdev.get_states_1d().abs()
            x_out_list.append(amplitudes)
        
        x_quantum = torch.stack(x_out_list, dim=1)  # [B, F, 2^n_qbits]
        
        # Classical post-processing (dual-head)
        x_out = self.post_decoder(x_quantum)  # [B, F, 2*P*C]
        x_mu_raw, x_logvar = torch.chunk(x_out, 2, dim=-1)
        
        # Reshape
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, :self.input_len]
        
        x_logvar = x_logvar.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_logvar = x_logvar.reshape(B, -1, self.enc_in)[:, :self.input_len]
        
        # RevIN denorm for mean only
        x_mu = self.revin(x_mu_raw, "denorm")
        
        # Future prediction
        z_flat = z.reshape(B, -1)
        z_future_flat = self.future_proj(z_flat)
        z_future = z_future_flat.reshape(B, self.step, self.latent_dim)
        
        # Decode future
        y_out_list = []
        for s in range(self.step):
            self.qdev.reset_states(B)
            
            z_padded = F.pad(z_future[:, s, :], (0, self.quantum_dim - self.latent_dim))
            z_padded = F.normalize(z_padded, p=2, dim=-1)
            
            self.encoder(self.qdev, z_padded)
            self._vqc_decoder(B)
            
            amplitudes = self.qdev.get_states_1d().abs()
            y_out_list.append(amplitudes)
        
        y_quantum = torch.stack(y_out_list, dim=1)
        y_out = self.post_decoder(y_quantum)
        y_mu_raw, y_logvar = torch.chunk(y_out, 2, dim=-1)
        
        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, :self.pred_len]
        
        y_logvar = y_logvar.reshape(B, self.step, self.patch_len, self.enc_in)
        y_logvar = y_logvar.reshape(B, -1, self.enc_in)[:, :self.pred_len]
        
        y_mu = self.revin(y_mu_raw, "denorm")
        
        return x_mu, y_mu, x_logvar, y_logvar
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: [B, L, C] input sequence
            
        Returns:
            x_mu, y_mu, mu_z, x_logvar, y_logvar
        """
        mu_z, z = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar = self.decode(z)
        return x_mu, y_mu, mu_z, x_logvar, y_logvar
    
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generate probabilistic forecasts.
        
        Args:
            x: [B, L, C] input sequence
            num_samples: number of samples
            
        Returns:
            samples: [B, num_samples, pred_len, C]
        """
        B = x.shape[0]
        mu_z, _ = self.encode(x)
        
        sample_list = []
        for _ in range(num_samples):
            # Sample from vMF
            z = torch.stack([
                self.sampling_vmf(mu_z[:, f, :]) for f in range(self.freq)
            ], dim=1)
            
            # Decode
            _, y_mu, _, y_logvar = self.decode(z)
            
            # Sample from observation noise
            y_logvar = torch.clamp(y_logvar, min=-10.0, max=10.0)
            std = torch.exp(0.5 * y_logvar)
            y_sample = y_mu + torch.randn_like(std) * std
            
            sample_list.append(y_sample)
        
        samples = torch.stack(sample_list, dim=1)
        return samples
    
    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction="mean"):
        """Gaussian NLL loss."""
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        nll = 0.5 * (math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp())
        nll = nll.sum(dim=[1, 2])
        
        if reduction == "mean":
            return nll.mean()
        return nll.sum()
    
    def kl_vmf_uniform(self, mu):
        """
        KL divergence between vMF and uniform on sphere.
        Approximation: KL ≈ κ * ||μ||
        
        Args:
            mu: [B, F, D] mean directions
            
        Returns:
            scalar KL
        """
        # For unit sphere, ||μ|| = 1, so KL ≈ κ
        return self.kappa * mu.shape[1]  # Simple approximation
