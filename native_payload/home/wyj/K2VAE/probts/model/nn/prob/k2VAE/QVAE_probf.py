import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Try importing standard layers from the project if available, otherwise use standard PyTorch
try:
    from probts.model.nn.prob.k2VAE.koopman import MLP
    from probts.model.nn.prob.k2VAE.RevIN import RevIN
except ImportError:
    # Fallback to local definitions if imports fail (for standalone testing)
    class MLP(nn.Module):
        def __init__(self, f_in, f_out, activation="relu", hidden_dim=128, hidden_layers=2):
            super().__init__()
            layers = []
            dims = [f_in] + [hidden_dim] * hidden_layers + [f_out]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class RevIN(nn.Module):
        def __init__(self, num_features, eps=1e-5, affine=True):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            if self.affine:
                self._init_params()
        def _init_params(self):
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        def forward(self, x, mode):
            if mode == 'norm':
                self._mean = torch.mean(x, dim=1, keepdim=True).detach()
                self._stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
                x = (x - self._mean) / self._stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
                return x
            elif mode == 'denorm':
                if self.affine:
                    x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
                x = x * self._stdev + self._mean
                return x

class QVAE_Probf_NN(tq.QuantumModule):
    """
    QVAE with Channel Independence encoding.
    
    Instead of flattening [B, L, C] -> [B, L*C], we:
    1. Transpose: [B, L, C] -> [B, C, L] (channel-first)
    2. Process each channel independently: [B*C, L] -> quantum -> [B*C, latent_dim]
    3. Aggregate: [B, C * latent_dim] for fusion
    4. Decode back to per-channel outputs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Time Series Configs ---
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.n_vars
        
        # --- Quantum Configs ---
        self.n_qbits = getattr(config, 'n_qbits', 8)
        self.n_blocks = getattr(config, 'n_blocks', 1)
        self.bottleneck_qbits = getattr(config, 'bottleneck_qbits', 4)

        # Latent dim per channel (after geometric compression)
        self.latent_dim_per_channel = pow(2, self.n_qbits-self.bottleneck_qbits)
        # Total latent dim = per_channel * num_channels
        self.total_latent_dim = self.latent_dim_per_channel * self.enc_in
        
        # --- Per-Channel Input/Output Projection ---
        # Each channel: [L] -> [2^n_qbits]
        self.MLP_in = nn.Linear(self.seq_len, 2 ** self.n_qbits)
        # Dual-head output: mean and logvar for NLL
        self.MLP_out_mu = nn.Linear(2 ** self.n_qbits, self.seq_len)  # Reconstruction mean
        self.MLP_out_logvar = nn.Linear(2 ** self.n_qbits, self.seq_len)  # Reconstruction logvar
        self.MLP_pred_mu = nn.Linear(2 ** self.n_qbits, self.pred_len)  # Prediction mean
        self.MLP_pred_logvar = nn.Linear(2 ** self.n_qbits, self.pred_len)  # Prediction logvar
        
        # --- Gaussian VAE: mu and logvar projection (per-channel) ---
        self.fc_mu = nn.Linear(self.latent_dim_per_channel, self.latent_dim_per_channel)
        self.fc_logvar = nn.Linear(self.latent_dim_per_channel, self.latent_dim_per_channel)

        # --- RevIN ---
        self.revin = RevIN(self.enc_in)

        # --- Quantum Encoder ---
        self.encoder_q = tq.AmplitudeEncoder()
        
        self.ry_layer = tq.QuantumModuleList()
        self.rz1_layer = tq.QuantumModuleList()
        self.rz2_layer = tq.QuantumModuleList()
        self.crx_layer = tq.QuantumModuleList()
        
        for k in range(self.n_blocks+1):
            for i in range(self.n_qbits):
                self.rz1_layer.append(tq.RZ(trainable=True, has_params=True))
                self.ry_layer.append(tq.RY(trainable=True, has_params=True))
                self.rz2_layer.append(tq.RZ(trainable=True, has_params=True))
                if k != self.n_blocks:
                    self.crx_layer.append(tq.CRX(trainable=True, has_params=True))

        # --- Quantum Decoder ---
        self.ry_layer_d = tq.QuantumModuleList()
        self.rz1_layer_d = tq.QuantumModuleList()
        self.rz2_layer_d = tq.QuantumModuleList()
        self.crx_layer_d = tq.QuantumModuleList()
        
        for k in range(self.n_blocks+1):
            for i in range(self.n_qbits):
                self.rz1_layer_d.append(tq.RZ(trainable=True, has_params=True))
                self.ry_layer_d.append(tq.RY(trainable=True, has_params=True))
                self.rz2_layer_d.append(tq.RZ(trainable=True, has_params=True))
                if k != self.n_blocks:
                    self.crx_layer_d.append(tq.CRX(trainable=True, has_params=True))

        # --- Latent Projection (Classical) ---
        # Operates on aggregated latent: [B, C * latent_dim_per_channel]
        self.latent_proj = MLP(
            f_in=self.total_latent_dim,
            f_out=self.total_latent_dim,
            hidden_dim=self.total_latent_dim * 2,
            hidden_layers=2
        )

        self.qdev = tq.QuantumDevice(n_wires=self.n_qbits)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def reparameterize(self, mu, logvar):
        """
        Gaussian reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def run_pqc(self, qdev, inverse=False):
        if not inverse:
            # Encoder PQC
            for k in range(self.n_blocks+1):
                for i in range(self.n_qbits):
                    self.rz1_layer[k*self.n_qbits+i](qdev, wires=i)
                    self.ry_layer[k*self.n_qbits+i](qdev, wires=i)
                    self.rz2_layer[k*self.n_qbits+i](qdev, wires=i)
                if k != self.n_blocks:
                    for i in range(self.n_qbits):
                        self.crx_layer[k*self.n_qbits+i](qdev, wires=[i, (i+1)%self.n_qbits])
        else:
            # Decoder PQC
            for k in range(self.n_blocks, -1, -1):
                if k == self.n_blocks:
                    for i in range(self.n_qbits):
                        self.rz2_layer_d[k*self.n_qbits+i](qdev, wires=i)
                        self.ry_layer_d[k*self.n_qbits+i](qdev, wires=i)
                        self.rz1_layer_d[k*self.n_qbits+i](qdev, wires=i)
                else:
                    for i in range(self.n_qbits-1, -1, -1):
                        self.crx_layer_d[k*self.n_qbits+i](qdev, wires=[i, (i+1)%self.n_qbits])
                    for i in range(self.n_qbits):
                        self.rz2_layer_d[k*self.n_qbits+i](qdev, wires=i)
                        self.ry_layer_d[k*self.n_qbits+i](qdev, wires=i)
                        self.rz1_layer_d[k*self.n_qbits+i](qdev, wires=i)

    def encode(self, x):
        """
        Channel Independence Encoding
        x: [B, L, C] (pre-normalized)
        Returns: mu [B, C*latent_dim], logvar [B, C*latent_dim], z [B, C*latent_dim]
        """
        bsz, seq_len, n_vars = x.shape
        
        # Step 1: Transpose to channel-first: [B, L, C] -> [B, C, L]
        x_t = x.permute(0, 2, 1)  # [B, C, L]
        
        # Step 2: Reshape for per-channel processing: [B, C, L] -> [B*C, L]
        x_flat = x_t.reshape(bsz * n_vars, seq_len)  # [B*C, L]
        
        # Step 3: Project each channel to quantum dim: [B*C, L] -> [B*C, 2^N]
        x_embed = self.MLP_in(x_flat)  # [B*C, 2^N]
        
        # Step 4: Quantum encoding (process all channels as batch)
        self.qdev.reset_states(bsz * n_vars)
        self.encoder_q(self.qdev, x_embed)
        self.run_pqc(self.qdev, inverse=False)
        
        # Step 5: Get latent per channel
        output = self.qdev.get_states_1d().abs()  # [B*C, 2^N]
        output = output ** 2
        # Geometric compression: [B*C, 2^(N-B), 2^B] -> sum(dim=-1) -> sqrt
        output = output.reshape(
            -1, 
            pow(2, self.n_qbits - self.bottleneck_qbits), 
            pow(2, self.bottleneck_qbits)
        ).sum(axis=-1).sqrt()  # [B*C, latent_dim_per_channel]
        
        # Step 6: Gaussian VAE projection (per-channel)
        mu = self.fc_mu(output)  # [B*C, latent_dim_per_channel]
        logvar = self.fc_logvar(output)  # [B*C, latent_dim_per_channel]
        z = self.reparameterize(mu, logvar)  # [B*C, latent_dim_per_channel]
        
        # Step 7: Reshape back and aggregate: [B*C, latent] -> [B, C*latent]
        mu = mu.reshape(bsz, n_vars * self.latent_dim_per_channel)
        logvar = logvar.reshape(bsz, n_vars * self.latent_dim_per_channel)
        z = z.reshape(bsz, n_vars * self.latent_dim_per_channel)
        
        return mu, logvar, z

    def decode(self, z, output_len):
        """
        Channel Independence Decoding with Dual-Head Output
        z: [B, C*latent_dim]
        output_len: seq_len (reconstruction) or pred_len (prediction)
        Returns: mu [B, output_len, C], logvar [B, output_len, C]
        """
        bsz = z.shape[0]
        n_vars = self.enc_in
        
        # Step 1: Reshape to per-channel: [B, C*latent] -> [B*C, latent]
        z_per_channel = z.reshape(bsz * n_vars, self.latent_dim_per_channel)
        
        # Step 2: Pad to quantum dim if needed
        target_dim = 2 ** self.n_qbits
        if z_per_channel.shape[-1] < target_dim:
            padding = torch.zeros(bsz * n_vars, target_dim - z_per_channel.shape[-1], device=z.device)
            z_padded = torch.cat([z_per_channel, padding], dim=-1)
        else:
            z_padded = z_per_channel
        
        # Step 3: Quantum decoding
        self.qdev.reset_states(bsz * n_vars)
        self.encoder_q(self.qdev, z_padded)
        self.run_pqc(self.qdev, inverse=True)
        
        # Step 4: Get output
        output = self.qdev.get_states_1d().abs()  # [B*C, 2^N]
        
        # Step 5: Project to output length (dual-head: mu and logvar)
        if output_len == self.seq_len:
            mu_proj = self.MLP_out_mu(output)  # [B*C, seq_len]
            logvar_proj = self.MLP_out_logvar(output)  # [B*C, seq_len]
        else:
            mu_proj = self.MLP_pred_mu(output)  # [B*C, pred_len]
            logvar_proj = self.MLP_pred_logvar(output)  # [B*C, pred_len]
        
        # Step 6: Reshape back: [B*C, L] -> [B, C, L] -> [B, L, C]
        mu_proj = mu_proj.reshape(bsz, n_vars, output_len).permute(0, 2, 1)  # [B, L, C]
        logvar_proj = logvar_proj.reshape(bsz, n_vars, output_len).permute(0, 2, 1)  # [B, L, C]
        
        return mu_proj, logvar_proj

    def forward(self, x):
        """
        x: [B, L, C]
        """
        # RevIN Norm
        x_norm = self.revin(x, 'norm')  # [B, L, C]
        
        # Encode (per-channel)
        mu_past, logvar_past, z_past = self.encode(x_norm)  # [B, C*latent]
        
        # Predict Future Latent (on aggregated latent)
        z_future = self.latent_proj(z_past)  # [B, C*latent]
        
        # Decode Past (Reconstruction) - returns mu and logvar
        x_rec_mu, x_rec_logvar = self.decode(z_past, self.seq_len)  # [B, seq_len, C] each
        
        # Decode Future (Prediction) - returns mu and logvar
        y_pred_mu, y_pred_logvar = self.decode(z_future, self.pred_len)  # [B, pred_len, C] each
        
        # RevIN Denorm (only for mu, logvar stays in normalized space for NLL)
        x_rec_mu = self.revin(x_rec_mu, 'denorm')
        y_pred_mu = self.revin(y_pred_mu, 'denorm')
        
        return x_rec_mu, x_rec_logvar, y_pred_mu, y_pred_logvar, mu_past, logvar_past, z_past
