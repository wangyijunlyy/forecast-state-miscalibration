"""
Quantum-Regularized VAE (Quantum Bottleneck VAE) for Time Series Reconstruction

This model uses a Quantum Circuit as a strictly bounded information bottleneck.
Unlike standard VAE with Gaussian prior and KL-Divergence, the latent vector is
the expectation values of quantum measurements (PauliZ), which are naturally 
bounded in [-1, 1], acting as a regularization mechanism.

Architecture follows gaussian_vae_mse.py pattern:
1. Patching: Input [B, L, C] → Patches [B, F, P*C]
2. Classical Encoder + Quantum Bottleneck → z_rec [B, F, n_qubits]
3. Future Projection MLP: z_rec [B, F*n_qubits] → z_pred [B, S*n_qubits]
4. Decoder (shared or separate): z → reconstruction/prediction

Tech Stack:
- PyTorch (torch.nn)
- PennyLane (qml) with default.qubit device

Author: Quantum ML Implementation
Date: 2026-01-21
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Try importing from project
try:
    from probts.model.nn.prob.k2VAE.koopman import MLP
    from probts.model.nn.prob.k2VAE.RevIN import RevIN
except ImportError:
    # Fallback for standalone testing
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
                self._stdev = torch.sqrt(
                    torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
                ).detach()
                x = (x - self._mean) / self._stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
                return x
            elif mode == 'denorm':
                if self.affine:
                    x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
                x = x * self._stdev + self._mean
                return x


class QuantumRegularizedVAE(nn.Module):
    """
    Quantum-Regularized VAE for Time Series with Patching.
    
    Key Architectural Concept:
    - Uses Quantum Circuit as strictly bounded information bottleneck
    - Latent z_rec, z_pred ∈ [-1, 1] per patch (PauliZ expectations)
    - No KL-Divergence; uses regularization loss: α * mean(z²)
    - Prediction latent is obtained by projecting reconstruction latent via MLP
    - Supports shared or separate decoders for reconstruction and prediction
    
    Args:
        config: Configuration object with attributes:
            - seq_len: Input sequence length
            - pred_len: Prediction length
            - patch_len: Patch length for patching
            - n_vars: Number of input features/channels
            - n_qubits: Number of qubits per patch (default: 4)
            - n_layers: Number of variational layers (default: 2)
            - hidden_dim: MLP hidden dimension (default: 64)
            - hidden_layers: Number of hidden layers in MLP (default: 2)
            - share_decoder: Whether to share decoder (default: True)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === Basic Parameters ===
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.enc_in = config.n_vars  # input channels C
        
        # === Quantum Configs ===
        self.n_qubits = getattr(config, 'n_qubits', 4)
        self.n_layers = getattr(config, 'n_layers', 2)
        
        # === Network Configs ===
        self.hidden_dim = getattr(config, 'hidden_dim', 64)
        self.hidden_layers = getattr(config, 'hidden_layers', 2)
        self.share_decoder = getattr(config, 'share_decoder', True)
        
        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)  # number of input patches F
        self.step = math.ceil(self.pred_len / self.patch_len)   # number of output patches S
        self.padding_len = self.patch_len * self.freq - self.input_len
        
        # === RevIN ===
        self.revin = RevIN(self.enc_in)
        
        # =====================================================
        # CLASSICAL ENCODER: Patch → n_qubits * 2 (RX/RY angles)
        # =====================================================
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.n_qubits * 2,  # RX and RY angles
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        
        # =====================================================
        # QUANTUM BOTTLENECK: PennyLane QNode
        # Returns z ∈ [-1, 1]^n_qubits per patch
        # =====================================================
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_circuit(params):
            """Quantum circuit with RX/RY + CNOT entanglement."""
            n_q = self.n_qubits
            rx_angles = params[:n_q]
            ry_angles = params[n_q:]
            
            for layer in range(self.n_layers):
                for i in range(n_q):
                    qml.RX(rx_angles[i], wires=i)
                    qml.RY(ry_angles[i], wires=i)
                # CNOT entanglement
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
            
            # Final rotation
            for i in range(n_q):
                qml.RX(rx_angles[i], wires=i)
                qml.RY(ry_angles[i], wires=i)
            
            # PauliZ expectations in [-1, 1]
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
        
        self.quantum_circuit = quantum_circuit
        
        # =====================================================
        # FUTURE PROJECTION MLP
        # z_rec [B, F*n_qubits] → z_pred [B, S*n_qubits]
        # =====================================================
        self.future_proj = nn.Linear(
            self.freq * self.n_qubits, 
            self.step * self.n_qubits
        )
        
        # =====================================================
        # DECODER(S): From latent to patches
        # =====================================================
        if self.share_decoder:
            # Shared decoder for both reconstruction and prediction
            self.decoder = MLP(
                f_in=self.n_qubits,
                f_out=self.patch_len * self.enc_in,
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
            )
        else:
            # Separate decoders
            self.decoder_rec = MLP(
                f_in=self.n_qubits,
                f_out=self.patch_len * self.enc_in,
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
            )
            self.decoder_pred = MLP(
                f_in=self.n_qubits,
                f_out=self.patch_len * self.enc_in,
                activation="relu",
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
            )

    def quantum_layer(self, params):
        """
        Apply quantum circuit to get latent z ∈ [-1, 1].
        
        Args:
            params: [B, F, n_qubits*2] rotation parameters per patch
        
        Returns:
            z: [B, F, n_qubits] latent with values in [-1, 1]
        """
        B, F, D = params.shape
        z_list = []
        
        for b in range(B):
            z_patches = []
            for f in range(F):
                expectations = self.quantum_circuit(params[b, f])
                z_patches.append(torch.stack(expectations))
            z_list.append(torch.stack(z_patches))
        
        z = torch.stack(z_list)  # [B, F, n_qubits]
        return z.float()  # Cast to float32 for PyTorch layers

    def encode(self, x: torch.Tensor):
        """
        Encode input to quantum latent representation.
        
        Args:
            x: [B, L, C] input time series
        
        Returns:
            z: [B, F, n_qubits] latent in [-1, 1]
        """
        B, L, C = x.shape
        
        # RevIN Normalization
        x_norm = self.revin(x, "norm")
        
        # === Patch Padding ===
        padded = torch.cat([x_norm[:, L - self.padding_len:, :], x_norm], dim=1)
        patches = padded.chunk(self.freq, dim=1)  # freq * [B, P, C]
        patches = torch.stack(patches, dim=1).reshape(B, self.freq, -1)  # [B, F, P*C]
        
        # === Classical Encoder → Rotation Parameters ===
        params = self.encoder(patches)  # [B, F, n_qubits*2]
        
        # === Quantum Bottleneck ===
        z = self.quantum_layer(params)  # [B, F, n_qubits]
        
        return z

    def decode(self, z_rec: torch.Tensor, z_pred: torch.Tensor):
        """
        Decode latent to reconstruction and prediction.
        
        Args:
            z_rec: [B, F, n_qubits] reconstruction latent
            z_pred: [B, S, n_qubits] prediction latent
        
        Returns:
            x_rec: [B, L, C] reconstructed input
            y_rec: [B, pred_len, C] predicted future
        """
        B = z_rec.shape[0]
        
        # ----- 1. Reconstruct Past -----
        if self.share_decoder:
            x_rec_out = self.decoder(z_rec)  # [B, F, P*C]
        else:
            x_rec_out = self.decoder_rec(z_rec)  # [B, F, P*C]
        
        x_rec_out = x_rec_out.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_out = x_rec_out.reshape(B, -1, self.enc_in)[:, :self.input_len]  # [B, L, C]
        x_rec = self.revin(x_rec_out, "denorm")
        
        # ----- 2. Predict Future -----
        if self.share_decoder:
            y_rec_out = self.decoder(z_pred)  # [B, S, P*C]
        else:
            y_rec_out = self.decoder_pred(z_pred)  # [B, S, P*C]
        
        y_rec_out = y_rec_out.reshape(B, self.step, self.patch_len, self.enc_in)
        y_rec_out = y_rec_out.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]
        y_rec = self.revin(y_rec_out, "denorm")
        
        return x_rec, y_rec

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: [B, L, C] input time series
        
        Returns:
            x_rec: [B, L, C] reconstructed input
            y_rec: [B, pred_len, C] predicted future
            z_rec: [B, F, n_qubits] reconstruction latent (in [-1, 1])
        """
        B = x.shape[0]
        
        # Encode to reconstruction latent
        z_rec = self.encode(x)  # [B, F, n_qubits]
        
        # Project to prediction latent via MLP
        z_rec_flat = z_rec.reshape(B, -1)  # [B, F*n_qubits]
        z_pred_flat = self.future_proj(z_rec_flat)  # [B, S*n_qubits]
        z_pred = z_pred_flat.reshape(B, self.step, self.n_qubits)  # [B, S, n_qubits]
        
        # Decode
        x_rec, y_rec = self.decode(z_rec, z_pred)
        
        return x_rec, y_rec, z_rec

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generate multiple predictions by adding noise to latent.
        
        Args:
            x: [B, L, C]
            num_samples: Number of samples
        
        Returns:
            samples: [B, num_samples, pred_len, C]
        """
        B = x.shape[0]
        
        # Encode
        z_rec = self.encode(x)  # [B, F, n_qubits]
        z_rec_flat = z_rec.reshape(B, -1)  # [B, F*n_qubits]
        
        sample_list = []
        for _ in range(num_samples):
            # Add noise and clamp to [-1, 1]
            noise = torch.randn_like(z_rec_flat) * 0.1
            z_rec_noisy_flat = torch.clamp(z_rec_flat + noise, -1.0, 1.0)
            
            # Project to prediction latent
            z_pred_flat = self.future_proj(z_rec_noisy_flat)
            z_pred = z_pred_flat.reshape(B, self.step, self.n_qubits)
            z_rec_noisy = z_rec_noisy_flat.reshape(B, self.freq, self.n_qubits)
            
            # Decode
            _, y_sample = self.decode(z_rec_noisy, z_pred)
            sample_list.append(y_sample)
        
        samples = torch.stack(sample_list, dim=1)  # [B, num_samples, pred_len, C]
        return samples

    def regularization_loss(self, z: torch.Tensor, alpha: float = 0.01):
        """
        Quantum regularization loss to prevent z from saturating at ±1.
        
        Args:
            z: Latent tensor in [-1, 1]
            alpha: Regularization weight
        
        Returns:
            loss: α * mean(z²)
        """
        return alpha * torch.mean(z ** 2)


# =====================================================
# STANDALONE TESTING
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing QuantumRegularizedVAE (Patching + Shared/Separate Decoder)")
    print("=" * 70)
    
    class DummyConfig:
        seq_len = 96
        pred_len = 24
        patch_len = 8
        n_vars = 3
        n_qubits = 4
        n_layers = 2
        hidden_dim = 32
        hidden_layers = 2
        share_decoder = True  # Test shared decoder
    
    config = DummyConfig()
    model = QuantumRegularizedVAE(config)
    
    print(f"\nModel Configuration:")
    print(f"  seq_len: {config.seq_len}")
    print(f"  pred_len: {config.pred_len}")
    print(f"  patch_len: {config.patch_len}")
    print(f"  n_qubits: {config.n_qubits}")
    print(f"  share_decoder: {config.share_decoder}")
    print(f"  freq (input patches): {model.freq}")
    print(f"  step (output patches): {model.step}")
    
    # Generate random data
    B = 2  # Small batch for testing
    x = torch.randn(B, config.seq_len, config.n_vars)
    y = torch.randn(B, config.pred_len, config.n_vars)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass
    print("\n[Test 1] Forward pass...")
    x_rec, y_rec, z_rec = model(x)
    print(f"  x_rec shape: {x_rec.shape}")
    print(f"  y_rec shape: {y_rec.shape}")
    print(f"  z_rec shape: {z_rec.shape}")
    print(f"  z_rec range: [{z_rec.min().item():.4f}, {z_rec.max().item():.4f}]")
    
    assert x_rec.shape == (B, config.seq_len, config.n_vars)
    assert y_rec.shape == (B, config.pred_len, config.n_vars)
    assert z_rec.min() >= -1.0 and z_rec.max() <= 1.0
    print("  ✓ All shapes correct, z_rec bounded in [-1, 1]")
    
    # Test loss computation
    print("\n[Test 2] Loss computation...")
    mse_rec = F.mse_loss(x_rec, x)
    mse_pred = F.mse_loss(y_rec, y)
    reg_loss = model.regularization_loss(z_rec, alpha=0.01)
    total_loss = mse_rec + mse_pred + reg_loss
    
    print(f"  MSE Reconstruction: {mse_rec.item():.4f}")
    print(f"  MSE Prediction: {mse_pred.item():.4f}")
    print(f"  Regularization Loss: {reg_loss.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    
    # Test backward pass
    print("\n[Test 3] Backward pass...")
    total_loss.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient sum: {grad_sum:.4f}")
    assert grad_sum > 0
    print("  ✓ Gradients computed correctly")
    
    # Test separate decoder
    print("\n[Test 4] Separate decoder mode...")
    config.share_decoder = False
    model_sep = QuantumRegularizedVAE(config)
    x_rec2, y_rec2, z_rec2 = model_sep(x)
    print(f"  x_rec shape: {x_rec2.shape}")
    print(f"  y_rec shape: {y_rec2.shape}")
    assert x_rec2.shape == (B, config.seq_len, config.n_vars)
    assert y_rec2.shape == (B, config.pred_len, config.n_vars)
    print("  ✓ Separate decoder works correctly")
    
    # Test sampling
    print("\n[Test 5] Sampling...")
    samples = model.sample(x, num_samples=5)
    print(f"  Samples shape: {samples.shape}")
    assert samples.shape == (B, 5, config.pred_len, config.n_vars)
    variance = torch.var(samples, dim=1).mean().item()
    print(f"  Sample variance: {variance:.6f}")
    print("  ✓ Sampling works correctly")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
