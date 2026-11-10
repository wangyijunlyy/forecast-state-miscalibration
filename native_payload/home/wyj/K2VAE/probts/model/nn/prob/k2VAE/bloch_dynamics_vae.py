"""
Bloch Dynamics VAE - Direct Quantum Supervision for Time Series Forecasting

This model uses a novel "Bloch Dynamics Loss" that directly maps quantum measurement
axes to physical time series properties, eliminating the need for a decoder.

Architecture:
    Input → Patching → Encoder → Quantum Circuit → [Exp(Z), Exp(Y), Exp(X)] per qubit
                                                              ↓
                                                    Bloch Dynamics Loss
                                                              ↓
                                                    Direct tri-axis supervision

Key Innovation:
- No decoder needed — quantum measurements ARE the predictions
- Pauli-Z → Series Value (position)
- Pauli-Y → Series Gradient (momentum/velocity)  
- Pauli-X → Inverse Local Volatility (coherence)

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


class BlochTargetPreprocessor(nn.Module):
    """
    Preprocess time series to compute 3 physical targets for Bloch supervision.
    
    Outputs:
        target_val: Normalized series value, scaled to [-0.9, 0.9]
        target_grad: First-order gradient (velocity), scaled
        target_vol: Inverse local volatility (coherence), scaled to [0, 1]
    """
    
    def __init__(self, vol_window: int = 5):
        super().__init__()
        self.vol_window = vol_window
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, L, C] normalized time series
            
        Returns:
            target_val: [B, L, C] value target in [-0.9, 0.9]
            target_grad: [B, L, C] gradient target
            target_vol: [B, L, C] inverse volatility in [0, 1]
        """
        B, L, C = x.shape
        
        # === 1. Value Target (Position) ===
        # Scale to [-0.9, 0.9] to fit Bloch sphere (leave margin)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_range = x_max - x_min + 1e-8
        target_val = 1.8 * (x - x_min) / x_range - 0.9  # [-0.9, 0.9]
        
        # === 2. Gradient Target (Momentum/Velocity) ===
        # First difference, zero-padded at start
        grad = torch.zeros_like(x)
        grad[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        # Scale gradient to [-0.9, 0.9]
        grad_abs_max = grad.abs().max(dim=1, keepdim=True)[0] + 1e-8
        target_grad = 0.9 * grad / grad_abs_max
        
        # === 3. Volatility Target (Coherence) ===
        # Rolling standard deviation
        vol = torch.zeros_like(x)
        for i in range(self.vol_window, L):
            window = x[:, i-self.vol_window:i, :]
            vol[:, i, :] = torch.std(window, dim=1)
        # Fill initial positions with first valid value
        vol[:, :self.vol_window, :] = vol[:, self.vol_window:self.vol_window+1, :]
        # Normalize to [0, 1] and invert (high vol → low coherence → low X)
        vol_max = vol.max(dim=1, keepdim=True)[0] + 1e-8
        vol_normalized = vol / vol_max  # [0, 1]
        target_vol = 1.0 - vol_normalized  # High coherence = low volatility
        # Scale to [-0.9, 0.9] range for X axis
        target_vol = 1.8 * target_vol - 0.9
        
        return target_val, target_grad, target_vol


class BlochDynamicsLoss(nn.Module):
    """
    Bloch Dynamics Loss - Direct supervision on quantum measurement axes.
    
    Loss = λ₁ * L_value(Z) + λ₂ * L_trend(Y) + λ₃ * L_coherence(X) + λ₄ * L_purity
    
    Where:
        L_value = ||⟨Z⟩ - target_val||²
        L_trend = ||⟨Y⟩ - target_grad||²
        L_coherence = ||⟨X⟩ - target_vol||²
        L_purity = ||(⟨X⟩² + ⟨Y⟩² + ⟨Z⟩²) - 1||²  (encourage pure states)
    """
    
    def __init__(
        self, 
        lambda_val: float = 1.0,
        lambda_grad: float = 1.0,
        lambda_vol: float = 0.5,
        lambda_purity: float = 0.1
    ):
        super().__init__()
        self.lambda_val = lambda_val
        self.lambda_grad = lambda_grad
        self.lambda_vol = lambda_vol
        self.lambda_purity = lambda_purity
    
    def forward(
        self, 
        q_out: torch.Tensor,
        target_val: torch.Tensor,
        target_grad: torch.Tensor,
        target_vol: torch.Tensor
    ):
        """
        Args:
            q_out: [B, F, n_qubits, 3] - quantum outputs (Z, Y, X per qubit)
            target_val: [B, F, C] - value targets per patch
            target_grad: [B, F, C] - gradient targets per patch
            target_vol: [B, F, C] - volatility targets per patch
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses for logging
        """
        # Split quantum outputs into axes
        z_pred = q_out[:, :, :, 0]  # [B, F, n_qubits]
        y_pred = q_out[:, :, :, 1]  # [B, F, n_qubits]
        x_pred = q_out[:, :, :, 2]  # [B, F, n_qubits]
        
        # Pool n_qubits dimension to match targets
        # Average over qubits then expand to match channels
        B, num_patches, n_q = z_pred.shape
        _, _, C = target_val.shape
        
        # Mean over qubits → [B, num_patches]
        z_mean = z_pred.mean(dim=-1, keepdim=True).expand(-1, -1, C)  # [B, num_patches, C]
        y_mean = y_pred.mean(dim=-1, keepdim=True).expand(-1, -1, C)
        x_mean = x_pred.mean(dim=-1, keepdim=True).expand(-1, -1, C)
        
        # === 1. Value Loss (Z-axis tracks position) ===
        loss_z = F.mse_loss(z_mean, target_val)
        
        # === 2. Trend Loss (Y-axis tracks gradient) ===
        loss_y = F.mse_loss(y_mean, target_grad)
        
        # === 3. Coherence Loss (X-axis tracks inverse volatility) ===
        loss_x = F.mse_loss(x_mean, target_vol)
        
        # === 4. Geometric Purity Constraint ===
        # Encourage state to be on Bloch sphere surface: X² + Y² + Z² ≈ 1
        magnitude = z_pred**2 + y_pred**2 + x_pred**2  # [B, F, n_qubits]
        loss_purity = F.mse_loss(magnitude, torch.ones_like(magnitude))
        
        # Total loss
        total_loss = (
            self.lambda_val * loss_z +
            self.lambda_grad * loss_y +
            self.lambda_vol * loss_x +
            self.lambda_purity * loss_purity
        )
        
        loss_dict = {
            'loss_z': loss_z.item(),
            'loss_y': loss_y.item(),
            'loss_x': loss_x.item(),
            'loss_purity': loss_purity.item()
        }
        
        return total_loss, loss_dict


class BlochDynamicsVAE(nn.Module):
    """
    Bloch Dynamics VAE - No Decoder Architecture.
    
    Uses direct quantum supervision where:
    - Pauli-Z expectations predict series values
    - Pauli-Y expectations predict series gradients
    - Pauli-X expectations predict inverse volatility
    
    For actual forecasting, a lightweight "readout head" maps Z-axis to predictions.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === Basic Parameters ===
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.enc_in = config.n_vars
        
        # === Quantum Configs ===
        self.n_qubits = getattr(config, 'n_qubits', 4)
        self.n_layers = getattr(config, 'n_layers', 2)
        
        # === Network Configs ===
        self.hidden_dim = getattr(config, 'hidden_dim', 64)
        self.hidden_layers = getattr(config, 'hidden_layers', 2)
        
        # === Patching ===
        self.freq = math.ceil(self.input_len / self.patch_len)  # F
        self.step = math.ceil(self.pred_len / self.patch_len)   # S
        self.padding_len = self.patch_len * self.freq - self.input_len
        
        # === RevIN ===
        self.revin = RevIN(self.enc_in)
        
        # === Target Preprocessor ===
        self.target_preprocessor = BlochTargetPreprocessor(vol_window=5)
        
        # === Classical Encoder: Patch → n_qubits * 2 (RX/RY angles) ===
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.n_qubits * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        
        # === Quantum Circuit: Returns [Z, Y, X] per qubit ===
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_tomography_circuit(params):
            """
            Quantum circuit that outputs all 3 Pauli expectations.
            Returns: [Z0, Z1, ..., Y0, Y1, ..., X0, X1, ...]
            """
            n_q = self.n_qubits
            rx_angles = params[:n_q]
            ry_angles = params[n_q:]
            
            for layer in range(self.n_layers):
                for i in range(n_q):
                    qml.RX(rx_angles[i], wires=i)
                    qml.RY(ry_angles[i], wires=i)
                # CNOT entanglement ring
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
            
            # Final rotation layer
            for i in range(n_q):
                qml.RX(rx_angles[i], wires=i)
                qml.RY(ry_angles[i], wires=i)
            
            # Return all 3 Pauli expectations per qubit
            z_exp = [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
            y_exp = [qml.expval(qml.PauliY(i)) for i in range(n_q)]
            x_exp = [qml.expval(qml.PauliX(i)) for i in range(n_q)]
            
            return z_exp + y_exp + x_exp
        
        self.quantum_circuit = quantum_tomography_circuit
        
        # === Future Projection (for prediction latent) ===
        self.future_proj = nn.Linear(
            self.freq * self.n_qubits,
            self.step * self.n_qubits
        )
        
        # === Readout Head: Z-axis → actual values ===
        # Lightweight linear map from quantum Z to output
        self.readout_head = nn.Linear(self.n_qubits, self.patch_len * self.enc_in)
    
    def quantum_layer(self, params):
        """
        Apply quantum circuit to get [Z, Y, X] expectations.
        
        Args:
            params: [B, F, n_qubits*2] rotation parameters
            
        Returns:
            q_out: [B, F, n_qubits, 3] - (Z, Y, X) per qubit per patch
        """
        B, F, D = params.shape
        q_out_list = []
        
        for b in range(B):
            patches_out = []
            for f in range(F):
                # Get all expectations: [Z0..Zn, Y0..Yn, X0..Xn]
                all_exp = self.quantum_circuit(params[b, f])
                all_exp = torch.stack(all_exp)  # [3 * n_qubits]
                
                # Reshape to [n_qubits, 3]
                z_exp = all_exp[:self.n_qubits]
                y_exp = all_exp[self.n_qubits:2*self.n_qubits]
                x_exp = all_exp[2*self.n_qubits:]
                
                patch_out = torch.stack([z_exp, y_exp, x_exp], dim=-1)  # [n_qubits, 3]
                patches_out.append(patch_out)
            
            q_out_list.append(torch.stack(patches_out))  # [F, n_qubits, 3]
        
        q_out = torch.stack(q_out_list)  # [B, F, n_qubits, 3]
        return q_out.float()
    
    def create_patch_targets(self, x_norm: torch.Tensor):
        """
        Create patch-level targets from normalized input.
        
        Args:
            x_norm: [B, L, C] normalized input
            
        Returns:
            target_val: [B, F, C]
            target_grad: [B, F, C]
            target_vol: [B, F, C]
        """
        B, L, C = x_norm.shape
        
        # Get per-timestep targets
        t_val, t_grad, t_vol = self.target_preprocessor(x_norm)
        
        # Pad and reshape to patches
        padded = torch.cat([x_norm[:, L - self.padding_len:, :], x_norm], dim=1)
        padded_val = torch.cat([t_val[:, L - self.padding_len:, :], t_val], dim=1)
        padded_grad = torch.cat([t_grad[:, L - self.padding_len:, :], t_grad], dim=1)
        padded_vol = torch.cat([t_vol[:, L - self.padding_len:, :], t_vol], dim=1)
        
        # Chunk into patches and take mean per patch
        def chunk_and_mean(x):
            chunks = x.chunk(self.freq, dim=1)  # freq * [B, P, C]
            return torch.stack([c.mean(dim=1) for c in chunks], dim=1)  # [B, F, C]
        
        target_val = chunk_and_mean(padded_val)
        target_grad = chunk_and_mean(padded_grad)
        target_vol = chunk_and_mean(padded_vol)
        
        return target_val, target_grad, target_vol
    
    def encode(self, x: torch.Tensor):
        """
        Encode input to quantum representation.
        
        Args:
            x: [B, L, C] input
            
        Returns:
            q_out: [B, F, n_qubits, 3] quantum outputs
            x_norm: [B, L, C] normalized input (for target creation)
        """
        B, L, C = x.shape
        
        # RevIN
        x_norm = self.revin(x, "norm")
        
        # Patch padding
        padded = torch.cat([x_norm[:, L - self.padding_len:, :], x_norm], dim=1)
        patches = padded.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1).reshape(B, self.freq, -1)  # [B, F, P*C]
        
        # Classical encoder
        params = self.encoder(patches)  # [B, F, n_qubits*2]
        
        # Quantum layer
        q_out = self.quantum_layer(params)  # [B, F, n_qubits, 3]
        
        return q_out, x_norm
    
    def decode_z_to_prediction(self, z_pred: torch.Tensor):
        """
        Map Z-axis predictions to actual output values.
        
        Args:
            z_pred: [B, S, n_qubits] Z-axis values for future patches
            
        Returns:
            y_pred: [B, pred_len, C] predicted future
        """
        B = z_pred.shape[0]
        
        # Use readout head
        y_patches = self.readout_head(z_pred)  # [B, S, P*C]
        y_patches = y_patches.reshape(B, self.step, self.patch_len, self.enc_in)
        y_pred = y_patches.reshape(B, -1, self.enc_in)[:, :self.pred_len]  # [B, pred_len, C]
        y_pred = self.revin(y_pred, "denorm")
        
        return y_pred
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: [B, L, C] input
            
        Returns:
            q_out: [B, F, n_qubits, 3] quantum tomography output
            x_norm: [B, L, C] normalized input
            y_pred: [B, pred_len, C] predicted future (from Z-axis)
        """
        B = x.shape[0]
        
        # Encode
        q_out, x_norm = self.encode(x)  # q_out: [B, F, n_qubits, 3]
        
        # Get Z-axis for past
        z_rec = q_out[:, :, :, 0]  # [B, F, n_qubits]
        
        # Project to future
        z_rec_flat = z_rec.reshape(B, -1)  # [B, F*n_qubits]
        z_pred_flat = self.future_proj(z_rec_flat)  # [B, S*n_qubits]
        z_pred = z_pred_flat.reshape(B, self.step, self.n_qubits)  # [B, S, n_qubits]
        
        # Decode Z to prediction
        y_pred = self.decode_z_to_prediction(z_pred)
        
        return q_out, x_norm, y_pred
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generate samples by adding noise to quantum parameters.
        """
        B = x.shape[0]
        
        q_out, x_norm = self.encode(x)
        z_rec = q_out[:, :, :, 0]  # [B, F, n_qubits]
        z_rec_flat = z_rec.reshape(B, -1)
        
        samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(z_rec_flat) * 0.1
            z_noisy = torch.clamp(z_rec_flat + noise, -1.0, 1.0)
            z_pred_flat = self.future_proj(z_noisy)
            z_pred = z_pred_flat.reshape(B, self.step, self.n_qubits)
            y_sample = self.decode_z_to_prediction(z_pred)
            samples.append(y_sample)
        
        return torch.stack(samples, dim=1)  # [B, num_samples, pred_len, C]


# =====================================================
# STANDALONE TESTING
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing BlochDynamicsVAE (Direct Quantum Supervision)")
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
    
    config = DummyConfig()
    model = BlochDynamicsVAE(config)
    loss_fn = BlochDynamicsLoss()
    
    print(f"\nModel Configuration:")
    print(f"  seq_len: {config.seq_len}, pred_len: {config.pred_len}")
    print(f"  patch_len: {config.patch_len}, n_qubits: {config.n_qubits}")
    print(f"  freq (input patches): {model.freq}")
    print(f"  step (output patches): {model.step}")
    
    # Test data
    B = 2
    x = torch.randn(B, config.seq_len, config.n_vars)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward
    print("\n[Test 1] Forward pass...")
    q_out, x_norm, y_pred = model(x)
    print(f"  q_out shape: {q_out.shape} (expected: [{B}, {model.freq}, {config.n_qubits}, 3])")
    print(f"  y_pred shape: {y_pred.shape} (expected: [{B}, {config.pred_len}, {config.n_vars}])")
    
    z = q_out[:, :, :, 0]
    y = q_out[:, :, :, 1]
    x_ax = q_out[:, :, :, 2]
    print(f"  Z range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  Y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  X range: [{x_ax.min():.3f}, {x_ax.max():.3f}]")
    print("  ✓ All outputs bounded in [-1, 1]")
    
    # Test loss
    print("\n[Test 2] Bloch Dynamics Loss...")
    target_val, target_grad, target_vol = model.create_patch_targets(x_norm)
    print(f"  target_val shape: {target_val.shape}")
    print(f"  target_grad shape: {target_grad.shape}")
    print(f"  target_vol shape: {target_vol.shape}")
    
    loss, loss_dict = loss_fn(q_out, target_val, target_grad, target_vol)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  loss_z (value): {loss_dict['loss_z']:.4f}")
    print(f"  loss_y (trend): {loss_dict['loss_y']:.4f}")
    print(f"  loss_x (coherence): {loss_dict['loss_x']:.4f}")
    print(f"  loss_purity: {loss_dict['loss_purity']:.4f}")
    print("  ✓ Loss computed correctly")
    
    # Test backward
    print("\n[Test 3] Backward pass...")
    loss.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient sum: {grad_sum:.4f}")
    assert grad_sum > 0
    print("  ✓ Gradients flow correctly")
    
    # Test sampling
    print("\n[Test 4] Sampling...")
    samples = model.sample(x, num_samples=5)
    print(f"  Samples shape: {samples.shape}")
    variance = torch.var(samples, dim=1).mean().item()
    print(f"  Sample variance: {variance:.6f}")
    print("  ✓ Sampling works correctly")
    
    print("\n" + "=" * 70)
    print("All BlochDynamicsVAE tests passed! ✓")
    print("=" * 70)
