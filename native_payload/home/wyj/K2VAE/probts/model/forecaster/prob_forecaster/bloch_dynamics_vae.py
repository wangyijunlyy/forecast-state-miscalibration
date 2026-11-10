"""
Bloch Dynamics VAE Forecaster Wrapper

ProbTS framework wrapper for the BlochDynamicsVAE model.
Uses direct Bloch sphere supervision instead of traditional decoder + MSE.

Author: Quantum ML Implementation
Date: 2026-01-21
"""

import torch
import torch.nn.functional as F

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.bloch_dynamics_vae import (
    BlochDynamicsVAE,
    BlochDynamicsLoss
)


class ConvertedParams:
    """Convert kwargs dict to object with attributes."""
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class BlochDynamicsVAEModel(Forecaster):
    """
    Bloch Dynamics VAE Forecaster - Direct Quantum Supervision.
    
    Key Innovation:
    - No decoder: quantum measurements ARE predictions
    - Tri-axis physical supervision:
        * Z-axis → Series Value (position)
        * Y-axis → Series Gradient (momentum)
        * X-axis → Inverse Volatility (coherence)
    - Geometric purity constraint on Bloch sphere
    
    Args:
        patch_len: Length of each patch
        n_qubits: Number of qubits in quantum circuit
        n_layers: Number of variational layers
        hidden_dim: MLP hidden dimension
        hidden_layers: Number of hidden layers in MLP
        lambda_val: Weight for value loss (Z-axis)
        lambda_grad: Weight for gradient loss (Y-axis)
        lambda_vol: Weight for volatility loss (X-axis)
        lambda_purity: Weight for purity constraint
        weight_pred: Weight for prediction MSE loss
    """
    
    def __init__(
        self,
        patch_len: int = 8,
        n_qubits: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        lambda_val: float = 1.0,
        lambda_grad: float = 1.0,
        lambda_vol: float = 0.5,
        lambda_purity: float = 0.1,
        weight_pred: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Build config
        config = ConvertedParams(kwargs)
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.n_vars = self.input_size
        config.patch_len = patch_len
        config.n_qubits = n_qubits
        config.n_layers = n_layers
        config.hidden_dim = hidden_dim
        config.hidden_layers = hidden_layers
        
        # Store loss weights
        self.lambda_val = lambda_val
        self.lambda_grad = lambda_grad
        self.lambda_vol = lambda_vol
        self.lambda_purity = lambda_purity
        self.weight_pred = weight_pred
        
        # Instantiate model
        self.model = BlochDynamicsVAE(config)
        
        # Loss function
        self.bloch_loss = BlochDynamicsLoss(
            lambda_val=lambda_val,
            lambda_grad=lambda_grad,
            lambda_vol=lambda_vol,
            lambda_purity=lambda_purity
        )
    
    def forward(self, input: torch.Tensor):
        """
        Forward pass.
        
        Args:
            input: [B, L, C] input time series
            
        Returns:
            q_out: [B, F, n_qubits, 3] quantum tomography output
            x_norm: [B, L, C] normalized input
            y_pred: [B, pred_len, C] predicted future
        """
        return self.model(input)
    
    def loss(self, batch_data, **kwargs):
        """
        Compute Bloch Dynamics Loss.
        
        Loss = BlochLoss(q_out, targets) + weight_pred * MSE(y_pred, target)
        
        The Bloch Loss provides:
        - Direct supervision on quantum measurements
        - Physical interpretability
        - Geometric constraint via purity term
        """
        self.model.train()
        
        # Get data
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        
        # Forward
        q_out, x_norm, y_pred = self.forward(x)
        
        # Create patch-level targets
        target_val, target_grad, target_vol = self.model.create_patch_targets(x_norm)
        
        # === Bloch Dynamics Loss ===
        bloch_loss, loss_dict = self.bloch_loss(
            q_out, target_val, target_grad, target_vol
        )
        
        # === Prediction MSE Loss ===
        pred_loss = F.mse_loss(y_pred, target)
        
        # Total loss
        total_loss = bloch_loss + self.weight_pred * pred_loss
        
        # Logging
        print(f"loss_z: {loss_dict['loss_z']:.4f}, "
              f"loss_y: {loss_dict['loss_y']:.4f}, "
              f"loss_x: {loss_dict['loss_x']:.4f}, "
              f"purity: {loss_dict['loss_purity']:.4f}, "
              f"pred: {pred_loss.item():.4f}")
        
        return total_loss
    
    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts.
        
        Args:
            batch_data: Batch data containing past_target_cdf
            num_samples: Number of samples (optional)
            
        Returns:
            outputs: [B, num_samples, pred_len, C] or [B, 1, pred_len, C]
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                _, _, y_pred = self.forward(x)
                outputs = y_pred.unsqueeze(1)
        
        return outputs
    
    def get_quantum_state(self, batch_data):
        """
        Get the full quantum tomography output.
        
        Returns:
            q_out: [B, F, n_qubits, 3] with (Z, Y, X) per qubit per patch
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            q_out, _, _ = self.forward(x)
        
        return q_out
