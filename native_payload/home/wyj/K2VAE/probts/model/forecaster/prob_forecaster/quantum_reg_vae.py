"""
Quantum-Regularized VAE Forecaster Wrapper

This is the ProbTS framework wrapper for the QuantumRegularizedVAE model.
Uses MSE reconstruction + MSE prediction + α * mean(z²) regularization.

Author: Quantum ML Implementation
Date: 2026-01-21
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from probts import Forecaster
from probts.model.nn.prob.k2VAE.quantum_reg_vae import QuantumRegularizedVAE


class ConvertedParams:
    """Convert kwargs dict to object with attributes."""
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class QuantumRegVAEModel(Forecaster):
    """
    Quantum-Regularized VAE Forecaster for probabilistic time series prediction.
    
    Key Features:
    - Quantum Bottleneck: Latent z ∈ [-1, 1] (PauliZ expectations)
    - Patching: Input divided into patches for efficient processing
    - Future Projection: z_rec → MLP → z_pred
    - Shared/Separate Decoder: Configurable via share_decoder parameter
    - Loss: MSE(rec) + MSE(pred) + α * mean(z²)
    
    Args:
        patch_len: Length of each patch (default: 8)
        n_qubits: Number of qubits in quantum circuit (default: 4)
        n_layers: Number of variational layers (default: 2)
        hidden_dim: MLP hidden dimension (default: 64)
        hidden_layers: Number of hidden layers in MLP (default: 2)
        share_decoder: Whether to share decoder for rec/pred (default: True)
        alpha: Regularization weight for z (default: 0.01)
        weight_rec: Reconstruction loss weight (default: 1.0)
        weight_pred: Prediction loss weight (default: 1.0)
    """
    
    def __init__(
        self,
        patch_len: int = 8,
        n_qubits: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        share_decoder: bool = True,
        alpha: float = 0.01,
        weight_rec: float = 1.0,
        weight_pred: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Build config object
        config = ConvertedParams(kwargs)
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.n_vars = self.input_size
        config.patch_len = patch_len
        config.n_qubits = n_qubits
        config.n_layers = n_layers
        config.hidden_dim = hidden_dim
        config.hidden_layers = hidden_layers
        config.share_decoder = share_decoder
        
        # Loss weights
        self.alpha = alpha
        self.weight_rec = weight_rec
        self.weight_pred = weight_pred
        
        # Instantiate the NN model
        self.model = QuantumRegularizedVAE(config)
    
    def forward(self, input):
        """
        Forward pass.
        
        Args:
            input: [B, L, C] input time series
            
        Returns:
            x_rec: [B, L, C] reconstructed input
            y_rec: [B, pred_len, C] predicted future
            z_rec: [B, F, n_qubits] reconstruction latent
        """
        return self.model(input)
    
    def loss(self, batch_data, **kwargs):
        """
        Compute loss.
        
        Loss = weight_rec * MSE(x_rec, x) 
             + weight_pred * MSE(y_rec, target) 
             + alpha * mean(z_rec²)
        
        The regularization term keeps z from saturating at ±1.
        """
        self.model.train()
        
        # Extract data
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        
        # Forward pass
        x_rec, y_rec, z_rec = self.forward(x)
        
        # === MSE Reconstruction Loss ===
        rec_loss = F.mse_loss(x_rec, x)
        
        # === MSE Prediction Loss ===
        pred_loss = F.mse_loss(y_rec, target)
        
        # === Quantum Regularization Loss ===
        reg_loss = self.model.regularization_loss(z_rec, self.alpha)
        
        # Total loss
        total_loss = self.weight_rec * rec_loss + self.weight_pred * pred_loss + reg_loss
        
        # Debug logging
        z_min = z_rec.min().item()
        z_max = z_rec.max().item()
        print(f"rec_loss: {rec_loss:.4f}, pred_loss: {pred_loss:.4f}, "
              f"reg_loss: {reg_loss:.4f}, z_range: [{z_min:.3f}, {z_max:.3f}]")
        
        return total_loss
    
    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts.
        
        Args:
            batch_data: Batch data containing past_target_cdf
            num_samples: Number of samples per prediction (optional)
            
        Returns:
            outputs: [B, num_samples, pred_len, C] or [B, 1, pred_len, C]
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            if num_samples is not None:
                samples = self.model.sample(x, num_samples)
                outputs = samples  # Already [B, num_samples, pred_len, C]
            else:
                _, y_rec, _ = self.forward(x)
                outputs = y_rec.unsqueeze(1)  # [B, 1, pred_len, C]
        
        return outputs
    
    def get_latent(self, batch_data):
        """
        Get the quantum latent representation.
        
        Args:
            batch_data: Batch data containing past_target_cdf
            
        Returns:
            z_rec: [B, F, n_qubits] reconstruction latent
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            _, _, z_rec = self.forward(x)
        
        return z_rec
