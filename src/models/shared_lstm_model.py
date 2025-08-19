#!/usr/bin/env python3
"""
Shared LSTM Model for Earthquake Forecasting

Implements a single shared LSTM model across all quadtree bins with:
- Dual output heads: max magnitude (continuous) and frequency (Poisson log-rate)
- LSTM(64, return_sequences=True) → LSTM(32, return_sequences=False)
- Concatenation with bin metadata features
- Weighted loss combining MSE (magnitude) and Poisson NLL (frequency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging


class SharedLSTMModel(nn.Module):
    """
    Shared LSTM model for earthquake forecasting across all quadtree bins.
    """
    
    def __init__(self, 
                 input_seq_features: int,
                 metadata_features: int,
                 lookback_years: int = 10,
                 lstm_hidden_1: int = 64,
                 lstm_hidden_2: int = 32,
                 dense_hidden: int = 32,
                 dropout_rate: float = 0.25):
        super(SharedLSTMModel, self).__init__()
        
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.dense_hidden = dense_hidden
        self.dropout_rate = dropout_rate
        
        # LSTM layers for sequential features
        self.lstm1 = nn.LSTM(
            input_size=input_seq_features,
            hidden_size=lstm_hidden_1,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            batch_first=True
        )
        
        # Feature concatenation layer
        combined_features = lstm_hidden_2 + metadata_features
        
        # Dense layers after concatenation with enhanced dropout and normalization
        self.dense1 = nn.Linear(combined_features, dense_hidden)
        self.bn1 = nn.BatchNorm1d(dense_hidden)  # 🔧 FIX: Add batch normalization
        self.dropout1 = nn.Dropout(dropout_rate * 1.5)  # 🔧 FIX: Increase dropout
        
        # Dual output heads with enhanced regularization
        # Magnitude head (continuous, linear activation) - UNCHANGED
        self.magnitude_head = nn.Sequential(
            nn.Linear(dense_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),  # 🔧 FIX: Increase dropout for output
            nn.Linear(16, 1)
        )
        
        # FREQUENCY HEAD: UPDATED TO LOG1P + MSE FORMULATION
        # Output: log(λ) where λ is the Poisson rate parameter
        # For inference: expected counts = exp(log(λ)) = λ
        self.frequency_head = nn.Sequential(
            nn.Linear(dense_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),  # 🔧 FIX: Increase dropout for output
            nn.Linear(16, 1)  # Linear activation for log-rate output
        )
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SharedLSTMModel initialized with {sum(p.numel() for p in self.parameters())} parameters")
        self.logger.info("Frequency head updated to Poisson formulation (log-rate output)")
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, 
                input_sequence: torch.Tensor, 
                metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_sequence: Input sequence tensor of shape (batch_size, lookback_years, input_seq_features)
            metadata: Bin metadata tensor of shape (batch_size, metadata_features)
            
        Returns:
            Tuple of (magnitude_pred, frequency_log_rate_pred)
            - magnitude_pred: continuous magnitude prediction
            - frequency_log_rate_pred: log of Poisson rate parameter (log(λ))
        """
        # LSTM processing
        lstm_out, _ = self.lstm1(input_sequence)
        lstm_out, (hidden, cell) = self.lstm2(lstm_out)
        
        # Take the last hidden state
        lstm_final = hidden[-1]  # (batch_size, lstm_hidden_2)
        
        # Concatenate LSTM output with metadata
        combined = torch.cat([lstm_final, metadata], dim=1)
        
        # Dense layers
        dense_out = self.dense1(combined)
        dense_out = self.bn1(dense_out)  # 🔧 FIX: Apply batch normalization
        dense_out = F.relu(dense_out)
        dense_out = self.dropout1(dense_out)
        
        # Dual output heads
        magnitude_pred = self.magnitude_head(dense_out)
        frequency_log_rate_pred = self.frequency_head(dense_out)  # log(λ)
        
        return magnitude_pred, frequency_log_rate_pred
    
    def predict_frequency_counts(self, frequency_log_rate_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert log-rate predictions to expected count predictions.
        
        Args:
            frequency_log_rate_pred: log(λ) predictions
            
        Returns:
            Expected count predictions: exp(log(λ)) = λ
        """
        return torch.exp(frequency_log_rate_pred)


class WeightedEarthquakeLoss(nn.Module):
    """
    Custom loss function combining MSE for magnitude and log1p + MSE for frequency.
    
    🔧 IMPROVEMENT: Added configurable weights α, β, γ and optional correlation penalty.
    Total Loss = α*MSE + β*log1p_MSE - γ*PearsonCorrelation
    """
    
    def __init__(self, 
                 magnitude_weight: float = 1.0,      # α: weight for magnitude loss
                 frequency_weight: float = 4.0,      # β: weight for frequency loss
                 correlation_weight: float = 0.0,    # γ: weight for correlation penalty (default: disabled)
                 dynamic_beta: bool = True):         # Enable dynamic β adjustment
        super(WeightedEarthquakeLoss, self).__init__()
        
        self.magnitude_weight = magnitude_weight      # α
        self.base_frequency_weight = frequency_weight # Base β value
        self.frequency_weight = frequency_weight      # Current β value (can be adjusted)
        self.correlation_weight = correlation_weight  # γ
        self.dynamic_beta = dynamic_beta              # Enable dynamic β
        self.epoch_count = 0                         # Track epochs for dynamic adjustment
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"WeightedEarthquakeLoss: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
        self.logger.info("Frequency loss updated to use log1p + MSE for better stability")
        if self.dynamic_beta:
            self.logger.info("Dynamic β enabled: β will adjust based on training progress")
        if correlation_weight > 0:
            self.logger.info("Correlation penalty enabled: Total Loss = alpha*MSE + beta*log1p_MSE - gamma*Correlation")
    
    def forward(self, 
                magnitude_pred: torch.Tensor, 
                frequency_log_rate_pred: torch.Tensor,
                magnitude_true: torch.Tensor, 
                frequency_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted loss with optional correlation penalty.
        
        Args:
            magnitude_pred: magnitude predictions
            frequency_log_rate_pred: log(λ) predictions (Poisson log-rate)
            magnitude_true: true magnitude values
            frequency_true: true frequency counts
        """
        # Magnitude loss (MSE) - UNCHANGED
        magnitude_loss = F.mse_loss(magnitude_pred, magnitude_true)
        
        # 🔧 IMPROVEMENT: Apply log-transform to frequency targets (log1p) for better scaling
        # This ensures both trend and scale are learned properly
        frequency_true_log = torch.log1p(frequency_true)  # log(1 + frequency)
        
        # FREQUENCY LOSS: UPDATED TO LOG1P + MSE for better stability
        # 🔧 FIX: Add clipping to prevent extreme values
        frequency_log_rate_clipped = torch.clamp(frequency_log_rate_pred, -10, 10)  # Prevent extreme log values
        
        # Convert predictions to log1p space for comparison
        frequency_pred_log = torch.log1p(torch.exp(frequency_log_rate_clipped))  # log(1 + exp(log(λ)))
        
        # MSE loss in log1p space
        frequency_loss = F.mse_loss(frequency_pred_log, frequency_true_log)
        
        # 🔧 IMPROVEMENT: Calculate correlation penalty if enabled
        correlation_penalty = 0.0
        if self.correlation_weight > 0:
            # Calculate Pearson correlation for both outputs
            mag_corr = self._pearson_correlation(magnitude_pred, magnitude_true)
            freq_corr = self._pearson_correlation(frequency_log_rate_pred, frequency_true)
            
            # Use average correlation as penalty
            correlation_penalty = (mag_corr + freq_corr) / 2.0
        
        # 🔧 IMPROVEMENT: Weighted combination with configurable weights (α, β, γ)
        total_loss = (self.magnitude_weight * magnitude_loss + 
                     self.frequency_weight * frequency_loss - 
                     self.correlation_weight * correlation_penalty)
        
        return total_loss
    
    def _pearson_correlation(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation coefficient between predictions and targets."""
        # Remove batch dimension if present
        if pred.dim() > 1:
            pred = pred.squeeze()
        if true.dim() > 1:
            true = true.squeeze()
        
        # Calculate means
        pred_mean = torch.mean(pred)
        true_mean = torch.mean(true)
        
        # Calculate correlation
        numerator = torch.sum((pred - pred_mean) * (true - true_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((true - true_mean) ** 2))
        
        # Handle division by zero
        if denominator == 0:
            return torch.tensor(0.0, device=pred.device)
        
        correlation = numerator / denominator
        return correlation
    
    def update_beta(self, epoch: int, val_loss: float = None):
        """
        Dynamically adjust β weight based on training progress.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss for adaptive adjustment
        """
        if not self.dynamic_beta:
            return
        
        self.epoch_count = epoch
        
        # Strategy 1: Gradual increase in early training
        if epoch < 10:
            # Start with lower β, gradually increase
            self.frequency_weight = self.base_frequency_weight * (0.5 + 0.5 * epoch / 10)
        
        # Strategy 2: Adaptive adjustment based on validation loss
        elif val_loss is not None and epoch > 10:
            # If validation loss is high, increase β to focus on frequency
            if val_loss > 1.0:
                self.frequency_weight = min(self.base_frequency_weight * 1.5, 10.0)
            else:
                # Gradually return to base value
                self.frequency_weight = max(self.base_frequency_weight, 
                                         self.frequency_weight * 0.95)
        
        # Strategy 3: Final convergence
        if epoch > 30:
            # Stabilize β near base value
            self.frequency_weight = self.base_frequency_weight
    
    def get_loss_components(self, 
                           magnitude_pred: torch.Tensor, 
                           frequency_log_rate_pred: torch.Tensor,
                           magnitude_true: torch.Tensor, 
                           frequency_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual loss components for monitoring."""
        # Magnitude loss (MSE) - UNCHANGED
        magnitude_loss = F.mse_loss(magnitude_pred, magnitude_true)
        
        # 🔧 IMPROVEMENT: Apply log-transform to frequency targets for consistency
        frequency_true_log = torch.log1p(frequency_true)  # log(1 + frequency)
        
        # Frequency loss (log1p + MSE) - UPDATED for better stability
        frequency_pred_log = torch.log1p(torch.exp(frequency_log_rate_pred))  # log(1 + exp(log(λ)))
        frequency_loss = F.mse_loss(frequency_pred_log, frequency_true_log)
        
        # 🔧 IMPROVEMENT: Calculate correlation penalty if enabled
        correlation_penalty = 0.0
        if self.correlation_weight > 0:
            mag_corr = self._pearson_correlation(magnitude_pred, magnitude_true)
            freq_corr = self._pearson_correlation(frequency_log_rate_pred, frequency_true)
            correlation_penalty = (mag_corr + freq_corr) / 2.0
        
        return {
            'magnitude_loss': magnitude_loss,
            'frequency_loss': frequency_loss,
            'correlation_penalty': correlation_penalty,
            'total_loss': (self.magnitude_weight * magnitude_loss + 
                          self.frequency_weight * frequency_loss - 
                          self.correlation_weight * correlation_penalty)
        }
