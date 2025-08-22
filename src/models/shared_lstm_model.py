#!/usr/bin/env python3
"""
Shared LSTM Model for Earthquake Forecasting

Implements a single shared LSTM model across all quadtree bins with:
- Dual output heads: max magnitude (continuous) and frequency (log-frequency)
- LSTM(64, return_sequences=True) -> LSTM(32, return_sequences=False)
- Concatenation with bin metadata features
- Weighted loss combining MSE (magnitude) and MSE (log-frequency)
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
                 lstm_hidden_2: int = 64,
                 dense_hidden: int = 32,
                 dropout_rate: float = 0.25,
                 freq_head_type: str = "linear"):
        super(SharedLSTMModel, self).__init__()
        
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.dense_hidden = dense_hidden
        self.dropout_rate = dropout_rate
        self.freq_head_type = freq_head_type
        
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
        
        # Dense layers after concatenation
        self.dense1 = nn.Linear(combined_features, dense_hidden)
        self.bn1 = nn.BatchNorm1d(dense_hidden)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # [REFACTOR] Deeper MLP funnel for output heads (120 -> 90 -> 30 -> 30 -> 1)
        # Magnitude head: deeper MLP funnel with Linear output (no activation)
        self.magnitude_head = nn.Sequential(
            nn.Linear(dense_hidden, 120),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(120, 90),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(30, 1)  # No activation for magnitude
        )
        
        # [REFACTOR] Frequency head: enhanced architecture with learnable scaling
        if freq_head_type == "linear":
            # Enhanced frequency head: light hidden layer for better expressiveness
            self.frequency_head = nn.Sequential(
                nn.Linear(dense_hidden, 120),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(120, 90),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(90, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.2),
                nn.Linear(30, 8),  # NEW: Light hidden layer
                nn.ReLU(),
                nn.Linear(8, 1)    # NEW: Output layer
            )
            # FIX: Add learnable scaling parameters for linear mode to prevent prediction collapse
            self.frequency_scale = nn.Parameter(torch.tensor(1.0))
            self.frequency_bias = nn.Parameter(torch.tensor(0.0))
        else:
            # Legacy scaled mode (kept for comparison)
            self.frequency_head = nn.Sequential(
                nn.Linear(dense_hidden, 120),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(120, 90),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(90, 60),  # Added intermediate layer
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.2),
                nn.Linear(60, 30),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.1),
                nn.Linear(30, 1),
                nn.Identity()  # No activation - let the model learn the full range
            )
            # Learnable scaling parameters for scaled mode
            self.frequency_scale = nn.Parameter(torch.tensor(1.0))
            self.frequency_bias = nn.Parameter(torch.tensor(0.0))
        
        # FIX: Create logger before calling _init_weights to avoid AttributeError
        self.logger = logging.getLogger(__name__)
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"SharedLSTMModel initialized with {sum(p.numel() for p in self.parameters())} parameters")
        self.logger.info(f"[REFACTORING] Frequency head type: {freq_head_type}")
        if freq_head_type == "linear":
            self.logger.info("  - Linear frequency head: direct log-frequency prediction")
            self.logger.info("  - FIX: Added learnable scaling parameters to prevent collapse")
            self.logger.info("  - Stable regression in log-space with adaptive scaling")
        else:
            self.logger.info("  - Scaled frequency head: learnable scaling parameters")
            self.logger.info("  - Legacy mode for comparison")
        self.logger.info("  - Magnitude head: Linear output (no activation)")
        self.logger.info("  - Deeper MLP funnel: 120 -> 90 -> 30 -> 30 -> 1")
    
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
        
        # FIX: Initialize frequency scaling parameters for better training stability
        if hasattr(self, 'frequency_scale') and self.frequency_scale is not None:
            # IMPROVED: Initialize scale to a larger value to encourage wider prediction ranges
            # Start with scale=5.0 instead of 2.0 to help model learn much broader frequency distributions
            nn.init.constant_(self.frequency_scale, 5.0)
            self.logger.info("  - Frequency scale initialized to 5.0 (aggressive initialization)")
        
        if hasattr(self, 'frequency_bias') and self.frequency_bias is not None:
            # IMPROVED: Initialize bias to a larger positive value to shift predictions up
            # Start with bias=1.0 instead of 0.5 to help model learn higher frequency values
            nn.init.constant_(self.frequency_bias, 1.0)
            self.logger.info("  - Frequency bias initialized to 1.0 (aggressive initialization)")
    
    def forward(self, 
                input_sequence: torch.Tensor, 
                metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_sequence: Input sequence tensor of shape (batch_size, lookback_years, input_seq_features)
            metadata: Bin metadata tensor of shape (batch_size, metadata_features)
            
        Returns:
            Tuple of (magnitude_pred, frequency_pred)
            - magnitude_pred: continuous magnitude prediction
            - frequency_pred: log-frequency prediction (raw linear output)
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
        dense_out = self.bn1(dense_out)
        dense_out = F.relu(dense_out)
        dense_out = self.dropout1(dense_out)
        
        # Dual output heads
        magnitude_pred = self.magnitude_head(dense_out)
        frequency_pred = self.frequency_head(dense_out)  # Raw output (no activation)
        
        # Apply scaling based on head type
        if self.freq_head_type == "scaled" and self.frequency_scale is not None:
            # Legacy scaled mode
            frequency_pred = self.frequency_scale * frequency_pred + self.frequency_bias
            frequency_pred = F.softplus(frequency_pred) + 1e-6  # Softplus for positivity
        # FIX: Apply scaling to linear mode as well to prevent prediction collapse
        elif self.freq_head_type == "linear" and self.frequency_scale is not None:
            # Linear mode: apply scaling to match target ranges
            frequency_pred = self.frequency_scale * frequency_pred + self.frequency_bias
        # For both modes, return scaled output
        
        return magnitude_pred, frequency_pred
    
    def predict_frequency_counts(self, frequency_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert frequency predictions to expected count predictions.
        
        Args:
            frequency_pred: frequency predictions (scaled log-frequency for both modes)
            
        Returns:
            Expected count predictions converted to raw counts
        """
        # FIX: frequency_pred is already scaled by the model's learnable parameters
        # Both modes now output scaled values that need to be converted to raw counts
        
        if self.freq_head_type == "linear":
            # Linear mode: frequency_pred is scaled log(λ), convert to raw counts
            # First denormalize the scaling, then convert to raw counts
            denorm_pred = (frequency_pred - self.frequency_bias) / self.frequency_scale
            raw_counts = torch.exp(denorm_pred)
        else:
            # Scaled mode: frequency_pred has scaling + Softplus, convert to raw counts
            # First denormalize the scaling, then convert to raw counts
            denorm_pred = (frequency_pred - self.frequency_bias) / self.frequency_scale
            raw_counts = torch.expm1(denorm_pred)
        
        # Add small epsilon to prevent zero values
        return raw_counts + 1e-6


class WeightedEarthquakeLoss(nn.Module):
    """
    Custom loss function combining MSE for magnitude and MSE for frequency.
    
    REFACTOR: Updated to use rebalanced weights and direct log-frequency comparison.
    Total Loss = alpha*MSE + beta*MSE (frequency targets preprocessed with log1p)
    """
    
    def __init__(self, 
                 magnitude_weight: float = 2.0,      # alpha: weight for magnitude loss (increased)
                 frequency_weight: float = 1.0,      # beta: weight for frequency loss (increased from 0.5)
                 correlation_weight: float = 0.0,    # gamma: weight for correlation penalty (disabled)
                 dynamic_beta: bool = False):        # Disable dynamic beta adjustment
        super(WeightedEarthquakeLoss, self).__init__()
        
        self.magnitude_weight = magnitude_weight      # alpha
        self.frequency_weight = frequency_weight      # beta
        self.correlation_weight = correlation_weight  # gamma
        self.dynamic_beta = dynamic_beta              # Disabled
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"WeightedEarthquakeLoss: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
        self.logger.info("[REFACTORING] APPLIED:")
        self.logger.info("  - Rebalanced loss weights: alpha=2.0, beta=1.0, gamma=0.0")
        self.logger.info("  - Frequency targets preprocessed with log1p")
        self.logger.info("  - Direct MSE comparison in log-space")
        self.logger.info("  - Total Loss = alpha*MSE + beta*MSE (log1p frequency)")
    
    def forward(self, 
                magnitude_pred: torch.Tensor, 
                frequency_pred: torch.Tensor,
                magnitude_true: torch.Tensor, 
                frequency_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted loss with log1p preprocessing for frequency.
        
        Args:
            magnitude_pred: magnitude predictions
            frequency_pred: frequency predictions (scaled log-frequency for both modes)
            magnitude_true: true magnitude values
            frequency_true: true frequency counts
        """
        # Magnitude loss (MSE)
        magnitude_loss = F.mse_loss(magnitude_pred, magnitude_true)
        
        # FIX: Ensure frequency predictions are properly scaled before loss computation
        # frequency_pred is already scaled by the model's learnable parameters
        
        # REFACTOR: Preprocess frequency targets with log1p and train directly on that scale
        frequency_true_log1p = torch.log1p(frequency_true)  # log(1 + frequency)
        
        # FIX: frequency_pred is now scaled, compare directly with log1p targets
        # The model learns to output values in the same scale as log1p targets
        
        # Frequency loss: MSE on log1p scale
        frequency_loss = F.mse_loss(frequency_pred, frequency_true_log1p)
        
        # REFACTOR: Weighted combination with rebalanced weights
        total_loss = (self.magnitude_weight * magnitude_loss + 
                     self.frequency_weight * frequency_loss)
        
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
    
    def get_loss_components(self, 
                           magnitude_pred: torch.Tensor, 
                           frequency_pred: torch.Tensor,
                           magnitude_true: torch.Tensor, 
                           frequency_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual loss components for monitoring."""
        # Magnitude loss (MSE)
        magnitude_loss = F.mse_loss(magnitude_pred, magnitude_true)
        
        # Frequency loss: MSE on log1p scale
        frequency_true_log1p = torch.log1p(frequency_true)
        frequency_loss = F.mse_loss(frequency_pred, frequency_true_log1p)
        
        return {
            'magnitude_loss': magnitude_loss,
            'frequency_loss': frequency_loss,
            'correlation_penalty': torch.tensor(0.0, device=magnitude_pred.device),
            'total_loss': (self.magnitude_weight * magnitude_loss + 
                          self.frequency_weight * frequency_loss)
        }
