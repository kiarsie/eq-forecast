#!/usr/bin/env python3
"""
Attention-Enhanced Shared LSTM Model for Earthquake Forecasting

Extends the shared LSTM with attention mechanism using the same configurations:
- LSTM(64, return_sequences=True) -> LSTM(32, return_sequences=True) with attention
- Concatenation with bin metadata features
- Dual output heads: max magnitude (continuous) and frequency (log-frequency)
- Weighted loss combining MSE (magnitude) and MSE (log-frequency)
- Same configurable loss weights alpha=2.0, beta=1.0, gamma=0.0
- Identical frequency head architecture and scaling parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from .shared_lstm_model import WeightedEarthquakeLoss  # Use the same refactored loss function


class SimpleWeightedAttention(nn.Module):
    """
    Extremely simple attention: just learnable weights for each timestep.
    No complex projections, no multi-head complexity - just weights that sum to 1.
    """
    
    def __init__(self, seq_length: int = 10):
        super(SimpleWeightedAttention, self).__init__()
        self.seq_length = seq_length
        
        # Learnable weights for each timestep (initialize to uniform)
        self.timestep_weights = nn.Parameter(torch.ones(seq_length) / seq_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned weights to each timestep.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Weighted output tensor
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Ensure weights sum to 1 (softmax for stability)
        weights = F.softmax(self.timestep_weights, dim=0)
        
        # Apply weights to each timestep
        # Shape: (batch_size, seq_len, embed_dim)
        weighted_output = x * weights.unsqueeze(0).unsqueeze(-1)
        
        return weighted_output


class AttentionSharedLSTMModel(nn.Module):
    """
    Attention-enhanced shared LSTM model for earthquake forecasting.
    """
    
    def __init__(self, 
                 input_seq_features: int,
                 metadata_features: int,
                 lookback_years: int = 10,
                 lstm_hidden_1: int = 64,
                 lstm_hidden_2: int = 32,  # FIXED: Match shared LSTM (32, not 64)
                 dense_hidden: int = 32,
                 dropout_rate: float = 0.25,
                 freq_head_type: str = "linear"):
        super(AttentionSharedLSTMModel, self).__init__()
        
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.dense_hidden = dense_hidden
        self.dropout_rate = dropout_rate
        self.freq_head_type = freq_head_type
        
        # LSTM layers for sequential features - EXACTLY like shared LSTM
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
        
        # Attention mechanism over the sequence
        self.attention = SimpleWeightedAttention(
            seq_length=lookback_years
        )
        
        # Feature concatenation layer - EXACTLY like shared LSTM
        combined_features = lstm_hidden_2 + metadata_features
        
        # Dense layers after concatenation - EXACTLY like shared LSTM
        self.dense1 = nn.Linear(combined_features, dense_hidden)
        self.bn1 = nn.BatchNorm1d(dense_hidden)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # [REFACTOR] Deeper MLP funnel for output heads (120 -> 90 -> 30 -> 30 -> 1) - EXACTLY like shared LSTM
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
        
        # [REFACTOR] Frequency head: enhanced architecture with learnable scaling - EXACTLY like shared LSTM
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
                nn.Linear(30, 8),  # Light hidden layer
                nn.ReLU(),
                nn.Linear(8, 1)    # Output layer
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
        
        self.logger.info(f"AttentionSharedLSTMModel initialized with {sum(p.numel() for p in self.parameters())} parameters")
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
        self.logger.info("  - Attention mechanism: Simple learnable timestep weights")
    
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
            # IMPROVED: Initialize scale to a much larger value to encourage wider prediction ranges
            # Start with scale=10.0 instead of 5.0 to help model learn much broader frequency distributions
            nn.init.constant_(self.frequency_scale, 10.0)
            self.logger.info("  - Frequency scale initialized to 10.0 (very aggressive initialization)")
        
        if hasattr(self, 'frequency_bias') and self.frequency_bias is not None:
            # IMPROVED: Initialize bias to a larger positive value to shift predictions up
            # Start with bias=2.0 instead of 1.0 to help model learn higher frequency values
            nn.init.constant_(self.frequency_bias, 2.0)
            self.logger.info("  - Frequency bias initialized to 2.0 (very aggressive initialization)")
    
    def forward(self, 
                input_sequence: torch.Tensor, 
                metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention-enhanced model.
        
        Args:
            input_sequence: Input sequence tensor of shape (batch_size, lookback_years, input_seq_features)
            metadata: Bin metadata tensor of shape (batch_size, metadata_features)
            
        Returns:
            Tuple of (magnitude_pred, frequency_pred)
            - magnitude_pred: continuous magnitude prediction
            - frequency_pred: log-frequency prediction (raw linear output)
        """
        # LSTM processing - EXACTLY like shared LSTM
        lstm_out, _ = self.lstm1(input_sequence)
        lstm_out, (hidden, cell) = self.lstm2(lstm_out)
        
        # Apply simple weighted attention over the sequence
        attended_seq = self.attention(lstm_out)
        
        # Debug: Log attention weights during training
        if self.training:
            weights = F.softmax(self.attention.timestep_weights, dim=0)
            self.logger.info(f"DEBUG: Attention weights: {weights.detach().cpu().numpy()}")
        
        # Take the last hidden state - EXACTLY like shared LSTM
        lstm_final = attended_seq[:, -1, :]  # (batch_size, lstm_hidden_2)
        
        # Concatenate LSTM output with metadata - EXACTLY like shared LSTM
        combined = torch.cat([lstm_final, metadata], dim=1)
        
        # Dense layers - EXACTLY like shared LSTM
        dense_out = self.dense1(combined)
        dense_out = self.bn1(dense_out)
        dense_out = F.relu(dense_out)
        dense_out = self.dropout1(dense_out)
        
        # Dual output heads - EXACTLY like shared LSTM
        magnitude_pred = self.magnitude_head(dense_out)
        frequency_pred = self.frequency_head(dense_out)  # Raw output (no activation)
        
        # Apply scaling based on head type - EXACTLY like shared LSTM
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
