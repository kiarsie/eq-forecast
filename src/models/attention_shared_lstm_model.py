#!/usr/bin/env python3
"""
Multi-Head Attention-Enhanced Shared LSTM Model for Earthquake Forecasting

Extends the shared LSTM with multi-head attention mechanism using the same configurations:
- LSTM(64, return_sequences=True) -> LSTM(32, return_sequences=True) with multi-head attention (2 heads)
- Concatenation with bin metadata features
- Dual output heads: max magnitude (continuous) and frequency (log-frequency)
- Weighted loss combining MSE (magnitude) and MSE (log-frequency)
- Same configurable loss weights alpha=2.0, beta=1.0, gamma=0.0
- Identical frequency head architecture and scaling parameters
- Enhanced attention with learnable Q, K, V projections and scaled dot-product attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from .shared_lstm_model import WeightedEarthquakeLoss  # Use the same refactored loss function


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with 2 attention heads.
    Implements scaled dot-product attention with learnable projections.
    """
    
    def __init__(self, embed_dim: int = 32, num_heads: int = 2, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Attended output tensor
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


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
                 freq_head_type: str = "linear",
                 frequency_scale_init: float = 2.0,
                 frequency_bias_init: float = 0.5):
        super(AttentionSharedLSTMModel, self).__init__()
        
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.dense_hidden = dense_hidden
        self.dropout_rate = dropout_rate
        self.freq_head_type = freq_head_type
        self.frequency_scale_init = frequency_scale_init
        self.frequency_bias_init = frequency_bias_init
        
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
        
        # Multi-head attention mechanism over the sequence
        self.attention = MultiHeadAttention(
            embed_dim=lstm_hidden_2,
            num_heads=2,
            dropout=dropout_rate * 0.5
        )
        
        # Layer normalization for attention output
        self.layer_norm = nn.LayerNorm(lstm_hidden_2)
        
        # Lightweight FeedForward Network after attention
        self.ffn = nn.Sequential(
            nn.Linear(lstm_hidden_2, lstm_hidden_2 * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(lstm_hidden_2 * 2, lstm_hidden_2)
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
            self.frequency_scale = nn.Parameter(torch.tensor(frequency_scale_init))
            self.frequency_bias = nn.Parameter(torch.tensor(frequency_bias_init))
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
            self.frequency_scale = nn.Parameter(torch.tensor(frequency_scale_init))
            self.frequency_bias = nn.Parameter(torch.tensor(frequency_bias_init))
        
        # FIX: Create logger before calling _init_weights to avoid AttributeError
        self.logger = logging.getLogger(__name__)
        
        # Initialize weights
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        self.logger.info(f"AttentionSharedLSTMModel initialized with {total_params} parameters")
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
        self.logger.info("  - Attention mechanism: Multi-head attention (2 heads) with residual connections")
        self.logger.info("  - Post-attention: LayerNorm + FeedForward Network (64->32) with residual connections")
    
    def set_normalization_params(self, params: dict):
        """Set normalization parameters for consistency across evaluations."""
        # Store normalization parameters for potential use in forward pass
        self.normalization_params = params
        self.logger.info("Normalization parameters set for consistency")
    
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
        
        # Initialize attention mechanism weights
        if hasattr(self, 'attention'):
            for module in self.attention.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # FIX: Initialize frequency scaling parameters for better training stability
        if hasattr(self, 'frequency_scale') and self.frequency_scale is not None:
            # Use the values passed to constructor
            nn.init.constant_(self.frequency_scale, self.frequency_scale_init)
            self.logger.info(f"  - Frequency scale initialized to {self.frequency_scale_init} (config-driven initialization)")
        
        if hasattr(self, 'frequency_bias') and self.frequency_bias is not None:
            # Use the values passed to constructor
            nn.init.constant_(self.frequency_bias, self.frequency_bias_init)
            self.logger.info(f"  - Frequency bias initialized to {self.frequency_bias_init} (config-driven initialization)")
    
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
        
        # Apply multi-head attention over the sequence with residual connection
        attended_seq = self.attention(lstm_out) + lstm_out  # Residual connection
        
        # Apply LayerNorm after residual connection
        attended_seq = self.layer_norm(attended_seq)
        
        # Apply FeedForward Network with residual connection
        ffn_out = self.ffn(attended_seq) + attended_seq  # Residual connection
        
        # Take the last hidden state - EXACTLY like shared LSTM
        lstm_final = ffn_out[:, -1, :]  # (batch_size, lstm_hidden_2)
        
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
