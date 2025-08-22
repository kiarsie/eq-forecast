#!/usr/bin/env python3
"""
Attention-Based Shared LSTM Model for Earthquake Forecasting

Extends the shared LSTM with attention mechanism:
- LSTM(64, return_sequences=True) → LSTM(32, return_sequences=True) with attention
- Concatenation with bin metadata features
- Dual output heads: magnitude (regression) and frequency (Softplus activation)
- Weighted loss combining MSE (magnitude) and MSE (frequency with log1p preprocessing)
- 🔧 REFACTOR: Same configurable loss weights alpha=1.0, beta=1.0, gamma=0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from .shared_lstm_model import WeightedEarthquakeLoss  # 🔧 REFACTOR: Use the same refactored loss function


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for sequence modeling.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Attended output tensor
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
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
                 lstm_hidden_2: int = 64,
                 dense_hidden: int = 32,
                 dropout_rate: float = 0.25,
                 num_attention_heads: int = 4,
                 freq_head_type: str = "linear"):
        super(AttentionSharedLSTMModel, self).__init__()
        
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.dense_hidden = dense_hidden
        self.dropout_rate = dropout_rate
        self.num_attention_heads = num_attention_heads
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
        
        # Attention mechanism over the sequence
        self.attention = MultiHeadAttention(
            embed_dim=lstm_hidden_2,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Feature concatenation layer
        combined_features = lstm_hidden_2 + metadata_features
        
        # Dense layers after concatenation
        self.dense1 = nn.Linear(combined_features, dense_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        
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
        
        # [REFACTOR] Frequency head: same funnel with Softplus activation (+1e-6)
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
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(30, 1),
            nn.Softplus()  # Softplus activation for frequency
        )
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AttentionSharedLSTMModel initialized with {sum(p.numel() for p in self.parameters())} parameters")
        self.logger.info("[REFACTORING] APPLIED:")
        self.logger.info(f"  - Attention heads: {num_attention_heads}")
        self.logger.info("  - Attention output aggregation: Mean pooling across timesteps")
        self.logger.info("  - Deeper MLP funnel: 120 → 90 → 30 → 30 → 1")
        self.logger.info("  - Magnitude head: Linear output (no activation)")
        self.logger.info("  - Frequency head: Softplus activation (+1e-6)")
        self.logger.info("  - Removed BatchNorm from final output layers")
    
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
        Forward pass through the attention-enhanced model.
        
        Args:
            input_sequence: Input sequence tensor of shape (batch_size, lookback_years, input_seq_features)
            metadata: Bin metadata tensor of shape (batch_size, metadata_features)
            
        Returns:
            Tuple of (magnitude_pred, frequency_pred)
        """
        # LSTM processing
        lstm_out, _ = self.lstm1(input_sequence)
        lstm_out, (hidden, cell) = self.lstm2(lstm_out)
        
        # Apply attention over the sequence
        attended_seq = self.attention(lstm_out)
        
        # 🔧 REFACTOR: Replace attended_seq[:, -1, :] with mean pooling across timesteps
        # This uses all timesteps instead of just the last one, providing more stable representations
        lstm_final = torch.mean(attended_seq, dim=1)  # (batch_size, lstm_hidden_2)
        
        # Alternative: Weighted sum across timesteps (uncomment if preferred)
        # weights = F.softmax(torch.randn(attended_seq.size(1), device=attended_seq.device), dim=0)
        # lstm_final = torch.sum(attended_seq * weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        # Concatenate LSTM output with metadata
        combined = torch.cat([lstm_final, metadata], dim=1)
        
        # Dense layers
        dense_out = self.dense1(combined)
        dense_out = F.relu(dense_out)
        dense_out = self.dropout(dense_out)
        
        # Dual output heads
        magnitude_pred = self.magnitude_head(dense_out)
        frequency_pred = self.frequency_head(dense_out)  # Already has Softplus activation
        frequency_pred = frequency_pred + 1e-6  # Add epsilon to prevent zero values
        
        return magnitude_pred, frequency_pred
    
    def predict_frequency_counts(self, frequency_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert frequency predictions to expected count predictions.
        
        Args:
            frequency_pred: frequency predictions with Softplus activation (in log1p space)
            
        Returns:
            Expected count predictions converted from log1p space to raw counts
        """
        # Convert from log1p space back to raw counts using expm1
        # frequency_pred is log(1 + λ), so λ = expm1(frequency_pred)
        raw_counts = torch.expm1(frequency_pred)
        
        # Add small epsilon to prevent zero values
        return raw_counts + 1e-6
