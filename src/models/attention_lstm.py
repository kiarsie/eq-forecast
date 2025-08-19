#!/usr/bin/env python3
"""
Attention LSTM Model for Earthquake Forecasting

Extends the Simple LSTM with attention mechanism:
- 4 hidden layers: 120, 90, 30, 30 neurons (same as Simple LSTM)
- Attention mechanism over the sequence
- Output layer: 1 neuron with sigmoid activation
- Input: 10-year lookback of earthquake variables
- Output: Forecasted earthquake variables (frequency, max magnitude)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionLSTM(nn.Module):
    """
    Attention LSTM model extending the Simple LSTM architecture.
    
    Architecture:
    - Same LSTM layers as Simple LSTM: 120, 90, 30, 30 neurons
    - Attention mechanism over the sequence
    - Output layer: 1 neuron with sigmoid
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: Tuple[int, int, int, int] = (32, 32, 16, 16),
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Initialize the Attention LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: Tuple of 4 hidden layer sizes (default: 120, 90, 30, 30)
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AttentionLSTM, self).__init__()
        
        if len(hidden_sizes) != 4:
            raise ValueError("hidden_sizes must be a tuple of exactly 4 values")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers following the paper's architecture (same as Simple LSTM)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0] * self.num_directions,
            hidden_size=hidden_sizes[1],
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0
        )
        
        self.lstm3 = nn.LSTM(
            input_size=hidden_sizes[1] * self.num_directions,
            hidden_size=hidden_sizes[2],
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0
        )
        
        self.lstm4 = nn.LSTM(
            input_size=hidden_sizes[2] * self.num_directions,
            hidden_size=hidden_sizes[3],
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
        # Attention mechanism
        # Ensure embed_dim is divisible by num_heads
        self.embed_dim = hidden_sizes[3] * self.num_directions
        
        # Find the largest number of heads that divides embed_dim evenly
        if self.embed_dim % 8 == 0:
            self.num_heads = 8  # 8 heads if divisible by 8
        elif self.embed_dim % 4 == 0:
            self.num_heads = 4  # 4 heads if divisible by 4
        elif self.embed_dim % 2 == 0:
            self.num_heads = 2  # 2 heads if divisible by 2
        else:
            self.num_heads = 1  # 1 head if not divisible by 2
        

            
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention output projection
        self.attention_projection = nn.Linear(
            hidden_sizes[3] * self.num_directions,
            hidden_sizes[3] * self.num_directions
        )
        
        # Output layer (2 neurons for frequency and magnitude)
        self.output_layer = nn.Linear(
            hidden_sizes[3] * self.num_directions, 
            2  # Changed from 1 to 2 for frequency and magnitude
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                else:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Attention LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 2) for frequency and magnitude
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM Layer 1: 120 neurons
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # LSTM Layer 2: 90 neurons
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # LSTM Layer 3: 30 neurons
        lstm3_out, (h3, c3) = self.lstm3(lstm2_out)
        lstm3_out = self.dropout3(lstm3_out)
        
        # LSTM Layer 4: 30 neurons
        lstm4_out, (h4, c4) = self.lstm4(lstm3_out)
        lstm4_out = self.dropout4(lstm4_out)
        
        # Apply attention mechanism
        # lstm4_out shape: (batch_size, seq_len, hidden_size[3] * num_directions)
        attended_output, attention_weights = self.attention(
            query=lstm4_out,
            key=lstm4_out,
            value=lstm4_out
        )
        
        # Project attention output
        attended_output = self.attention_projection(attended_output)
        
        # Global average pooling over sequence dimension
        # Shape: (batch_size, hidden_size[3] * num_directions)
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Pass through output layer (2 neurons for frequency and magnitude)
        output = self.output_layer(pooled_output)
        
        # Apply sigmoid activation as per the paper
        output = torch.sigmoid(output)
        return output
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM layers
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        lstm3_out, (h3, c3) = self.lstm3(lstm2_out)
        lstm3_out = self.dropout3(lstm3_out)
        
        lstm4_out, (h4, c4) = self.lstm4(lstm3_out)
        lstm4_out = self.dropout4(lstm4_out)
        
        # Attention mechanism
        attended_output, attention_weights = self.attention(
            query=lstm4_out,
            key=lstm4_out,
            value=lstm4_out
        )
        
        # Project attention output
        attended_output = self.attention_projection(attended_output)
        
        # Global average pooling
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Output layer
        output = self.output_layer(pooled_output)
        # Apply sigmoid activation as per the paper
        output = torch.sigmoid(output)
        
        return output, attention_weights
    
    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Get hidden states from all LSTM layers for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of hidden states from each LSTM layer
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM Layer 1
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # LSTM Layer 2
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # LSTM Layer 3
        lstm3_out, (h3, c3) = self.lstm3(lstm2_out)
        lstm3_out = self.dropout3(lstm3_out)
        
        # LSTM Layer 4
        lstm4_out, (h4, c4) = self.lstm4(lstm3_out)
        lstm4_out = self.dropout4(lstm4_out)
        
        return (lstm1_out, lstm2_out, lstm3_out, lstm4_out), (h1, h2, h3, h4), (c1, c2, c3, c4)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_type': 'AttentionLSTM',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters(),
            'architecture': {
                'lstm1': f"LSTM({self.input_size}, {self.hidden_sizes[0]})",
                'lstm2': f"LSTM({self.hidden_sizes[0] * self.num_directions}, {self.hidden_sizes[1]})",
                'lstm3': f"LSTM({self.hidden_sizes[1] * self.num_directions}, {self.hidden_sizes[2]})",
                'lstm4': f"LSTM({self.hidden_sizes[2] * self.num_directions}, {self.hidden_sizes[3]})",
                'attention': f"MultiheadAttention({self.embed_dim}, {self.num_heads} heads)",
                'output': f"Linear({self.embed_dim}, 2) for frequency and magnitude"
            }
        }
