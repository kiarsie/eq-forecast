import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output


class AttentionLSTM(nn.Module):
    """
    Attention-enhanced LSTM model for earthquake forecasting.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: Tuple[int, ...] = (64, 48, 24, 24),
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 num_attention_heads: int = 8):
        super(AttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_attention_heads = num_attention_heads
        
        # Calculate the multiplier for bidirectional LSTM
        self.bidirectional_multiplier = 2 if bidirectional else 1
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if len(hidden_sizes) > 1 else 0
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_sizes[i-1] * self.bidirectional_multiplier,
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if i < len(hidden_sizes) - 1 else 0
                )
            )
        
        # Attention mechanism over the sequence
        final_hidden_size = hidden_sizes[-1] * self.bidirectional_multiplier
        self.attention = MultiHeadAttention(
            embed_dim=final_hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Final output layer
        self.output_layer = nn.Linear(final_hidden_size, 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AttentionLSTM initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention-enhanced model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Take the last output from the sequence
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Final output layer
        x = self.output_layer(x)
        
        return x
