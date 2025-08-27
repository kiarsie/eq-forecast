import torch
import torch.nn as nn
import logging
from typing import Tuple


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for earthquake forecasting.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: Tuple[int, ...] = (64, 48, 24, 24),
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super(SimpleLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.bidirectional = bidirectional
        
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
        
        # Final output layer
        final_hidden_size = hidden_sizes[-1] * self.bidirectional_multiplier
        self.output_layer = nn.Linear(final_hidden_size, 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SimpleLSTM initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
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
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Take the last output from the sequence
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Final output layer
        x = self.output_layer(x)
        
        return x
