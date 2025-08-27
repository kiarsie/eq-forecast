#!/usr/bin/env python3
"""
Attention Shared LSTM Trainer

Identical to SharedLSTMTrainer, but runs with AttentionSharedLSTMModel
so that hyperparameters, optimizer, scheduler, and evaluation are
completely consistent for comparison purposes.
"""

from .shared_lstm_trainer import SharedLSTMTrainer
from .attention_shared_lstm_model import AttentionSharedLSTMModel
from torch.utils.data import DataLoader


class AttentionSharedLSTMTrainer(SharedLSTMTrainer):
    def __init__(self,
                 model: AttentionSharedLSTMModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader = None,
                 learning_rate: float = 4e-4,      # Updated: was 5e-4, now 4e-4 for better performance
                 weight_decay: float = 5e-5,       # Updated: was 1e-4, now 5e-5 for better performance
                 magnitude_weight: float = 1.5,    # Updated: was 2.0, now 1.5 for better balance
                 frequency_weight: float = 2.0,    # Updated: was 1.0, now 2.0 for better frequency prediction
                 correlation_weight: float = 0.0,  # gamma: weight for correlation penalty (disabled)
                 variance_penalty_weight: float = 0.05,  # Weight for variance penalty
                 warmup_epochs: int = 20,              # Epochs before activating variance penalty
                 device: str = 'auto',
                 save_dir: str = None,
                 scaling_lr_multiplier: float = 8.0,
                 scaling_wd_multiplier: float = 1.0):
        """
        Initialize the attention trainer.
        
        REFACTOR: Updated loss weights to alpha=1.5, beta=2.0, gamma=0.0 for better performance.
        Based on successful previous run with these hyperparameters.
        """
        super().__init__(model,
                         train_loader,
                         val_loader,
                         test_loader,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         magnitude_weight=magnitude_weight,
                         frequency_weight=frequency_weight,
                         correlation_weight=correlation_weight,
                         variance_penalty_weight=variance_penalty_weight,
                         warmup_epochs=warmup_epochs,
                         device=device,
                         save_dir=save_dir,
                         scaling_lr_multiplier=scaling_lr_multiplier,
                         scaling_wd_multiplier=scaling_wd_multiplier)
        self.logger.info("AttentionSharedLSTMTrainer initialized with identical setup to SharedLSTMTrainer")
