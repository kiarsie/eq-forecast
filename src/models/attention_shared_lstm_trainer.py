#!/usr/bin/env python3
"""
Attention Shared LSTM Trainer

Identical to SharedLSTMTrainer, but runs with AttentionSharedLSTMModel
so that hyperparameters, optimizer, scheduler, and evaluation are
completely consistent for comparison purposes.
"""

from .shared_lstm_trainer import SharedLSTMTrainer
from .attention_shared_lstm_model import AttentionSharedLSTMModel


class AttentionSharedLSTMTrainer(SharedLSTMTrainer):
    def __init__(self,
                 model: AttentionSharedLSTMModel,
                 train_loader,
                 val_loader,
                 test_loader,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 magnitude_weight: float = 2.0,
                 frequency_weight: float = 1.0,  # FIXED: Changed from 0.5 to 1.0 for consistency
                 correlation_weight: float = 0.0,
                 device: str = 'auto',
                 save_dir: str = None):
        super().__init__(model,
                         train_loader,
                         val_loader,
                         test_loader,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         magnitude_weight=magnitude_weight,
                         frequency_weight=frequency_weight,
                         correlation_weight=correlation_weight,
                         device=device,
                         save_dir=save_dir)
        self.logger.info("AttentionSharedLSTMTrainer initialized with identical setup to SharedLSTMTrainer")
