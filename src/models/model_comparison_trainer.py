import logging
import torch.nn as nn
from typing import Dict
from .shared_lstm_trainer import SharedLSTMTrainer
from .attention_shared_lstm_trainer import AttentionSharedLSTMTrainer


class ModelComparisonTrainer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 magnitude_weight: float = 2.0,
                 frequency_weight: float = 0.5,
                 correlation_weight: float = 0.0,
                 device: str = 'auto'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.magnitude_weight = magnitude_weight
        self.frequency_weight = frequency_weight
        self.correlation_weight = correlation_weight
        self.device = device

        # 🔹 set up logger
        self.logger = logging.getLogger("ModelComparisonTrainer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initialized ModelComparisonTrainer with identical hyperparameters.")


    def train_model(self, model: nn.Module, model_name: str,
                    max_epochs: int = 300, patience: int = 12) -> Dict:
        """
        Train either a SharedLSTM or AttentionSharedLSTM using the correct trainer.
        Returns the training history dictionary for plotting/analysis.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training for: {model_name}")
        self.logger.info(f"    - Learning Rate: {self.learning_rate}")
        self.logger.info(f"    - Weight Decay: {self.weight_decay}")
        self.logger.info(f"    - Loss Weights: mag={self.magnitude_weight}, "
                         f"freq={self.frequency_weight}, corr={self.correlation_weight}")
        self.logger.info(f"    - Device: {self.device}")
        self.logger.info("=" * 60)

        if "Attention" in model_name:
            trainer = AttentionSharedLSTMTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                magnitude_weight=self.magnitude_weight,
                frequency_weight=self.frequency_weight,
                correlation_weight=self.correlation_weight,
                device=self.device,
                save_dir=None  # No save_dir needed for comparison runs
            )
        else:
            trainer = SharedLSTMTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                magnitude_weight=self.magnitude_weight,
                frequency_weight=self.frequency_weight,
                correlation_weight=self.correlation_weight,
                device=self.device,
                save_dir=None  # No save_dir needed for comparison runs
            )

        # Run training (don’t save checkpoints during comparison runs)
        training_history = trainer.train(
            max_epochs=max_epochs,
            save_path=None,
            save_best=False
        )

        self.logger.info(f"Finished training {model_name}. Best val_loss: {min(training_history['val_losses']):.4f}")
        return training_history
