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
                 input_seq_features: int = 12,
                 metadata_features: int = 4,
                 lookback_years: int = 10,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 magnitude_weight: float = 2.0,
                 frequency_weight: float = 1.0,  # FIXED: Changed from 0.5 to 1.0 for consistency
                 correlation_weight: float = 0.0,
                 device: str = 'auto',
                 output_dir: str = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_seq_features = input_seq_features
        self.metadata_features = metadata_features
        self.lookback_years = lookback_years
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.magnitude_weight = magnitude_weight
        self.frequency_weight = frequency_weight
        self.correlation_weight = correlation_weight
        self.device = device
        self.output_dir = output_dir

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

    def run_comparison(self, max_epochs: int = 300, patience: int = 15) -> Dict:
        """
        Run the complete comparison between Simple LSTM and Attention LSTM models.
        
        Args:
            max_epochs: Maximum training epochs for each model
            patience: Early stopping patience
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info("Starting Model Comparison: Simple LSTM vs Attention LSTM")
        self.logger.info("=" * 80)
        
        # Import models here to avoid circular imports
        from .shared_lstm_model import SharedLSTMModel
        from .attention_shared_lstm_model import AttentionSharedLSTMModel
        
        # Create both models
        simple_model = SharedLSTMModel(
            input_seq_features=self.input_seq_features,
            metadata_features=self.metadata_features,
            lookback_years=self.lookback_years,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=self.input_seq_features,
            metadata_features=self.metadata_features,
            lookback_years=self.lookback_years,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        # Train both models
        self.logger.info("Training Simple LSTM Model...")
        simple_results = self.train_model(simple_model, "SimpleLSTM", max_epochs, patience)
        
        self.logger.info("📚 Training Attention LSTM Model...")
        attention_results = self.train_model(attention_model, "AttentionSharedLSTM", max_epochs, patience)
        
        # Evaluate both models
        self.logger.info("Evaluating both models...")
        simple_metrics = self.evaluate_model(simple_model, "SimpleLSTM")
        attention_metrics = self.evaluate_model(attention_model, "AttentionSharedLSTM")
        
        # Compile comparison results
        comparison_results = {
            "simple_lstm": {
                "training_history": simple_results,
                "test_metrics": simple_metrics,
                "model": simple_model
            },
            "attention_lstm": {
                "training_history": attention_results,
                "test_metrics": attention_metrics,
                "model": attention_model
            },
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "magnitude_weight": self.magnitude_weight,
                "frequency_weight": self.frequency_weight,
                "correlation_weight": self.correlation_weight
            }
        }
        
        # Save comparison results if output_dir is specified
        if self.output_dir:
            self.save_comparison_results(comparison_results)
        
        return comparison_results
    
    def evaluate_model(self, model: nn.Module, model_name: str) -> Dict:
        """Evaluate a trained model on the test set."""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Create appropriate trainer for evaluation
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
                device=self.device
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
                device=self.device
            )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(self.test_loader)
        return test_metrics
    
    def save_comparison_results(self, results: Dict):
        """Save comparison results to files."""
        import json
        import torch
        from pathlib import Path
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(results["simple_lstm"]["model"].state_dict(), 
                  output_path / "simple_lstm_model.pth")
        torch.save(results["attention_lstm"]["model"].state_dict(), 
                  output_path / "attention_lstm_model.pth")
        
        # Save metrics
        with open(output_path / "comparison_metrics.json", 'w') as f:
            json.dump({
                "simple_lstm_metrics": self._convert_to_serializable(results["simple_lstm"]["test_metrics"]),
                "attention_lstm_metrics": self._convert_to_serializable(results["attention_lstm"]["test_metrics"]),
                "hyperparameters": self._convert_to_serializable(results["hyperparameters"])
            }, f, indent=2)
        
        # Save training histories
        with open(output_path / "simple_lstm_training_history.json", 'w') as f:
            json.dump(self._convert_to_serializable(results["simple_lstm"]["training_history"]), f, indent=2)
        
        with open(output_path / "attention_lstm_training_history.json", 'w') as f:
            json.dump(self._convert_to_serializable(results["attention_lstm"]["training_history"]), f, indent=2)
        
        self.logger.info(f"Comparison results saved to: {output_path}")
    
    def print_comparison_summary(self):
        """Print a summary of the comparison results."""
        self.logger.info("MODEL COMPARISON SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Hyperparameters used for both models:")
        self.logger.info(f"  Learning Rate: {self.learning_rate}")
        self.logger.info(f"  Weight Decay: {self.weight_decay}")
        self.logger.info(f"  Magnitude Weight: {self.magnitude_weight}")
        self.logger.info(f"  Frequency Weight: {self.frequency_weight}")
        self.logger.info(f"  Correlation Weight: {self.correlation_weight}")
        self.logger.info("")
        self.logger.info("Both models were trained with identical hyperparameters for fair comparison.")
        self.logger.info("Check the output directory for detailed results and model files.")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pytorch types to JSON-serializable Python types."""
        import numpy as np
        import torch
        
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj

    def train_model(self, model: nn.Module, model_name: str,
                    max_epochs: int = 300, patience: int = 15) -> Dict:  # FIXED: Changed from 12 to 15 for consistency
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

        # Run training (don't save checkpoints during comparison runs)
        training_history = trainer.train(
            max_epochs=max_epochs,
            save_path=None,
            save_best=False
        )

        self.logger.info(f"Finished training {model_name}. Best val_loss: {min(training_history['val_losses']):.4f}")
        return training_history
