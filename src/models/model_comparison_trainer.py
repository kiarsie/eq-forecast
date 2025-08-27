import logging
import torch.nn as nn
from typing import Dict
from pathlib import Path
from .shared_lstm_trainer import SharedLSTMTrainer
from .attention_shared_lstm_trainer import AttentionSharedLSTMTrainer
from torch.utils.data import DataLoader
import torch


class ModelComparisonTrainer:
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 learning_rate: float = 4e-4,      # Updated: was 5e-4, now 4e-4 for better performance
                 weight_decay: float = 5e-5,       # Updated: was 1e-4, now 5e-5 for better performance
                 magnitude_weight: float = 1.5,    # Updated: was 2.0, now 1.5 for better balance
                 frequency_weight: float = 2.0,    # Updated: was 1.0, now 2.0 for better frequency prediction
                 correlation_weight: float = 0.0,  # gamma: weight for correlation penalty (disabled)
                 device: str = 'auto',
                 output_dir: str = None,
                 config: dict = None):
        """Initialize the model comparison trainer."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.magnitude_weight = magnitude_weight
        self.frequency_weight = frequency_weight
        self.correlation_weight = correlation_weight
        self.device = device
        self.output_dir = output_dir
        self.config = config or {}
        
        # Setup logger first
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler for .log files
            if output_dir:
                log_dir = Path(output_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "training_comparison.log"
                file_handler = logging.FileHandler(log_file, mode='w')
                file_handler.setFormatter(console_formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Log file created at: {log_file}")
            
            self.logger.setLevel(logging.INFO)
        
        # 🔧 NEW: Store normalization parameters for consistency
        self.normalization_params = None
        if hasattr(train_loader.dataset, 'get_normalization_params'):
            self.normalization_params = train_loader.dataset.get_normalization_params()
            self.logger.info("Normalization parameters captured for consistency")
        
        # 🔧 NEW: Validate target ranges at initialization
        self.expected_target_ranges = None
        if hasattr(train_loader.dataset, 'validate_target_ranges'):
            self.expected_target_ranges = train_loader.dataset.validate_target_ranges('test')
            self.logger.info("Expected target ranges captured for validation")
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
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
        
        # Create both models with config values
        model_config = self.config.get('model_architecture', {})
        freq_scaling = self.config.get('frequency_scaling', {})
        
        # 🔧 FIX: Get feature dimensions from the dataset
        input_features, target_features, metadata_features = self.train_loader.dataset.get_feature_dimensions()
        lookback_years = 10  # Default value for comparison
        
        simple_model = SharedLSTMModel(
            input_seq_features=input_features,
            metadata_features=metadata_features,
            lookback_years=lookback_years,
            lstm_hidden_1=model_config.get('lstm_hidden_1', 64),
            lstm_hidden_2=model_config.get('lstm_hidden_2', 32),
            dense_hidden=model_config.get('dense_hidden', 32),
            dropout_rate=model_config.get('dropout_rate', 0.25),
            freq_head_type=model_config.get('freq_head_type', "linear"),
            frequency_scale_init=freq_scaling.get('frequency_scale_init', 2.0),
            frequency_bias_init=freq_scaling.get('frequency_bias_init', 0.5)
        )
        
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=input_features,
            metadata_features=metadata_features,
            lookback_years=lookback_years,
            lstm_hidden_1=model_config.get('lstm_hidden_1', 64),
            lstm_hidden_2=model_config.get('lstm_hidden_2', 32),
            dense_hidden=model_config.get('dense_hidden', 32),
            dropout_rate=model_config.get('dropout_rate', 0.25),
            freq_head_type=model_config.get('freq_head_type', "linear"),
            frequency_scale_init=freq_scaling.get('frequency_scale_init', 2.0),
            frequency_bias_init=freq_scaling.get('frequency_bias_init', 0.5)
        )
        
        # Train both models
        self.logger.info("Training Simple LSTM Model...")
        simple_results = self.train_model(simple_model, "SimpleLSTM", max_epochs, patience)
        
        self.logger.info("Training Attention LSTM Model...")
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
        
        # 🔧 NEW: Validate target ranges before evaluation
        if self.expected_target_ranges and hasattr(self.test_loader.dataset, 'assert_target_consistency'):
            try:
                self.test_loader.dataset.assert_target_consistency(self.expected_target_ranges)
                self.logger.info(f"[OK] Target range consistency validated for {model_name}")
            except ValueError as e:
                self.logger.error(f"[ERROR] Target range inconsistency for {model_name}: {e}")
                # Continue with evaluation but log the issue
        
        # Create appropriate trainer for evaluation
        freq_scaling = self.config.get('frequency_scaling', {})
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
                scaling_lr_multiplier=freq_scaling.get('scaling_lr_multiplier', 8.0),
                scaling_wd_multiplier=freq_scaling.get('scaling_wd_multiplier', 1.0)
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
                scaling_lr_multiplier=freq_scaling.get('scaling_lr_multiplier', 8.0),
                scaling_wd_multiplier=freq_scaling.get('scaling_wd_multiplier', 1.0)
            )
        
        # 🔧 NEW: Ensure consistent normalization parameters
        if self.normalization_params and hasattr(trainer, 'set_normalization_params'):
            trainer.set_normalization_params(self.normalization_params)
            self.logger.info(f"Normalization parameters synchronized for {model_name}")
        
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

        # Get frequency scaling config for trainers
        freq_scaling = self.config.get('frequency_scaling', {})
        
        # Get variance penalty config for trainers
        loss_weights = self.config.get('loss_weights', {})
        variance_penalty_weight = loss_weights.get('variance_penalty_weight', 0.05)
        warmup_epochs = loss_weights.get('warmup_epochs', 20)
        
        # Create save directory for this model
        if self.output_dir:
            model_save_dir = Path(self.output_dir) / f"{model_name.lower().replace(' ', '_')}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create individual log file for this model
            model_log_file = model_save_dir / f"{model_name.lower().replace(' ', '_')}_training.log"
            model_logger = logging.getLogger(f"{__name__}.{model_name}")
            if not model_logger.handlers:
                file_handler = logging.FileHandler(model_log_file, mode='w')
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(formatter)
                model_logger.addHandler(file_handler)
                model_logger.setLevel(logging.INFO)
                self.logger.info(f"Model-specific log file created at: {model_log_file}")
        else:
            model_save_dir = None
            model_logger = None
            
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
                variance_penalty_weight=variance_penalty_weight,
                warmup_epochs=warmup_epochs,
                device=self.device,
                save_dir=str(model_save_dir) if model_save_dir else None,
                scaling_lr_multiplier=freq_scaling.get('scaling_lr_multiplier', 8.0),
                scaling_wd_multiplier=freq_scaling.get('scaling_wd_multiplier', 1.0)
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
                variance_penalty_weight=variance_penalty_weight,
                warmup_epochs=warmup_epochs,
                device=self.device,
                save_dir=str(model_save_dir) if model_save_dir else None,
                scaling_lr_multiplier=freq_scaling.get('scaling_lr_multiplier', 8.0),
                scaling_wd_multiplier=freq_scaling.get('scaling_wd_multiplier', 1.0)
            )

        # Run training with saving enabled
        save_path = None
        if model_save_dir:
            save_path = model_save_dir / f"{model_name.lower().replace(' ', '_')}_best_model.pth"
            
        training_history = trainer.train(
            max_epochs=max_epochs,
            save_path=str(save_path) if save_path else None,
            save_best=True
        )

        self.logger.info(f"Finished training {model_name}. Best val_loss: {min(training_history['val_losses']):.4f}")
        
        # Save training history to file
        if model_save_dir:
            import json
            history_file = model_save_dir / f"{model_name.lower().replace(' ', '_')}_training_history.json"
            with open(history_file, 'w') as f:
                json.dump(self._convert_to_serializable(training_history), f, indent=2)
            self.logger.info(f"Training history saved to: {history_file}")
            
            # Save training history plot
            try:
                plot_dir = model_save_dir / "training_plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"{model_name.lower().replace(' ', '_')}_training_history.png"
                trainer.plot_training_history(str(plot_path))
                self.logger.info(f"Training history plot saved to: {plot_path}")
            except Exception as e:
                self.logger.warning(f"Could not save training history plot for {model_name}: {e}")
            
        return training_history
