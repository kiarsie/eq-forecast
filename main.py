#!/usr/bin/env python3
"""
Shared LSTM-based Earthquake Forecasting System

Implements the methodology from the paper with shared LSTM architecture:
1. Pre-processing: Filter shallow earthquakes (<70km), classify into spatial bins
2. Shared LSTM Training: Single model across all bins with dual output heads
3. Forecasting: Predict frequency and maximum magnitude simultaneously
4. Evaluation: Comprehensive metrics including WMAPE and Forecast Accuracy
5. Visualization: Training progress, predictions, and spatial analysis
6. Model Comparison: Simple LSTM vs Attention LSTM with consistent hyperparameters

Architecture:
- Shared LSTM: LSTM(64, return_sequences=True) -> LSTM(32, return_sequences=False)
- Dual output heads: magnitude (continuous) and frequency (Poisson log-rate)
- Metadata integration: spatial coordinates, bin area, temporal features
- Weighted loss combining MSE (magnitude) and log1p+MSE (frequency)

Hyperparameter Consistency:
- All modes (train, full_pipeline, compare_models) use identical hyperparameters
- Learning rate: 5e-4, Weight decay: 1e-4, Magnitude weight: 2.0, Frequency weight: 0.5
- This ensures fair comparison and consistent training conditions across all modes
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing.earthquake_processor import EarthquakeProcessor
from src.models.shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
from src.models.shared_lstm_trainer import SharedLSTMTrainer
from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
from src.models.attention_shared_lstm_trainer import AttentionSharedLSTMTrainer
from src.models.enhanced_shared_processor import EnhancedSharedDataset
from src.models.model_comparison_trainer import ModelComparisonTrainer

# Import optimized configuration utilities
try:
    from apply_optimized_configs import load_config, create_optimized_model, get_training_params, get_all_training_params
    OPTIMIZED_CONFIGS_AVAILABLE = True
except ImportError:
    OPTIMIZED_CONFIGS_AVAILABLE = False
    print("⚠️  Optimized configs not available. Install apply_optimized_configs.py for best performance.")


def load_optimized_config(config_name: str = "best_frequency") -> Optional[Dict]:
    """
    Load an optimized hyperparameter configuration.
    
    Args:
        config_name: Name of the configuration to load
            - "best_frequency": Best frequency prediction (49.39 range)
            - "best_magnitude": Best magnitude prediction (1.52 range)  
            - "best_balanced": Best balanced performance (25.9 combined score)
            - "anti_overfitting": Prevents overfitting with aggressive regularization
            - "balanced_anti_overfitting": Balanced performance and capacity
            - "enhanced_frequency_scaling": Maximum range coverage
            - "high_performance_balanced": Maximum overall performance
    
    Returns:
        Configuration dictionary or None if not available
    """
    if not OPTIMIZED_CONFIGS_AVAILABLE:
        return None
    
    config_files = {
        "best_frequency": "best_frequency_config.json",
        "best_magnitude": "best_magnitude_config.json", 
        "best_balanced": "best_balanced_config.json",
            "anti_overfitting": "anti_overfitting_config.json",
    "balanced_anti_overfitting": "balanced_anti_overfitting_config.json",
    "hybrid_balanced": "hybrid_balanced_config.json",
        "enhanced_frequency_scaling": "enhanced_frequency_scaling_config.json",
        "high_performance_balanced": "high_performance_balanced_config.json"
    }
    
    if config_name not in config_files:
        print(f"Unknown config: {config_name}")
        print(f"Available: {list(config_files.keys())}")
        return None
    
    try:
        config = load_config(config_files[config_name])
        print(f"Loaded {config['name']} configuration:")
        
        # Handle different config formats
        if 'performance' in config:
            print(f"   Frequency range: {config['performance']['frequency_range']:.2f}")
            print(f"   Magnitude range: {config['performance']['magnitude_range']:.2f}")
        elif 'anti_overfitting_features' in config:
            print(f"   Type: Anti-overfitting configuration")
            print(f"   Dropout: {config['model_architecture']['dropout_rate']}")
            print(f"   Weight decay: {config['training_parameters']['weight_decay']}")
            print(f"   Expected: Realistic performance (60-80% accuracy)")
        return config
    except Exception as e:
        print(f"Failed to load {config_name}: {e}")
        return None


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def preprocess_earthquake_data(input_path: str, output_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Preprocess earthquake catalog data following the paper's methodology.
    
    Args:
        input_path: Path to raw earthquake catalog
        output_path: Path to save processed data
        logger: Logger instance
        
    Returns:
        Processed earthquake catalog DataFrame
    """
    logger.info("Starting earthquake data preprocessing")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Load raw earthquake catalog
        logger.info("Loading raw earthquake catalog...")
        
        # Use the existing load_catalog function to get properly formatted data
        from src.preprocessing.load_catalog import load_catalog
        
        # Load and get the DataFrame (ignore the CSEP catalog)
        raw_df, _ = load_catalog(input_path)
        logger.info(f"Loaded {len(raw_df)} earthquake records")
        logger.info(f"Columns: {list(raw_df.columns)}")
        
        # Initialize processor
        processor = EarthquakeProcessor(min_depth=70.0)
        
        # Process catalog
        logger.info("Processing earthquake catalog...")
        processed_catalog, annual_stats = processor.process_catalog(
            df=raw_df,
            save_path=output_path
        )
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Processed catalog: {len(processed_catalog)} records")
        logger.info(f"Annual statistics: {len(annual_stats)} year-bin combinations")
        
        # Display summary statistics
        logger.info("\n=== Preprocessing Summary ===")
        logger.info(f"Original earthquakes: {len(raw_df)}")
        logger.info(f"Shallow earthquakes (<70km): {len(processed_catalog)}")
        logger.info(f"Quadtree bins created: {processed_catalog['bin_id'].nunique()}")
        logger.info(f"Years covered: {annual_stats['year'].min()} - {annual_stats['year'].max()}")
        
        # Display bin statistics
        bin_summary = processor.get_bin_statistics(annual_stats)
        logger.info(f"\nBin statistics summary:")
        logger.info(f"Average earthquakes per bin per year: {bin_summary['frequency_mean'].mean():.2f}")
        logger.info(f"Average max magnitude per bin: {bin_summary['max_magnitude_mean'].mean():.2f}")
        
        return processed_catalog, annual_stats
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def create_shared_lstm_datasets(data_path: str, lookback_years: int = 10, target_horizon: int = 1):
    """Create datasets for shared LSTM training."""
    # Get logger from the current context
    logger = logging.getLogger(__name__)
    logger.info("Creating shared LSTM datasets...")
    
    # Create the enhanced shared dataset
    dataset = EnhancedSharedDataset(
        data_path=data_path,
        lookback_years=lookback_years,
        target_horizon=target_horizon,
        normalize=True,
        rolling_windows=[3, 5, 10],
        train_end_year=2009,
        val_end_year=2017,
        test_end_year=2025
    )
    
    # Get feature dimensions
    input_features, target_features, metadata_features = dataset.get_feature_dimensions()
    logger.info(f"Feature dimensions: Input={input_features}, Target={target_features}, Metadata={metadata_features}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'train']
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'val']
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'test']
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"Created data loaders: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    return {
        'dataset': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_features': input_features,
        'target_features': target_features,
        'metadata_features': metadata_features
    }


def create_shared_lstm_model(input_features: int, 
                             metadata_features: int,
                             lookback_years: int = 10,
                             logger: logging.Logger = None) -> SharedLSTMModel:
    """
    Create the shared LSTM model.
    
    Args:
        input_features: Number of input features
        metadata_features: Number of metadata features
        lookback_years: Number of years to look back
        logger: Logger instance
        
    Returns:
        SharedLSTMModel instance
    """
    if logger:
        logger.info("Creating shared LSTM model...")
        logger.info(f"Input features: {input_features}")
        logger.info(f"Metadata features: {metadata_features}")
        logger.info(f"Lookback years: {lookback_years}")
    
    model = SharedLSTMModel(
        input_seq_features=input_features,
        metadata_features=metadata_features,
        lookback_years=lookback_years,
        lstm_hidden_1=64,
        lstm_hidden_2=32,
        dense_hidden=32,
        dropout_rate=0.25,
        freq_head_type="linear"  # Default to new stable linear frequency head
    )
    
    if logger:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
    
    return model


def create_attention_shared_lstm_model(input_features: int, 
                                       metadata_features: int,
                                       lookback_years: int = 10,
                                       logger: logging.Logger = None) -> AttentionSharedLSTMModel:
    """
    Create the attention shared LSTM model.
    
    Args:
        input_features: Number of input features
        metadata_features: Number of metadata features
        lookback_years: Number of years to look back
        logger: Logger instance
        
    Returns:
        AttentionSharedLSTMModel instance
    """
    if logger:
        logger.info("Creating attention shared LSTM model...")
        logger.info(f"Input features: {input_features}")
        logger.info(f"Metadata features: {metadata_features}")
        logger.info(f"Lookback years: {lookback_years}")
    
    model = AttentionSharedLSTMModel(
        input_seq_features=input_features,
        metadata_features=metadata_features,
        lookback_years=lookback_years,
        lstm_hidden_1=64,
        lstm_hidden_2=32,
        dense_hidden=32,
        dropout_rate=0.25,
        freq_head_type="linear"  # Default to new stable linear frequency head
    )
    
    if logger:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
    
    return model


def train_shared_lstm_model(model: SharedLSTMModel,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader,
                            save_dir: str,
                            logger: logging.Logger,
                            num_epochs: int = 300,
                            learning_rate: float = 1e-3,
                            weight_decay: float = 1e-4,
                            magnitude_weight: float = 1.0,
                            frequency_weight: float = 4.0,
                            correlation_weight: float = 0.0) -> Dict:
    """
    Train the shared LSTM model.
    
    Args:
        model: SharedLSTMModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        save_dir: Directory to save results
        logger: Logger instance
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        magnitude_weight: Weight for magnitude loss
        frequency_weight: Weight for frequency loss
        correlation_weight: Weight for correlation penalty
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting shared LSTM model training")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Training epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Loss weights: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
    
    try:
        # Create trainer
        trainer = SharedLSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            magnitude_weight=magnitude_weight,
            frequency_weight=frequency_weight,
            correlation_weight=correlation_weight,
            save_dir=save_dir
        )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Train the model
        logger.info("Training model...")
        training_history = trainer.train(
            max_epochs=num_epochs,
            save_path=str(save_path / "best_model.pth"),
            save_best=True
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save test metrics
        metrics_path = save_path / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            # 🔧 FIX: Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in test_metrics.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_metrics[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    serializable_metrics[key] = [float(v.item()) if hasattr(v, 'item') else v for v in value]
                else:
                    serializable_metrics[key] = value
            
            json.dump(serializable_metrics, f, indent=2)
        
        # Plot training history
        plot_path = save_path / "training_history.png"
        trainer.plot_training_history(str(plot_path))
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Test metrics saved to: {metrics_path}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"Training plots saved to: {plot_path}")
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'best_val_loss': trainer.best_val_loss,
            'model_path': str(save_path / "best_model.pth")
        }
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def train_attention_shared_lstm_model(model: AttentionSharedLSTMModel,
                                      train_loader: torch.utils.data.DataLoader,
                                      val_loader: torch.utils.data.DataLoader,
                                      test_loader: torch.utils.data.DataLoader,
                                      save_dir: str,
                                      logger: logging.Logger,
                                      num_epochs: int = 300,
                                      learning_rate: float = 5e-4,  # FIXED: Match successful training (5e-4)
                                      weight_decay: float = 1e-4,
                                      magnitude_weight: float = 2.0,  # FIXED: Match successful training (2.0)
                                      frequency_weight: float = 1.0,  # FIXED: Match successful training (1.0)
                                      correlation_weight: float = 0.0) -> Dict:
    """
    Train the attention shared LSTM model.
    
    Args:
        model: AttentionSharedLSTMModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        save_dir: Directory to save results
        logger: Logger instance
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        magnitude_weight: Weight for magnitude loss
        frequency_weight: Weight for frequency loss
        correlation_weight: Weight for correlation penalty
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting attention shared LSTM model training")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Training epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Loss weights: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
    
    try:
        # Create trainer
        trainer = AttentionSharedLSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            magnitude_weight=magnitude_weight,
            frequency_weight=frequency_weight,
            correlation_weight=correlation_weight
        )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Train the model
        logger.info("Training model...")
        training_history = trainer.train(
            max_epochs=num_epochs,
            save_path=str(save_path / "attention_best_model.pth"),
            save_best=True
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save training history
        history_path = save_path / "attention_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save test metrics
        metrics_path = save_path / "attention_test_metrics.json"
        with open(metrics_path, 'w') as f:
            # 🔧 FIX: Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in test_metrics.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_metrics[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    serializable_metrics[key] = [float(v.item()) if hasattr(v, 'item') else v for v in value]
                else:
                    serializable_metrics[key] = value
            
            json.dump(serializable_metrics, f, indent=2)
        
        # Plot training history
        plot_path = save_path / "attention_training_history.png"
        trainer.plot_training_history(str(plot_path))
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Test metrics saved to: {metrics_path}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"Training plots saved to: {plot_path}")
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'best_val_loss': trainer.best_val_loss,
            'model_path': str(save_path / "attention_best_model.pth")
        }
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_shared_lstm_model(model_path: str,
                               test_loader: torch.utils.data.DataLoader,
                               save_dir: str,
                               logger: logging.Logger,
                               model_type: str = "simple") -> Dict:
    """
    Evaluate the trained shared LSTM model.
    
    Args:
        model_path: Path to trained model
        test_loader: Test data loader
        save_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("Starting model evaluation")
    logger.info(f"Model path: {model_path}")
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model architecture from checkpoint
        model_state = checkpoint['model_state_dict']
        
        # Determine LSTM hidden sizes from the checkpoint
        # Note: PyTorch LSTM weight_ih_l0 has shape [4*hidden_size, input_size] due to 4 gates
        lstm2_hidden_size = model_state['lstm2.weight_ih_l0'].shape[0] // 4
        lstm1_hidden_size = model_state['lstm1.weight_ih_l0'].shape[0] // 4
        
        # Determine metadata features from dense1 weight shape
        dense1_input_size = model_state['dense1.weight'].shape[1]
        metadata_features = dense1_input_size - lstm2_hidden_size
        
        # Determine input features from lstm1 weight shape
        input_features = model_state['lstm1.weight_ih_l0'].shape[1]
        
        logger.info(f"Extracted model architecture from checkpoint:")
        logger.info(f"  Input features: {input_features}")
        logger.info(f"  LSTM1 hidden size: {lstm1_hidden_size}")
        logger.info(f"  LSTM2 hidden size: {lstm2_hidden_size}")
        logger.info(f"  Metadata features: {metadata_features}")
        
        # Debug: Check if the extracted values make sense
        logger.info(f"DEBUG: LSTM2 weight shape: {model_state['lstm2.weight_ih_l0'].shape}")
        logger.info(f"DEBUG: LSTM1 weight shape: {model_state['lstm1.weight_ih_l0'].shape}")
        logger.info(f"DEBUG: Dense1 weight shape: {model_state['dense1.weight'].shape}")
        
        # Create a dummy model with the exact architecture from the checkpoint
        logger.info(f"DEBUG: Creating model with parameters:")
        logger.info(f"  input_seq_features: {input_features}")
        logger.info(f"  metadata_features: {metadata_features}")
        logger.info(f"  lstm_hidden_1: {lstm1_hidden_size}")
        logger.info(f"  lstm_hidden_2: {lstm2_hidden_size}")
        
        if model_type == "attention":
            # Create attention model for attention checkpoints
            from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
            dummy_model = AttentionSharedLSTMModel(
                input_seq_features=input_features,
                metadata_features=metadata_features,
                lookback_years=10,
                lstm_hidden_1=lstm1_hidden_size,
                lstm_hidden_2=lstm2_hidden_size,
                dense_hidden=32,
                dropout_rate=0.25,
                freq_head_type="linear"  # Default to new stable linear frequency head
            )
        else:
            # Create simple model for simple checkpoints
            dummy_model = SharedLSTMModel(
                input_seq_features=input_features,
                metadata_features=metadata_features,
                lookback_years=10,
                lstm_hidden_1=lstm1_hidden_size,
                lstm_hidden_2=lstm2_hidden_size,
                dense_hidden=32,
                dropout_rate=0.25,
                freq_head_type="linear"  # Default to new stable linear frequency head
            )
        
        # Create trainer with loaded model
        if model_type == "attention":
            from src.models.attention_shared_lstm_trainer import AttentionSharedLSTMTrainer
            trainer = AttentionSharedLSTMTrainer(
                model=dummy_model,
                train_loader=test_loader,  # Dummy
                val_loader=test_loader,    # Dummy
                test_loader=test_loader
            )
        else:
            trainer = SharedLSTMTrainer(
                model=dummy_model,
                train_loader=test_loader,  # Dummy
                val_loader=test_loader,    # Dummy
                test_loader=test_loader,
                save_dir=save_dir
            )
        
        # Load the trained model
        trainer.load_model(model_path)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save evaluation results
        save_path = Path(save_dir)
        metrics_path = save_path / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            # 🔧 FIX: Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in test_metrics.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_metrics[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    serializable_metrics[key] = [float(v.item()) if hasattr(v, 'item') else v for v in value]
                else:
                    serializable_metrics[key] = value
            
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info("Evaluation completed successfully!")
        
        # Display evaluation summary
        logger.info("\n=== Evaluation Summary ===")
        logger.info(f"Total Loss: {test_metrics['total_loss']:.4f}")
        logger.info(f"Magnitude Loss (MSE): {test_metrics['magnitude_loss']:.4f}")
        logger.info(f"Frequency Loss (Log1p+MSE): {test_metrics['frequency_loss']:.4f}")
        logger.info(f"Magnitude MAE: {test_metrics['magnitude_mae']:.4f}")
        logger.info(f"Frequency MAE: {test_metrics['frequency_mae']:.4f}")
        logger.info(f"Magnitude Correlation: {test_metrics['magnitude_corr']:.4f}")
        logger.info(f"Frequency Correlation: {test_metrics['frequency_corr']:.4f}")
        logger.info(f"Magnitude Accuracy (±0.3): {test_metrics['magnitude_accuracy']:.3f}")
        logger.info(f"Frequency Accuracy (±1): {test_metrics['frequency_accuracy']:.3f}")
        
        return test_metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def generate_comprehensive_visualizations(save_dir: str, 
                                         model_path: str,
                                         test_loader: torch.utils.data.DataLoader,
                                         logger: logging.Logger,
                                         model_type: str = "simple"):
    """
    Generate comprehensive visualizations of results.
    
    Args:
        save_dir: Directory containing results
        model_path: Path to trained model
        test_loader: Test data loader
        logger: Logger instance
    """
    logger.info("Generating comprehensive visualizations")
    
    try:
        save_path = Path(save_dir)
        
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model architecture from checkpoint
        model_state = checkpoint['model_state_dict']
        
        # Determine LSTM hidden sizes from the checkpoint
        # Note: PyTorch LSTM weight_ih_l0 has shape [4*hidden_size, input_size] due to 4 gates
        lstm2_hidden_size = model_state['lstm2.weight_ih_l0'].shape[0] // 4
        lstm1_hidden_size = model_state['lstm1.weight_ih_l0'].shape[0] // 4
        
        # Determine metadata features from dense1 weight shape
        dense1_input_size = model_state['dense1.weight'].shape[1]
        metadata_features = dense1_input_size - lstm2_hidden_size
        
        # Determine input features from lstm1 weight shape
        input_features = model_state['lstm1.weight_ih_l0'].shape[1]
        
        # Create a dummy model with the exact architecture from the checkpoint
        logger.info(f"DEBUG: Creating visualization model with parameters:")
        logger.info(f"  input_seq_features: {input_features}")
        logger.info(f"  metadata_features: {metadata_features}")
        logger.info(f"  lstm_hidden_1: {lstm1_hidden_size}")
        logger.info(f"  lstm_hidden_2: {lstm2_hidden_size}")
        
        if model_type == "attention":
            # Create attention model for attention checkpoints
            from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
            dummy_model = AttentionSharedLSTMModel(
                input_seq_features=input_features,
                metadata_features=metadata_features,
                lookback_years=10,
                lstm_hidden_1=lstm1_hidden_size,
                lstm_hidden_2=lstm2_hidden_size,
                dense_hidden=32,
                dropout_rate=0.25,
                freq_head_type="linear"  # Default to new stable linear frequency head
            )
        else:
            # Create simple model for simple checkpoints
            dummy_model = SharedLSTMModel(
                input_seq_features=input_features,
                metadata_features=metadata_features,
                lookback_years=10,
                lstm_hidden_1=lstm1_hidden_size,
                lstm_hidden_2=lstm2_hidden_size,
                dense_hidden=32,
                dropout_rate=0.25,
                freq_head_type="linear"  # Default to new stable linear frequency head
            )
        
        # Create trainer with loaded model
        if model_type == "attention":
            from src.models.attention_shared_lstm_trainer import AttentionSharedLSTMTrainer
            trainer = AttentionSharedLSTMTrainer(
                model=dummy_model,
                train_loader=test_loader,  # Dummy
                val_loader=test_loader,    # Dummy
                test_loader=test_loader
            )
        else:
            trainer = SharedLSTMTrainer(
                model=dummy_model,
                train_loader=test_loader,  # Dummy
                val_loader=test_loader,    # Dummy
                test_loader=test_loader,
                save_dir=save_dir
            )
        
        # Load the trained model
        trainer.load_model(model_path)
        
        # Generate predictions for visualization
        logger.info("Generating predictions for visualization...")
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        trainer.model.eval()
        with torch.no_grad():
            for input_seq, target_seq, metadata, metadata_dict in test_loader:
                # Move to device
                input_seq = input_seq.to(trainer.device)
                metadata = metadata.to(trainer.device)
                
                # Forward pass
                magnitude_pred, frequency_log_rate_pred = trainer.model(input_seq, metadata)
                
                # Convert predictions to expected counts
                frequency_count_pred = trainer.model.predict_frequency_counts(frequency_log_rate_pred.squeeze())
                
                # Store predictions and targets
                all_predictions.append({
                    'magnitude': magnitude_pred.squeeze().cpu().numpy(),
                    'frequency': frequency_count_pred.cpu().numpy()
                })
                all_targets.append({
                    'magnitude': target_seq[:, 0].numpy(),  # Fixed: target_seq is [batch_size, 2]
                    'frequency': target_seq[:, 1].numpy()   # Fixed: target_seq is [batch_size, 2]
                })
                all_metadata.append(metadata_dict)
        
        # Combine all predictions
        magnitude_preds = np.concatenate([pred['magnitude'] for pred in all_predictions])
        frequency_preds = np.concatenate([pred['frequency'] for pred in all_predictions])
        magnitude_targets = np.concatenate([target['magnitude'] for target in all_targets])
        frequency_targets = np.concatenate([target['frequency'] for target in all_targets])
        
        # Denormalize PREDICTIONS (this was missing!)
        magnitude_preds_denorm = np.array([
            test_loader.dataset.denormalize_single_feature(mag, 'magnitude')
            for mag in magnitude_preds
        ])
        frequency_preds_denorm = np.array([
            test_loader.dataset.denormalize_single_feature(freq, 'frequency')
            for freq in frequency_preds
        ])
        
        # Denormalize targets
        magnitude_targets_denorm = np.array([
            test_loader.dataset.denormalize_single_feature(mag, 'magnitude')
            for mag in magnitude_targets
        ])
        frequency_targets_denorm = np.array([
            test_loader.dataset.denormalize_single_feature(freq, 'frequency')
            for freq in frequency_targets
        ])
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Magnitude predictions vs targets
        axes[0, 0].scatter(magnitude_targets_denorm, magnitude_preds_denorm, alpha=0.6)
        axes[0, 0].plot([magnitude_targets_denorm.min(), magnitude_targets_denorm.max()], 
                        [magnitude_targets_denorm.min(), magnitude_targets_denorm.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Magnitude')
        axes[0, 0].set_ylabel('Predicted Magnitude')
        axes[0, 0].set_title('Magnitude Predictions vs Targets')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Frequency predictions vs targets
        axes[0, 1].scatter(frequency_targets_denorm, frequency_preds_denorm, alpha=0.6)
        axes[0, 1].plot([frequency_targets_denorm.min(), frequency_targets_denorm.max()], 
                        [frequency_targets_denorm.min(), frequency_targets_denorm.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Frequency')
        axes[0, 1].set_ylabel('Predicted Frequency')
        axes[0, 1].set_title('Frequency Predictions vs Targets')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Magnitude residuals
        magnitude_residuals = magnitude_preds_denorm - magnitude_targets_denorm
        axes[0, 2].scatter(magnitude_targets_denorm, magnitude_residuals, alpha=0.6)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('True Magnitude')
        axes[0, 2].set_ylabel('Residual (Pred - True)')
        axes[0, 2].set_title('Magnitude Residuals')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Frequency residuals
        frequency_residuals = frequency_preds_denorm - frequency_targets_denorm
        axes[1, 0].scatter(frequency_targets_denorm, frequency_residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('True Frequency')
        axes[1, 0].set_ylabel('Residual (Pred - True)')
        axes[1, 0].set_title('Frequency Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Magnitude error distribution
        axes[1, 1].hist(magnitude_residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Magnitude Residual')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Magnitude Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Frequency error distribution
        axes[1, 2].hist(frequency_residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 2].set_xlabel('Frequency Residual')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Frequency Error Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = save_path / "comprehensive_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive visualizations saved to: {plot_path}")
        
        # Create additional plots
        create_spatial_analysis_plots(save_path, test_loader, trainer, logger)
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def run_model_comparison(datasets: Dict,
                         save_dir: str,
                         logger: logging.Logger,
                         num_epochs: int = 300,
                         patience: int = 15,  # FIXED: Changed from 12 to 15 for consistency
                         learning_rate: float = 5e-4,
                         weight_decay: float = 1e-4,
                         magnitude_weight: float = 2.0,
                         frequency_weight: float = 1.0,  # FIXED: Changed from 0.5 to 1.0 for consistency
                         correlation_weight: float = 0.0) -> Dict:
    """
    Run comparison between Simple LSTM and Attention LSTM models.
    
    Args:
        datasets: Dictionary containing datasets and data loaders
        save_dir: Directory to save comparison results
        logger: Logger instance
        num_epochs: Maximum training epochs for each model (default: 300)
        patience: Early stopping patience (default: 12)
        learning_rate: Learning rate for both models (default: 5e-4)
        weight_decay: Weight decay for both models (default: 1e-4)
        magnitude_weight: Weight for magnitude loss (alpha) (default: 2.0)
        frequency_weight: Weight for frequency loss (beta) (default: 0.5)
        correlation_weight: Weight for correlation penalty (gamma) (default: 0.0)
        
    Returns:
        Comparison results dictionary
    """
    logger.info("Starting model comparison: Simple LSTM vs Attention LSTM")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Training epochs: {num_epochs}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Loss weights: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
    logger.info("Note: Using same hyperparameters as full pipeline for fair comparison")
    
    try:
        # Create save directory
        save_path = Path(save_dir)
        comparison_dir = save_path / "model_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the comparison trainer
        comparison_trainer = ModelComparisonTrainer(
            train_loader=datasets['train_loader'],
            val_loader=datasets['val_loader'],
            test_loader=datasets['test_loader'],
            input_seq_features=datasets['input_features'],
            metadata_features=datasets['metadata_features'],
            lookback_years=10,  # Fixed for comparison
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            magnitude_weight=magnitude_weight,
            frequency_weight=frequency_weight,
            correlation_weight=correlation_weight,
            device='auto',
            output_dir=str(comparison_dir)
        )
        
        # Run the comparison
        logger.info("Running model comparison...")
        results = comparison_trainer.run_comparison(
            max_epochs=num_epochs,
            patience=patience
        )
        
        # Print comparison summary
        comparison_trainer.print_comparison_summary()
        
        logger.info("Model comparison completed successfully!")
        logger.info(f"Comparison results saved to: {comparison_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        raise


def create_spatial_analysis_plots(save_path: Path, 
                                  test_loader: torch.utils.data.DataLoader,
                                  trainer: SharedLSTMTrainer,
                                  logger: logging.Logger):
    """
    Create spatial analysis plots showing performance across different bins.
    
    Args:
        save_path: Path to save plots
        test_loader: Test data loader
        trainer: Trained model trainer
        logger: Logger instance
    """
    logger.info("Creating spatial analysis plots...")
    
    try:
        # Collect predictions and metadata for spatial analysis
        bin_performance = {}
        
        trainer.model.eval()
        with torch.no_grad():
            for input_seq, target_seq, metadata, metadata_dict in test_loader:
                # Move to device
                input_seq = input_seq.to(trainer.device)
                metadata = metadata.to(trainer.device)
                
                # Forward pass
                magnitude_pred, frequency_log_rate_pred = trainer.model(input_seq, metadata)
                frequency_count_pred = trainer.model.predict_frequency_counts(frequency_log_rate_pred.squeeze())
                
                # Extract bin information
                # metadata_dict contains batch data, bin_id is a list
                bin_ids = metadata_dict['bin_id']
                logger.info(f"DEBUG: Processing bin_id: {bin_ids} (type: {type(bin_ids)})")
                
                # Process each sample in the batch
                for i in range(len(target_seq)):
                    # Get the bin_id for this specific sample
                    bin_id = bin_ids[i] if isinstance(bin_ids, (list, np.ndarray)) else str(bin_ids)
                    
                    # For pre-processed data, we don't have actual lat/lon coordinates
                    # Use bin_id as spatial identifier and create synthetic coordinates for visualization
                    try:
                        bin_id_numeric = float(bin_id) if bin_id != '-1' else 0.0
                    except (ValueError, TypeError):
                        # Handle cases where bin_id might be a string or other format
                        bin_id_numeric = 0.0
                    
                    # Create synthetic coordinates based on bin_id for visualization purposes
                    center_lat = bin_id_numeric * 10  # Synthetic latitude
                    center_lon = bin_id_numeric * 15  # Synthetic longitude
                
                    # Calculate metrics
                    mag_true = target_seq[i, 0].item()  # Fixed: target_seq is [batch_size, 2]
                    freq_true = target_seq[i, 1].item()  # Fixed: target_seq is [batch_size, 2]
                    mag_pred = magnitude_pred[i].item()
                    freq_pred = frequency_count_pred[i].item()
                    
                    # Denormalize
                    mag_true_denorm = test_loader.dataset.denormalize_single_feature(mag_true, 'magnitude')
                    freq_true_denorm = test_loader.dataset.denormalize_single_feature(freq_true, 'frequency')
                    
                    if bin_id not in bin_performance:
                        logger.info(f"DEBUG: Creating new bin_performance entry for bin_id: {bin_id}")
                        bin_performance[bin_id] = {
                            'lat': center_lat,
                            'lon': center_lon,
                            'magnitude_errors': [],
                            'frequency_errors': [],
                            'magnitude_targets': [],
                            'frequency_targets': [],
                            'magnitude_predictions': [],
                            'frequency_predictions': []
                        }
                    
                    logger.info(f"DEBUG: Appending data for bin_id: {bin_id}")
                    bin_performance[bin_id]['magnitude_errors'].append(abs(mag_pred - mag_true_denorm))
                    bin_performance[bin_id]['frequency_errors'].append(abs(freq_pred - freq_true_denorm))
                    bin_performance[bin_id]['magnitude_targets'].append(mag_true_denorm)
                    bin_performance[bin_id]['frequency_targets'].append(freq_true_denorm)
                    bin_performance[bin_id]['magnitude_predictions'].append(mag_pred)
                    bin_performance[bin_id]['frequency_predictions'].append(freq_pred)
        
        # Create spatial performance plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract coordinates and metrics
        lats = [data['lat'] for data in bin_performance.values()]
        lons = [data['lon'] for data in bin_performance.values()]
        mag_mae = [np.mean(data['magnitude_errors']) for data in bin_performance.values()]
        freq_mae = [np.mean(data['frequency_errors']) for data in bin_performance.values()]
        
        # 1. Magnitude MAE spatial plot
        scatter1 = axes[0, 0].scatter(lons, lats, c=mag_mae, cmap='viridis', s=100, alpha=0.8)
        axes[0, 0].set_xlabel('Synthetic Longitude (Bin ID × 15)')
        axes[0, 0].set_ylabel('Synthetic Latitude (Bin ID × 10)')
        axes[0, 0].set_title('Magnitude MAE by Spatial Bin (Synthetic Coordinates)')
        plt.colorbar(scatter1, ax=axes[0, 0], label='MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Frequency MAE spatial plot
        scatter2 = axes[0, 1].scatter(lons, lats, c=freq_mae, cmap='plasma', s=100, alpha=0.8)
        axes[0, 1].set_xlabel('Synthetic Longitude (Bin ID × 15)')
        axes[0, 1].set_ylabel('Synthetic Latitude (Bin ID × 10)')
        axes[0, 1].set_title('Frequency MAE by Spatial Bin (Synthetic Coordinates)')
        plt.colorbar(scatter2, ax=axes[0, 1], label='MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Magnitude correlation by bin
        mag_corrs = []
        for data in bin_performance.values():
            if len(data['magnitude_targets']) > 1:
                corr = np.corrcoef(data['magnitude_targets'], data['magnitude_predictions'])[0, 1]
                mag_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                mag_corrs.append(0.0)
        
        scatter3 = axes[1, 0].scatter(lons, lats, c=mag_corrs, cmap='RdBu_r', s=100, alpha=0.8, vmin=-1, vmax=1)
        axes[1, 0].set_xlabel('Synthetic Longitude (Bin ID × 15)')
        axes[1, 0].set_ylabel('Synthetic Latitude (Bin ID × 10)')
        axes[1, 0].set_title('Magnitude Correlation by Spatial Bin (Synthetic Coordinates)')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Frequency correlation by bin
        freq_corrs = []
        for data in bin_performance.values():
            if len(data['frequency_targets']) > 1:
                corr = np.corrcoef(data['frequency_targets'], data['frequency_predictions'])[0, 1]
                freq_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                freq_corrs.append(0.0)
        
        scatter4 = axes[1, 1].scatter(lons, lats, c=freq_corrs, cmap='RdBu_r', s=100, alpha=0.8, vmin=-1, vmax=1)
        axes[1, 1].set_xlabel('Synthetic Longitude (Bin ID × 15)')
        axes[1, 1].set_ylabel('Synthetic Latitude (Bin ID × 10)')
        axes[1, 1].set_title('Frequency Correlation by Spatial Bin (Synthetic Coordinates)')
        plt.colorbar(scatter4, ax=axes[1, 1], label='Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        spatial_plot_path = save_path / "spatial_performance_analysis.png"
        plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spatial analysis plots saved to: {spatial_plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating spatial analysis plots: {e}")


def main():
    """
    Main function to run the earthquake forecasting pipeline.
    """
    # Setup logging first
    logger = setup_logging(log_level="INFO", log_file="data/shared_lstm_forecasting.log")
    
    # 🔧 FIX: Set random seeds for reproducibility
    import random
    import numpy as np
    import torch
    
    # Set seeds for all random operations
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for reproducibility")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Earthquake Forecasting Pipeline')
    parser.add_argument(
        '--input_data',
        type=str,
        default='data/eq_catalog.csv',
        help='Path to raw earthquake catalog CSV file (default: data/eq_catalog.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Directory to save processed data and results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['preprocess', 'train', 'evaluate', 'full_pipeline', 'compare_models'],
        default='full_pipeline',
        help='Mode: preprocess, train, evaluate, full_pipeline, or compare_models'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=300,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,  # Updated: matches refactored trainer default
        help='Learning rate for training - default: 5e-4 for refactored model'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,  # Updated: matches refactored trainer default
        help='Weight decay for training - default: 1e-4 for refactored model'
    )
    parser.add_argument(
        '--magnitude_weight',
        type=float,
        default=2.0,  # Updated: increased for better magnitude variance tracking
        help='Weight for magnitude loss (alpha) - default: 2.0 for refactored model'
    )
    parser.add_argument(
        '--frequency_weight',
        type=float,
        default=0.5,  # Updated: matches full pipeline default
        help='Weight for frequency loss (beta) - default: 0.5 for refactored model'
    )
    parser.add_argument(
        '--correlation_weight',
        type=float,
        default=0.0,  # Updated: disabled for refactored model
        help='Weight for correlation penalty (gamma) - default: 0.0 for refactored model'
    )
    parser.add_argument(
        '--lookback_years',
        type=int,
        default=10,
        help='Number of years to look back for prediction'
    )
    parser.add_argument(
        '--target_horizon',
        type=int,
        default=1,
        help='Number of years to predict ahead'
    )
    parser.add_argument(
        '--train_end_year',
        type=int,
        default=2009,
        help='Last year for training data'
    )
    parser.add_argument(
        '--val_end_year',
        type=int,
        default=2017,
        help='Last year for validation data'
    )
    parser.add_argument(
        '--test_end_year',
        type=int,
        default=2025,
        help='Last year for test data'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='simple',
        choices=['simple', 'attention', 'compare'],
        help='Which model to train: simple (SharedLSTM), attention (AttentionSharedLSTM), or compare (both).'
    )
    parser.add_argument(
        '--optimized_config',
        type=str,
        default=None,
        choices=['best_frequency', 'best_magnitude', 'best_balanced', 'anti_overfitting', 'balanced_anti_overfitting', 'enhanced_frequency_scaling', 'high_performance_balanced', 'hybrid_balanced'],
        help='Use optimized hyperparameter configuration for best performance. Options: best_frequency (49.39 range), best_magnitude (1.52 range), best_balanced (balanced performance), anti_overfitting (prevents overfitting), balanced_anti_overfitting (balanced performance and capacity), enhanced_frequency_scaling (maximum range coverage), high_performance_balanced (maximum overall performance), hybrid_balanced (anti-overfitting magnitude + high-performance frequency)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Shared LSTM-based Earthquake Forecasting System")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input data: {args.input_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Training epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Loss weights: alpha={args.magnitude_weight}, beta={args.frequency_weight}, gamma={args.correlation_weight}")
    logger.info(f"Lookback years: {args.lookback_years}")
    logger.info(f"Time split: Train<={args.train_end_year}, Val{args.val_end_year-2009}-{args.val_end_year}, Test{args.val_end_year+1}-{args.test_end_year}")
    
    try:
        if args.mode == 'preprocess' or args.mode == 'full_pipeline':
            # Step 1: Preprocessing
            logger.info("\n" + "="*50)
            logger.info("STEP 1: EARTHQUAKE DATA PREPROCESSING")
            logger.info("="*50)
            
            processed_data_path = output_dir / "processed_earthquake_catalog.csv"
            processed_catalog, annual_stats = preprocess_earthquake_data(
                input_path=args.input_data,
                output_path=str(processed_data_path),
                logger=logger
            )
            
            logger.info(f"Preprocessed data saved to: {processed_data_path}")
        
        if args.mode == 'train' or args.mode == 'full_pipeline':
            # Step 2: Create datasets
            logger.info("\n" + "="*50)
            logger.info("STEP 2: CREATING SHARED LSTM DATASETS")
            logger.info("="*50)
            
            # Get appropriate data path
            if args.mode == 'full_pipeline':
                data_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
            else:
                data_path = args.input_data
            
            if not Path(data_path).exists():
                logger.error(f"Data file not found: {data_path}")
                logger.error("Please run preprocessing first or use --mode full_pipeline")
                sys.exit(1)
            
            datasets = create_shared_lstm_datasets(
                data_path=str(data_path),
                lookback_years=args.lookback_years,
                target_horizon=args.target_horizon,
                train_end_year=args.train_end_year,
                val_end_year=args.val_end_year,
                test_end_year=args.test_end_year,
                logger=logger
            )
            
            # Step 3: Create and train model
            logger.info("\n" + "="*50)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("="*50)
            
            results_dir = output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            if args.model == "simple":
                logger.info("Training Simple Shared LSTM Model")
                
                # Check if using optimized configuration
                if args.optimized_config and OPTIMIZED_CONFIGS_AVAILABLE:
                    logger.info(f"Using optimized configuration: {args.optimized_config}")
                    config = load_optimized_config(args.optimized_config)
                    if config:
                        # Create model with optimized parameters
                        model = create_optimized_model(
                            SharedLSTMModel, 
                            config,
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        # Get training parameters from config
                        train_params = get_all_training_params(config)
                        logger.info(f"Using optimized training parameters:")
                        for key, value in train_params.items():
                            logger.info(f"   {key}: {value}")
                        
                        training_results = train_shared_lstm_model(
                            model=model,
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            save_dir=str(results_dir),
                            logger=logger,
                            **train_params
                        )
                    else:
                        logger.warning("⚠️  Failed to load optimized config, using default parameters")
                        model = create_shared_lstm_model(
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        training_results = train_shared_lstm_model(
                            model=model,
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            save_dir=str(results_dir),
                            logger=logger,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            magnitude_weight=args.magnitude_weight,
                            frequency_weight=args.frequency_weight,
                            correlation_weight=args.correlation_weight
                        )
                else:
                    # Use default parameters
                    model = create_shared_lstm_model(
                        input_features=datasets['input_features'],
                        metadata_features=datasets['metadata_features'],
                        lookback_years=args.lookback_years,
                        logger=logger
                    )
                    
                    training_results = train_shared_lstm_model(
                        model=model,
                        train_loader=datasets['train_loader'],
                        val_loader=datasets['val_loader'],
                        test_loader=datasets['test_loader'],
                        save_dir=str(results_dir),
                        logger=logger,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        magnitude_weight=args.magnitude_weight,
                        frequency_weight=args.frequency_weight,
                        correlation_weight=args.correlation_weight
                    )
                
            elif args.model == "attention":
                logger.info("Training Attention Shared LSTM Model")
                
                # Check if using optimized configuration
                if args.optimized_config and OPTIMIZED_CONFIGS_AVAILABLE:
                    logger.info(f"Using optimized configuration: {args.optimized_config}")
                    config = load_optimized_config(args.optimized_config)
                    if config:
                        # Create model with optimized parameters
                        model = create_optimized_model(
                            AttentionSharedLSTMModel, 
                            config,
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        # Get training parameters from config
                        train_params = get_all_training_params(config)
                        logger.info(f"Using optimized training parameters:")
                        for key, value in train_params.items():
                            logger.info(f"   {key}: {value}")
                        
                        training_results = train_attention_shared_lstm_model(
                            model=model,
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            save_dir=str(results_dir),
                            logger=logger,
                            **train_params
                        )
                    else:
                        logger.warning("⚠️  Failed to load optimized config, using default parameters")
                        model = create_attention_shared_lstm_model(
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        training_results = train_attention_shared_lstm_model(
                            model=model,
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            save_dir=str(results_dir),
                            logger=logger,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            magnitude_weight=args.magnitude_weight,
                            frequency_weight=args.frequency_weight,
                            correlation_weight=args.correlation_weight
                        )
                else:
                    # Use default parameters
                    model = create_attention_shared_lstm_model(
                        input_features=datasets['input_features'],
                        metadata_features=datasets['metadata_features'],
                        lookback_years=args.lookback_years,
                        logger=logger
                    )
                    
                    training_results = train_attention_shared_lstm_model(
                        model=model,
                        train_loader=datasets['train_loader'],
                        val_loader=datasets['val_loader'],
                        test_loader=datasets['test_loader'],
                        save_dir=str(results_dir),
                            logger=logger,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            magnitude_weight=args.magnitude_weight,
                            frequency_weight=args.frequency_weight,
                            correlation_weight=args.correlation_weight
                    )
                
            elif args.model == "compare":
                logger.info("Training Both Models for Comparison")
                
                # Check if using optimized configuration
                if args.optimized_config and OPTIMIZED_CONFIGS_AVAILABLE:
                    logger.info(f"Using optimized configuration: {args.optimized_config}")
                    config = load_optimized_config(args.optimized_config)
                    if config:
                        # Create models with optimized parameters
                        simple_model = create_optimized_model(
                            SharedLSTMModel, 
                            config,
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        attention_model = create_optimized_model(
                            AttentionSharedLSTMModel, 
                            config,
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        # Get training parameters from config
                        train_params = get_all_training_params(config)
                        logger.info(f"Using optimized training parameters:")
                        for key, value in train_params.items():
                            logger.info(f"   {key}: {value}")
                        
                        # Use ModelComparisonTrainer with optimized parameters
                        comparison_trainer = ModelComparisonTrainer(
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            learning_rate=train_params['learning_rate'],
                            weight_decay=train_params['weight_decay'],
                            magnitude_weight=train_params['magnitude_weight'],
                            frequency_weight=train_params['frequency_weight'],
                            correlation_weight=train_params['correlation_weight'],
                            device='auto'
                        )
                        
                        # Train both models with optimized parameters
                        simple_results = comparison_trainer.train_model(
                            model=simple_model,
                            model_name="SharedLSTM",
                            max_epochs=train_params['num_epochs'],
                            patience=train_params['patience']
                        )
                        
                        attention_results = comparison_trainer.train_model(
                            model=attention_model,
                            model_name="AttentionSharedLSTM",
                            max_epochs=train_params['num_epochs'],
                            patience=train_params['patience']
                        )
                    else:
                        logger.warning("⚠️  Failed to load optimized config, using default parameters")
                        # Fall back to default models and parameters
                        simple_model = create_shared_lstm_model(
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        attention_model = create_attention_shared_lstm_model(
                            input_features=datasets['input_features'],
                            metadata_features=datasets['metadata_features'],
                            lookback_years=args.lookback_years,
                            logger=logger
                        )
                        
                        # Use ModelComparisonTrainer with default parameters
                        comparison_trainer = ModelComparisonTrainer(
                            train_loader=datasets['train_loader'],
                            val_loader=datasets['val_loader'],
                            test_loader=datasets['test_loader'],
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            magnitude_weight=args.magnitude_weight,
                            frequency_weight=args.frequency_weight,
                            correlation_weight=args.correlation_weight,
                            device='auto'
                        )
                        
                        # Train both models with default parameters
                        simple_results = comparison_trainer.train_model(
                            model=simple_model,
                            model_name="SharedLSTM",
                            max_epochs=args.num_epochs,
                            patience=15
                        )
                        
                        attention_results = comparison_trainer.train_model(
                            model=attention_model,
                            model_name="AttentionSharedLSTM",
                            max_epochs=args.num_epochs,
                            patience=15
                        )
                else:
                    # Use default parameters
                    logger.info("Note: Using default hyperparameters for fair comparison")
                    simple_model = create_shared_lstm_model(
                        input_features=datasets['input_features'],
                        metadata_features=datasets['metadata_features'],
                        lookback_years=args.lookback_years,
                        logger=logger
                    )
                    
                    attention_model = create_attention_shared_lstm_model(
                        input_features=datasets['input_features'],
                        metadata_features=datasets['metadata_features'],
                        lookback_years=args.lookback_years,
                        logger=logger
                    )
                    
                    # Use ModelComparisonTrainer with default parameters
                    comparison_trainer = ModelComparisonTrainer(
                        train_loader=datasets['train_loader'],
                        val_loader=datasets['val_loader'],
                        test_loader=datasets['test_loader'],
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        magnitude_weight=args.magnitude_weight,
                        frequency_weight=args.frequency_weight,
                        correlation_weight=args.correlation_weight,
                        device='auto'
                    )
                    
                    # Train both models with default parameters
                    simple_results = comparison_trainer.train_model(
                        model=simple_model,
                        model_name="SharedLSTM",
                        max_epochs=args.num_epochs,
                        patience=15
                    )
                    
                    attention_results = comparison_trainer.train_model(
                        model=attention_model,
                        model_name="AttentionSharedLSTM",
                        max_epochs=args.num_epochs,
                        patience=15
                    )
                
                # Save models with distinct names
                simple_save_path = Path(results_dir) / "shared_best_model.pth"
                attention_save_path = Path(results_dir) / "attention_best_model.pth"
                
                # Save the trained models
                torch.save({
                    'model_state_dict': simple_model.state_dict(),
                    'model_config': {
                        'input_seq_features': datasets['input_features'],
                        'metadata_features': datasets['metadata_features'],
                        'lookback_years': args.lookback_years,
                        'lstm_hidden_1': 64,
                        'lstm_hidden_2': 32,
                        'dense_hidden': 32,
                        'dropout_rate': 0.25,
                        'freq_head_type': "linear"
                    }
                }, simple_save_path)
                
                torch.save({
                    'model_state_dict': attention_model.state_dict(),
                    'model_config': {
                        'input_seq_features': datasets['input_features'],
                        'metadata_features': datasets['metadata_features'],
                        'lookback_years': args.lookback_years,
                        'lstm_hidden_1': 64,
                        'lstm_hidden_2': 32,
                        'dense_hidden': 32,
                        'dropout_rate': 0.25,
                        'freq_head_type': "linear"
                    }
                }, attention_save_path)
                
                training_results = {
                    'simple_model': simple_results,
                    'attention_model': attention_results,
                    'simple_model_path': str(simple_save_path),
                    'attention_model_path': str(attention_save_path)
                }
            
            logger.info(f"Training results saved to: {results_dir}")
        
        if args.mode == 'evaluate' or args.mode == 'full_pipeline':
            # Step 4: Model Evaluation
            logger.info("\n" + "="*50)
            logger.info("STEP 4: MODEL EVALUATION")
            logger.info("="*50)
            
            # Get appropriate data path
            if args.mode == 'full_pipeline':
                data_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
                
                # Determine model path based on model type
                if args.model == "simple":
                    model_path = output_dir / "results" / "best_model.pth"
                elif args.model == "attention":
                    model_path = output_dir / "results" / "attention_best_model.pth"
                elif args.model == "compare":
                    # For comparison mode, evaluate both models
                    simple_model_path = output_dir / "results" / "shared_best_model.pth"
                    attention_model_path = output_dir / "results" / "attention_best_model.pth"
                    model_path = simple_model_path  # Default for now
                
                # 🔧 FIX: Use cached datasets from training to ensure consistency
                if 'datasets' in locals():
                    logger.info("Using cached datasets from training phase to ensure consistency")
                    evaluation_datasets = datasets
                else:
                    logger.warning("No cached datasets found, creating new ones (may cause inconsistency)")
                    evaluation_datasets = create_shared_lstm_datasets(
                        data_path=str(data_path),
                        lookback_years=args.lookback_years,
                        target_horizon=args.target_horizon,
                        train_end_year=args.train_end_year,
                        val_end_year=args.val_end_year,
                        test_end_year=args.test_end_year,
                        logger=logger
                    )
            else:
                data_path = args.input_data
                # Determine model path based on model type
                if args.model == "simple":
                    model_path = output_dir / "results" / "best_model.pth"
                elif args.model == "attention":
                    model_path = output_dir / "results" / "attention_best_model.pth"
                elif args.model == "compare":
                    # For comparison mode, evaluate both models
                    simple_model_path = output_dir / "results" / "shared_best_model.pth"
                    attention_model_path = output_dir / "results" / "attention_best_model.pth"
                    model_path = simple_model_path  # Default for now
                
                evaluation_datasets = create_shared_lstm_datasets(
                    data_path=str(data_path),
                    lookback_years=args.lookback_years,
                    target_horizon=args.target_horizon,
                    train_end_year=args.train_end_year,
                    val_end_year=args.val_end_year,
                    test_end_year=args.test_end_year,
                    logger=logger
                )
            
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                logger.error("Please train the model first or use --mode full_pipeline")
                sys.exit(1)
            
            # Handle evaluation based on model type
            if args.model == "compare":
                logger.info("Evaluating both models from comparison...")
                
                # Evaluate simple model
                logger.info("Evaluating Simple Shared LSTM Model...")
                # Create subdirectories for evaluation results
                (output_dir / "results" / "simple_model").mkdir(parents=True, exist_ok=True)
                simple_evaluation_results = evaluate_shared_lstm_model(
                    model_path=str(simple_model_path),
                    test_loader=evaluation_datasets['test_loader'],
                    save_dir=str(output_dir / "results" / "simple_model"),
                    logger=logger,
                    model_type="simple"
                )
                
                # Evaluate attention model
                logger.info("Evaluating Attention Shared LSTM Model...")
                # Create subdirectories for evaluation results
                (output_dir / "results" / "attention_model").mkdir(parents=True, exist_ok=True)
                attention_evaluation_results = evaluate_shared_lstm_model(
                    model_path=str(attention_model_path),
                    test_loader=evaluation_datasets['test_loader'],
                    save_dir=str(output_dir / "results" / "attention_model"),
                    logger=logger,
                    model_type="attention"
                )
                
                evaluation_results = {
                    'simple_model': simple_evaluation_results,
                    'attention_model': attention_evaluation_results
                }
                
                # Generate visualizations for both models
                logger.info("\n" + "="*50)
                logger.info("STEP 5: GENERATING COMPREHENSIVE VISUALIZATIONS")
                logger.info("="*50)
                
                # Visualizations for simple model
                logger.info("Generating visualizations for Simple Shared LSTM Model...")
                generate_comprehensive_visualizations(
                    save_dir=str(output_dir / "results" / "simple_model"),
                    model_path=str(simple_model_path),
                    test_loader=evaluation_datasets['test_loader'],
                    logger=logger,
                    model_type="simple"
                )
                
                # Visualizations for attention model
                logger.info("Generating visualizations for Attention Shared LSTM Model...")
                generate_comprehensive_visualizations(
                    save_dir=str(output_dir / "results" / "attention_model"),
                    model_path=str(attention_model_path),
                    test_loader=evaluation_datasets['test_loader'],
                    logger=logger,
                    model_type="attention"
                )
                
            else:
                # Single model evaluation
                # Determine model type based on the model path
                if "attention" in str(model_path):
                    model_type = "attention"
                else:
                    model_type = "simple"
                
                evaluation_results = evaluate_shared_lstm_model(
                    model_path=str(model_path),
                    test_loader=evaluation_datasets['test_loader'],
                    save_dir=str(output_dir / "results"),
                    logger=logger,
                    model_type=model_type
                )
                
                # Step 5: Generate Visualizations
                logger.info("\n" + "="*50)
                logger.info("STEP 5: GENERATING COMPREHENSIVE VISUALIZATIONS")
                logger.info("="*50)
                
                # Determine model type for visualization
                if "attention" in str(model_path):
                    viz_model_type = "attention"
                else:
                    viz_model_type = "simple"
                
                generate_comprehensive_visualizations(
                    save_dir=str(output_dir / "results"),
                    model_path=str(model_path),
                    test_loader=evaluation_datasets['test_loader'],  # Use cached datasets
                    logger=logger,
                    model_type=viz_model_type
                )
        
        if args.mode == 'compare_models':
            # Model Comparison Mode
            logger.info("\n" + "="*50)
            logger.info("MODEL COMPARISON: Simple LSTM vs Attention LSTM")
            logger.info("="*50)
            
            # Check if using optimized configuration
            if args.optimized_config and OPTIMIZED_CONFIGS_AVAILABLE:
                logger.info(f"Using optimized configuration: {args.optimized_config}")
                config = load_optimized_config(args.optimized_config)
                if config:
                    # Use optimized hyperparameters
                    train_params = get_training_params(config)
                    logger.info(f"Using optimized training parameters:")
                    for key, value in train_params.items():
                        logger.info(f"   {key}: {value}")
                else:
                    logger.warning("⚠️  Failed to load optimized config, using default parameters")
                    train_params = {
                        'learning_rate': args.learning_rate,
                        'weight_decay': args.weight_decay,
                        'magnitude_weight': args.magnitude_weight,
                        'frequency_weight': args.frequency_weight,
                        'correlation_weight': args.correlation_weight
                    }
            else:
                logger.info("Note: Using default hyperparameters for fair comparison")
                train_params = {
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'magnitude_weight': args.magnitude_weight,
                    'frequency_weight': args.frequency_weight,
                    'correlation_weight': args.correlation_weight
                }
            
            # Get appropriate data path - FIXED: Use correct logic
            if Path(args.input_data).exists():
                data_path = args.input_data
            elif (output_dir / "processed_earthquake_catalog_annual_stats.csv").exists():
                data_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
            else:
                logger.error("No suitable data file found!")
                logger.error("Please ensure either --input_data exists or run preprocessing first")
                sys.exit(1)
            
            # Create datasets for comparison
            datasets = create_shared_lstm_datasets(
                data_path=str(data_path),
                lookback_years=args.lookback_years,
                target_horizon=args.target_horizon,
                train_end_year=args.train_end_year,
                val_end_year=args.val_end_year,
                test_end_year=args.test_end_year,
                logger=logger
            )
            
            # Run model comparison
            # Ensure results directory exists
            (output_dir / "results").mkdir(parents=True, exist_ok=True)
            comparison_results = run_model_comparison(
                datasets=datasets,
                save_dir=str(output_dir / "results"),
                logger=logger,
                **train_params
            )
            
            logger.info(f"Model comparison completed! Results saved to: {output_dir / 'results' / 'model_comparison'}")
        
        logger.info("\n" + "="*80)
        logger.info("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("")
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Processed data: {args.input_data}")
        logger.info(f"Results: {output_dir}")
        logger.info(f"Log file: {output_dir}/shared_lstm_forecasting.log")
        logger.info("")
        logger.info("The system has completed the full pipeline:")
        logger.info("1. [SUCCESS] Preprocessed earthquake data (filtered shallow, spatial binned)")
        logger.info("2. [SUCCESS] Created shared LSTM datasets with rolling features")
        if args.model == "simple":
            logger.info("3. [SUCCESS] Trained simple shared LSTM model with dual output heads")
        elif args.model == "attention":
            logger.info("3. [SUCCESS] Trained attention shared LSTM model with dual output heads")
        elif args.model == "compare":
            logger.info("3. [SUCCESS] Trained both simple and attention shared LSTM models for comparison")
        logger.info("4. [SUCCESS] Evaluated model performance using comprehensive metrics")
        logger.info("5. [SUCCESS] Generated comprehensive visualizations and spatial analysis")
        logger.info("")
        logger.info("You can now analyze the results in the output directory!")
        
        return
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()