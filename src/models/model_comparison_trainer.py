#!/usr/bin/env python3
"""
Model Comparison Trainer for Earthquake Forecasting

Trains and compares the performance of:
1. Simple Shared LSTM Model
2. Attention-Enhanced Shared LSTM Model

This script provides a fair comparison by:
- Using identical training data and splits
- Same hyperparameters as the full pipeline (ensuring consistency)
- Same evaluation metrics
- Comprehensive performance analysis
- Consistent training conditions for architectural comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import torch.nn.functional as F
import time
import json
from datetime import datetime

from .shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
from .shared_lstm_trainer import SharedLSTMTrainer  # 🔧 ADD: Use your proven trainer
from .attention_shared_lstm_model import AttentionSharedLSTMModel
from .quadtree_data_loader import QuadtreeDataLoader
from .quadtree_trainer import QuadtreeModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparisonTrainer:
    """
    Trainer that compares Simple Shared LSTM vs Attention Shared LSTM performance.
    
    Uses the same hyperparameters as the full pipeline to ensure fair comparison
    and consistent training conditions.
    """
    
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 input_seq_features: int,
                 metadata_features: int,
                 lookback_years: int = 10,
                 learning_rate: float = 5e-4,        # Default: matches full pipeline
                 weight_decay: float = 1e-4,         # Default: matches full pipeline
                 magnitude_weight: float = 2.0,      # alpha: weight for magnitude loss
                 frequency_weight: float = 0.5,      # beta: weight for frequency loss
                 correlation_weight: float = 0.0,    # gamma: weight for correlation penalty (default: disabled)
                 device: str = 'auto',
                 output_dir: str = 'model_comparison_results'):
        """
        Initialize the comparison trainer.
        
        🔧 BOTH MODELS will use hyperparameters from main pipeline for fair comparison!
        This ensures consistent training conditions with the full pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            input_seq_features: Number of input sequence features
            metadata_features: Number of metadata features
            lookback_years: Number of years to look back
            learning_rate: Learning rate for both models (default: 5e-4, matches full pipeline)
            weight_decay: Weight decay for both models (default: 1e-4, matches full pipeline)
            magnitude_weight: Weight for magnitude loss (alpha) (default: 2.0, matches full pipeline)
            frequency_weight: Weight for frequency loss (beta) (default: 0.5, matches full pipeline)
            correlation_weight: Weight for correlation penalty (gamma) (default: 0.0, matches full pipeline)
            device: Device to use for training
            output_dir: Directory to save comparison results
        """
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
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            'simple_lstm': {},
            'attention_lstm': {},
            'comparison': {}
        }
        
        logger.info(f"ModelComparisonTrainer initialized on device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_models(self) -> Tuple[SharedLSTMModel, AttentionSharedLSTMModel]:
        """
        Create both models with identical configurations.
        
        Returns:
            Tuple of (simple_lstm_model, attention_lstm_model)
        """
        logger.info("Creating models for comparison...")
        
        # Simple Shared LSTM Model
        simple_lstm = SharedLSTMModel(
            input_seq_features=self.input_seq_features,
            metadata_features=self.metadata_features,
            lookback_years=self.lookback_years,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"  # Default to new stable linear frequency head
        )
        
        # Attention Shared LSTM Model
        attention_lstm = AttentionSharedLSTMModel(
            input_seq_features=self.input_seq_features,
            metadata_features=self.metadata_features,
            lookback_years=self.lookback_years,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            num_attention_heads=8
        )
        
        # Move models to device
        simple_lstm.to(self.device)
        attention_lstm.to(self.device)
        
        # Log model information
        simple_params = sum(p.numel() for p in simple_lstm.parameters())
        attention_params = sum(p.numel() for p in attention_lstm.parameters())
        
        logger.info(f"Simple LSTM parameters: {simple_params:,}")
        logger.info(f"Attention LSTM parameters: {attention_params:,}")
        logger.info(f"Parameter difference: {attention_params - simple_params:,}")
        
        return simple_lstm, attention_lstm
    
    def train_model(self, 
                   model: nn.Module, 
                   model_name: str,
                   max_epochs: int = 300,
                   patience: int = 12) -> Dict:
        """
        Train a single model using the proven SharedLSTMTrainer approach.
        
        🔧 BOTH MODELS now use hyperparameters from main pipeline for fair comparison!
        
        Args:
            model: The model to train (Simple LSTM or Attention LSTM)
            model_name: Name of the model for logging
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience (not used with SharedLSTMTrainer)
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {model_name} using proven SharedLSTMTrainer...")
        
                # 🔧 USE PROVEN TRAINER: Both models now use hyperparameters from main pipeline
        # This ensures consistent training approach and fair comparison
        
        # Use your proven SharedLSTMTrainer with hyperparameters from main pipeline
        trainer = SharedLSTMTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            learning_rate=self.learning_rate,        # Use hyperparameters from main pipeline
            weight_decay=self.weight_decay,         # Use hyperparameters from main pipeline
            magnitude_weight=self.magnitude_weight, # Use hyperparameters from main pipeline
            frequency_weight=self.frequency_weight, # Use hyperparameters from main pipeline
            correlation_weight=self.correlation_weight, # Use hyperparameters from main pipeline
            device=self.device
        )
        
        # 🔧 TRAIN: Use your proven training method
        training_history = trainer.train(
            max_epochs=max_epochs,
            save_path=None,  # Don't save during comparison
            save_best=False
        )
        
        logger.info(f"{model_name} training completed using main pipeline hyperparameters")
        return training_history
    
    def evaluate_model(self, 
                      model: nn.Module, 
                      model_name: str) -> Dict:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: The trained model to evaluate
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # 🔧 ADDITIONAL CHECK: Verify model state and data
        model.eval()
        
        # Check if model has the required methods
        if not hasattr(model, 'predict_frequency_counts'):
            logger.error(f"Model {model_name} does not have predict_frequency_counts method!")
            raise AttributeError(f"Model {model_name} missing required method predict_frequency_counts")
        
        # Log model parameters for debugging
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model {model_name} has {total_params:,} parameters")
        
        # Check first batch to understand data flow
        try:
            first_batch = next(iter(self.test_loader))
            sequences, targets, metadata, metadata_dict = first_batch
            logger.info(f"Test data shapes - Sequences: {sequences.shape}, Targets: {targets.shape}, Metadata: {metadata.shape}")
            logger.info(f"Target range - Magnitude: [{targets[:, 0].min():.3f}, {targets[:, 0].max():.3f}], Frequency: [{targets[:, 1].min():.3f}, {targets[:, 1].max():.3f}]")
        except Exception as e:
            logger.warning(f"Could not inspect first batch: {e}")
        
        all_magnitude_preds = []
        all_frequency_preds = []
        all_frequency_log1p_preds = []  # Store log1p predictions for logging
        all_magnitude_targets = []
        all_frequency_targets = []
        
        with torch.no_grad():
            for batch_idx, (sequences, targets, metadata, metadata_dict) in enumerate(self.test_loader):
                sequences = sequences.to(self.device)
                metadata = metadata.to(self.device)
                magnitude_targets = targets[:, 0].to(self.device)
                frequency_targets = targets[:, 1].to(self.device)
                
                # Forward pass
                magnitude_pred, frequency_log_rate_pred = model(sequences, metadata)
                
                # 🔧 REFACTOR: Convert frequency predictions from log1p space to raw counts for evaluation
                # The model outputs log1p(λ), we need to convert to λ = expm1(log1p(λ))
                try:
                    frequency_pred = np.expm1(frequency_log_rate_pred.cpu().numpy())
                    
                    # Add debugging for frequency conversion
                    if batch_idx == 0:  # Log first batch info
                        logger.debug(f"Frequency conversion - Log1p range: [{frequency_log_rate_pred.min():.3f}, {frequency_log_rate_pred.max():.3f}]")
                        logger.debug(f"Frequency conversion - Raw count range: [{frequency_pred.min():.3f}, {frequency_pred.max():.3f}]")
                except Exception as e:
                    logger.warning(f"Error in frequency conversion: {e}")
                    # Fallback: use raw log1p predictions
                    frequency_pred = frequency_log_rate_pred.cpu().numpy()
                
                # Store predictions and targets
                try:
                    all_magnitude_preds.append(magnitude_pred.cpu().numpy())
                    all_frequency_preds.append(frequency_pred)  # frequency_pred is already numpy array
                    all_frequency_log1p_preds.append(frequency_log_rate_pred.cpu().numpy())  # Store log1p predictions
                    all_magnitude_targets.append(magnitude_targets.cpu().numpy())
                    all_frequency_targets.append(frequency_targets.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Error storing predictions for batch {batch_idx}: {e}")
                    # Skip this batch if there's an error
                    continue
        
        # Check if we have any data before concatenating
        if not all_magnitude_preds or not all_frequency_preds:
            logger.error("No predictions collected during evaluation!")
            raise RuntimeError("No predictions collected during evaluation")
        
        # Concatenate all predictions and targets
        magnitude_preds = np.concatenate(all_magnitude_preds)
        frequency_preds = np.concatenate(all_frequency_preds)
        frequency_log1p_preds = np.concatenate(all_frequency_log1p_preds)  # Concatenate log1p predictions
        magnitude_targets = np.concatenate(all_magnitude_targets)
        frequency_targets = np.concatenate(all_frequency_targets)
        
        # 🔧 CRITICAL FIX: Denormalize targets and predictions for proper evaluation
        # The model was trained on normalized data, so we need to denormalize for evaluation
        try:
            # Get the dataset from the test loader to access denormalization methods
            dataset = self.test_loader.dataset
            if hasattr(dataset, 'denormalize_single_feature'):
                # Denormalize targets
                magnitude_targets_denorm = np.array([
                    dataset.denormalize_single_feature(mag, 'magnitude')
                    for mag in magnitude_targets
                ])
                frequency_targets_denorm = np.array([
                    dataset.denormalize_single_feature(freq, 'frequency')
                    for freq in frequency_targets
                ])
                
                # Denormalize predictions for both magnitude and frequency
                # Magnitude predictions need to be denormalized from normalized space back to actual magnitude scale
                magnitude_preds_denorm = np.array([
                    dataset.denormalize_single_feature(mag, 'magnitude')
                    for mag in magnitude_preds
                ])
                
                # For frequency, we need to convert from log-rate to counts, then denormalize
                frequency_preds_denorm = np.array([
                    dataset.denormalize_single_feature(freq, 'frequency')
                    for freq in frequency_preds
                ])
                
                # Use denormalized predictions for evaluation
                magnitude_preds = magnitude_preds_denorm
                
                # Use denormalized values for evaluation
                magnitude_targets = magnitude_targets_denorm
                frequency_targets = frequency_targets_denorm
                frequency_preds = frequency_preds_denorm
                
                # 🔧 DEBUG: Log the denormalization process
                logger.info(f"Denormalization debug:")
                logger.info(f"  Magnitude - Before: mean={np.mean(magnitude_preds_denorm):.3f}, range=[{np.min(magnitude_preds_denorm):.3f}, {np.max(magnitude_preds_denorm):.3f}]")
                logger.info(f"  Frequency - Before: mean={np.mean(frequency_preds_denorm):.3f}, range=[{np.min(frequency_preds_denorm):.3f}, {np.max(frequency_preds_denorm):.3f}]")
                
                logger.info(f"Successfully denormalized data for evaluation")
                logger.info(f"Magnitude range: {magnitude_targets.min():.3f} to {magnitude_targets.max():.3f}")
                logger.info(f"Frequency range: {frequency_targets.min():.3f} to {frequency_targets.max():.3f}")
                
                # 🔧 ADDITIONAL CHECK: Validate prediction ranges
                if frequency_preds.min() < 0:
                    logger.warning(f"Frequency predictions contain negative values! Clipping to 0.")
                    frequency_preds = np.clip(frequency_preds, 0, None)
                
                if magnitude_preds.min() < 0:
                    logger.warning(f"Magnitude predictions contain negative values after denormalization: {magnitude_preds.min():.3f}. This might indicate training issues.")
                
                # Check for extreme values that might indicate training issues
                if frequency_preds.max() > 1000:
                    logger.warning(f"Frequency predictions have very high values: {frequency_preds.max():.1f}")
                
                if magnitude_preds.max() > 10:
                    logger.warning(f"Magnitude predictions have very high values: {magnitude_preds.max():.1f}")
            else:
                logger.warning("Dataset does not have denormalization methods. Using normalized data for evaluation.")
        except Exception as e:
            logger.warning(f"Could not denormalize data: {e}. Using normalized data for evaluation.")
        
        # 🔧 REFACTOR: Log both log-space and raw-space ranges for transparency
        freq_log_pred_min, freq_log_pred_max = np.min(frequency_log1p_preds), np.max(frequency_log1p_preds)
        freq_raw_pred_min, freq_raw_pred_max = np.min(frequency_preds), np.max(frequency_preds)
        freq_true_min, freq_true_max = np.min(frequency_targets), np.max(frequency_targets)
        
        logger.info(f"Freq log-space preds [{freq_log_pred_min:.2f},{freq_log_pred_max:.2f}], targets [{freq_true_min:.1f},{freq_true_max:.1f}]")
        logger.info(f"Freq raw-space preds [{freq_raw_pred_min:.1f},{freq_raw_pred_max:.1f}], targets [{freq_true_min:.1f},{freq_true_max:.1f}]")
        
        # 🔧 GUARDRAIL: Check if raw-space prediction std is <5% of target std
        freq_pred_std = np.std(frequency_preds)
        freq_target_std = np.std(frequency_targets)
        
        if freq_target_std > 0 and freq_pred_std < freq_target_std * 0.05:
            logger.warning(f"[WARNING] Frequency predictions collapsed (check scaling): pred_std={freq_pred_std:.3f}, target_std={freq_target_std:.3f}")
        
        # Calculate metrics with additional safeguards
        # Check for NaN or infinite values
        if np.any(np.isnan(magnitude_preds)) or np.any(np.isinf(magnitude_preds)):
            logger.warning(f"Magnitude predictions contain NaN or infinite values!")
            magnitude_preds = np.nan_to_num(magnitude_preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(frequency_preds)) or np.any(np.isinf(frequency_preds)):
            logger.warning(f"Frequency predictions contain NaN or infinite values!")
            frequency_preds = np.nan_to_num(frequency_preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(magnitude_targets)) or np.any(np.isinf(magnitude_targets)):
            logger.warning(f"Magnitude targets contain NaN or infinite values!")
            magnitude_targets = np.nan_to_num(magnitude_targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(frequency_targets)) or np.any(np.isinf(frequency_targets)):
            logger.warning(f"Frequency targets contain NaN or infinite values!")
            frequency_targets = np.nan_to_num(frequency_targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics
        magnitude_mse = np.mean((magnitude_preds - magnitude_targets) ** 2)
        magnitude_mae = np.mean(np.abs(magnitude_preds - magnitude_targets))
        magnitude_rmse = np.sqrt(magnitude_mse)
        
        frequency_mse = np.mean((frequency_preds - frequency_targets) ** 2)
        frequency_mae = np.mean(np.abs(frequency_preds - frequency_targets))
        frequency_rmse = np.sqrt(frequency_mse)
        
        # Calculate R-squared for both outputs
        # Handle potential division by zero and ensure proper scaling
        magnitude_ss_res = np.sum((magnitude_targets - magnitude_preds) ** 2)
        magnitude_ss_tot = np.sum((magnitude_targets - np.mean(magnitude_targets)) ** 2)
        magnitude_r2 = 1 - (magnitude_ss_res / magnitude_ss_tot) if magnitude_ss_tot > 0 else 0.0
        
        frequency_ss_res = np.sum((frequency_targets - frequency_preds) ** 2)
        frequency_ss_tot = np.sum((frequency_targets - np.mean(frequency_targets)) ** 2)
        frequency_r2 = 1 - (frequency_ss_res / frequency_ss_tot) if frequency_ss_tot > 0 else 0.0
        
        # Calculate correlation coefficients with proper error handling
        try:
            magnitude_corr = np.corrcoef(magnitude_preds.flatten(), magnitude_targets.flatten())[0, 1]
            if np.isnan(magnitude_corr):
                magnitude_corr = 0.0
        except Exception as e:
            logger.warning(f"Could not calculate magnitude correlation: {e}")
            magnitude_corr = 0.0
            
        try:
            frequency_corr = np.corrcoef(frequency_preds.flatten(), frequency_targets.flatten())[0, 1]
            if np.isnan(frequency_corr):
                frequency_corr = 0.0
        except Exception as e:
            logger.warning(f"Could not calculate frequency correlation: {e}")
            frequency_corr = 0.0
        
        # 🔧 IMPROVED METRICS: More meaningful accuracy and error metrics
        # Magnitude accuracy (within ±0.5 magnitude units) - but only count as accurate if prediction is positive
        magnitude_accurate = np.logical_and(
            np.abs(magnitude_preds - magnitude_targets) <= 0.5,
            magnitude_preds >= 0  # Must predict positive magnitudes
        )
        magnitude_accuracy = np.mean(magnitude_accurate)
        
        # Frequency accuracy (within ±1 count) - but only count as accurate if prediction is reasonable
        frequency_accurate = np.logical_and(
            np.abs(frequency_preds - frequency_targets) <= 1.0,
            frequency_preds >= 0  # Must predict non-negative frequencies
        )
        frequency_accuracy = np.mean(frequency_accurate)
        
        # More meaningful relative error metrics
        # For magnitude: relative error should be reasonable (not astronomical)
        magnitude_rel_error = np.mean(np.abs(magnitude_preds - magnitude_targets) / (np.abs(magnitude_targets) + 1e-8))
        
        # For frequency: relative error should be reasonable
        frequency_rel_error = np.mean(np.abs(frequency_preds - frequency_targets) / (np.abs(frequency_targets) + 1e-8))
        
        # 🔧 NEW: Add scale consistency checks
        magnitude_scale_consistent = np.std(magnitude_preds) > 0.1 and np.std(magnitude_targets) > 0.1
        frequency_scale_consistent = np.std(frequency_preds) > 0.1 and np.std(frequency_targets) > 0.1
        
        # 🔧 NEW: Add prediction range validation
        magnitude_in_range = np.logical_and(
            magnitude_preds >= 0,
            magnitude_preds <= 10  # Reasonable earthquake magnitude range
        )
        frequency_in_range = np.logical_and(
            frequency_preds >= 0,
            frequency_preds <= 100  # Reasonable frequency range
        )
        
        evaluation_results = {
            'model_name': model_name,
            'magnitude': {
                'mse': float(magnitude_mse),
                'mae': float(magnitude_mae),
                'rmse': float(magnitude_rmse),
                'r2': float(magnitude_r2),
                'correlation': float(magnitude_corr),
                'accuracy': float(magnitude_accuracy),  # IMPROVED: Accuracy within ±0.5 + positive predictions
                'relative_error': float(magnitude_rel_error),  # Relative error
                'scale_consistent': bool(magnitude_scale_consistent),  # NEW: Scale consistency check
                'predictions_in_range': float(np.mean(magnitude_in_range))  # NEW: Predictions in reasonable range
            },
            'frequency': {
                'mse': float(frequency_mse),
                'mae': float(frequency_mae),
                'rmse': float(frequency_rmse),
                'r2': float(frequency_r2),
                'correlation': float(frequency_corr),
                'accuracy': float(frequency_accuracy),  # IMPROVED: Accuracy within ±1 + non-negative predictions
                'relative_error': float(frequency_rel_error),  # Relative error
                'scale_consistent': bool(frequency_scale_consistent),  # NEW: Scale consistency check
                'predictions_in_range': float(np.mean(frequency_in_range))  # NEW: Predictions in reasonable range
            },
            'predictions': {
                'magnitude_preds': magnitude_preds.tolist(),
                'frequency_preds': frequency_preds.tolist(),
                'magnitude_targets': magnitude_targets.tolist(),
                'frequency_targets': frequency_targets.tolist()
            }
        }
        
        logger.info(f"{model_name} evaluation completed:")
        logger.info(f"  Magnitude - MSE: {magnitude_mse:.6f}, R²: {magnitude_r2:.6f}, Corr: {magnitude_corr:.6f}")
        logger.info(f"  Frequency - MSE: {frequency_mse:.6f}, R²: {frequency_r2:.6f}, Corr: {frequency_corr:.6f}")
        
        # 🔧 NEW: Performance summary with interpretation (Windows-safe)
        logger.info(f"  Performance Summary:")
        if magnitude_r2 < 0 and frequency_r2 < 0:
            logger.error(f"    [CRITICAL] Both outputs show negative R² - model is performing worse than predicting the mean!")
        elif magnitude_r2 < 0:
            logger.error(f"    [CRITICAL] Magnitude: Negative R² - model is performing worse than predicting the mean!")
        elif frequency_r2 < 0:
            logger.error(f"    [CRITICAL] Frequency: Negative R² - model is performing worse than predicting the mean!")
        
        if magnitude_corr < 0.1 and frequency_corr < 0.1:
            logger.warning(f"    [WARNING] Both outputs show very low correlation (<0.1) - model is not learning meaningful patterns!")
        
        # Check for the accuracy vs R² inconsistency
        if magnitude_accuracy > 0.5 and magnitude_r2 < 0:
            logger.warning(f"    [WARNING] Magnitude accuracy ({magnitude_accuracy:.1%}) is misleading - R² is negative!")
        if frequency_accuracy > 0.5 and frequency_r2 < 0:
            logger.warning(f"    [WARNING] Frequency accuracy ({frequency_accuracy:.1%}) is misleading - R² is negative!")
        
        # 🔧 ADDITIONAL DEBUGGING: Show data statistics
        logger.info(f"  Data Statistics:")
        logger.info(f"    Magnitude - Targets: mean={magnitude_targets.mean():.3f}, std={magnitude_targets.std():.3f}, range=[{magnitude_targets.min():.3f}, {magnitude_targets.max():.3f}]")
        logger.info(f"    Magnitude - Preds:  mean={magnitude_preds.mean():.3f}, std={magnitude_preds.std():.3f}, range=[{magnitude_preds.min():.3f}, {magnitude_preds.max():.3f}]")
        logger.info(f"    Frequency - Targets: mean={frequency_targets.mean():.3f}, std={frequency_targets.std():.3f}, range=[{frequency_targets.min():.3f}, {frequency_targets.max():.3f}]")
        logger.info(f"    Frequency - Preds:  mean={frequency_preds.mean():.3f}, std={frequency_preds.std():.3f}, range=[{frequency_preds.min():.3f}, {frequency_preds.max():.3f}]")
        
        # 🔧 ENHANCED: Log interpretability metrics with scale validation
        logger.info(f"  Interpretability Metrics:")
        logger.info(f"    Magnitude Accuracy (±0.5): {magnitude_accuracy:.3f} ({magnitude_accuracy*100:.1f}%)")
        logger.info(f"    Frequency Accuracy (±1): {frequency_accuracy:.3f} ({frequency_accuracy*100:.1f}%)")
        logger.info(f"    Magnitude Relative Error: {magnitude_rel_error:.3f}")
        logger.info(f"    Frequency Relative Error: {frequency_rel_error:.3f}")
        
        # 🔧 NEW: Log scale consistency and prediction quality
        logger.info(f"  Scale & Quality Checks:")
        logger.info(f"    Magnitude Scale Consistent: {magnitude_scale_consistent} (std_pred={np.std(magnitude_preds):.3f}, std_target={np.std(magnitude_targets):.3f})")
        logger.info(f"    Frequency Scale Consistent: {frequency_scale_consistent} (std_pred={np.std(frequency_preds):.3f}, std_target={np.std(frequency_targets):.3f})")
        logger.info(f"    Magnitude Predictions in Range [0,10]: {np.mean(magnitude_in_range):.1%}")
        logger.info(f"    Frequency Predictions in Range [0,100]: {np.mean(frequency_in_range):.1%}")
        
        # 🔧 NEW: Explain why accuracy might be misleading (Windows-safe)
        if magnitude_accuracy > 0.5 and magnitude_r2 < 0:
            logger.warning(f"  [WARNING] HIGH ACCURACY BUT POOR R²: Magnitude accuracy is misleading!")
            logger.warning(f"      This suggests predictions are wrong scale but happen to be close to some targets by chance.")
        
        if frequency_accuracy > 0.5 and frequency_r2 < 0:
            logger.warning(f"  [WARNING] HIGH ACCURACY BUT POOR R²: Frequency accuracy is misleading!")
            logger.warning(f"      This suggests predictions are wrong scale but happen to be close to some targets by chance.")
        
        return evaluation_results
    
    def run_comparison(self, max_epochs: int = 300, patience: int = 12) -> Dict:
        """
        Run the complete comparison between Simple LSTM and Attention LSTM.
        
        Args:
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary containing all comparison results
        """
        logger.info("Starting model comparison...")
        
        # Create models
        simple_lstm, attention_lstm = self.create_models()
        
        # Train Simple LSTM
        logger.info("=" * 60)
        logger.info("TRAINING SIMPLE SHARED LSTM")
        logger.info("=" * 60)
        simple_results = self.train_model(
            simple_lstm, "Simple Shared LSTM", max_epochs, patience
        )
        
        # Evaluate Simple LSTM
        simple_eval = self.evaluate_model(simple_lstm, "Simple Shared LSTM")
        
        # Train Attention LSTM
        logger.info("=" * 60)
        logger.info("TRAINING ATTENTION SHARED LSTM")
        logger.info("=" * 60)
        attention_results = self.train_model(
            attention_lstm, "Attention Shared LSTM", max_epochs, patience
        )
        
        # Evaluate Attention LSTM
        attention_eval = self.evaluate_model(attention_lstm, "Attention Shared LSTM")
        
        # Store results
        self.results['simple_lstm'] = {
            'training': simple_results,
            'evaluation': simple_eval
        }
        
        self.results['attention_lstm'] = {
            'training': attention_results,
            'evaluation': attention_eval
        }
        
        # Generate comparison
        self.results['comparison'] = self._generate_comparison()
        
        # Save results
        self._save_results()
        
        # Generate plots
        self._generate_comparison_plots()
        
        logger.info("Model comparison completed!")
        return self.results
    
    def _generate_comparison(self) -> Dict:
        """
        Generate comparison metrics between the two models.
        
        Returns:
            Dictionary containing comparison metrics
        """
        simple_eval = self.results['simple_lstm']['evaluation']
        attention_eval = self.results['attention_lstm']['evaluation']
        
        # Calculate improvements
        magnitude_improvements = {}
        frequency_improvements = {}
        
        for metric in ['mse', 'mae', 'rmse']:
            # Lower is better for these metrics
            simple_val = simple_eval['magnitude'][metric]
            attention_val = attention_eval['magnitude'][metric]
            improvement = ((simple_val - attention_val) / simple_val) * 100
            magnitude_improvements[metric] = improvement
        
        for metric in ['r2', 'correlation']:
            # Higher is better for these metrics
            simple_val = simple_eval['magnitude'][metric]
            attention_val = attention_eval['magnitude'][metric]
            improvement = ((attention_val - simple_val) / simple_val) * 100
            magnitude_improvements[metric] = improvement
        
        for metric in ['mse', 'mae', 'rmse']:
            simple_val = simple_eval['frequency'][metric]
            attention_val = attention_eval['frequency'][metric]
            improvement = ((simple_val - attention_val) / simple_val) * 100
            frequency_improvements[metric] = improvement
        
        for metric in ['r2', 'correlation']:
            simple_val = simple_eval['frequency'][metric]
            attention_val = attention_eval['frequency'][metric]
            improvement = ((attention_val - simple_val) / simple_val) * 100
            frequency_improvements[metric] = improvement
        
        # Training time comparison
        # 🔧 FIX: Add fallback for missing training_time to prevent KeyError
        simple_time = self.results['simple_lstm']['training'].get('training_time', 0.0)
        attention_time = self.results['attention_lstm']['training'].get('training_time', 0.0)
        
        # Handle case where training time is 0 (not recorded)
        if simple_time == 0.0 or attention_time == 0.0:
            logger.warning("Training time not recorded for one or both models. Using placeholder values.")
            simple_time = simple_time if simple_time > 0.0 else 1.0
            attention_time = attention_time if attention_time > 0.0 else 1.0
        
        time_ratio = attention_time / simple_time
        
        comparison = {
            'magnitude_improvements': magnitude_improvements,
            'frequency_improvements': frequency_improvements,
            'training_time_comparison': {
                'simple_lstm_time': simple_time,
                'attention_lstm_time': attention_time,
                'time_ratio': time_ratio
            },
            'summary': {
                'attention_better_magnitude': sum(1 for v in magnitude_improvements.values() if v > 0),
                'attention_better_frequency': sum(1 for v in frequency_improvements.values() if v > 0),
                'total_metrics': len(magnitude_improvements) + len(frequency_improvements)
            }
        }
        
        return comparison
    
    def _save_results(self):
        """Save all results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"model_comparison_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_copy = self.results.copy()
        
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
    
    def _generate_comparison_plots(self):
        """Generate comparison plots and save them."""
        # Training loss comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training losses
        simple_train = self.results['simple_lstm']['training']['train_losses']
        simple_val = self.results['simple_lstm']['training']['val_losses']
        attention_train = self.results['attention_lstm']['training']['train_losses']
        attention_val = self.results['attention_lstm']['training']['val_losses']
        
        # Create separate x-axes for each model since they may have trained for different numbers of epochs
        simple_epochs = range(1, len(simple_train) + 1)
        attention_epochs = range(1, len(attention_train) + 1)
        
        # Overall loss
        axes[0, 0].plot(simple_epochs, simple_train, 'b-', label='Simple LSTM Train', alpha=0.7)
        axes[0, 0].plot(simple_epochs, simple_val, 'b--', label='Simple LSTM Val', alpha=0.7)
        axes[0, 0].plot(attention_epochs, attention_train, 'r-', label='Attention LSTM Train', alpha=0.7)
        axes[0, 0].plot(attention_epochs, attention_val, 'r--', label='Attention LSTM Val', alpha=0.7)
        axes[0, 0].set_title('Training Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Magnitude losses
        simple_mag_train = self.results['simple_lstm']['training']['train_magnitude_losses']
        simple_mag_val = self.results['simple_lstm']['training']['val_magnitude_losses']
        attention_mag_train = self.results['attention_lstm']['training']['train_magnitude_losses']
        attention_mag_val = self.results['attention_lstm']['training']['val_magnitude_losses']
        
        axes[0, 1].plot(simple_epochs, simple_mag_train, 'b-', label='Simple LSTM Train', alpha=0.7)
        axes[0, 1].plot(simple_epochs, simple_mag_val, 'b--', label='Simple LSTM Val', alpha=0.7)
        axes[0, 1].plot(attention_epochs, attention_mag_train, 'r-', label='Attention LSTM Train', alpha=0.7)
        axes[0, 1].plot(attention_epochs, attention_mag_val, 'r--', label='Attention LSTM Val', alpha=0.7)
        axes[0, 1].set_title('Magnitude Loss Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency losses
        simple_freq_train = self.results['simple_lstm']['training']['train_frequency_losses']
        simple_freq_val = self.results['simple_lstm']['training']['val_frequency_losses']
        attention_freq_train = self.results['attention_lstm']['training']['train_frequency_losses']
        attention_freq_val = self.results['attention_lstm']['training']['val_frequency_losses']
        
        axes[1, 0].plot(simple_epochs, simple_freq_train, 'b-', label='Simple LSTM Train', alpha=0.7)
        axes[1, 0].plot(simple_epochs, simple_freq_val, 'b--', label='Simple LSTM Val', alpha=0.7)
        axes[1, 0].plot(attention_epochs, attention_freq_train, 'r-', label='Attention LSTM Train', alpha=0.7)
        axes[1, 0].plot(attention_epochs, attention_freq_val, 'r--', label='Attention LSTM Val', alpha=0.7)
        axes[1, 0].set_title('Frequency Loss Comparison')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance comparison bar chart
        metrics = ['MSE', 'MAE', 'RMSE', 'R²', 'Correlation']
        simple_mag_scores = [
            self.results['simple_lstm']['evaluation']['magnitude']['mse'],
            self.results['simple_lstm']['evaluation']['magnitude']['mae'],
            self.results['simple_lstm']['evaluation']['magnitude']['rmse'],
            self.results['simple_lstm']['evaluation']['magnitude']['r2'],
            self.results['simple_lstm']['evaluation']['magnitude']['correlation']
        ]
        attention_mag_scores = [
            self.results['attention_lstm']['evaluation']['magnitude']['mse'],
            self.results['attention_lstm']['evaluation']['magnitude']['mae'],
            self.results['attention_lstm']['evaluation']['magnitude']['rmse'],
            self.results['attention_lstm']['evaluation']['magnitude']['r2'],
            self.results['attention_lstm']['evaluation']['magnitude']['correlation']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, simple_mag_scores, width, label='Simple LSTM', alpha=0.7)
        axes[1, 1].bar(x + width/2, attention_mag_scores, width, label='Attention LSTM', alpha=0.7)
        axes[1, 1].set_title('Magnitude Performance Comparison')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"model_comparison_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to: {plot_file}")
    
    def print_comparison_summary(self):
        """Print a summary of the comparison results."""
        if not self.results.get('comparison'):
            logger.warning("No comparison results available. Run run_comparison() first.")
            return
        
        comparison = self.results['comparison']
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        # Magnitude improvements
        print("\nMAGNITUDE PREDICTION IMPROVEMENTS:")
        print("-" * 40)
        for metric, improvement in comparison['magnitude_improvements'].items():
            if improvement > 0:
                print(f"  {metric.upper()}: +{improvement:.2f}% (Attention LSTM better)")
            else:
                print(f"  {metric.upper()}: {improvement:.2f}% (Simple LSTM better)")
        
        # Frequency improvements
        print("\nFREQUENCY PREDICTION IMPROVEMENTS:")
        print("-" * 40)
        for metric, improvement in comparison['frequency_improvements'].items():
            if improvement > 0:
                print(f"  {metric.upper()}: +{improvement:.2f}% (Attention LSTM better)")
            else:
                print(f"  {metric.upper()}: {improvement:.2f}% (Simple LSTM better)")
        
        # Training time comparison
        time_comp = comparison['training_time_comparison']
        print(f"\nTRAINING TIME COMPARISON:")
        print("-" * 40)
        print(f"  Simple LSTM: {time_comp['simple_lstm_time']:.2f}s")
        print(f"  Attention LSTM: {time_comp['attention_lstm_time']:.2f}s")
        print(f"  Time ratio: {time_comp['time_ratio']:.2f}x")
        
        # Summary
        summary = comparison['summary']
        print(f"\nOVERALL SUMMARY:")
        print("-" * 40)
        print(f"  Attention LSTM better on {summary['attention_better_magnitude']}/{len(comparison['magnitude_improvements'])} magnitude metrics")
        print(f"  Attention LSTM better on {summary['attention_better_frequency']}/{len(comparison['frequency_improvements'])} frequency metrics")
        
        total_better = summary['attention_better_magnitude'] + summary['attention_better_frequency']
        total_metrics = summary['total_metrics']
        print(f"  Total: Attention LSTM better on {total_better}/{total_metrics} metrics")
        
        if total_better > total_metrics / 2:
            print("  🎯 CONCLUSION: Attention LSTM shows overall better performance!")
        elif total_better < total_metrics / 2:
            print("  ⚠️  CONCLUSION: Simple LSTM shows overall better performance!")
        else:
            print("  🤝 CONCLUSION: Models show similar performance!")
        
        print("=" * 80)


def main():
    """
    Main function to run the model comparison.
    This can be used as a standalone script or imported as a module.
    """
    # Example usage
    logger.info("Model Comparison Trainer - Example Usage")
    logger.info("This module provides tools to compare Simple LSTM vs Attention LSTM performance")
    
    # You would typically use this like:
    # trainer = ModelComparisonTrainer(train_loader, val_loader, test_loader, ...)
    # results = trainer.run_comparison()
    # trainer.print_comparison_summary()


if __name__ == "__main__":
    main()
