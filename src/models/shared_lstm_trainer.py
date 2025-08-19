#!/usr/bin/env python3
"""
Shared LSTM Trainer for Earthquake Forecasting

Implements training pipeline with:
- Time-based data splitting
- Early stopping with patience=12
- Adam optimizer with weight decay
- Weighted loss (MSE + Poisson NLL)
- Training for 300 epochs
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

from .shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss


class SharedLSTMTrainer:
    """
    Trainer for the shared LSTM earthquake forecasting model.
    """
    
    def __init__(self,
                 model: SharedLSTMModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 magnitude_weight: float = 1.0,      # α: weight for magnitude loss
                 frequency_weight: float = 3.0,      # β: weight for frequency loss
                 correlation_weight: float = 0.0,    # γ: weight for correlation penalty (default: disabled)
                 device: str = 'auto'):
        """
        Initialize the trainer.
        
        🔧 IMPROVEMENT: Added configurable loss weights α, β, γ and correlation penalty support.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.magnitude_weight = magnitude_weight      # α
        self.frequency_weight = frequency_weight      # β
        self.correlation_weight = correlation_weight  # γ
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = WeightedEarthquakeLoss(
            magnitude_weight=magnitude_weight,
            frequency_weight=frequency_weight,
            correlation_weight=correlation_weight
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 🔧 FIX: Add gradient clipping to prevent exploding gradients
        self.max_grad_norm = 1.0
        
        # 🔧 IMPROVEMENT: Add CosineAnnealingWarmRestarts scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,             # 🔧 FIX: First restart after 5 epochs (faster adaptation)
            T_mult=2,          # Double the restart interval each time
            eta_min=1e-7       # 🔧 FIX: Lower minimum learning rate
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_magnitude_losses = []
        self.train_frequency_losses = []
        self.val_magnitude_losses = []
        self.val_frequency_losses = []
        
        # 🔧 IMPROVEMENT: Enhanced metrics tracking
        self.train_magnitude_mae = []
        self.train_frequency_mae = []
        self.train_magnitude_corr = []
        self.train_frequency_corr = []
        self.val_magnitude_mae = []
        self.val_frequency_mae = []
        self.val_magnitude_corr = []
        self.val_frequency_corr = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10  # 🔧 IMPROVEMENT: Reduced to 10 (training done by epoch 30)
        self.patience_counter = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SharedLSTMTrainer initialized on {self.device}")
        self.logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        self.logger.info(f"Loss weights: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
        self.logger.info(f"Early stopping patience: {self.patience}")
        self.logger.info("CosineAnnealingWarmRestarts scheduler enabled (T_0=5, T_mult=2)")
        self.logger.info("FIXES APPLIED: Log1p+MSE loss, Dynamic beta, Cosine LR, early stopping patience=10")
        self.logger.info("ENHANCED: Lower LR (1e-4), Higher weight decay (1e-3), Gradient clipping, BatchNorm, Increased dropout")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_magnitude_loss = 0.0
        total_frequency_loss = 0.0
        
        # 🔧 IMPROVEMENT: Enhanced metrics tracking
        total_magnitude_mae = 0.0
        total_frequency_mae = 0.0
        total_magnitude_corr = 0.0
        total_frequency_corr = 0.0
        
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (input_seq, target_seq, metadata, _) in enumerate(progress_bar):
            # Move to device
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            metadata = metadata.to(self.device)
            
            # Forward pass
            magnitude_pred, frequency_log_rate_pred = self.model(input_seq, metadata)
            
            # Extract targets
            magnitude_true = target_seq[:, 0, 0]  # (batch_size,)
            frequency_true_normalized = target_seq[:, 0, 1]  # (batch_size,) - NORMALIZED
            
            # 🔧 FIX: Denormalize frequency for Poisson loss using single feature method
            frequency_true = torch.tensor([
                self.train_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                for freq in frequency_true_normalized
            ], dtype=torch.float32).to(self.device)
            
            # Compute loss
            loss = self.criterion(magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                                magnitude_true, frequency_true)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 🔧 FIX: Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Get loss components for monitoring
            loss_components = self.criterion.get_loss_components(
                magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                magnitude_true, frequency_true
            )
            
            # 🔧 IMPROVEMENT: Calculate additional metrics
            magnitude_mae = F.l1_loss(magnitude_pred.squeeze(), magnitude_true)
            frequency_mae = F.l1_loss(torch.exp(frequency_log_rate_pred.squeeze()), frequency_true)
            
            # Calculate correlations
            magnitude_corr = self._pearson_correlation(magnitude_pred.squeeze(), magnitude_true)
            frequency_corr = self._pearson_correlation(torch.exp(frequency_log_rate_pred.squeeze()), frequency_true)
            
            # Update totals
            total_loss += loss.item()
            total_magnitude_loss += loss_components['magnitude_loss'].item()
            total_frequency_loss += loss_components['frequency_loss'].item()
            total_magnitude_mae += magnitude_mae.item()
            total_frequency_mae += frequency_mae.item()
            total_magnitude_corr += magnitude_corr.item()
            total_frequency_corr += frequency_corr.item()
            num_batches += 1
            
            # Update progress bar with enhanced metrics
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Mag': f"{loss_components['magnitude_loss'].item():.4f}",
                'Freq': f"{loss_components['frequency_loss'].item():.4f}",
                'Mag_MAE': f"{magnitude_mae.item():.4f}",
                'Freq_MAE': f"{frequency_mae.item():.4f}"
            })
        
        # Return average losses and metrics
        return {
            'total_loss': total_loss / num_batches,
            'magnitude_loss': total_magnitude_loss / num_batches,
            'frequency_loss': total_frequency_loss / num_batches,
            'magnitude_mae': total_magnitude_mae / num_batches,
            'frequency_mae': total_frequency_mae / num_batches,
            'magnitude_correlation': total_magnitude_corr / num_batches,
            'frequency_correlation': total_frequency_corr / num_batches
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_magnitude_loss = 0.0
        total_frequency_loss = 0.0
        
        # 🔧 IMPROVEMENT: Enhanced metrics tracking
        total_magnitude_mae = 0.0
        total_frequency_mae = 0.0
        total_magnitude_corr = 0.0
        total_frequency_corr = 0.0
        
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq, metadata, _ in self.val_loader:
                # Move to device
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                metadata = metadata.to(self.device)
                
                # Forward pass
                magnitude_pred, frequency_log_rate_pred = self.model(input_seq, metadata)
                
                # Extract targets
                magnitude_true = target_seq[:, 0, 0]
                frequency_true_normalized = target_seq[:, 0, 1]
                
                # 🔧 FIX: Denormalize frequency for Poisson loss using single feature method
                frequency_true = torch.tensor([
                    self.val_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                    for freq in frequency_true_normalized
                ], dtype=torch.float32).to(self.device)
                
                # Compute loss
                loss = self.criterion(magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                                    magnitude_true, frequency_true)
                
                # Get loss components
                loss_components = self.criterion.get_loss_components(
                    magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                    magnitude_true, frequency_true
                )
                
                # 🔧 IMPROVEMENT: Calculate additional metrics
                magnitude_mae = F.l1_loss(magnitude_pred.squeeze(), magnitude_true)
                frequency_mae = F.l1_loss(torch.exp(frequency_log_rate_pred.squeeze()), frequency_true)
                
                # Calculate correlations
                magnitude_corr = self._pearson_correlation(magnitude_pred.squeeze(), magnitude_true)
                frequency_corr = self._pearson_correlation(torch.exp(frequency_log_rate_pred.squeeze()), frequency_true)
                
                # Update totals
                total_loss += loss.item()
                total_magnitude_loss += loss_components['magnitude_loss'].item()
                total_frequency_loss += loss_components['frequency_loss'].item()
                total_magnitude_mae += magnitude_mae.item()
                total_frequency_mae += frequency_mae.item()
                total_magnitude_corr += magnitude_corr.item()
                total_frequency_corr += frequency_corr.item()
                num_batches += 1
        
        # Return average losses and metrics
        return {
            'total_loss': total_loss / num_batches,
            'magnitude_loss': total_magnitude_loss / num_batches,
            'frequency_loss': total_frequency_loss / num_batches,
            'magnitude_mae': total_magnitude_mae / num_batches,
            'frequency_mae': total_frequency_mae / num_batches,
            'magnitude_correlation': total_magnitude_corr / num_batches,
            'frequency_correlation': total_frequency_corr / num_batches
        }
    
    def train(self, 
              max_epochs: int = 300,
              save_path: Optional[str] = None,
              save_best: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and enhanced logging.
        
        🔧 IMPROVEMENT: Added scheduler step, enhanced metrics tracking, and comprehensive logging.
        """
        self.logger.info(f"Starting training for up to {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # 🔧 IMPROVEMENT: Store enhanced metrics
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.train_magnitude_losses.append(train_metrics['magnitude_loss'])
            self.train_frequency_losses.append(train_metrics['frequency_loss'])
            self.val_magnitude_losses.append(val_metrics['magnitude_loss'])
            self.val_frequency_losses.append(val_metrics['frequency_loss'])
            
            # 🔧 IMPROVEMENT: Store additional metrics
            self.train_magnitude_mae.append(train_metrics['magnitude_mae'])
            self.train_frequency_mae.append(train_metrics['frequency_mae'])
            self.train_magnitude_corr.append(train_metrics['magnitude_correlation'])
            self.train_frequency_corr.append(train_metrics['frequency_correlation'])
            self.val_magnitude_mae.append(val_metrics['magnitude_mae'])
            self.val_frequency_mae.append(val_metrics['frequency_mae'])
            self.val_magnitude_corr.append(val_metrics['magnitude_correlation'])
            self.val_frequency_corr.append(val_metrics['frequency_correlation'])
            
            # 🔧 IMPROVEMENT: Comprehensive logging for each epoch
            self.logger.info(
                f"Epoch {epoch+1}/{max_epochs}:\n"
                f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                f"Mag: {train_metrics['magnitude_loss']:.4f}, "
                f"Freq: {train_metrics['frequency_loss']:.4f}, "
                f"Mag_MAE: {train_metrics['magnitude_mae']:.4f}, "
                f"Freq_MAE: {train_metrics['frequency_mae']:.4f}, "
                f"Mag_Corr: {train_metrics['magnitude_correlation']:.4f}, "
                f"Freq_Corr: {train_metrics['frequency_correlation']:.4f}\n"
                f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                f"Mag: {val_metrics['magnitude_loss']:.4f}, "
                f"Freq: {val_metrics['frequency_loss']:.4f}, "
                f"Mag_MAE: {val_metrics['magnitude_mae']:.4f}, "
                f"Freq_MAE: {val_metrics['frequency_mae']:.4f}, "
                f"Mag_Corr: {val_metrics['magnitude_correlation']:.4f}, "
                f"Freq_Corr: {val_metrics['frequency_correlation']:.4f}"
            )
            
            # 🔧 IMPROVEMENT: Step the scheduler (CosineAnnealingWarmRestarts doesn't need validation loss)
            self.scheduler.step()
            
            # 🔧 IMPROVEMENT: Update dynamic β weight
            self.criterion.update_beta(epoch, val_metrics['total_loss'])
            
            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                # Save best model
                if save_best and save_path:
                    self.save_model(save_path)
                    self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_magnitude_losses': self.train_magnitude_losses,
            'train_frequency_losses': self.train_frequency_losses,
            'val_magnitude_losses': self.val_magnitude_losses,
            'val_frequency_losses': self.val_frequency_losses,
            # 🔧 IMPROVEMENT: Return enhanced metrics
            'train_magnitude_mae': self.train_magnitude_mae,
            'train_frequency_mae': self.train_frequency_mae,
            'train_magnitude_corr': self.train_magnitude_corr,
            'train_frequency_corr': self.train_frequency_corr,
            'val_magnitude_mae': self.val_magnitude_mae,
            'val_frequency_mae': self.val_frequency_mae,
            'val_magnitude_corr': self.val_magnitude_corr,
            'val_frequency_corr': self.val_frequency_corr
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.
        """
        self.model.eval()
        total_loss = 0.0
        total_magnitude_loss = 0.0
        total_frequency_loss = 0.0
        magnitude_predictions = []
        frequency_predictions = []
        frequency_count_predictions = []  # NEW: exp(log(λ)) predictions
        magnitude_targets = []
        frequency_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq, metadata, _ in data_loader:
                # Move to device
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                metadata = metadata.to(self.device)
                
                # Forward pass
                magnitude_pred, frequency_log_rate_pred = self.model(input_seq, metadata)
                
                # Extract targets
                magnitude_true = target_seq[:, 0, 0]
                frequency_true_normalized = target_seq[:, 0, 1]
                
                # 🔧 FIX: Denormalize frequency for Poisson loss using single feature method
                frequency_true = torch.tensor([
                    data_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                    for freq in frequency_true_normalized
                ], dtype=torch.float32).to(self.device)
                
                # Compute loss
                loss = self.criterion(magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                                    magnitude_true, frequency_true)
                
                # Get loss components
                loss_components = self.criterion.get_loss_components(
                    magnitude_pred.squeeze(), frequency_log_rate_pred.squeeze(),
                    magnitude_true, frequency_true
                )
                
                # Store predictions and targets
                magnitude_predictions.extend(magnitude_pred.squeeze().cpu().numpy())
                frequency_predictions.extend(frequency_log_rate_pred.squeeze().cpu().numpy())
                
                # NEW: Convert log-rate to expected counts for evaluation
                frequency_count_pred = self.model.predict_frequency_counts(frequency_log_rate_pred.squeeze())
                frequency_count_predictions.extend(frequency_count_pred.cpu().numpy())
                
                magnitude_targets.extend(magnitude_true.cpu().numpy())
                frequency_targets.extend(frequency_true.cpu().numpy())
                
                # Update totals
                total_loss += loss.item()
                total_magnitude_loss += loss_components['magnitude_loss'].item()
                total_frequency_loss += loss_components['frequency_loss'].item()
                num_batches += 1
        
        # Calculate additional metrics
        magnitude_predictions = np.array(magnitude_predictions)
        frequency_predictions = np.array(frequency_predictions)
        frequency_count_predictions = np.array(frequency_count_predictions)  # NEW
        magnitude_targets = np.array(magnitude_targets)
        frequency_targets = np.array(frequency_targets)
        
        # Magnitude metrics (UNCHANGED)
        magnitude_mse = np.mean((magnitude_predictions - magnitude_targets) ** 2)
        magnitude_mae = np.mean(np.abs(magnitude_predictions - magnitude_targets))
        
        # 🔧 FIX: Better correlation calculation with error handling
        try:
            magnitude_corr = np.corrcoef(magnitude_predictions, magnitude_targets)[0, 1]
            if np.isnan(magnitude_corr):
                magnitude_corr = 0.0
        except (IndexError, ValueError):
            magnitude_corr = 0.0
        
        # FREQUENCY METRICS: UPDATED FOR LOG1P + MSE FORMULATION
        # Log1p + MSE loss (already computed above)
        frequency_log1p_mse = total_frequency_loss / num_batches
        
        # 🔧 DEBUG: Check frequency prediction ranges
        freq_pred_min, freq_pred_max = np.min(frequency_count_predictions), np.max(frequency_count_predictions)
        freq_true_min, freq_true_max = np.min(frequency_targets), np.max(frequency_targets)
        self.logger.info(f"Frequency predictions range: {freq_pred_min:.2f} to {freq_pred_max:.2f}")
        self.logger.info(f"Frequency targets range: {freq_true_min:.2f} to {freq_true_max:.2f}")
        
        # 🔧 ENHANCED: Add interpretability metrics
        # Magnitude accuracy (within ±0.5 magnitude units)
        magnitude_accuracy = np.mean(np.abs(magnitude_predictions - magnitude_targets) <= 0.5)
        
        # Frequency accuracy (within ±1 count)
        frequency_accuracy = np.mean(np.abs(frequency_count_predictions - frequency_targets) <= 1.0)
        
        # Relative error metrics
        magnitude_rel_error = np.mean(np.abs(magnitude_predictions - magnitude_targets) / (np.abs(magnitude_targets) + 1e-8))
        frequency_rel_error = np.mean(np.abs(frequency_count_predictions - frequency_targets) / (np.abs(frequency_targets) + 1e-8))
        
        # Log interpretability metrics
        self.logger.info(f"Magnitude Accuracy (±0.5): {magnitude_accuracy:.3f} ({magnitude_accuracy*100:.1f}%)")
        self.logger.info(f"Frequency Accuracy (±1): {frequency_accuracy:.3f} ({frequency_accuracy*100:.1f}%)")
        self.logger.info(f"Magnitude Relative Error: {magnitude_rel_error:.3f}")
        self.logger.info(f"Frequency Relative Error: {frequency_rel_error:.3f}")
        
        # MAE on expected counts (exp(log(λ)))
        frequency_mae = np.mean(np.abs(frequency_count_predictions - frequency_targets))
        
        # 🔧 FIX: Better correlation calculation with error handling
        try:
            frequency_corr = np.corrcoef(frequency_count_predictions, frequency_targets)[0, 1]
            if np.isnan(frequency_corr):
                frequency_corr = 0.0
        except (IndexError, ValueError):
            frequency_corr = 0.0
        
        # MSE on expected counts (for comparison with old method)
        frequency_mse = np.mean((frequency_count_predictions - frequency_targets) ** 2)
        
        return {
            'total_loss': total_loss / num_batches,
            'magnitude_loss': total_magnitude_loss / num_batches,
            'frequency_loss': total_frequency_loss / num_batches,
            'magnitude_mse': magnitude_mse,
            'magnitude_mae': magnitude_mae,
            'magnitude_correlation': magnitude_corr,
            'frequency_log1p_mse': frequency_log1p_mse,     # NEW: Log1p + MSE
            'frequency_mae': frequency_mae,                  # NEW: MAE on exp(log(λ))
            'frequency_correlation': frequency_corr,         # NEW: Correlation on exp(log(λ))
            'frequency_mse': frequency_mse,                  # NEW: MSE on exp(log(λ)) for comparison
            # 🔧 ENHANCED: Interpretability metrics
            'magnitude_accuracy': magnitude_accuracy,        # NEW: Accuracy within ±0.5
            'frequency_accuracy': frequency_accuracy,        # NEW: Accuracy within ±1
            'magnitude_rel_error': magnitude_rel_error,      # NEW: Relative error
            'frequency_rel_error': frequency_rel_error       # NEW: Relative error
        }
    
    def save_model(self, path: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_magnitude_losses': self.train_magnitude_losses,
                'train_frequency_losses': self.train_frequency_losses,
                'val_magnitude_losses': self.val_magnitude_losses,
                'val_frequency_losses': self.val_frequency_losses,
                # 🔧 IMPROVEMENT: Store enhanced metrics
                'train_magnitude_mae': self.train_magnitude_mae,
                'train_frequency_mae': self.train_frequency_mae,
                'train_magnitude_corr': self.train_magnitude_corr,
                'train_frequency_corr': self.train_frequency_corr,
                'val_magnitude_mae': self.val_magnitude_mae,
                'val_frequency_mae': self.val_frequency_mae,
                'val_magnitude_corr': self.val_magnitude_corr,
                'val_frequency_corr': self.val_frequency_corr
            }
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Restore training history
        history = checkpoint['training_history']
        self.train_losses = history['train_losses']
        self.val_losses = history['val_losses']
        self.train_magnitude_losses = history['train_magnitude_losses']
        self.train_frequency_losses = history['train_frequency_losses']
        self.val_magnitude_losses = history['val_magnitude_losses']
        self.val_frequency_losses = history['val_frequency_losses']
        
        # 🔧 IMPROVEMENT: Restore enhanced metrics
        if 'train_magnitude_mae' in history:
            self.train_magnitude_mae = history['train_magnitude_mae']
            self.train_frequency_mae = history['train_frequency_mae']
            self.train_magnitude_corr = history['train_magnitude_corr']
            self.train_frequency_corr = history['train_frequency_corr']
            self.val_magnitude_mae = history['val_magnitude_mae']
            self.val_frequency_mae = history['val_frequency_mae']
            self.val_magnitude_corr = history['val_magnitude_corr']
            self.val_frequency_corr = history['val_frequency_corr']
        
        self.logger.info(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history with enhanced metrics."""
        # 🔧 IMPROVEMENT: Create a larger figure to accommodate all metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Magnitude loss
        axes[0, 1].plot(self.train_magnitude_losses, label='Train')
        axes[0, 1].plot(self.val_magnitude_losses, label='Validation')
        axes[0, 1].set_title('Magnitude Loss (MSE)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Frequency loss
        axes[0, 2].plot(self.train_frequency_losses, label='Train')
        axes[0, 2].plot(self.val_frequency_losses, label='Validation')
        axes[0, 2].set_title('Frequency Loss (Poisson NLL)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 🔧 IMPROVEMENT: Magnitude MAE
        if hasattr(self, 'train_magnitude_mae') and self.train_magnitude_mae:
            axes[1, 0].plot(self.train_magnitude_mae, label='Train')
            axes[1, 0].plot(self.val_magnitude_mae, label='Validation')
            axes[1, 0].set_title('Magnitude MAE')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 🔧 IMPROVEMENT: Frequency MAE
        if hasattr(self, 'train_frequency_mae') and self.train_frequency_mae:
            axes[1, 1].plot(self.train_frequency_mae, label='Train')
            axes[1, 1].plot(self.val_frequency_mae, label='Validation')
            axes[1, 1].set_title('Frequency MAE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # 🔧 IMPROVEMENT: Magnitude Correlation
        if hasattr(self, 'train_magnitude_corr') and self.train_magnitude_corr:
            axes[1, 2].plot(self.train_magnitude_corr, label='Train')
            axes[1, 2].plot(self.val_magnitude_corr, label='Validation')
            axes[1, 2].set_title('Magnitude Correlation')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Correlation')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        # 🔧 IMPROVEMENT: Frequency Correlation
        if hasattr(self, 'train_frequency_corr') and self.train_frequency_corr:
            axes[2, 0].plot(self.train_frequency_corr, label='Train')
            axes[2, 0].plot(self.val_frequency_corr, label='Validation')
            axes[2, 0].set_title('Frequency Correlation')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Correlation')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        
        # Combined view
        axes[2, 1].plot(self.train_losses, label='Train Total', alpha=0.7)
        axes[2, 1].plot(self.val_losses, label='Val Total', alpha=0.7)
        axes[2, 1].plot(self.train_magnitude_losses, label='Train Mag', alpha=0.5)
        axes[2, 1].plot(self.train_frequency_losses, label='Train Freq', alpha=0.5)
        axes[2, 1].set_title('All Losses')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Learning rate (if scheduler is used)
        if hasattr(self, 'scheduler'):
            axes[2, 2].plot([self.optimizer.param_groups[0]['lr']] * len(self.train_losses), 
                           label='Learning Rate', color='red', linestyle='--')
            axes[2, 2].set_title('Learning Rate')
            axes[2, 2].set_xlabel('Epoch')
            axes[2, 2].set_ylabel('LR')
            axes[2, 2].legend()
            axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        # plt.show()  # Commented out to prevent terminal issues
        plt.close()   # Close figure to free memory

    def _pearson_correlation(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation coefficient between predictions and targets."""
        # Remove batch dimension if present
        if pred.dim() > 1:
            pred = pred.squeeze()
        if true.dim() > 1:
            true = true.squeeze()
        
        # Calculate means
        pred_mean = torch.mean(pred)
        true_mean = torch.mean(true)
        
        # Calculate correlation
        numerator = torch.sum((pred - pred_mean) * (true - true_mean))
        denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((true - true_mean) ** 2))
        
        # Handle division by zero
        if denominator == 0:
            return torch.tensor(0.0, device=pred.device)
        
        correlation = numerator / denominator
        return correlation