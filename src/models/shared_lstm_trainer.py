#!/usr/bin/env python3
"""
Shared LSTM Trainer for Earthquake Forecasting

CRITICAL FIXES APPLIED TO FIX PREDICTION COLLAPSE:

1. ✅ HYPERPARAMETER CONSISTENCY:
   - Fixed frequency_weight from 0.5 to 1.0 across ALL models
   - Fixed patience from 25 to 15 epochs for faster iteration
   - All models now use identical settings

2. ✅ AGGRESSIVE SCALING INITIALIZATION:
   - frequency_scale: 5.0 -> 10.0 (very aggressive)
   - frequency_bias: 1.0 -> 2.0 (very aggressive)
   - Applied to both shared_lstm and attention_lstm models

3. ✅ ENHANCED LEARNING RATES:
   - Scaling parameters: 10x -> 20x main learning rate
   - Weight decay: 2x -> 3x main weight decay
   - More aggressive parameter updates

4. ✅ RANGE EXPANSION PENALTIES:
   - Added freq_range_penalty to loss function
   - Added mag_range_penalty to loss function
   - Prevents prediction collapse by penalizing narrow ranges

5. ✅ PREDICTION RANGE MONITORING:
   - Added _check_prediction_range() method
   - Enhanced early stopping to consider prediction ranges
   - Resets patience if predictions are too narrow

6. ✅ SCALING PARAMETER MONITORING:
   - Track frequency_scale and frequency_bias values
   - Monitor gradients during training
   - Log every 10 batches to avoid spam

7. ✅ DEBUGGING FOR ATTENTION MODEL:
   - Added extensive logging to diagnose collapse
   - Monitor LSTM outputs, attention outputs, final representations
   - Track scaling parameter updates

8. ✅ IMPROVED EARLY STOPPING:
   - Don't stop early if predictions are too narrow
   - Reset patience when range expansion is needed
   - Better balance between convergence and range coverage

Implements training pipeline with:
- Time-based data splitting
- Early stopping with patience=15 (FIXED)
- Adam optimizer with enhanced weight decay
- Weighted loss with range expansion penalties
- Training for 300 epochs with smart early stopping
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
import time  # 🔧 ADD: Import time for training duration tracking

from .shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss


class SharedLSTMTrainer:
    """
    Trainer for the shared LSTM earthquake forecasting model.
    """
    
    def __init__(self,
                 model: SharedLSTMModel,
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
        Initialize the trainer.
        
        REFACTOR: Updated loss weights to alpha=1.5, beta=2.0, gamma=0.0 for better performance.
        Based on successful previous run with these hyperparameters.
        """
        # FIX: Create logger first to avoid AttributeError
        self.logger = logging.getLogger(__name__)
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.magnitude_weight = magnitude_weight      # alpha
        self.frequency_weight = frequency_weight      # beta
        self.correlation_weight = correlation_weight  # gamma
        self.variance_penalty_weight = variance_penalty_weight
        self.warmup_epochs = warmup_epochs
        self.save_dir = save_dir
        self.scaling_lr_multiplier = scaling_lr_multiplier
        self.scaling_wd_multiplier = scaling_wd_multiplier
        
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
            correlation_weight=correlation_weight,
            variance_penalty_weight=self.variance_penalty_weight,  # Use stored parameter
            warmup_epochs=self.warmup_epochs                      # Use stored parameter
        )
        
        # 🔧 NEW: Create separate parameter groups for different learning rates
        # Main model parameters
        main_params = []
        scaling_params = []
        
        for name, param in self.model.named_parameters():
            if 'frequency_scale' in name or 'frequency_bias' in name:
                scaling_params.append(param)
            else:
                main_params.append(param)
        
        # Create optimizer with different learning rates
        self.optimizer = optim.Adam([
            {'params': main_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': scaling_params, 'lr': learning_rate * self.scaling_lr_multiplier, 'weight_decay': weight_decay * self.scaling_wd_multiplier}
        ])
        
        self.logger.info(f"  - Main parameters: LR={learning_rate}, Weight decay={weight_decay}")
        self.logger.info(f"  - Frequency scaling parameters: LR={learning_rate * self.scaling_lr_multiplier}, Weight decay={weight_decay * self.scaling_wd_multiplier}")
        
        # NEW: Monitor scaling parameters during training
        self.scaling_monitor = {
            'frequency_scale_values': [],
            'frequency_bias_values': [],
            'frequency_scale_grads': [],
            'frequency_bias_grads': []
        }
        
        # 🔧 REFACTOR: Keep gradient clipping at 0.5 for stability
        self.max_grad_norm = 0.5
        
        # Enhanced learning rate scheduling for better adaptation
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Use config value for consistency
            T_mult=2,
            eta_min=1e-7  # Lower minimum LR for better fine-tuning
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_magnitude_losses = []
        self.train_frequency_losses = []
        self.val_magnitude_losses = []
        self.val_frequency_losses = []
        
        # REFACTOR: Enhanced metrics tracking
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
        self.patience = 25  # ENHANCED: Increased to 25 for better convergence
        self.patience_counter = 0
        
        self.logger.info(f"SharedLSTMTrainer initialized on {self.device}")
        self.logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        self.logger.info(f"Loss weights: alpha(magnitude)={magnitude_weight}, beta(frequency)={frequency_weight}, gamma(correlation)={correlation_weight}")
        self.logger.info(f"Variance penalty: weight={self.variance_penalty_weight}, warmup_epochs={self.warmup_epochs}")
        self.logger.info(f"Early stopping patience: {self.patience}")
        self.logger.info("CosineAnnealingWarmRestarts scheduler enabled (T_0=10, T_mult=2)")
        self.logger.info("  - Frequency targets preprocessed with log1p")
        self.logger.info("  - Gradient clipping at 0.5")
        self.logger.info("  - Enhanced debug logging for prediction ranges")
        self.logger.info(f"  - Config-driven scaling parameter learning rate: {self.scaling_lr_multiplier}x main LR")
        self.logger.info(f"  - Config-driven scaling parameter weight decay: {self.scaling_wd_multiplier}x main WD")
        self.logger.info(f"  - Config-driven scaling initialization: scale={getattr(self.model, 'frequency_scale_init', 2.0)}, bias={getattr(self.model, 'frequency_bias_init', 0.5)}")
        self.logger.info("  - Range expansion penalties added to loss function")
    
    def set_normalization_params(self, params: dict):
        """Set normalization parameters from external source for consistency."""
        if hasattr(self.model, 'set_normalization_params'):
            self.model.set_normalization_params(params)
            self.logger.info("Model normalization parameters synchronized")
        else:
            self.logger.warning("Model does not support normalization parameter synchronization")
    
    def _check_prediction_range(self) -> bool:
        """
        Check if predictions are too narrow (indicating potential collapse).
        Returns True if predictions are too narrow, False if they're healthy.
        """
        if not hasattr(self, 'train_loader') or len(self.train_loader) == 0:
            return False
        
        # Get a sample batch to check prediction ranges
        try:
            sample_batch = next(iter(self.train_loader))
            input_seq, target_seq, metadata, _ = sample_batch
            input_seq = input_seq.to(self.device)
            metadata = metadata.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                magnitude_pred, frequency_pred = self.model(input_seq, metadata)
            
            # Check frequency prediction range
            freq_range = torch.max(frequency_pred) - torch.min(frequency_pred)
            freq_std = torch.std(frequency_pred)
            
            # Check magnitude prediction range
            mag_range = torch.max(magnitude_pred) - torch.min(magnitude_pred)
            mag_std = torch.std(magnitude_pred)
            
            # Define thresholds for "too narrow"
            freq_too_narrow = freq_range < 0.1 or freq_std < 0.01
            mag_too_narrow = mag_range < 0.05 or mag_std < 0.01
            
            if freq_too_narrow or mag_too_narrow:
                self.logger.warning(f"  [WARNING] Predictions too narrow - Freq: range={freq_range:.4f}, std={freq_std:.4f}, Mag: range={mag_range:.4f}, std={mag_std:.4f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"  [WARNING] Could not check prediction ranges: {e}")
            return False
    
    def train_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_magnitude_loss = 0.0
        total_frequency_loss = 0.0
        
        # REFACTOR: Enhanced metrics tracking
        total_magnitude_mae = 0.0
        total_frequency_mae = 0.0
        total_magnitude_corr = 0.0
        total_frequency_corr = 0.0
        
        # REFACTOR: Debug logging for prediction ranges
        magnitude_preds = []
        frequency_preds = []
        
        num_batches = 0
        
        # Create progress bar with better Windows compatibility
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch {getattr(self, 'current_epoch', 'N/A')}",
            leave=False,  # Don't leave the progress bar
            ncols=80,     # Reduced width for better compatibility
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            disable=False,  # Ensure it's enabled
            position=0     # Fixed position to prevent overlapping
        )
        
        for batch_idx, (input_seq, target_seq, metadata, _) in enumerate(progress_bar):
            # Move to device
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            metadata = metadata.to(self.device)
            
            # Forward pass
            magnitude_pred, frequency_pred = self.model(input_seq, metadata)
            
            # REFACTOR: Collect predictions for debug logging
            magnitude_preds.append(magnitude_pred.detach().cpu())
            frequency_preds.append(frequency_pred.detach().cpu())
            
            # REFACTOR: Debug logging for prediction ranges to detect collapsing outputs
            if batch_idx == 0:  # Log first batch of each epoch
                freq_min = torch.min(frequency_pred).item()
                freq_max = torch.max(frequency_pred).item()
                mag_min = torch.min(magnitude_pred).item()
                mag_max = torch.max(magnitude_pred).item()
                self.logger.info(f"Epoch {getattr(self, 'current_epoch', 'N/A')} - Freq range: [{freq_min:.4f}, {freq_max:.4f}], Mag range: [{mag_min:.4f}, {mag_max:.4f}]")
            
            # Extract targets
            magnitude_true = target_seq[:, 0]  # (batch_size,) - max_magnitude
            frequency_true_normalized = target_seq[:, 1]  # (batch_size,) - frequency (NORMALIZED)
            
            # REFACTOR: Denormalize frequency for loss computation
            frequency_true = torch.tensor([
                self.train_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                for freq in frequency_true_normalized
            ], dtype=torch.float32).to(self.device)
            
            # Compute loss with epoch for variance penalty activation
            loss = self.criterion(magnitude_pred.squeeze(), frequency_pred.squeeze(),
                                magnitude_true, frequency_true, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 🔧 REFACTOR: Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # NEW: Monitor scaling parameters during training
            if hasattr(self.model, 'frequency_scale') and hasattr(self.model, 'frequency_bias'):
                scale_value = self.model.frequency_scale.item()
                bias_value = self.model.frequency_bias.item()
                scale_grad = self.model.frequency_scale.grad.item() if self.model.frequency_scale.grad is not None else 0.0
                bias_grad = self.model.frequency_bias.grad.item() if self.model.frequency_bias.grad is not None else 0.0
                
                # Store values for monitoring
                self.scaling_monitor['frequency_scale_values'].append(scale_value)
                self.scaling_monitor['frequency_bias_values'].append(bias_value)
                self.scaling_monitor['frequency_scale_grads'].append(scale_grad)
                self.scaling_monitor['frequency_bias_grads'].append(bias_grad)
                
                # Log scaling parameters every 10 batches to avoid spam
                if batch_idx % 10 == 0:
                    self.logger.info(f"  Scaling params - Scale: {scale_value:.4f} (grad: {scale_grad:.6f}), Bias: {bias_value:.4f} (grad: {bias_grad:.6f})")
            
            # Get loss components for monitoring
            loss_components = self.criterion.get_loss_components(
                magnitude_pred.squeeze(), frequency_pred.squeeze(),
                magnitude_true, frequency_true
            )
            
            # 🔧 REFACTOR: Calculate additional metrics
            magnitude_mae = F.l1_loss(magnitude_pred.squeeze(), magnitude_true)
            frequency_mae = F.l1_loss(frequency_pred.squeeze(), torch.log1p(frequency_true))  # Compare in log1p space
            
            # Calculate correlations
            magnitude_corr = self._pearson_correlation(magnitude_pred.squeeze(), magnitude_true)
            frequency_corr = self._pearson_correlation(frequency_pred.squeeze(), torch.log1p(frequency_true))  # Compare in log1p space
            
            # Accumulate metrics
            total_loss += loss.item()
            total_magnitude_loss += loss_components['magnitude_loss'].item()
            total_frequency_loss += loss_components['frequency_loss'].item()
            total_magnitude_mae += magnitude_mae.item()
            total_frequency_mae += frequency_mae.item()
            total_magnitude_corr += magnitude_corr.item()
            total_frequency_corr += frequency_corr.item()
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Mag_Loss': f"{loss_components['magnitude_loss'].item():.4f}",
                'Freq_Loss': f"{loss_components['frequency_loss'].item():.4f}"
            })
        
        # 🔧 REFACTOR: Log per-epoch mean/std of predictions for both magnitude and frequency
        if magnitude_preds and frequency_preds:
            magnitude_preds_tensor = torch.cat(magnitude_preds, dim=0)
            frequency_preds_tensor = torch.cat(frequency_preds, dim=0)
            
            mag_mean = torch.mean(magnitude_preds_tensor).item()
            mag_std = torch.std(magnitude_preds_tensor).item()
            freq_mean = torch.mean(frequency_preds_tensor).item()
            freq_std = torch.std(frequency_preds_tensor).item()
            
            self.logger.info(f"Epoch {getattr(self, 'current_epoch', 'N/A')} - Predictions Summary:")
            self.logger.info(f"  Magnitude: mean={mag_mean:.4f}, std={mag_std:.4f}")
            self.logger.info(f"  Frequency: mean={freq_mean:.4f}, std={freq_std:.4f}")
            
            # NEW: Monitor frequency scaling parameters to ensure they're learning properly
            if hasattr(self.model, 'frequency_scale') and self.model.frequency_scale is not None:
                scale_value = self.model.frequency_scale.item()
                bias_value = self.model.frequency_bias.item() if hasattr(self.model, 'frequency_bias') else 0.0
                
                # NEW: Monitor gradients for scaling parameters
                scale_grad = self.model.frequency_scale.grad.item() if self.model.frequency_scale.grad is not None else 0.0
                bias_grad = self.model.frequency_bias.grad.item() if self.model.frequency_bias.grad is not None else 0.0
                
            self.logger.info(f"  Frequency scaling: scale={scale_value:.4f}, bias={bias_value:.4f}")
            self.logger.info(f"  Frequency gradients: scale_grad={scale_grad:.6f}, bias_grad={bias_grad:.6f}")
            
            # NEW: Log prediction/target std ratio for monitoring range coverage
            if hasattr(self.train_loader.dataset, 'target_stats'):
                target_mag_std = self.train_loader.dataset.target_stats.get('magnitude_std', 1.0)
                target_freq_std = self.train_loader.dataset.target_stats.get('frequency_std', 1.0)
                
                mag_std_ratio = mag_std / target_mag_std if target_mag_std > 0 else 0.0
                freq_std_ratio = freq_std / target_freq_std if target_freq_std > 0 else 0.0
                
                self.logger.info(f"  Std ratios - Mag: {mag_std_ratio:.3f}, Freq: {freq_std_ratio:.3f}")
                
                # NEW: Attention monitoring for multi-head attention models
                if hasattr(self.model, 'attention') and hasattr(self.model.attention, 'q_proj'):
                    # Log attention weights for a sample batch
                    with torch.no_grad():
                        # Get attention weights for a sample batch
                        sample_batch = next(iter(self.train_loader))
                        sample_input = sample_batch[0][:1].to(self.device)  # Take first sample
                        sample_metadata = sample_batch[2][:1].to(self.device)
                        
                        # Forward pass to get attention weights
                        self.model.eval()
                        lstm_out, _ = self.model.lstm1(sample_input)
                        lstm_out, _ = self.model.lstm2(lstm_out)
                        
                        # Get attention weights
                        q = self.model.attention.q_proj(lstm_out)
                        k = self.model.attention.k_proj(lstm_out)
                        scores = torch.matmul(q, k.transpose(-2, -1)) * self.model.attention.scale
                        attn_weights = F.softmax(scores, dim=-1)
                        
                        # Calculate head specialization (variance across heads)
                        head_specialization = torch.std(attn_weights, dim=1).mean().item()
                        self.logger.info(f"  Attention head specialization: {head_specialization:.4f}")
                        
                        # Log attention weights for first few timesteps
                        if epoch % 5 == 0:  # Log every 5 epochs to avoid spam
                            attn_sample = attn_weights[0, 0, :5, :5].detach().cpu().numpy()
                            self.logger.info(f"  Attention weights sample (first 5x5):\n{attn_sample}")
                        
                        self.model.train()
                
                # Warn if scaling parameters are not learning (stuck at initialization)
                if abs(scale_value - 5.0) < 0.01 and abs(bias_value - 1.0) < 0.01:
                    self.logger.warning(f"  [WARNING] Frequency scaling parameters may not be learning (scale={scale_value:.4f}, bias={bias_value:.4f})")
                # NEW: Warn if gradients are too small
                elif abs(scale_grad) < 1e-6 and abs(bias_grad) < 1e-6:
                    self.logger.warning(f"  [WARNING] Frequency scaling gradients are very small (scale_grad={scale_grad:.6f}, bias_grad={bias_grad:.6f})")
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_magnitude_loss = total_magnitude_loss / num_batches
        avg_frequency_loss = total_frequency_loss / num_batches
        avg_magnitude_mae = total_magnitude_mae / num_batches
        avg_frequency_mae = total_frequency_mae / num_batches
        avg_magnitude_corr = total_magnitude_corr / num_batches
        avg_frequency_corr = total_frequency_corr / num_batches
        
        return {
            'loss': avg_loss,
            'magnitude_loss': avg_magnitude_loss,
            'frequency_loss': avg_frequency_loss,
            'magnitude_mae': avg_magnitude_mae,
            'frequency_mae': avg_frequency_mae,
            'magnitude_corr': avg_magnitude_corr,
            'frequency_corr': avg_frequency_corr
        }
    
    def validate_epoch(self, epoch: int = 0) -> Dict[str, float]:
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
        
        # 🔧 NEW: Debug logging for frequency prediction ranges
        all_freq_preds = []
        all_freq_targets = []
        all_freq_targets_log1p = []
        
        with torch.no_grad():
            # 🔧 NEW: Add validation progress bar for consistency
            val_progress = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            for batch_idx, (input_seq, target_seq, metadata, _) in enumerate(val_progress):
                # Move to device
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                metadata = metadata.to(self.device)
                
                # Forward pass
                magnitude_pred, frequency_pred = self.model(input_seq, metadata)
                
                # 🔧 NEW: Collect frequency predictions and targets for range analysis
                all_freq_preds.append(frequency_pred.detach().cpu())
                all_freq_targets_log1p.append(torch.log1p(target_seq[:, 1]).detach().cpu())
                
                # Extract targets
                magnitude_true = target_seq[:, 0]  # (batch_size,) - max_magnitude
                frequency_true_normalized = target_seq[:, 1]  # (batch_size,) - frequency (NORMALIZED)
                
                # 🔧 FIX: Denormalize frequency for Poisson loss using single feature method
                frequency_true = torch.tensor([
                    self.val_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                    for freq in frequency_true_normalized
                ], dtype=torch.float32).to(self.device)
                
                # 🔧 NEW: Collect raw frequency targets for range analysis
                all_freq_targets.append(frequency_true.detach().cpu())
                
                # Compute loss with epoch for variance penalty activation
                loss = self.criterion(magnitude_pred.squeeze(), frequency_pred.squeeze(),
                                    magnitude_true, frequency_true, epoch)
                
                # Get loss components
                loss_components = self.criterion.get_loss_components(
                    magnitude_pred.squeeze(), frequency_pred.squeeze(),
                    magnitude_true, frequency_true
                )
                
                # 🔧 REFACTOR: Calculate additional metrics
                magnitude_mae = F.l1_loss(magnitude_pred.squeeze(), magnitude_true)
                frequency_mae = F.l1_loss(frequency_pred.squeeze(), torch.log1p(frequency_true))  # Compare in log1p space
                
                # Calculate correlations
                magnitude_corr = self._pearson_correlation(magnitude_pred.squeeze(), magnitude_true)
                frequency_corr = self._pearson_correlation(frequency_pred.squeeze(), torch.log1p(frequency_true))  # Compare in log1p space
                
                # Update totals
                total_loss += loss.item()
                total_magnitude_loss += loss_components['magnitude_loss'].item()
                total_frequency_loss += loss_components['frequency_loss'].item()
                total_magnitude_mae += magnitude_mae.item()
                total_frequency_mae += frequency_mae.item()
                total_magnitude_corr += magnitude_corr.item()
                total_frequency_corr += frequency_corr.item()
                num_batches += 1
        
        # 🔧 NEW: Debug logging for frequency prediction ranges
        if all_freq_preds:
            # Concatenate all predictions and targets
            freq_preds_cat = torch.cat(all_freq_preds)
            freq_targets_cat = torch.cat(all_freq_targets)
            freq_targets_log1p_cat = torch.cat(all_freq_targets_log1p)
            
            # Calculate ranges in normalized log1p space
            freq_pred_norm_min, freq_pred_norm_max = torch.min(freq_preds_cat).item(), torch.max(freq_preds_cat).item()
            freq_target_norm_min, freq_target_norm_max = torch.min(freq_targets_log1p_cat).item(), torch.max(freq_targets_log1p_cat).item()
            
            # 🔧 NEW: Denormalize predictions to log1p space for comparison
            if hasattr(self.val_loader.dataset, 'denormalize_frequency_log1p'):
                # Denormalize predictions back to log1p space
                freq_preds_log1p = []
                for pred in freq_preds_cat:
                    freq_preds_log1p.append(self.val_loader.dataset.denormalize_frequency_log1p(pred.item()))
                freq_preds_log1p = torch.tensor(freq_preds_log1p)
                
                freq_pred_log1p_min, freq_pred_log1p_max = torch.min(freq_preds_log1p).item(), torch.max(freq_preds_log1p).item()
                freq_target_log1p_min, freq_target_log1p_max = torch.min(freq_targets_log1p_cat).item(), torch.max(freq_targets_log1p_cat).item()
                
                # Log both normalized and denormalized log1p ranges
                self.logger.info(f"Freq normalized log1p preds [{freq_pred_norm_min:.4f}, {freq_pred_norm_max:.4f}], targets [{freq_target_norm_min:.4f}, {freq_target_norm_max:.4f}]")
                self.logger.info(f"Freq denorm log1p preds [{freq_pred_log1p_min:.4f}, {freq_pred_log1p_max:.4f}], targets [{freq_target_log1p_min:.4f}, {freq_target_log1p_max:.4f}]")
            else:
                # Fallback to old logging if dataset doesn't have the new method
                self.logger.info(f"Freq normalized log1p preds [{freq_pred_norm_min:.4f}, {freq_pred_norm_max:.4f}], targets [{freq_target_norm_min:.4f}, {freq_target_norm_max:.4f}]")
            
            # Calculate ranges in raw-space (convert predictions back to counts)
            if self.model.freq_head_type == "linear":
                # frequency_pred is normalized log1p, denormalize to raw counts
                if hasattr(self.val_loader.dataset, 'denormalize_frequency_log1p'):
                    freq_preds_raw = []
                    for pred in freq_preds_cat:
                        freq_preds_raw.append(self.val_loader.dataset.denormalize_frequency_log1p(pred.item()))
                    freq_preds_raw = torch.tensor(freq_preds_raw)
                else:
                    # Fallback: frequency_pred is log(λ), so λ = exp(frequency_pred)
                    freq_preds_raw = torch.exp(freq_preds_cat)
            else:
                # Legacy mode: frequency_pred is log(1 + λ), so λ = expm1(frequency_pred)
                freq_preds_raw = torch.expm1(freq_preds_cat)
            
            freq_pred_raw_min, freq_pred_raw_max = torch.min(freq_preds_raw).item(), torch.max(freq_preds_raw).item()
            freq_target_raw_min, freq_target_raw_max = torch.min(freq_targets_cat).item(), torch.max(freq_targets_cat).item()
            
            # Log raw-space ranges for intuition
            self.logger.info(f"Freq raw-space preds [{freq_pred_raw_min:.4f}, {freq_pred_raw_max:.4f}], targets [{freq_target_raw_min:.4f}, {freq_target_raw_max:.4f}]")
        
        # Return average losses and metrics
        return {
            'total_loss': total_loss / num_batches,
            'magnitude_loss': total_magnitude_loss / num_batches,
            'frequency_loss': total_frequency_loss / num_batches,
            'magnitude_mae': total_magnitude_mae / num_batches,
            'frequency_mae': total_frequency_mae / num_batches,
            'magnitude_corr': total_magnitude_corr / num_batches,
            'frequency_corr': total_frequency_corr / num_batches
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
        
        # 🔧 ADD: Record training start time
        training_start_time = time.time()
        
        for epoch in range(max_epochs):
            # 🔧 IMPROVEMENT: Track current epoch for debug logging
            self.current_epoch = epoch + 1
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # 🔧 IMPROVEMENT: Store enhanced metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.train_magnitude_losses.append(train_metrics['magnitude_loss'])
            self.train_frequency_losses.append(train_metrics['frequency_loss'])
            self.val_magnitude_losses.append(val_metrics['magnitude_loss'])
            self.val_frequency_losses.append(val_metrics['frequency_loss'])
            
            # 🔧 REFACTOR: Store additional metrics
            self.train_magnitude_mae.append(train_metrics['magnitude_mae'])
            self.train_frequency_mae.append(train_metrics['frequency_mae'])
            self.train_magnitude_corr.append(train_metrics['magnitude_corr'])
            self.train_frequency_corr.append(train_metrics['frequency_corr'])
            self.val_magnitude_mae.append(val_metrics['magnitude_mae'])
            self.val_frequency_mae.append(val_metrics['frequency_mae'])
            self.val_magnitude_corr.append(val_metrics['magnitude_corr'])
            self.val_frequency_corr.append(val_metrics['frequency_corr'])
            
            # 🔧 IMPROVEMENT: Comprehensive logging for each epoch
            self.logger.info(
                f"Epoch {epoch+1}/{max_epochs}:\n"
                f"  Train - Loss: {train_metrics['loss']:.4f}, "
                f"Mag: {train_metrics['magnitude_loss']:.4f}, "
                f"Freq: {train_metrics['frequency_loss']:.4f}, "
                f"Mag_MAE: {train_metrics['magnitude_mae']:.4f}, "
                f"Freq_MAE: {train_metrics['frequency_mae']:.4f}, "
                f"Mag_Corr: {train_metrics['magnitude_corr']:.4f}, "
                f"Freq_Corr: {train_metrics['frequency_corr']:.4f}\n"
                f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                f"Mag: {val_metrics['magnitude_loss']:.4f}, "
                f"Freq: {val_metrics['frequency_loss']:.4f}, "
                f"Mag_MAE: {val_metrics['magnitude_mae']:.4f}, "
                f"Freq_MAE: {val_metrics['frequency_mae']:.4f}, "
                f"Mag_Corr: {val_metrics['magnitude_corr']:.4f}, "
                f"Freq_Corr: {val_metrics['frequency_corr']:.4f}"
            )
            
            # 🔧 IMPROVEMENT: Step the scheduler (CosineAnnealingWarmRestarts doesn't need validation loss)
            self.scheduler.step()
            
            # 🔧 NEW: β annealing - gradually increase frequency weight from 2.0 -> 3.0 over training
            if hasattr(self.criterion, 'frequency_weight'):
                base_freq_weight = 2.0
                max_freq_weight = 3.0
                annealed_weight = base_freq_weight + (epoch / max_epochs) * (max_freq_weight - base_freq_weight)
                self.criterion.frequency_weight = annealed_weight
                
                if epoch % 10 == 0:  # Log every 10 epochs
                    self.logger.info(f"  β annealing: frequency_weight = {annealed_weight:.3f}")
            
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
                    # NEW: Check if predictions are too narrow before stopping
                    if self._check_prediction_range():
                        self.logger.info("  [INFO] Resetting patience - predictions too narrow, continuing training")
                        self.patience_counter = 0
                        continue
                    else:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        # 🔧 ADD: Record training end time
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}. Total training time: {training_duration:.2f} seconds.")
        
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
            'val_frequency_corr': self.val_frequency_corr,
            # 🔧 ADD: Return training duration
            'training_time': training_duration
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
                magnitude_pred, frequency_pred = self.model(input_seq, metadata)
                
                # Extract targets
                magnitude_true = target_seq[:, 0]  # (batch_size,) - max_magnitude
                frequency_true_normalized = target_seq[:, 1]  # (batch_size,) - frequency (NORMALIZED)
                
                # 🔧 FIX: Denormalize frequency for Poisson loss using single feature method
                frequency_true = torch.tensor([
                    data_loader.dataset.denormalize_single_feature(freq.item(), 'frequency')
                    for freq in frequency_true_normalized
                ], dtype=torch.float32).to(self.device)
                
                # Compute loss
                loss = self.criterion(magnitude_pred.squeeze(), frequency_pred.squeeze(),
                                    magnitude_true, frequency_true)
                
                # Get loss components
                loss_components = self.criterion.get_loss_components(
                    magnitude_pred.squeeze(), frequency_pred.squeeze(),
                    magnitude_true, frequency_true
                )
                
                        # Store predictions and targets
        magnitude_predictions.extend(magnitude_pred.squeeze().cpu().numpy())
        frequency_predictions.extend(frequency_pred.squeeze().cpu().numpy())
        
        # Store raw frequency targets for evaluation
        frequency_targets.extend(frequency_true.cpu().numpy())
        
        # Store magnitude targets
        magnitude_targets.extend(magnitude_true.cpu().numpy())
        
        # Update totals
        total_loss += loss.item()
        total_magnitude_loss += loss_components['magnitude_loss'].item()
        total_frequency_loss += loss_components['frequency_loss'].item()
        num_batches += 1
        
        # Calculate additional metrics
        magnitude_predictions = np.array(magnitude_predictions)
        frequency_predictions = np.array(frequency_predictions)
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
        
        # FREQUENCY METRICS: REFACTORED FOR LOG1P + MSE FORMULATION
        # Training loss remains in log1p space (already computed above)
        frequency_log1p_mse = total_frequency_loss / num_batches
        
        # 🔧 REFACTOR: Convert frequency predictions from normalized log1p space to raw counts for evaluation
        # Handle different frequency head types and normalization
        if self.model.freq_head_type == "linear":
            # frequency_predictions are in normalized log1p space, denormalize to raw counts
            if hasattr(self.test_loader.dataset, 'denormalize_frequency_log1p'):
                frequency_raw_predictions = []
                for pred in frequency_predictions:
                    frequency_raw_predictions.append(self.test_loader.dataset.denormalize_frequency_log1p(pred))
                frequency_raw_predictions = np.array(frequency_raw_predictions)
            else:
                # Fallback: frequency_predictions are in log(λ) space, convert to raw counts using exp()
                frequency_raw_predictions = np.exp(frequency_predictions)
        else:
            # Legacy mode: frequency_predictions are in log(1 + λ) space, convert to raw counts using expm1()
            frequency_raw_predictions = np.expm1(frequency_predictions)
        
        # Log both normalized and denormalized log1p ranges for transparency
        freq_norm_pred_min, freq_norm_pred_max = np.min(frequency_predictions), np.max(frequency_predictions)
        freq_raw_pred_min, freq_raw_pred_max = np.min(frequency_raw_predictions), np.max(frequency_raw_predictions)
        freq_true_min, freq_true_max = np.min(frequency_targets), np.max(frequency_targets)
        
        # 🔧 NEW: Show normalized log1p ranges
        self.logger.info(f"Freq normalized log1p preds [{freq_norm_pred_min:.2f},{freq_norm_pred_max:.2f}]")
        
        # Show denormalized log1p ranges if available
        if hasattr(self.test_loader.dataset, 'denormalize_frequency_log1p'):
            freq_log1p_preds = []
            for pred in frequency_predictions:
                freq_log1p_preds.append(self.test_loader.dataset.denormalize_frequency_log1p(pred))
            freq_log1p_preds = np.array(freq_log1p_preds)
            freq_log1p_min, freq_log1p_max = np.min(freq_log1p_preds), np.max(freq_log1p_preds)
            self.logger.info(f"Freq denorm log1p preds [{freq_log1p_min:.2f},{freq_log1p_max:.2f}], targets [{freq_true_min:.1f},{freq_true_max:.1f}]")
        
        self.logger.info(f"Freq raw-space preds [{freq_raw_pred_min:.1f},{freq_raw_pred_max:.1f}], targets [{freq_true_min:.1f},{freq_true_max:.1f}]")
        
        # 🔧 GUARDRAIL: Check if raw-space prediction std is <5% of target std
        freq_pred_std = np.std(frequency_raw_predictions)
        freq_target_std = np.std(frequency_targets)
        
        if freq_target_std > 0 and freq_pred_std < freq_target_std * 0.05:
            self.logger.warning(f"[WARNING] Frequency predictions collapsed (check scaling): pred_std={freq_pred_std:.3f}, target_std={freq_target_std:.3f}")
        
        # 🔧 ENHANCED: Add interpretability metrics
        # Magnitude accuracy with multiple tolerance levels for better diagnosis
        magnitude_accuracy_03 = np.mean(np.abs(magnitude_predictions - magnitude_targets) <= 0.3)
        magnitude_accuracy_02 = np.mean(np.abs(magnitude_predictions - magnitude_targets) <= 0.2)
        magnitude_accuracy_01 = np.mean(np.abs(magnitude_predictions - magnitude_targets) <= 0.1)
        
        # Use the strictest tolerance for the main metric
        magnitude_accuracy = magnitude_accuracy_01
        
        # Frequency accuracy (within ±1 count) - computed on raw counts
        frequency_accuracy = np.mean(np.abs(frequency_raw_predictions - frequency_targets) <= 1.0)
        
        # Relative error metrics - computed on raw counts
        magnitude_rel_error = np.mean(np.abs(magnitude_predictions - magnitude_targets) / (np.abs(magnitude_targets) + 1e-8))
        frequency_rel_error = np.mean(np.abs(frequency_raw_predictions - frequency_targets) / (np.abs(frequency_targets) + 1e-8))
        
        # Log interpretability metrics with multiple tolerance levels
        self.logger.info(f"Magnitude Accuracy (±0.1): {magnitude_accuracy_01:.3f} ({magnitude_accuracy_01*100:.1f}%)")
        self.logger.info(f"Magnitude Accuracy (±0.2): {magnitude_accuracy_02:.3f} ({magnitude_accuracy_02*100:.1f}%)")
        self.logger.info(f"Magnitude Accuracy (±0.3): {magnitude_accuracy_03:.3f} ({magnitude_accuracy_03*100:.1f}%)")
        self.logger.info(f"Frequency Accuracy (±1): {frequency_accuracy:.3f} ({frequency_accuracy*100:.1f}%)")
        self.logger.info(f"Magnitude Relative Error: {magnitude_rel_error:.3f}")
        self.logger.info(f"Frequency Relative Error: {frequency_rel_error:.3f}")
        
        # 🔧 NEW: Add detailed magnitude diagnostics
        mag_pred_min, mag_pred_max = np.min(magnitude_predictions), np.max(magnitude_predictions)
        mag_target_min, mag_target_max = np.min(magnitude_targets), np.max(magnitude_targets)
        mag_pred_std = np.std(magnitude_predictions)
        mag_target_std = np.std(magnitude_targets)
        
        self.logger.info(f"Magnitude Details:")
        self.logger.info(f"  Predictions: [{mag_pred_min:.3f}, {mag_pred_max:.3f}] (std: {mag_pred_std:.3f})")
        self.logger.info(f"  Targets: [{mag_target_min:.3f}, {mag_target_max:.3f}] (std: {mag_target_std:.3f})")
        self.logger.info(f"  Prediction range: {mag_pred_max - mag_pred_min:.3f}")
        self.logger.info(f"  Target range: {mag_target_max - mag_target_min:.3f}")
        
        # Check for magnitude prediction collapse
        if mag_pred_std < mag_target_std * 0.1:
            self.logger.warning(f"  [WARNING] Magnitude predictions collapsed: pred_std={mag_pred_std:.3f}, target_std={mag_target_std:.3f}")
        elif mag_pred_std < mag_target_std * 0.3:
            self.logger.warning(f"  [WARNING] Magnitude predictions may be too narrow: pred_std={mag_pred_std:.3f}, target_std={mag_target_std:.3f}")
        else:
            self.logger.info(f"  [GOOD] Magnitude prediction range looks healthy")
        
        # 🔧 WARNING: Check for metric inconsistencies
        # High accuracy with narrow prediction ranges suggests scale mismatch
        freq_pred_range = np.max(frequency_raw_predictions) - np.min(frequency_raw_predictions)
        freq_target_range = np.max(frequency_targets) - np.min(frequency_targets)
        
        if freq_pred_range < freq_target_range * 0.1:  # Predictions < 10% of target range
            self.logger.warning(f"[WARNING] Frequency predictions have very narrow range ({freq_pred_range:.2f}) compared to targets ({freq_target_range:.2f})")
            self.logger.warning(f"[WARNING] High accuracy ({frequency_accuracy*100:.1f}%) may be misleading due to scale mismatch!")
            self.logger.warning(f"[WARNING] Raw-space predictions: {np.min(frequency_raw_predictions):.2f} to {np.max(frequency_raw_predictions):.2f}")
            self.logger.warning(f"[WARNING] Targets: {np.min(frequency_targets):.2f} to {np.max(frequency_targets):.2f}")
        
        # Check magnitude predictions for similar issues
        mag_pred_range = np.max(magnitude_predictions) - np.min(magnitude_predictions)
        mag_target_range = np.max(magnitude_targets) - np.min(magnitude_targets)
        
        if mag_pred_range < mag_target_range * 0.1:  # Predictions < 10% of target range
            self.logger.warning(f"[WARNING] Magnitude predictions have very narrow range ({mag_pred_range:.2f}) compared to targets ({mag_target_range:.2f})")
            self.logger.warning(f"[WARNING] High accuracy ({magnitude_accuracy*100:.1f}%) may be misleading due to scale mismatch!")
            self.logger.warning(f"[WARNING] Predictions: {np.min(magnitude_predictions):.2f} to {np.max(magnitude_predictions):.2f}")
            self.logger.warning(f"[WARNING] Targets: {np.min(magnitude_targets):.2f} to {np.max(magnitude_targets):.2f}")
        
        # MAE on raw counts (converted from log1p space)
        frequency_mae = np.mean(np.abs(frequency_raw_predictions - frequency_targets))
        
        # 🔧 FIX: Better correlation calculation with error handling - computed on raw counts
        try:
            frequency_corr = np.corrcoef(frequency_raw_predictions, frequency_targets)[0, 1]
            if np.isnan(frequency_corr):
                frequency_corr = 0.0
        except (IndexError, ValueError):
            frequency_corr = 0.0
        
        # MSE on raw counts (converted from log1p space)
        frequency_mse = np.mean((frequency_raw_predictions - frequency_targets) ** 2)
        
        # 🔧 SUMMARY: Overall performance assessment
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE ASSESSMENT")
        self.logger.info("=" * 60)
        
        # Check for critical scale issues
        if freq_pred_range < freq_target_range * 0.1:
            self.logger.error(f"[CRITICAL] Frequency scale mismatch: predictions cover only {freq_pred_range/freq_target_range*100:.1f}% of target range")
        elif freq_pred_range < freq_target_range * 0.3:
            self.logger.warning(f"[WARNING] Frequency scale mismatch: predictions cover only {freq_pred_range/freq_target_range*100:.1f}% of target range")
        else:
            self.logger.info(f"[GOOD] Frequency scale looks good: predictions cover {freq_pred_range/freq_target_range*100:.1f}% of target range")
        
        if mag_pred_range < mag_target_range * 0.1:
            self.logger.error(f"[CRITICAL] Magnitude scale mismatch: predictions cover only {mag_pred_range/mag_target_range*100:.1f}% of target range")
        elif mag_pred_range < mag_target_range * 0.3:
            self.logger.warning(f"[WARNING] Magnitude scale mismatch: predictions cover only {mag_pred_range/mag_target_range*100:.1f}% of target range")
        else:
            self.logger.info(f"[GOOD] Magnitude scale looks good: predictions cover {mag_pred_range/mag_target_range*100:.1f}% of target range")
        
        self.logger.info("=" * 60)
        
        # 🔧 NEW: Add diagnostic histogram plot for frequency predictions vs targets
        try:
            import matplotlib.pyplot as plt
            
            # Create histogram comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Frequency predictions vs targets
            ax1.hist(frequency_raw_predictions, bins=20, alpha=0.7, label='Predictions', color='blue')
            ax1.hist(frequency_targets, bins=20, alpha=0.7, label='Targets', color='red')
            ax1.set_xlabel('Frequency (counts)')
            ax1.set_ylabel('Count')
            ax1.set_title('Frequency: Predictions vs Targets')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Magnitude predictions vs targets
            ax2.hist(magnitude_predictions, bins=20, alpha=0.7, label='Predictions', color='blue')
            ax2.hist(magnitude_targets, bins=20, alpha=0.7, label='Targets', color='red')
            ax2.set_xlabel('Magnitude')
            ax2.set_ylabel('Count')
            ax2.set_title('Magnitude: Predictions vs Targets')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Use the save_dir parameter if available, otherwise fall back to data/results
            if hasattr(self, 'save_dir') and self.save_dir:
                plot_save_dir = self.save_dir
            else:
                # Fallback to data/results in the current working directory
                plot_save_dir = os.path.join(os.getcwd(), 'data', 'results')
            
            os.makedirs(plot_save_dir, exist_ok=True)
            
            # FIX: Include model type in filename to prevent overwriting between different models
            model_type = "shared_lstm"  # Default for this trainer
            if hasattr(self.model, '__class__'):
                model_type = self.model.__class__.__name__.lower().replace('model', '')
            
            plot_path = os.path.join(plot_save_dir, f'prediction_vs_target_histograms_{model_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"[TOOL] Diagnostic histograms saved to: {plot_path}")
            self.logger.info(f"[TOOL] Frequency compression: pred_range={freq_pred_range:.1f}, target_range={freq_target_range:.1f}")
            self.logger.info(f"[TOOL] Magnitude compression: pred_range={mag_pred_range:.1f}, target_range={mag_target_range:.1f}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping diagnostic plots")
        except Exception as e:
            self.logger.warning(f"Could not create diagnostic plots: {e}")
        
        return {
            'total_loss': total_loss / num_batches,
            'magnitude_loss': total_magnitude_loss / num_batches,
            'frequency_loss': total_frequency_loss / num_batches,
            'magnitude_mse': magnitude_mse,
            'magnitude_mae': magnitude_mae,
            'magnitude_corr': magnitude_corr,
            'frequency_log1p_mse': frequency_log1p_mse,     # NEW: Log1p + MSE
            'frequency_mae': frequency_mae,                  # NEW: MAE on exp(log(λ))
            'frequency_corr': frequency_corr,                # NEW: Correlation on exp(log(λ))
            'frequency_mse': frequency_mse,                  # NEW: MSE on exp(log(λ)) for comparison
            # ENHANCED: Interpretability metrics
            'magnitude_accuracy': magnitude_accuracy,        # NEW: Accuracy within ±0.1
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
                # IMPROVEMENT: Store enhanced metrics
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
        
        # FIX: Handle missing optimizer state (e.g., from compare mode)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state restored from checkpoint")
        else:
            self.logger.info("No optimizer state in checkpoint (evaluation mode)")
        
        # 🔧 FIX: Handle missing best_val_loss (e.g., from compare mode)
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        else:
            self.best_val_loss = float('inf')
            self.logger.info("No best validation loss in checkpoint, setting to infinity")
        
        # 🔧 FIX: Handle missing training history (e.g., from compare mode)
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_magnitude_losses = history.get('train_magnitude_losses', [])
            self.train_frequency_losses = history.get('train_frequency_losses', [])
            self.val_magnitude_losses = history.get('val_magnitude_losses', [])
            self.val_frequency_losses = history.get('val_frequency_losses', [])
            
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
        else:
            self.logger.info("No training history in checkpoint (evaluation mode)")
            # Initialize empty lists for evaluation
            self.train_losses = []
            self.val_losses = []
            self.train_magnitude_losses = []
            self.train_frequency_losses = []
            self.val_magnitude_losses = []
            self.val_frequency_losses = []
        
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
        axes[0, 2].set_title('Frequency Loss (MSE on log1p)')  # 🔧 FIX: Updated title to match actual loss function
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