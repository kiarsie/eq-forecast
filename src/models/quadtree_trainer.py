#!/usr/bin/env python3
"""
Quadtree-based Trainer for Earthquake Forecasting

Implements the methodology from the paper:
- Separate LSTM network for each quadtree bin
- Training with 10-year lookback data
- Comparison between Simple LSTM and Attention LSTM
- Evaluation using WMAPE and Forecast Accuracy metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

from .simple_lstm import SimpleLSTM
from .attention_lstm import AttentionLSTM
from .quadtree_data_loader import QuadtreeDataLoader


class QuadtreeModelTrainer:
    """
    Trainer for quadtree-based earthquake forecasting models.
    
    Creates and trains separate LSTM networks for each quadtree bin.
    """
    
    def __init__(self, 
                 data_path: str,
                 save_dir: str = "results",
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to processed earthquake data
            save_dir: Directory to save results
            lookback_years: Number of years to look back
            target_horizon: Number of years to predict ahead
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            device: Device to use for training
        """
        self.data_path = data_path
        self.save_dir = Path(save_dir)
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.data_loader = QuadtreeDataLoader(
            data_path=data_path,
            lookback_years=lookback_years,
            target_horizon=target_horizon,
            batch_size=batch_size
        )
        
        # Get feature dimensions
        self.input_size, self.output_size = self.data_loader.get_feature_dimensions()
        self.bin_count = self.data_loader.get_bin_count()
        
        self.logger.info(f"Input size: {self.input_size}, Output size: {self.output_size}")
        self.logger.info(f"Number of quadtree bins: {self.bin_count}")
        
        # Initialize models for each bin
        self.models = {}
        self.optimizers = {}
        
        # Training history tracking for visualization
        self.training_history = {}
        for bin_id in range(self.bin_count):
            self.training_history[bin_id] = {
                'simple_lstm': {
                    'train_losses': [],
                    'val_losses': [],
                    'learning_rates': [],
                    'epochs': []
                },
                'attention_lstm': {
                    'train_losses': [],
                    'val_losses': [],
                    'learning_rates': [],
                    'epochs': []
                }
            }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Simple LSTM and Attention LSTM models for each bin."""
        self.logger.info("Initializing models for each quadtree bin...")
        
        for bin_id in range(self.bin_count):
            # Simple LSTM
            simple_lstm = SimpleLSTM(
                input_size=self.input_size,
                hidden_sizes=(64, 48, 24, 24),  # Balanced capacity - not too small, not too large
                dropout=0.2,  # Reduced dropout for better capacity utilization
                bidirectional=False
            ).to(self.device)
            
            # Attention LSTM
            attention_lstm = AttentionLSTM(
                input_size=self.input_size,
                hidden_sizes=(64, 48, 24, 24),  # Balanced capacity - not too small, not too large
                dropout=0.2,  # Reduced dropout for better capacity utilization
                bidirectional=False
            ).to(self.device)
            
            # Store models
            self.models[bin_id] = {
                'simple_lstm': simple_lstm,
                'attention_lstm': attention_lstm
            }
            
            # Create optimizers
            self.optimizers[bin_id] = {
                'simple_lstm': optim.Adam(simple_lstm.parameters(), lr=self.learning_rate, weight_decay=1e-4),
                'attention_lstm': optim.Adam(attention_lstm.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            }
        
        self.logger.info(f"Initialized {self.bin_count * 2} models (2 per bin)")
        
        # Initialize learning rate schedulers for each model
        self.schedulers = {}
        for bin_id in range(self.bin_count):
            self.schedulers[bin_id] = {
                'simple_lstm': optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers[bin_id]['simple_lstm'], 
                    mode='min', 
                    factor=0.5, 
                    patience=10, 
                    verbose=True
                ),
                'attention_lstm': optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers[bin_id]['attention_lstm'], 
                    mode='min', 
                    factor=0.5, 
                    patience=10, 
                    verbose=True
                )
            }
    
    def plot_training_progress(self, bin_id: int, model_type: str):
        """
        Plot training progress for a specific bin and model type.
        
        Args:
            bin_id: ID of the quadtree bin
            model_type: Type of model ('simple_lstm' or 'attention_lstm')
        """
        if bin_id not in self.training_history or model_type not in self.training_history[bin_id]:
            self.logger.warning(f"No training history found for bin {bin_id}, {model_type}")
            return
        
        history = self.training_history[bin_id][model_type]
        
        if not history['train_losses']:
            self.logger.warning(f"No training data available for bin {bin_id}, {model_type}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss curves
        axes[0].plot(history['epochs'], history['train_losses'], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(history['epochs'], history['val_losses'], label='Val Loss', color='red', linewidth=2)
        axes[0].set_title(f'{model_type.replace("_", " ").title()} - Bin {bin_id} Training Progress')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale for better visualization
        
        # Learning rate
        axes[1].plot(history['epochs'], history['learning_rates'], color='green', linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')  # Log scale for learning rate
        
        # Loss difference (train - val)
        loss_diff = [t - v for t, v in zip(history['train_losses'], history['val_losses'])]
        axes[2].plot(history['epochs'], loss_diff, color='purple', linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Overfitting Indicator (Train - Val Loss)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Difference')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"training_progress_bin_{bin_id}_{model_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training progress plot saved: {plot_path}")
    
    def plot_all_training_progress(self):
        """Plot training progress for all bins and model types."""
        self.logger.info("Generating training progress plots for all models...")
        
        for bin_id in range(self.bin_count):
            for model_type in ['simple_lstm', 'attention_lstm']:
                self.plot_training_progress(bin_id, model_type)
        
        self.logger.info("All training progress plots generated!")
    
    def generate_training_summary(self):
        """Generate comprehensive training summary with all plots and history."""
        self.logger.info("Generating comprehensive training summary...")
        
        # Generate all training progress plots
        self.plot_all_training_progress()
        
        # Save training history to JSON
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        self.logger.info(f"Training history saved: {history_path}")
        
        # Generate summary statistics
        summary_stats = {}
        for bin_id in range(self.bin_count):
            summary_stats[bin_id] = {}
            for model_type in ['simple_lstm', 'attention_lstm']:
                if bin_id in self.training_history and model_type in self.training_history[bin_id]:
                    history = self.training_history[bin_id][model_type]
                    if history['train_losses']:
                        summary_stats[bin_id][model_type] = {
                            'final_train_loss': history['train_losses'][-1],
                            'final_val_loss': history['val_losses'][-1],
                            'best_val_loss': min(history['val_losses']),
                            'epochs_trained': len(history['epochs']),
                            'final_learning_rate': history['learning_rates'][-1],
                            'learning_rate_reductions': len([i for i in range(1, len(history['learning_rates'])) 
                                                         if history['learning_rates'][i] < history['learning_rates'][i-1]])
                        }
        
        # Save summary statistics
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
        
        return summary_stats
    
    def train_bin_models(self, 
                         bin_id: int, 
                         num_epochs: int = 100, 
                         patience: int = 20) -> Dict:
        """
        Train both Simple LSTM and Attention LSTM for a specific bin.
        
        Args:
            bin_id: ID of the quadtree bin
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training results for both models
        """
        self.logger.info(f"Training models for bin {bin_id}")
        
        # Get bin-specific data loaders
        bin_loaders = self.data_loader.get_bin_loaders()
        if bin_id not in bin_loaders:
            self.logger.error(f"Bin {bin_id} not found in data loaders")
            return {}
        
        train_loader, val_loader, test_loader = bin_loaders[bin_id]
        
        # Training results
        results = {}
        
        # Train Simple LSTM
        self.logger.info(f"Training Simple LSTM for bin {bin_id}")
        simple_results = self._train_single_model(
            model=self.models[bin_id]['simple_lstm'],
            optimizer=self.optimizers[bin_id]['simple_lstm'],
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=f"simple_lstm_bin_{bin_id}",
            num_epochs=num_epochs,
            patience=patience
        )
        results['simple_lstm'] = simple_results
        
        # Train Attention LSTM
        self.logger.info(f"Training Attention LSTM for bin {bin_id}")
        attention_results = self._train_single_model(
            model=self.models[bin_id]['attention_lstm'],
            optimizer=self.optimizers[bin_id]['attention_lstm'],
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=f"attention_lstm_bin_{bin_id}",
            num_epochs=num_epochs,
            patience=patience
        )
        results['attention_lstm'] = attention_results
        
        # Save models
        self._save_bin_models(bin_id)
        
        # Generate training progress plots
        self.plot_training_progress(bin_id, 'simple_lstm')
        self.plot_training_progress(bin_id, 'attention_lstm')
        
        return results
    
    def _train_single_model(self, 
                           model: nn.Module, 
                           optimizer: optim.Optimizer,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           model_name: str,
                           num_epochs: int,
                           patience: int) -> Dict:
        """
        Train a single model.
        
        Args:
            model: Model to train
            optimizer: Optimizer for the model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        # Extract bin_id and model_type from model_name
        parts = model_name.split('_')
        bin_id = int(parts[-1])  # Last part is bin_id
        model_type = '_'.join(parts[:-2])  # Everything except 'bin' and 'id'
        
        # Loss function (MSE for regression)
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        learning_rates = []
        epochs = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Reshape outputs to match targets
                if outputs.shape != targets.shape:
                    outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets, metadata in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    
                    # Reshape outputs to match targets
                    if outputs.shape != targets.shape:
                        outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store training history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(current_lr)
            epochs.append(epoch)
            
            # Update global training history for plotting
            if bin_id in self.training_history and model_type in self.training_history[bin_id]:
                self.training_history[bin_id][model_type]['train_losses'] = train_losses.copy()
                self.training_history[bin_id][model_type]['val_losses'] = val_losses.copy()
                self.training_history[bin_id][model_type]['learning_rates'] = learning_rates.copy()
                self.training_history[bin_id][model_type]['epochs'] = epochs.copy()
            
            # Step the learning rate scheduler
            scheduler = self.schedulers[bin_id][model_type]
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.save_dir / f"{model_name}_best.pth")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'epochs': epochs,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
    
    def evaluate_bin_models(self, bin_id: int) -> Dict:
        """
        Evaluate both models for a specific bin.
        
        Args:
            bin_id: ID of the quadtree bin
            
        Returns:
            Evaluation results for both models
        """
        self.logger.info(f"Evaluating models for bin {bin_id}")
        
        # Get test data loader
        bin_loaders = self.data_loader.get_bin_loaders()
        if bin_id not in bin_loaders:
            self.logger.error(f"Bin {bin_id} not found in data loaders")
            return {}
        
        _, _, test_loader = bin_loaders[bin_id]
        
        results = {}
        
        # Evaluate Simple LSTM
        simple_results = self._evaluate_single_model(
            model=self.models[bin_id]['simple_lstm'],
            test_loader=test_loader,
            model_name=f"simple_lstm_bin_{bin_id}"
        )
        results['simple_lstm'] = simple_results
        
        # Evaluate Attention LSTM
        attention_results = self._evaluate_single_model(
            model=self.models[bin_id]['attention_lstm'],
            test_loader=test_loader,
            model_name=f"attention_lstm_bin_{bin_id}"
        )
        results['attention_lstm'] = attention_results
        
        return results
    
    def test_bin_models(self, bin_id: int) -> Dict:
        """
        Test both models for a specific bin on the test set.
        
        Args:
            bin_id: ID of the quadtree bin
            
        Returns:
            Test results for both models
        """
        self.logger.info(f"Testing models for bin {bin_id}")
        
        # Get test data loader
        bin_loaders = self.data_loader.get_bin_loaders()
        if bin_id not in bin_loaders:
            self.logger.error(f"Bin {bin_id} not found in data loaders")
            return {}
        
        _, _, test_loader = bin_loaders[bin_id]
        
        results = {}
        
        # Test Simple LSTM
        simple_results = self._test_single_model(
            model=self.models[bin_id]['simple_lstm'],
            test_loader=test_loader,
            model_name=f"simple_lstm_bin_{bin_id}"
        )
        results['simple_lstm'] = simple_results
        
        # Test Attention LSTM
        attention_results = self._test_single_model(
            model=self.models[bin_id]['attention_lstm'],
            test_loader=test_loader,
            model_name=f"attention_lstm_bin_{bin_id}"
        )
        results['attention_lstm'] = attention_results
        
        return results
    
    def validate_bin_models(self, bin_id: int) -> Dict:
        """
        Validate both models for a specific bin on the validation set.
        
        Args:
            bin_id: ID of the quadtree bin
            
        Returns:
            Validation results for both models
        """
        self.logger.info(f"Validating models for bin {bin_id}")
        
        # Get validation data loader
        bin_loaders = self.data_loader.get_bin_loaders()
        if bin_id not in bin_loaders:
            self.logger.error(f"Bin {bin_id} not found in data loaders")
            return {}
        
        _, val_loader, _ = bin_loaders[bin_id]
        
        results = {}
        
        # Validate Simple LSTM
        simple_results = self._validate_single_model(
            model=self.models[bin_id]['simple_lstm'],
            val_loader=val_loader,
            model_name=f"simple_lstm_bin_{bin_id}"
        )
        results['simple_lstm'] = simple_results
        
        # Validate Attention LSTM
        attention_results = self._validate_single_model(
            model=self.models[bin_id]['attention_lstm'],
            val_loader=val_loader,
            model_name=f"attention_lstm_bin_{bin_id}"
        )
        results['attention_lstm'] = attention_results
        
        return results
    
    def _evaluate_single_model(self, 
                              model: nn.Module, 
                              test_loader: DataLoader,
                              model_name: str) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model
            
        Returns:
            Evaluation results
        """
        model.eval()
        
        predictions = []
        actuals = []
        metadata_list = []
        
        with torch.no_grad():
            for inputs, targets, metadata in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Reshape outputs to match targets
                if outputs.shape != targets.shape:
                    outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
                
                # Convert to numpy
                pred_np = outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                predictions.extend(pred_np)
                actuals.extend(target_np)
                metadata_list.extend(metadata)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
        wmape = self._calculate_wmape(actuals, predictions)
        forecast_accuracy = 100.0 - wmape
        
        return {
            'mse': mse,
            'mae': mae,
            'wmape': wmape,
            'forecast_accuracy': forecast_accuracy,
            'predictions': predictions,
            'actuals': actuals,
            'metadata': metadata_list
        }
    
    def _test_single_model(self, 
                           model: nn.Module, 
                           test_loader: DataLoader,
                           model_name: str) -> Dict:
        """
        Test a single model on the test set.
        
        Args:
            model: Model to test
            test_loader: Test data loader
            model_name: Name of the model
            
        Returns:
            Test results
        """
        model.eval()
        
        predictions = []
        actuals = []
        metadata_list = []
        
        with torch.no_grad():
            for inputs, targets, metadata in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Reshape outputs to match targets
                if outputs.shape != targets.shape:
                    outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
                
                # Convert to numpy
                pred_np = outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                predictions.extend(pred_np)
                actuals.extend(target_np)
                metadata_list.extend(metadata)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
        wmape = self._calculate_wmape(actuals, predictions)
        forecast_accuracy = 100.0 - wmape
        
        return {
            'mse': mse,
            'mae': mae,
            'wmape': wmape,
            'forecast_accuracy': forecast_accuracy,
            'predictions': predictions,
            'actuals': actuals,
            'metadata': metadata_list
        }
    
    def _validate_single_model(self, 
                              model: nn.Module, 
                              val_loader: DataLoader,
                              model_name: str) -> Dict:
        """
        Validate a single model on the validation set.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            model_name: Name of the model
            
        Returns:
            Validation results
        """
        model.eval()
        
        predictions = []
        actuals = []
        metadata_list = []
        
        with torch.no_grad():
            for inputs, targets, metadata in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Reshape outputs to match targets
                if outputs.shape != targets.shape:
                    outputs = outputs.unsqueeze(1).expand(-1, targets.shape[1], -1)
                
                # Convert to numpy
                pred_np = outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                predictions.extend(pred_np)
                actuals.extend(target_np)
                metadata_list.extend(metadata)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
        wmape = self._calculate_wmape(actuals, predictions)
        forecast_accuracy = 100.0 - wmape
        
        return {
            'mse': mse,
            'mae': mae,
            'wmape': wmape,
            'forecast_accuracy': forecast_accuracy,
            'predictions': predictions,
            'actuals': actuals,
            'metadata': metadata_list
        }
    
    def _calculate_wmape(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate Weighted Mean Absolute Percentage Error (WMAPE).
        
        WMAPE = (Σ|A - F|) / (Σ|A|) * 100
        
        Args:
            actuals: Actual values
            predictions: Predicted values
            
        Returns:
            WMAPE value
        """
        absolute_errors = np.abs(actuals - predictions)
        absolute_actuals = np.abs(actuals)
        
        # Avoid division by zero
        if np.sum(absolute_actuals) == 0:
            return 0.0
        
        wmape = (np.sum(absolute_errors) / np.sum(absolute_actuals)) * 100
        return wmape
    
    def _save_bin_models(self, bin_id: int):
        """Save models for a specific bin."""
        # Save Simple LSTM
        simple_path = self.save_dir / f"simple_lstm_bin_{bin_id}_final.pth"
        torch.save(self.models[bin_id]['simple_lstm'].state_dict(), simple_path)
        
        # Save Attention LSTM
        attention_path = self.save_dir / f"attention_lstm_bin_{bin_id}_final.pth"
        torch.save(self.models[bin_id]['attention_lstm'].state_dict(), attention_path)
        
        self.logger.info(f"Saved models for bin {bin_id}")
    
    def train_all_bins(self, num_epochs: int = 100, patience: int = 20) -> Dict:
        """
        Train models for all quadtree bins.
        
        Args:
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training results for all bins
        """
        self.logger.info("Starting training for all quadtree bins")
        
        all_results = {}
        
        for bin_id in range(self.bin_count):
            try:
                bin_results = self.train_bin_models(bin_id, num_epochs, patience)
                all_results[bin_id] = bin_results
                self.logger.info(f"Completed training for bin {bin_id}")
            except Exception as e:
                self.logger.error(f"Error training bin {bin_id}: {e}")
                all_results[bin_id] = {'error': str(e)}
        
        # Save overall results
        self._save_overall_results(all_results)
        
        # Generate comprehensive training summary with plots
        self.logger.info("Generating comprehensive training summary...")
        summary_stats = self.generate_training_summary()
        
        return all_results
    
    def evaluate_all_bins(self) -> Dict:
        """
        Evaluate models for all quadtree bins.
        
        Returns:
            Evaluation results for all bins
        """
        self.logger.info("Starting evaluation for all quadtree bins")
        
        all_results = {}
        
        for bin_id in range(self.bin_count):
            try:
                bin_results = self.evaluate_bin_models(bin_id)
                all_results[bin_id] = bin_results
                self.logger.info(f"Completed evaluation for bin {bin_id}")
            except Exception as e:
                self.logger.error(f"Error evaluating bin {bin_id}: {e}")
                all_results[bin_id] = {'error': str(e)}
        
        # Save evaluation results
        self._save_evaluation_results(all_results)
        
        return all_results
    
    def test_all_bins(self) -> Dict:
        """
        Test models for all quadtree bins on the test set.
        
        Returns:
            Test results for all bins
        """
        self.logger.info("Starting testing for all quadtree bins")
        
        all_results = {}
        
        for bin_id in range(self.bin_count):
            try:
                bin_results = self.test_bin_models(bin_id)
                all_results[bin_id] = bin_results
                self.logger.info(f"Completed testing for bin {bin_id}")
            except Exception as e:
                self.logger.error(f"Error testing bin {bin_id}: {e}")
                all_results[bin_id] = {'error': str(e)}
        
        # Save test results
        self._save_test_results(all_results)
        
        return all_results
    
    def validate_all_bins(self) -> Dict:
        """
        Validate models for all quadtree bins on the validation set.
        
        Returns:
            Validation results for all bins
        """
        self.logger.info("Starting validation for all quadtree bins")
        
        all_results = {}
        
        for bin_id in range(self.bin_count):
            try:
                bin_results = self.validate_bin_models(bin_id)
                all_results[bin_id] = bin_results
                self.logger.info(f"Completed validation for bin {bin_id}")
            except Exception as e:
                self.logger.error(f"Error validating bin {bin_id}: {e}")
                all_results[bin_id] = {'error': str(e)}
        
        # Save validation results
        self._save_validation_results(all_results)
        
        return all_results
    
    def _save_overall_results(self, results: Dict):
        """Save overall training results."""
        results_path = self.save_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for bin_id, bin_results in results.items():
            serializable_results[str(bin_id)] = {}
            for model_name, model_results in bin_results.items():
                if isinstance(model_results, dict) and 'error' not in model_results:
                    serializable_results[str(bin_id)][model_name] = {
                        'best_val_loss': float(model_results['best_val_loss']),
                        'epochs_trained': model_results['epochs_trained']
                    }
                else:
                    serializable_results[str(bin_id)][model_name] = model_results
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved training results to {results_path}")
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results."""
        results_path = self.save_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for bin_id, bin_results in results.items():
            serializable_results[str(bin_id)] = {}
            for model_name, model_results in bin_results.items():
                if isinstance(model_results, dict) and 'error' not in model_results:
                    serializable_results[str(bin_id)][model_name] = {
                        'mse': float(model_results['mse']),
                        'mae': float(model_results['mae']),
                        'wmape': float(model_results['wmape']),
                        'forecast_accuracy': float(model_results['forecast_accuracy'])
                    }
                else:
                    serializable_results[str(bin_id)][model_name] = model_results
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {results_path}")
    
    def _save_test_results(self, results: Dict):
        """Save test results."""
        results_path = self.save_dir / "test_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for bin_id, bin_results in results.items():
            serializable_results[str(bin_id)] = {}
            for model_name, model_results in bin_results.items():
                if isinstance(model_results, dict) and 'error' not in model_results:
                    serializable_results[str(bin_id)][model_name] = {
                        'mse': float(model_results['mse']),
                        'mae': float(model_results['mae']),
                        'wmape': float(model_results['wmape']),
                        'forecast_accuracy': float(model_results['forecast_accuracy'])
                    }
                else:
                    serializable_results[str(bin_id)][model_name] = model_results
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved test results to {results_path}")
    
    def _save_validation_results(self, results: Dict):
        """Save validation results."""
        results_path = self.save_dir / "validation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for bin_id, bin_results in results.items():
            serializable_results[str(bin_id)] = {}
            for model_name, model_results in bin_results.items():
                if isinstance(model_results, dict) and 'error' not in model_results:
                    serializable_results[str(bin_id)][model_name] = {
                        'mse': float(model_results['mse']),
                        'mae': float(model_results['mae']),
                        'wmape': float(model_results['wmape']),
                        'forecast_accuracy': float(model_results['forecast_accuracy'])
                    }
                else:
                    serializable_results[str(bin_id)][model_name] = model_results
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved validation results to {results_path}")
    
    def generate_visualizations(self, results: Dict):
        """Generate visualizations of the results."""
        self.logger.info("Generating visualizations")
        
        # Create visualization directory
        viz_dir = self.save_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Training curves
        self._plot_training_curves(results, viz_dir)
        
        # 2. Model comparison
        self._plot_model_comparison(results, viz_dir)
        
        # 3. Bin performance analysis
        self._plot_bin_performance(results, viz_dir)
        
        self.logger.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_training_curves(self, results: Dict, viz_dir: Path):
        """Plot training curves for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves for Quadtree Bin Models')
        
        # Plot training curves for first few bins
        bins_to_plot = min(4, len(results))
        
        for i, (bin_id, bin_results) in enumerate(list(results.items())[:bins_to_plot]):
            row = i // 2
            col = i % 2
            
            if 'simple_lstm' in bin_results and 'error' not in bin_results['simple_lstm']:
                train_losses = bin_results['simple_lstm']['train_losses']
                val_losses = bin_results['simple_lstm']['val_losses']
                
                axes[row, col].plot(train_losses, label='Train Loss')
                axes[row, col].plot(val_losses, label='Val Loss')
                axes[row, col].set_title(f'Bin {bin_id} - Simple LSTM')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, results: Dict, viz_dir: Path):
        """Plot model comparison across bins."""
        # Extract metrics for comparison
        simple_metrics = []
        attention_metrics = []
        bin_ids = []
        
        for bin_id, bin_results in results.items():
            if 'simple_lstm' in bin_results and 'attention_lstm' in bin_results:
                if 'error' not in bin_results['simple_lstm'] and 'error' not in bin_results['attention_lstm']:
                    simple_metrics.append(bin_results['simple_lstm']['forecast_accuracy'])
                    attention_metrics.append(bin_results['attention_lstm']['forecast_accuracy'])
                    bin_ids.append(bin_id)
        
        if not simple_metrics:
            return
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Forecast accuracy comparison
        x = np.arange(len(bin_ids))
        width = 0.35
        
        ax1.bar(x - width/2, simple_metrics, width, label='Simple LSTM', alpha=0.8)
        ax1.bar(x + width/2, attention_metrics, width, label='Attention LSTM', alpha=0.8)
        ax1.set_xlabel('Quadtree Bin ID')
        ax1.set_ylabel('Forecast Accuracy (%)')
        ax1.set_title('Forecast Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # WMAPE comparison
        simple_wmape = [results[bin_id]['simple_lstm']['wmape'] for bin_id in bin_ids]
        attention_wmape = [results[bin_id]['attention_lstm']['wmape'] for bin_id in bin_ids]
        
        ax2.bar(x - width/2, simple_wmape, width, label='Simple LSTM', alpha=0.8)
        ax2.bar(x + width/2, attention_wmape, width, label='Attention LSTM', alpha=0.8)
        ax2.set_xlabel('Quadtree Bin ID')
        ax2.set_ylabel('WMAPE (%)')
        ax2.set_title('WMAPE Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_ids)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bin_performance(self, results: Dict, viz_dir: Path):
        """Plot performance analysis across bins."""
        # Extract performance metrics
        bin_ids = []
        simple_accuracies = []
        attention_accuracies = []
        
        for bin_id, bin_results in results.items():
            if 'simple_lstm' in bin_results and 'attention_lstm' in bin_results:
                if 'error' not in bin_results['simple_lstm'] and 'error' not in bin_results['attention_lstm']:
                    bin_ids.append(bin_id)
                    simple_accuracies.append(bin_results['simple_lstm']['forecast_accuracy'])
                    attention_accuracies.append(bin_results['attention_lstm']['forecast_accuracy'])
        
        if not bin_ids:
            return
        
        # Create performance heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create performance matrix
        performance_matrix = np.array([simple_accuracies, attention_accuracies])
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        
        # Customize plot
        ax.set_xticks(range(len(bin_ids)))
        ax.set_xticklabels(bin_ids)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Simple LSTM', 'Attention LSTM'])
        ax.set_xlabel('Quadtree Bin ID')
        ax.set_title('Forecast Accuracy Heatmap Across Bins')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Forecast Accuracy (%)')
        
        # Add text annotations
        for i in range(2):
            for j in range(len(bin_ids)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "bin_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
