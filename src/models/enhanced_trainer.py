#!/usr/bin/env python3
"""
Enhanced Quadtree Model Trainer

Supports both Simple LSTM and Attention LSTM models with comparison capabilities.
Implements the methodology from the paper with model comparison functionality.
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
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .simple_lstm import SimpleLSTM
from .attention_lstm import AttentionLSTM
from .quadtree_data_loader import QuadtreeDataLoader


class EnhancedQuadtreeTrainer:
    """
    Enhanced trainer that supports both Simple LSTM and Attention LSTM models.
    
    Features:
    - Train Simple LSTM models
    - Train Attention LSTM models  
    - Compare model performance
    - Generate comprehensive evaluation metrics
    - Save detailed results and visualizations
    """
    
    def __init__(self, 
                 data_path: str,
                 save_dir: str,
                 logger: logging.Logger,
                 model_types: List[str] = ['simple', 'attention'],
                 hidden_sizes: Tuple[int, int, int, int] = (120, 90, 30, 30),
                 lookback_years: int = 10,
                 num_epochs: int = 100,
                 patience: int = 20,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize the enhanced trainer.
        
        Args:
            data_path: Path to processed earthquake data
            save_dir: Directory to save results
            logger: Logger instance
            model_types: List of model types to train ('simple', 'attention')
            hidden_sizes: LSTM hidden layer sizes
            lookback_years: Number of years for lookback window
            num_epochs: Maximum training epochs
            patience: Early stopping patience
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            device: Device to use (auto-detect if None)
        """
        self.data_path = data_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.model_types = model_types
        self.hidden_sizes = hidden_sizes
        self.lookback_years = lookback_years
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load data
        self.data_loader = QuadtreeDataLoader(data_path, lookback_years)
        # Get unique bin IDs from the dataset data
        self.bin_ids = sorted(self.data_loader.dataset.data['bin_id'].unique())
        self.logger.info(f"Found {len(self.bin_ids)} quadtree bins")
        
        # Results storage
        self.results = {
            'simple': {},
            'attention': {},
            'comparison': {}
        }
        
        # Training history tracking for visualization
        self.training_history = {}
        for bin_id in self.bin_ids:
            self.training_history[bin_id] = {
                'simple': {
                    'train_losses': [],
                    'val_losses': [],
                    'learning_rates': [],
                    'epochs': []
                },
                'attention': {
                    'train_losses': [],
                    'val_losses': [],
                    'learning_rates': [],
                    'epochs': []
                }
            }
    
    def plot_training_progress(self, bin_id: int, model_type: str):
        """
        Plot training progress for a specific bin and model type.
        
        Args:
            bin_id: ID of the quadtree bin
            model_type: Type of model ('simple' or 'attention')
        """
        # Convert numpy types to standard Python types
        bin_id_int = int(bin_id)
        
        if bin_id_int not in self.training_history or model_type not in self.training_history[bin_id_int]:
            self.logger.warning(f"No training history found for bin {bin_id_int}, {model_type}")
            return
        
        history = self.training_history[bin_id_int][model_type]
        
        if not history['train_losses']:
            self.logger.warning(f"No training data available for bin {bin_id_int}, {model_type}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Convert numpy arrays to lists for plotting
        epochs = [int(e) for e in history['epochs']]
        train_losses = [float(l) for l in history['train_losses']]
        val_losses = [float(l) for l in history['val_losses']]
        learning_rates = [float(lr) for lr in history['learning_rates']]
        
        # Loss curves
        axes[0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
        axes[0].set_title(f'{model_type.title()} LSTM - Bin {bin_id_int} Training Progress')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale for better visualization
        
        # Learning rate
        axes[1].plot(epochs, learning_rates, color='green', linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')  # Log scale for learning rate
        
        # Loss difference (train - val)
        loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
        axes[2].plot(epochs, loss_diff, color='purple', linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Overfitting Indicator (Train - Val Loss)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Difference')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"training_progress_bin_{bin_id_int}_{model_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training progress plot saved: {plot_path}")
    
    def plot_all_training_progress(self):
        """Plot training progress for all bins and model types."""
        self.logger.info("Generating training progress plots for all models...")
        
        for bin_id in self.bin_ids:
            for model_type in ['simple', 'attention']:
                self.plot_training_progress(bin_id, model_type)
        
        self.logger.info("All training progress plots generated!")
    
    def generate_training_summary(self):
        """Generate comprehensive training summary with all plots and history."""
        self.logger.info("Generating comprehensive training summary...")
        
        # Generate all training progress plots
        self.plot_all_training_progress()
        
        # Convert numpy types to standard Python types for JSON serialization
        serializable_history = {}
        for bin_id in self.training_history:
            # Convert numpy.int64 to standard int
            bin_id_str = str(int(bin_id))
            serializable_history[bin_id_str] = {}
            
            for model_type in ['simple', 'attention']:
                if model_type in self.training_history[bin_id]:
                    history = self.training_history[bin_id][model_type]
                    if history['train_losses']:
                        serializable_history[bin_id_str][model_type] = {
                            'final_train_loss': float(history['train_losses'][-1]),
                            'final_val_loss': float(history['val_losses'][-1]),
                            'best_val_loss': float(min(history['val_losses'])),
                            'epochs_trained': int(len(history['epochs'])),
                            'final_learning_rate': float(history['learning_rates'][-1]),
                            'learning_rate_reductions': int(len([i for i in range(1, len(history['learning_rates'])) 
                                                         if history['learning_rates'][i] < history['learning_rates'][i-1]]))
                        }
        
        # Save training history to JSON
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2, default=str)
        
        self.logger.info(f"Training history saved: {history_path}")
        
        # Generate summary statistics
        summary_stats = {}
        for bin_id in self.bin_ids:
            bin_id_str = str(int(bin_id))
            summary_stats[bin_id_str] = {}
            for model_type in ['simple', 'attention']:
                if bin_id in self.training_history and model_type in self.training_history[bin_id]:
                    history = self.training_history[bin_id][model_type]
                    if history['train_losses']:
                        summary_stats[bin_id_str][model_type] = {
                            'final_train_loss': float(history['train_losses'][-1]),
                            'final_val_loss': float(history['val_losses'][-1]),
                            'best_val_loss': float(min(history['val_losses'])),
                            'epochs_trained': int(len(history['epochs'])),
                            'final_learning_rate': float(history['learning_rates'][-1]),
                            'learning_rate_reductions': int(len([i for i in range(1, len(history['learning_rates'])) 
                                                         if history['learning_rates'][i] < history['learning_rates'][i-1]]))
                        }
        
        # Save summary statistics
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
        
        return summary_stats
    
    def train_models(self) -> Dict:
        """
        Train both Simple LSTM and Attention LSTM models.
        
        Returns:
            Dictionary containing training results for all models
        """
        self.logger.info("Starting model training...")
        
        # Check for existing trained models to enable resume functionality
        existing_models_info = self._check_existing_models()
        self.logger.info(f"Existing trained models found: {existing_models_info}")
        
        for model_type in self.model_types:
            # Initialize results structure for this model type
            if model_type not in self.results:
                self.results[model_type] = {
                    'frequency_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}},
                    'magnitude_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}}
                }
            
            # Check if all bins for this model type are already trained
            completed_bin_ids = existing_models_info.get(model_type, [])
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training {model_type.upper()} LSTM models")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Resuming training for {model_type} on bins: {completed_bin_ids}")
            
            # Filter out completed bins from self.bin_ids
            remaining_bin_ids = [
                bin_id for bin_id in self.bin_ids if bin_id not in completed_bin_ids
            ]
            
            if not remaining_bin_ids:
                self.logger.info(f"All {model_type} models for all bins are already trained. Skipping.")
                # Initialize empty results structure for this model type since we're skipping training
                self.results[model_type] = {
                    'frequency_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}},
                    'magnitude_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}}
                }
                continue
            
            self.results[model_type] = self._train_model_type(model_type, remaining_bin_ids)
        
        # Compare models if both were trained
        if len(self.model_types) > 1:
            # Check if we have results to compare
            if any(model_type in self.results and self.results[model_type] for model_type in self.model_types):
                self.logger.info(f"\n{'='*50}")
                self.logger.info("Comparing model performance")
                self.logger.info(f"{'='*50}")
                
                self.results['comparison'] = self._compare_models()
            else:
                self.logger.info("No results to compare - skipping comparison")
        
        # Load existing models and merge results
        self._load_existing_models(existing_models_info)
        
        # Save comprehensive results
        self._save_results()
        
        # Generate comprehensive training summary with plots
        self.logger.info("Generating comprehensive training summary...")
        summary_stats = self.generate_training_summary()
        
        return self.results
    
    def _check_existing_models(self) -> Dict[str, List[int]]:
        """
        Check for existing trained models to enable resume functionality.
        
        Returns:
            Dictionary mapping model_type to list of completed bin IDs
        """
        existing_models = {}
        
        # Check both the current save_dir and the data/results directory where models might be saved
        search_dirs = [self.save_dir]
        
        # Also check data/results directory if it exists and is different from save_dir
        data_results_dir = Path("data/results")
        if data_results_dir.exists() and data_results_dir != self.save_dir:
            search_dirs.append(data_results_dir)
        
        self.logger.info(f"Searching for existing models in directories: {[str(d) for d in search_dirs]}")
        
        for model_type in self.model_types:
            existing_models[model_type] = []
            
            # Check for frequency models in all search directories
            freq_pattern = f"{model_type}_frequency_bin_*.pth"
            mag_pattern = f"{model_type}_magnitude_bin_*.pth"
            
            freq_files = []
            mag_files = []
            
            for search_dir in search_dirs:
                freq_files.extend(list(search_dir.glob(freq_pattern)))
                mag_files.extend(list(search_dir.glob(mag_pattern)))
            
            self.logger.info(f"Found {len(freq_files)} frequency files and {len(mag_files)} magnitude files for {model_type}")
            
            # Extract bin IDs from filenames
            for file_path in freq_files + mag_files:
                try:
                    # Extract bin ID from filename like "simple_frequency_bin_5.pth"
                    bin_id = int(file_path.stem.split('_')[-1])
                    if bin_id not in existing_models[model_type]:
                        existing_models[model_type].append(bin_id)
                except (ValueError, IndexError):
                    self.logger.warning(f"Failed to extract bin ID from filename: {file_path}")
                    continue
            
            existing_models[model_type].sort()
            self.logger.info(f"Model type {model_type}: Found bins {existing_models[model_type]}")
        
        return existing_models
    
    def _load_existing_models(self, existing_models_info: Dict[str, List[int]]):
        """
        Load existing trained models and merge them with current results.
        
        Args:
            existing_models_info: Dictionary mapping model_type to list of completed bin IDs
        """
        self.logger.info("Loading existing trained models...")
        
        # Check both the current save_dir and the data/results directory where models might be saved
        search_dirs = [self.save_dir]
        
        # Also check data/results directory if it exists and is different from save_dir
        data_results_dir = Path("data/results")
        if data_results_dir.exists() and data_results_dir != self.save_dir:
            search_dirs.append(data_results_dir)
        
        for model_type in self.model_types:
            if model_type not in self.results:
                self.results[model_type] = {
                    'frequency_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}},
                    'magnitude_models': {'models': {}, 'training_history': {}, 'test_metrics': {}, 'validation_metrics': {}}
                }
            
            completed_bins = existing_models_info.get(model_type, [])
            
            for bin_id in completed_bins:
                self.logger.info(f"Loading existing {model_type} model for bin {bin_id}")
                
                # Load frequency model - search in all directories
                freq_model_loaded = False
                for search_dir in search_dirs:
                    freq_path = search_dir / f"{model_type}_frequency_bin_{bin_id}.pth"
                    if freq_path.exists():
                        try:
                            # Create model with correct architecture
                            input_size = 2  # Default for your data
                            model = self._create_model(model_type, input_size)
                            model.load_state_dict(torch.load(freq_path, map_location=self.device))
                            self.results[model_type]['frequency_models']['models'][bin_id] = model
                            
                            # Create placeholder metrics for existing models
                            self.results[model_type]['frequency_models']['test_metrics'][bin_id] = {
                                'loss': 0.0,  # Placeholder - we don't have actual test metrics
                                'mae': 0.0
                            }
                            self.results[model_type]['frequency_models']['validation_metrics'][bin_id] = {
                                'loss': 0.0,  # Placeholder - we don't have actual validation metrics
                                'mae': 0.0
                            }
                            
                            self.logger.info(f"✅ Loaded {model_type} frequency model for bin {bin_id}")
                            freq_model_loaded = True
                            break
                        except Exception as e:
                            self.logger.warning(f"Failed to load {model_type} frequency model for bin {bin_id} from {search_dir}: {e}")
                
                if not freq_model_loaded:
                    self.logger.warning(f"Failed to load {model_type} frequency model for bin {bin_id} from any directory")
                
                # Load magnitude model - search in all directories
                mag_model_loaded = False
                for search_dir in search_dirs:
                    mag_path = search_dir / f"{model_type}_magnitude_bin_{bin_id}.pth"
                    if mag_path.exists():
                        try:
                            # Create model with correct architecture
                            input_size = 2  # Default for your data
                            model = self._create_model(model_type, input_size)
                            model.load_state_dict(torch.load(mag_path, map_location=self.device))
                            self.results[model_type]['magnitude_models']['models'][bin_id] = model
                            
                            # Create placeholder metrics for existing models
                            self.results[model_type]['magnitude_models']['test_metrics'][bin_id] = {
                                'loss': 0.0,  # Placeholder - we don't have actual test metrics
                                'mae': 0.0
                            }
                            self.results[model_type]['magnitude_models']['validation_metrics'][bin_id] = {
                                'loss': 0.0,  # Placeholder - we don't have actual validation metrics
                                'mae': 0.0
                            }
                            
                            self.logger.info(f"✅ Loaded {model_type} magnitude model for bin {bin_id}")
                            mag_model_loaded = True
                            break
                        except Exception as e:
                            self.logger.warning(f"Failed to load {model_type} magnitude model for bin {bin_id} from {search_dir}: {e}")
                
                if not mag_model_loaded:
                    self.logger.warning(f"Failed to load {model_type} magnitude model for bin {bin_id} from any directory")
    
    def _train_model_type(self, model_type: str, remaining_bin_ids: List[int] = None) -> Dict:
        """
        Train models of a specific type (simple or attention).
        
        Args:
            model_type: 'simple' or 'attention'
            remaining_bin_ids: List of bin IDs that still need training
            
        Returns:
            Dictionary containing training results for all models
        """
        if remaining_bin_ids is None:
            remaining_bin_ids = self.bin_ids
            
        results = {
            'frequency_models': {},
            'magnitude_models': {},
            'training_history': {},
            'test_metrics': {},
            'validation_metrics': {}
        }
        
        # Train frequency models
        self.logger.info(f"Training {model_type} LSTM models for earthquake frequency...")
        freq_results = self._train_target_models(model_type, 'frequency', remaining_bin_ids)
        
        # Train magnitude models  
        self.logger.info(f"Training {model_type} LSTM models for maximum magnitude...")
        mag_results = self._train_target_models(model_type, 'magnitude', remaining_bin_ids)
        
        return {
            'frequency_models': {
                'models': freq_results['models'],
                'training_history': freq_results['training_history'],
                'test_metrics': freq_results['test_metrics'],
                'validation_metrics': freq_results['validation_metrics']
            },
            'magnitude_models': {
                'models': mag_results['models'],
                'training_history': mag_results['training_history'],
                'test_metrics': mag_results['test_metrics'],
                'validation_metrics': mag_results['validation_metrics']
            }
        }
    
    def _train_target_models(self, model_type: str, target: str, remaining_bin_ids: List[int] = None) -> Dict:
        """
        Train models for a specific target (frequency or magnitude).
        
        Args:
            model_type: 'simple' or 'attention'
            target: 'frequency' or 'magnitude'
            remaining_bin_ids: List of bin IDs that still need training
            
        Returns:
            Dictionary containing model results
        """
        if remaining_bin_ids is None:
            remaining_bin_ids = self.bin_ids
            
        self.logger.info(f"Training {len(remaining_bin_ids)} {model_type} models for {target} on bins: {remaining_bin_ids}")
        
        models = {}
        training_history = {}
        test_metrics = {}
        validation_metrics = {}
        
        # Get data loaders for remaining bins
        bin_loaders = self.data_loader.get_bin_loaders()
        
        for bin_id in remaining_bin_ids:
            self.logger.info(f"Training {model_type} LSTM for bin {bin_id} ({target})...")
            
            try:
                # Get data loaders for this bin
                if bin_id not in bin_loaders:
                    self.logger.warning(f"Bin {bin_id} not found in data loaders, skipping...")
                    continue
                
                train_loader, val_loader, test_loader = bin_loaders[bin_id]
                
                # Check if we have data
                if len(train_loader.dataset) == 0:
                    self.logger.warning(f"No training data for bin {bin_id}, skipping...")
                    continue
                
                self.logger.info(f"Bin {bin_id}: Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
                
                # Create model
                # Get input size from the first batch
                try:
                    sample_batch = next(iter(train_loader))
                    input_size = sample_batch[0].shape[-1]  # Last dimension is input size
                    self.logger.info(f"Bin {bin_id}: Input size detected as {input_size}")
                except Exception as e:
                    self.logger.error(f"Error detecting input size for bin {bin_id}: {e}")
                    continue
                
                model = self._create_model(model_type, input_size)
                
                # Train model
                model, history, metrics = self._train_single_model(
                    model, train_loader, val_loader, test_loader, bin_id, target
                )
                
                # Store results
                models[bin_id] = model
                training_history[bin_id] = history
                test_metrics[bin_id] = metrics['test']
                validation_metrics[bin_id] = metrics['validation']
                
                # Save model
                model_path = self.save_dir / f"{model_type}_{target}_bin_{bin_id}.pth"
                torch.save(model.state_dict(), model_path)
                
            except Exception as e:
                self.logger.error(f"Error training {model_type} model for bin {bin_id}: {e}")
                continue
        
        return {
            'models': models,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'validation_metrics': validation_metrics
        }
    
    def _create_model(self, model_type: str, input_size: int) -> nn.Module:
        """
        Create a model of the specified type.
        
        Args:
            model_type: 'simple' or 'attention'
            input_size: Number of input features
            
        Returns:
            Initialized model
        """
        if model_type == 'simple':
            model = SimpleLSTM(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                dropout=0.2
            )
        elif model_type == 'attention':
            model = AttentionLSTM(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.to(self.device)
        return model
    
    def _train_single_model(self, 
                           model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: DataLoader,
                           bin_id: int,
                           target: str) -> Tuple[nn.Module, Dict, Dict]:
        """
        Train a single model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            bin_id: Bin ID for logging
            target: Target variable name
            
        Returns:
            Tuple of (trained_model, training_history, metrics)
        """
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # Initialize best model state
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_maes = []
            
            for batch_X, batch_y, batch_metadata in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Debug: Log shapes to understand the mismatch
                if epoch == 0:  # Only log first epoch to avoid spam
                    self.logger.info(f"Debug - batch_X shape: {batch_X.shape}, batch_y shape: {batch_y.shape}, outputs shape: {outputs.shape}")
                
                # Ensure proper shape alignment for loss calculation
                if outputs.dim() == 3:  # [batch, seq, features]
                    outputs = outputs.squeeze(1)  # Remove sequence dimension
                if batch_y.dim() == 3:  # [batch, seq, features]
                    batch_y = batch_y.squeeze(1)  # Remove sequence dimension
                
                # Now both should be [batch, features]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_maes.append(mean_absolute_error(
                    batch_y.cpu().numpy(),
                    outputs.detach().cpu().numpy()
                ))
            
            # Validation phase
            model.eval()
            val_losses = []
            val_maes = []
            
            with torch.no_grad():
                for batch_X, batch_y, batch_metadata in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    
                    # Ensure proper shape alignment for loss calculation
                    if outputs.dim() == 3:  # [batch, seq, features]
                        outputs = outputs.squeeze(1)  # Remove sequence dimension
                    if batch_y.dim() == 3:  # [batch, seq, features]
                        batch_y = batch_y.squeeze(1)  # Remove sequence dimension
                    
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
                    val_maes.append(mean_absolute_error(
                        batch_y.cpu().numpy(),
                        outputs.cpu().numpy()
                    ))
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_mae = np.mean(train_maes)
            avg_val_mae = np.mean(val_maes)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Store history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_mae'].append(avg_train_mae)
            training_history['val_mae'].append(avg_val_mae)
            
            # Track training history for enhanced plotting
            if bin_id in self.training_history:
                # Determine model type from the model class
                if 'SimpleLSTM' in str(type(model)):
                    model_type_key = 'simple'
                else:
                    model_type_key = 'attention'
                
                if model_type_key in self.training_history[bin_id]:
                    self.training_history[bin_id][model_type_key]['train_losses'].append(float(avg_train_loss))
                    self.training_history[bin_id][model_type_key]['val_losses'].append(float(avg_val_loss))
                    self.training_history[bin_id][model_type_key]['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
                    self.training_history[bin_id][model_type_key]['epochs'].append(int(epoch))
            
            # Log progress
            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            self.logger.warning(f"No best model state found for bin {bin_id}, keeping final model")
        
        # Calculate final metrics on test set
        model.eval()
        test_losses = []
        test_maes = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_metadata in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                
                # Ensure proper shape alignment for loss calculation
                if outputs.dim() == 3:  # [batch, seq, features]
                    outputs = outputs.squeeze(1)  # Remove sequence dimension
                if batch_y.dim() == 3:  # [batch, seq, features]
                    batch_y = batch_y.squeeze(1)  # Remove sequence dimension
                
                loss = criterion(outputs, batch_y)
                test_losses.append(loss.item())
                test_maes.append(mean_absolute_error(
                    batch_y.cpu().numpy(),
                    outputs.cpu().numpy()
                ))
        
        test_metrics = {
            'loss': np.mean(test_losses),
            'mae': np.mean(test_maes)
        }
        
        # Calculate validation metrics from the training history
        validation_metrics = {
            'loss': training_history['val_loss'][-1] if training_history['val_loss'] else float('inf'),
            'mae': training_history['val_mae'][-1] if training_history['val_mae'] else float('inf')
        }
        
        return model, training_history, {
            'test': test_metrics,
            'validation': validation_metrics
        }
    
    def _evaluate_model(self, model: nn.Module, test_data: Tuple, split: str) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            test_data: Test data (X, y)
            split: Data split name ('test' or 'validation')
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_tensor).cpu().numpy().squeeze()
            y_true = y_test.squeeze()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # WMAPE calculation
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        forecast_accuracy = 100 - wmape
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'wmape': wmape,
            'forecast_accuracy': forecast_accuracy,
            'predictions': y_pred,
            'actuals': y_true
        }
    
    def _compare_models(self) -> Dict:
        """
        Compare performance between Simple LSTM and Attention LSTM models.
        
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'frequency': {},
            'magnitude': {},
            'overall': {}
        }
        
        for target in ['frequency', 'magnitude']:
            comparison[target] = self._compare_target_models(target)
        
        # Overall comparison
        comparison['overall'] = self._calculate_overall_comparison()
        
        return comparison
    
    def _compare_target_models(self, target: str) -> Dict:
        """
        Compare models for a specific target.
        
        Args:
            target: 'frequency' or 'magnitude'
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Check if both model types have results
        if ('simple' not in self.results or 'attention' not in self.results or
            f'{target}_models' not in self.results['simple'] or 
            f'{target}_models' not in self.results['attention']):
            self.logger.warning(f"Cannot compare {target} models - missing results for one or both model types")
            return comparison
        
        for metric in ['mae', 'mse', 'rmse', 'wmape', 'forecast_accuracy']:
            simple_scores = []
            attention_scores = []
            
            for bin_id in self.bin_ids:
                if (bin_id in self.results['simple'][f'{target}_models']['test_metrics'] and
                    bin_id in self.results['attention'][f'{target}_models']['test_metrics']):
                    
                    # Skip placeholder metrics (0.0 values indicate loaded models without real metrics)
                    simple_score = self.results['simple'][f'{target}_models']['test_metrics'][bin_id][metric]
                    attention_score = self.results['attention'][f'{target}_models']['test_metrics'][bin_id][metric]
                    
                    # Only include real scores (not placeholder 0.0 values)
                    if simple_score > 0.0 and attention_score > 0.0:
                        simple_scores.append(simple_score)
                        attention_scores.append(attention_score)
            
            if simple_scores and attention_scores:
                simple_mean = np.mean(simple_scores)
                attention_mean = np.mean(attention_scores)
                
                # Prevent division by zero
                improvement_pct = 0.0
                if simple_mean > 0.0:
                    improvement_pct = ((attention_mean - simple_mean) / simple_mean) * 100
                
                comparison[metric] = {
                    'simple_mean': simple_mean,
                    'attention_mean': attention_mean,
                    'simple_std': np.std(simple_scores),
                    'attention_std': np.std(attention_scores),
                    'improvement': attention_mean - simple_mean,
                    'improvement_pct': improvement_pct
                }
        
        return comparison
    
    def _calculate_overall_comparison(self) -> Dict:
        """
        Calculate overall comparison metrics.
        
        Returns:
            Overall comparison results
        """
        overall = {}
        
        # Check if both model types have results
        if ('simple' not in self.results or 'attention' not in self.results or
            'frequency_models' not in self.results['simple'] or 
            'frequency_models' not in self.results['attention'] or
            'magnitude_models' not in self.results['simple'] or 
            'magnitude_models' not in self.results['attention']):
            self.logger.warning("Cannot calculate overall comparison - missing results for one or both model types")
            return overall
        
        for metric in ['mae', 'mse', 'rmse', 'wmape', 'forecast_accuracy']:
            freq_simple = self.results['simple']['frequency_models']['test_metrics']
            freq_attention = self.results['attention']['frequency_models']['test_metrics']
            mag_simple = self.results['simple']['magnitude_models']['test_metrics']
            mag_attention = self.results['attention']['magnitude_models']['test_metrics']
            
            simple_scores = []
            attention_scores = []
            
            # Collect all scores
            for bin_id in self.bin_ids:
                if (bin_id in freq_simple and bin_id in freq_attention and
                    bin_id in mag_simple and bin_id in mag_attention):
                    
                    # Skip placeholder metrics (0.0 values indicate loaded models without real metrics)
                    freq_simple_score = freq_simple[bin_id][metric]
                    freq_attention_score = freq_attention[bin_id][metric]
                    mag_simple_score = mag_simple[bin_id][metric]
                    mag_attention_score = mag_attention[bin_id][metric]
                    
                    # Only include real scores (not placeholder 0.0 values)
                    if (freq_simple_score > 0.0 and freq_attention_score > 0.0 and
                        mag_simple_score > 0.0 and mag_attention_score > 0.0):
                        
                        simple_scores.extend([freq_simple_score, mag_simple_score])
                        attention_scores.extend([freq_attention_score, mag_attention_score])
            
            if simple_scores and attention_scores:
                simple_mean = np.mean(simple_scores)
                attention_mean = np.mean(attention_scores)
                
                # Prevent division by zero
                improvement_pct = 0.0
                if simple_mean > 0.0:
                    improvement_pct = ((attention_mean - simple_mean) / simple_mean) * 100
                
                overall[metric] = {
                    'simple_mean': simple_mean,
                    'attention_mean': attention_mean,
                    'simple_std': np.std(simple_scores),
                    'attention_std': np.std(attention_scores),
                    'improvement': attention_mean - simple_mean,
                    'improvement_pct': improvement_pct
                }
        
        return overall
    
    def _save_results(self):
        """Save training results to files."""
        # Check if we have any results to save
        if not self.results or not any(model_type in self.results and self.results[model_type] for model_type in self.model_types):
            self.logger.warning("No training results to save - skipping save operation")
            return
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save results summary
        results_summary = {
            'model_types': self.model_types,
            'training_config': {
                'hidden_sizes': self.hidden_sizes,
                'lookback_years': self.lookback_years,
                'num_epochs': self.num_epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            },
            'results': convert_numpy_types(self.results)
        }
        
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Generate comparison plots only if we have results to compare
        if len(self.model_types) > 1 and any(model_type in self.results and self.results[model_type] for model_type in self.model_types):
            self._generate_comparison_plots()
        else:
            self.logger.info("No results to compare - skipping comparison plots")
        
        self.logger.info(f"Results saved to: {self.save_dir}")
    
    def _generate_comparison_plots(self):
        """Generate comparison plots between models."""
        # Check if comparison results exist
        if 'comparison' not in self.results or not self.results['comparison']:
            self.logger.warning("No comparison results available - skipping comparison plots")
            return
        
        # Create subplot grid that can accommodate all metrics
        # 5 metrics need 5 columns, 2 targets need 2 rows
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle('Simple LSTM vs Attention LSTM Performance Comparison', fontsize=16)
        
        metrics = ['mae', 'mse', 'rmse', 'wmape', 'forecast_accuracy']
        targets = ['frequency', 'magnitude']
        
        for i, target in enumerate(targets):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                if (target in self.results['comparison'] and 
                    metric in self.results['comparison'][target]):
                    comp = self.results['comparison'][target][metric]
                    
                    models = ['Simple LSTM', 'Attention LSTM']
                    means = [comp['simple_mean'], comp['attention_mean']]
                    stds = [comp['simple_std'], comp['attention_std']]
                    
                    bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
                    ax.set_title(f'{target.title()} - {metric.upper()}')
                    ax.set_ylabel(metric.upper())
                    
                    # Add improvement annotation
                    if comp['improvement_pct'] != 0:
                        color = 'green' if comp['improvement'] < 0 else 'red'
                        ax.text(0.5, 0.95, f"{comp['improvement_pct']:.1f}% {'improvement' if comp['improvement'] < 0 else 'worse'}",
                               transform=ax.transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                else:
                    # Handle case where comparison data is missing
                    ax.text(0.5, 0.5, f'No {target} {metric} data', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{target.title()} - {metric.upper()}')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Comparison plots generated")


def train_enhanced_quadtree_models(data_path: str,
                                  save_dir: str,
                                  logger: logging.Logger,
                                  model_types: List[str] = ['simple', 'attention'],
                                  **kwargs) -> Dict:
    """
    Train enhanced quadtree models with comparison capabilities.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save results
        logger: Logger instance
        model_types: List of model types to train
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary containing training results
    """
    trainer = EnhancedQuadtreeTrainer(
        data_path=data_path,
        save_dir=save_dir,
        logger=logger,
        model_types=model_types,
        **kwargs
    )
    
    return trainer.train_models()
