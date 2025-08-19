#!/usr/bin/env python3
"""
Main Training Script for Shared LSTM Earthquake Forecasting

This script demonstrates how to use the shared LSTM model for earthquake forecasting
with the specified architecture and training parameters.
"""

import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.enhanced_shared_processor import EnhancedSharedDataset
from src.models.shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
from src.models.shared_lstm_trainer import SharedLSTMTrainer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('shared_lstm_training.log'),
            logging.StreamHandler()
        ]
    )


def create_data_loaders(data_path: str, 
                       lookback_years: int = 10,
                       batch_size: int = 16,
                       **kwargs) -> tuple:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to earthquake catalog CSV
        lookback_years: Number of years to look back
        batch_size: Batch size for training
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = EnhancedSharedDataset(
        data_path=data_path,
        lookback_years=lookback_years,
        **kwargs
    )
    
    # Get sequences for each split
    train_sequences = dataset.get_split_sequences('train')
    val_sequences = dataset.get_split_sequences('val')
    test_sequences = dataset.get_split_sequences('test')
    
    print(f"Data split: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'train']
        ),
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'val']
        ),
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(
            [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'test']
        ),
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    print("=== STARTING TRAINING SCRIPT ===")
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("Logging setup complete")
    
    # Configuration
    config = {
        'data_path': 'data/eq_catalog.csv',  # Update with your data path
        'lookback_years': 10,
        'batch_size': 16,
        'learning_rate': 1e-4,        # 🔧 FIX: Reduced from 1e-3 to 1e-4 for better fine-tuning
        'weight_decay': 1e-3,         # 🔧 FIX: Increased from 1e-4 to 1e-3 for better regularization
        'magnitude_weight': 1.0,      # α = 1 (magnitude weight)
        'frequency_weight': 8.0,      # β = 6 (frequency weight)
        'max_epochs': 50,             # REDUCED to ~50 epochs (early stopping at ~30)
        'save_path': 'src/models/shared_lstm_best.pth',
        'rolling_windows': [3, 5, 10],
        'train_end_year': 2009,
        'val_end_year': 2017,
        'test_end_year': 2025
    }
    
    print(f"Configuration loaded: {config}")
    
    logger.info("Starting Shared LSTM Earthquake Forecasting Training")
    logger.info(f"Configuration: {config}")
    
    # Check if data file exists
    print("Checking data file...")
    if not os.path.exists(config['data_path']):
        logger.error(f"Data file not found: {config['data_path']}")
        logger.info("Please update the data_path in the config to point to your earthquake catalog CSV")
        return
    
    print("Data file found, proceeding...")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=config['data_path'],
            lookback_years=config['lookback_years'],
            batch_size=config['batch_size'],
            rolling_windows=config['rolling_windows'],
            train_end_year=config['train_end_year'],
            val_end_year=config['val_end_year'],
            test_end_year=config['test_end_year']
        )
        
        print("Data loaders created successfully")
        
        # Get feature dimensions from a sample batch
        print("Getting sample batch...")
        sample_batch = next(iter(train_loader))
        input_seq, target_seq, metadata, _ = sample_batch
        
        input_seq_features = input_seq.shape[2]  # Features per year
        metadata_features = metadata.shape[1]    # Metadata features
        target_features = target_seq.shape[2]    # Target features
        
        print(f"Feature dimensions: input={input_seq.shape}, target={target_seq.shape}, metadata={metadata.shape}")
        
        logger.info(f"Feature dimensions:")
        logger.info(f"  Input sequence: {input_seq.shape}")
        logger.info(f"  Metadata: {metadata.shape}")
        logger.info(f"  Target: {target_seq.shape}")
        
        # Create model
        print("Creating Shared LSTM model...")
        logger.info("Creating Shared LSTM model...")
        model = SharedLSTMModel(
            input_seq_features=input_seq_features,
            metadata_features=metadata_features,
            lookback_years=config['lookback_years'],
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25
        )
        
        print("Model created successfully")
        
        # Create trainer
        print("Creating trainer...")
        logger.info("Creating trainer...")
        # Note: Trainer includes CosineAnnealingWarmRestarts scheduler and dynamic β
        trainer = SharedLSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            magnitude_weight=config['magnitude_weight'],
            frequency_weight=config['frequency_weight']
        )
        
        print("Trainer created successfully")
        
        # Train model
        print("Starting training...")
        logger.info("Starting training...")
        training_history = trainer.train(
            max_epochs=config['max_epochs'],
            save_path=config['save_path'],
            save_best=True
        )
        
        print("Training completed")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Plot training history
        print("Plotting training history...")
        logger.info("Plotting training history...")
        trainer.plot_training_history(save_path='training_history.png')
        
        logger.info("Training completed successfully!")
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()