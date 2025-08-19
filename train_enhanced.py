#!/usr/bin/env python3
"""
Enhanced Training Script for Earthquake Forecasting

Uses the improved architecture and data processing:
- Balanced hidden sizes: (64, 48, 24, 24)
- Enhanced data processing with rolling features
- Better normalization for sparse data
- Improved training parameters
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
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.enhanced_trainer import EnhancedQuadtreeTrainer
from src.models.enhanced_data_processor import EnhancedQuadtreeDataLoader


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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


def train_enhanced_models(data_path: str, 
                         save_dir: str, 
                         logger: logging.Logger,
                         model_types: List[str] = ['simple', 'attention'],
                         **kwargs) -> dict:
    """
    Train enhanced earthquake forecasting models.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save results
        logger: Logger instance
        model_types: List of model types to train
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary containing training results
    """
    logger.info("=== ENHANCED EARTHQUAKE FORECASTING TRAINING ===")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Model types: {model_types}")
    
    # Enhanced training parameters
    enhanced_params = {
        'hidden_sizes': (64, 48, 24, 24),  # Balanced capacity
        'lookback_years': 10,
        'num_epochs': 100,
        'patience': 15,  # Increased patience
        'batch_size': 32,
        'learning_rate': 0.001,
        'add_rolling_features': True,
        'rolling_windows': [3, 5, 7],
        **kwargs
    }
    
    logger.info("Enhanced parameters:")
    for key, value in enhanced_params.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize enhanced trainer
    trainer = EnhancedQuadtreeTrainer(
        data_path=data_path,
        save_dir=save_dir,
        logger=logger,
        model_types=model_types,
        **enhanced_params
    )
    
    # Train models
    logger.info("Starting enhanced training...")
    results = trainer.train_models()
    
    logger.info("Enhanced training completed successfully!")
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced Earthquake Forecasting Training")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to processed earthquake data CSV")
    parser.add_argument("--save_dir", type=str, default="data/results/enhanced",
                       help="Directory to save enhanced training results")
    parser.add_argument("--model_types", nargs="+", default=["simple", "attention"],
                       choices=["simple", "attention"],
                       help="Model types to train")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Validate data path
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Train enhanced models
        results = train_enhanced_models(
            data_path=args.data_path,
            save_dir=str(save_dir),
            logger=logger,
            model_types=args.model_types,
            num_epochs=args.num_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Display results summary
        logger.info("\n=== ENHANCED TRAINING RESULTS SUMMARY ===")
        if isinstance(results, dict) and 'comparison' in results:
            for model_type in args.model_types:
                if model_type in results:
                    freq_models = len(results[model_type].get('frequency_models', {}).get('models', {}))
                    mag_models = len(results[model_type].get('magnitude_models', {}).get('models', {}))
                    logger.info(f"{model_type.title()} LSTM - Frequency models: {freq_models}, Magnitude models: {mag_models}")
        else:
            logger.info(f"Training completed for {len(results)} bins")
        
        logger.info(f"Results saved to: {save_dir}")
        
    except Exception as e:
        logger.error(f"Error during enhanced training: {e}")
        raise


if __name__ == "__main__":
    main()

