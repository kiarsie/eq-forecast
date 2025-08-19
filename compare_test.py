#!/usr/bin/env python3
"""
Model Comparison Script: Simple LSTM vs Attention LSTM

This script trains and compares both models to determine which performs better
for earthquake forecasting.
"""

import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.enhanced_shared_processor import EnhancedSharedDataset
from src.models.shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
from src.models.shared_lstm_trainer import SharedLSTMTrainer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_comparison.log'),
            logging.StreamHandler()
        ]
    )


def create_data_loaders(data_path: str, 
                       lookback_years: int = 10,
                       batch_size: int = 16,
                       **kwargs) -> tuple:
    """Create data loaders for training, validation, and testing."""
    dataset = EnhancedSharedDataset(
        data_path=data_path,
        lookback_years=lookback_years,
        **kwargs
    )
    
    train_sequences = dataset.get_split_sequences('train')
    val_sequences = dataset.get_split_sequences('val')
    test_sequences = dataset.get_split_sequences('test')
    
    print(f"Data split: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")
    
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


def train_and_evaluate_model(model_name: str,
                           model: torch.nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: DataLoader,
                           config: Dict) -> Dict:
    """Train and evaluate a single model."""
    print(f"\n=== Training {model_name} ===")
    
    # Create trainer
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
    
    # Train model
    training_history = trainer.train(
        max_epochs=config['max_epochs'],
        save_path=f"src/models/{model_name.lower().replace(' ', '_')}_best.pth",
        save_best=True
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history(save_path=f'{model_name.lower().replace(" ", "_")}_training_history.png')
    
    return {
        'model_name': model_name,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'trainer': trainer
    }


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """Compare model performance and create summary."""
    comparison_data = []
    
    for result in results:
        metrics = result['test_metrics']
        comparison_data.append({
            'Model': result['model_name'],
            'Total Loss': metrics['total_loss'],
            'Magnitude MSE': metrics['magnitude_mse'],
            'Magnitude MAE': metrics['magnitude_mae'],
            'Magnitude Correlation': metrics['magnitude_correlation'],
            'Frequency Poisson NLL': metrics['frequency_poisson_nll'],
            'Frequency MAE': metrics['frequency_mae'],
            'Frequency Correlation': metrics['frequency_correlation'],
            'Frequency MSE': metrics['frequency_mse']
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_comparison(results: List[Dict], save_path: str = 'model_comparison.png'):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = [r['model_name'] for r in results]
    
    # Magnitude metrics
    magnitude_mse = [r['test_metrics']['magnitude_mse'] for r in results]
    magnitude_mae = [r['test_metrics']['magnitude_mae'] for r in results]
    magnitude_corr = [r['test_metrics']['magnitude_correlation'] for r in results]
    
    # Frequency metrics
    freq_poisson_nll = [r['test_metrics']['frequency_poisson_nll'] for r in results]
    freq_mae = [r['test_metrics']['frequency_mae'] for r in results]
    freq_corr = [r['test_metrics']['frequency_correlation'] for r in results]
    
    # Plot 1: Magnitude Performance
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, magnitude_mse, width, label='MSE', alpha=0.8)
    axes[0, 0].bar(x + width/2, magnitude_mae, width, label='MAE', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].set_title('Magnitude Prediction Performance')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Magnitude Correlation
    axes[0, 1].bar(model_names, magnitude_corr, alpha=0.8, color='green')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Magnitude Prediction Correlation')
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Frequency Performance
    axes[1, 0].bar(x - width/2, freq_poisson_nll, width, label='Poisson NLL', alpha=0.8)
    axes[1, 0].bar(x + width/2, freq_mae, width, label='MAE', alpha=0.8)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Frequency Prediction Performance')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Frequency Correlation
    axes[1, 1].bar(model_names, freq_corr, alpha=0.8, color='orange')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Frequency Prediction Correlation')
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function."""
    print("=== MODEL COMPARISON: Simple LSTM vs Attention LSTM ===")
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    config = {
        'data_path': 'data/eq_catalog.csv',
        'lookback_years': 10,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'magnitude_weight': 1.0,
        'frequency_weight': 3.0,  # Updated default α = 3.0
        'max_epochs': 100,  # Reduced for comparison
        'rolling_windows': [3, 5, 10],
        'train_end_year': 2009,
        'val_end_year': 2017,
        'test_end_year': 2025
    }
    
    logger.info("Starting model comparison")
    logger.info(f"Configuration: {config}")
    
    # Check if data file exists
    if not os.path.exists(config['data_path']):
        logger.error(f"Data file not found: {config['data_path']}")
        return
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=config['data_path'],
            lookback_years=config['lookback_years'],
            batch_size=config['batch_size'],
            rolling_windows=config['rolling_windows'],
            train_end_year=config['train_end_year'],
            val_end_year=config['val_end_year'],
            test_end_year=config['test_end_year']
        )
        
        # Get feature dimensions
        sample_batch = next(iter(train_loader))
        input_seq, target_seq, metadata, _ = sample_batch
        input_seq_features = input_seq.shape[2]
        metadata_features = metadata.shape[1]
        
        # Create models
        simple_lstm = SharedLSTMModel(
            input_seq_features=input_seq_features,
            metadata_features=metadata_features,
            lookback_years=config['lookback_years']
        )
        
        attention_lstm = AttentionSharedLSTMModel(
            input_seq_features=input_seq_features,
            metadata_features=metadata_features,
            lookback_years=config['lookback_years']
        )
        
        # Train and evaluate both models
        results = []
        
        # Simple LSTM
        simple_result = train_and_evaluate_model(
            "Simple LSTM",
            simple_lstm,
            train_loader, val_loader, test_loader,
            config
        )
        results.append(simple_result)
        
        # Attention LSTM
        attention_result = train_and_evaluate_model(
            "Attention LSTM",
            attention_lstm,
            train_loader, val_loader, test_loader,
            config
        )
        results.append(attention_result)
        
        # Compare results
        comparison_df = compare_models(results)
        print("\n=== MODEL COMPARISON RESULTS ===")
        print(comparison_df.to_string(index=False))
        
        # Save comparison results
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        
        # Create comparison plots
        plot_comparison(results)
        
        # Determine winner
        simple_total_loss = simple_result['test_metrics']['total_loss']
        attention_total_loss = attention_result['test_metrics']['total_loss']
        
        if simple_total_loss < attention_total_loss:
            winner = "Simple LSTM"
            improvement = ((attention_total_loss - simple_total_loss) / attention_total_loss) * 100
        else:
            winner = "Attention LSTM"
            improvement = ((simple_total_loss - attention_total_loss) / simple_total_loss) * 100
        
        print(f"\n🏆 WINNER: {winner}")
        print(f"Improvement: {improvement:.2f}%")
        
        logger.info("Model comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()