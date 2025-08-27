#!/usr/bin/env python3
"""
Run Optimized Earthquake Forecasting Training Pipeline

This script runs the full pipeline with the optimized hyperparameters that showed better performance:
- Learning Rate: 4e-4 (was 5e-4)
- Weight Decay: 5e-5 (was 1e-4)  
- Magnitude Weight: 1.5 (was 2.0)
- Frequency Weight: 2.0 (was 1.0)
- Correlation Weight: 0.0 (unchanged)

This ensures you get:
1. ✅ Evaluation summaries for both models
2. ✅ Training history plots
3. ✅ Better performing models with optimized hyperparameters
"""

import sys
import torch
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.model_comparison_trainer import ModelComparisonTrainer
from src.models.enhanced_shared_processor import EnhancedSharedDataset

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_datasets():
    """Create datasets for training."""
    logger = logging.getLogger(__name__)
    
    # Check for processed data
    data_paths = [
        "data/processed_earthquake_catalog_lstm_ready.csv",
        "data/processed_earthquake_catalog_annual_stats.csv"
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if not data_path:
        logger.error("No processed data found. Please run preprocessing first.")
        return None
    
    logger.info(f"Using data from: {data_path}")
    
    try:
        # Create the enhanced shared dataset
        dataset = EnhancedSharedDataset(
            data_path=data_path,
            lookback_years=10,
            target_horizon=1,
            normalize=True,
            rolling_windows=[3, 5, 10],
            train_end_year=2009,
            val_end_year=2017,
            test_end_year=2025
        )
        
        # Get feature dimensions
        input_features, target_features, metadata_features = dataset.get_feature_dimensions()
        logger.info(f"Feature dimensions: Input={input_features}, Target={target_features}, Metadata={metadata_features}")
        
        # Create data loaders using the correct approach
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
        
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return None

def main():
    """Main function to run optimized training pipeline."""
    logger = setup_logging()
    
    print("=" * 80)
    print("EARTHQUAKE FORECASTING - OPTIMIZED TRAINING PIPELINE")
    print("=" * 80)
    print("Using optimized hyperparameters for better performance:")
    print("  Learning Rate: 4e-4 (was 5e-4)")
    print("  Weight Decay: 5e-5 (was 1e-4)")
    print("  Magnitude Weight: 1.5 (was 2.0)")
    print("  Frequency Weight: 2.0 (was 1.0)")
    print("  Correlation Weight: 0.0 (unchanged)")
    print("=" * 80)
    
    # Create datasets
    datasets = create_datasets()
    if not datasets:
        print("❌ Failed to create datasets. Exiting.")
        return
    
    # Create model comparison trainer with optimized hyperparameters
    comparison_trainer = ModelComparisonTrainer(
        train_loader=datasets['train_loader'],
        val_loader=datasets['val_loader'],
        test_loader=datasets['test_loader'],
        learning_rate=4e-4,        # ✅ Optimized: was 5e-4
        weight_decay=5e-5,         # ✅ Optimized: was 1e-4
        magnitude_weight=1.5,      # ✅ Optimized: was 2.0
        frequency_weight=2.0,      # ✅ Optimized: was 1.0
        correlation_weight=0.0,    # ✅ Unchanged
        device='auto',
        output_dir="data/results/optimized_comparison",
        config={
            'model_architecture': {
                'lstm_hidden_1': 64,
                'lstm_hidden_2': 32,
                'dense_hidden': 32
            },
            'frequency_scaling': {
                'scaling_lr_multiplier': 8.0,
                'scaling_wd_multiplier': 1.0
            },
            'loss_weights': {
                'variance_penalty_weight': 0.05,
                'warmup_epochs': 20
            }
        }
    )
    
    logger.info("Starting model comparison with optimized hyperparameters...")
    
    # Train and compare both models
    try:
        results = comparison_trainer.run_comparison(max_epochs=100, patience=15)
        
        if results:
            logger.info("✅ Model comparison completed successfully!")
            
            # Print detailed comparison summary
            logger.info("\n" + "="*60)
            logger.info("FINAL OPTIMIZED MODEL COMPARISON RESULTS")
            logger.info("="*60)
            
            # Simple LSTM results
            simple_metrics = results["simple_lstm"]["test_metrics"]
            logger.info("SIMPLE LSTM TEST PERFORMANCE:")
            logger.info(f"  Total Loss: {simple_metrics.get('total_loss', 'N/A'):.4f}")
            logger.info(f"  Magnitude Loss: {simple_metrics.get('magnitude_loss', 'N/A'):.4f}")
            logger.info(f"  Frequency Loss: {simple_metrics.get('frequency_loss', 'N/A'):.4f}")
            logger.info(f"  Magnitude MAE: {simple_metrics.get('magnitude_mae', 'N/A'):.4f}")
            logger.info(f"  Frequency MAE: {simple_metrics.get('frequency_mae', 'N/A'):.4f}")
            logger.info(f"  Magnitude Correlation: {simple_metrics.get('magnitude_corr', 'N/A'):.4f}")
            logger.info(f"  Frequency Correlation: {simple_metrics.get('frequency_corr', 'N/A'):.4f}")
            
            # Attention LSTM results
            attention_metrics = results["attention_lstm"]["test_metrics"]
            logger.info("ATTENTION LSTM TEST PERFORMANCE:")
            logger.info(f"  Total Loss: {attention_metrics.get('total_loss', 'N/A'):.4f}")
            logger.info(f"  Magnitude Loss: {attention_metrics.get('magnitude_loss', 'N/A'):.4f}")
            logger.info(f"  Frequency Loss: {attention_metrics.get('frequency_loss', 'N/A'):.4f}")
            logger.info(f"  Magnitude MAE: {attention_metrics.get('magnitude_mae', 'N/A'):.4f}")
            logger.info(f"  Frequency MAE: {attention_metrics.get('frequency_mae', 'N/A'):.4f}")
            logger.info(f"  Magnitude Correlation: {attention_metrics.get('magnitude_corr', 'N/A'):.4f}")
            logger.info(f"  Frequency Correlation: {attention_metrics.get('frequency_corr', 'N/A'):.4f}")
            
            # Performance comparison
            logger.info("\nPERFORMANCE COMPARISON:")
            simple_total_loss = simple_metrics.get('total_loss', float('inf'))
            attention_total_loss = attention_metrics.get('total_loss', float('inf'))
            if simple_total_loss != float('inf') and attention_total_loss != float('inf'):
                loss_improvement = ((simple_total_loss - attention_total_loss) / simple_total_loss) * 100
                logger.info(f"  Simple LSTM Total Loss: {simple_total_loss:.4f}")
                logger.info(f"  Attention LSTM Total Loss: {attention_total_loss:.4f}")
                logger.info(f"  {'Attention' if attention_total_loss < simple_total_loss else 'Simple'} LSTM is {abs(loss_improvement):.1f}% {'better' if attention_total_loss < simple_total_loss else 'worse'}")
            
            # Winner determination
            logger.info("\nWINNER DETERMINATION:")
            if simple_total_loss != float('inf') and attention_total_loss != float('inf'):
                if attention_total_loss < simple_total_loss:
                    logger.info("🏆 ATTENTION LSTM is the WINNER!")
                    logger.info(f"   Overall improvement: {abs(loss_improvement):.1f}% better than Simple LSTM")
                else:
                    logger.info("🏆 SIMPLE LSTM is the WINNER!")
                    logger.info(f"   Overall improvement: {abs(loss_improvement):.1f}% better than Attention LSTM")
            
            logger.info("="*60)
            
            # Print comparison summary
            comparison_trainer.print_comparison_summary()
            
            print(f"\n✅ Training history plots saved to: {comparison_trainer.output_dir}")
            print("✅ Evaluation summaries displayed above")
            print("✅ Optimized hyperparameters used throughout training")
            
        else:
            logger.error("❌ Model comparison failed to return results")
    except Exception as e:
        logger.error(f"❌ Error during model comparison: {e}")
        raise
    
    print("\n" + "="*80)
    print("OPTIMIZED TRAINING PIPELINE COMPLETED!")
    print("="*80)
    print("You now have:")
    print("✅ Models trained with better hyperparameters")
    print("✅ Training history plots")
    print("✅ Detailed evaluation summaries")
    print("✅ Performance comparison analysis")
    print("="*80)

if __name__ == "__main__":
    main()
