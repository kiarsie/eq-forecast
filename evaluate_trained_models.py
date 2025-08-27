#!/usr/bin/env python3
"""
Evaluate Trained Models and Show Final Results

This script loads the already trained models and evaluates them to show the final results.
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.shared_lstm_model import SharedLSTMModel
from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
from src.models.enhanced_shared_processor import EnhancedSharedDataset
from src.models.shared_lstm_trainer import SharedLSTMTrainer
from src.models.attention_shared_lstm_trainer import AttentionSharedLSTMTrainer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_trained_models():
    """Load the trained models from the results directory."""
    logger = logging.getLogger(__name__)
    
    results_dir = Path("data/results")
    
    # Check if models exist
    simple_model_path = results_dir / "shared_best_model.pth"
    attention_model_path = results_dir / "attention_best_model.pth"
    
    if not simple_model_path.exists():
        logger.error(f"Simple LSTM model not found at {simple_model_path}")
        return None, None
    
    if not attention_model_path.exists():
        logger.error(f"Attention LSTM model not found at {attention_model_path}")
        return None, None
    
    logger.info("Loading trained models...")
    
    try:
        # Load Simple LSTM model
        simple_model = SharedLSTMModel(
            input_seq_features=12,  # actual input features from saved model
            metadata_features=4,    # bin_id, center_lat, center_lon, bin_area
            lookback_years=10,
            lstm_hidden_1=48,      # actual LSTM1 hidden size from saved model
            lstm_hidden_2=24,      # actual LSTM2 hidden size from saved model
            dense_hidden=24,       # actual dense hidden size from saved model
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        # Load the saved model and extract state dict
        simple_checkpoint = torch.load(simple_model_path, map_location='cpu')
        if 'model_state_dict' in simple_checkpoint:
            simple_model.load_state_dict(simple_checkpoint['model_state_dict'])
        else:
            simple_model.load_state_dict(simple_checkpoint)
        simple_model.eval()
        logger.info("✅ Simple LSTM model loaded successfully")
        
        # Load Attention LSTM model
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=12,  # actual input features from saved model
            metadata_features=4,    # bin_id, center_lat, center_lon, bin_area
            lookback_years=10,
            lstm_hidden_1=48,      # actual LSTM1 hidden size from saved model
            lstm_hidden_2=24,      # actual LSTM2 hidden size from saved model
            dense_hidden=24,       # actual dense hidden size from saved model
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        # Load the saved model and extract state dict
        attention_checkpoint = torch.load(attention_model_path, map_location='cpu')
        if 'model_state_dict' in attention_checkpoint:
            attention_model.load_state_dict(attention_checkpoint['model_state_dict'])
        else:
            attention_model.load_state_dict(attention_checkpoint)
        attention_model.eval()
        logger.info("✅ Attention LSTM model loaded successfully")
        
        return simple_model, attention_model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

def create_test_datasets():
    """Create test datasets for evaluation."""
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
        # Create the enhanced shared dataset directly
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
        
        # Create test data loader
        test_sequences = dataset.get_split_sequences('test')
        logger.info(f"Test sequences: {len(test_sequences)}")
        
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            sampler=torch.utils.data.SubsetRandomSampler(
                [i for i, seq in enumerate(dataset.sequences) if seq['split'] == 'test']
            ),
            shuffle=False,
            num_workers=0,
            pin_memory=False  # Set to False for evaluation
        )
        
        logger.info("✅ Test datasets created successfully")
        return {
            'dataset': dataset,
            'test_loader': test_loader,
            'input_features': input_features,
            'target_features': target_features,
            'metadata_features': metadata_features
        }
        
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return None

def evaluate_model(model, test_loader, model_name, logger):
    """Evaluate a single model on the test set."""
    logger.info(f"Evaluating {model_name}...")
    
    # Create appropriate trainer for evaluation
    if "Attention" in model_name:
        trainer = AttentionSharedLSTMTrainer(
            model=model,
            train_loader=test_loader,  # Use test_loader for all to avoid issues
            val_loader=test_loader,
            test_loader=test_loader,
            learning_rate=4e-4,        # Updated: was 5e-4, now 4e-4 for better performance
            weight_decay=5e-5,         # Updated: was 1e-4, now 5e-5 for better performance
            magnitude_weight=1.5,      # Updated: was 2.0, now 1.5 for better balance
            frequency_weight=2.0,      # Updated: was 1.0, now 2.0 for better frequency prediction
            correlation_weight=0.0,    # gamma: weight for correlation penalty (disabled)
            device='cpu'
        )
    else:
        trainer = SharedLSTMTrainer(
            model=model,
            train_loader=test_loader,  # Use test_loader for all to avoid issues
            val_loader=test_loader,
            test_loader=test_loader,
            learning_rate=4e-4,        # Updated: was 5e-4, now 4e-4 for better performance
            weight_decay=5e-5,         # Updated: was 1e-4, now 5e-5 for better performance
            magnitude_weight=1.5,      # Updated: was 2.0, now 1.5 for better balance
            frequency_weight=2.0,      # Updated: was 1.0, now 2.0 for better frequency prediction
            correlation_weight=0.0,    # gamma: weight for correlation penalty (disabled)
            device='cpu'
        )
    
    # Evaluate on test set
    try:
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"✅ {model_name} evaluation completed")
        return test_metrics
    except Exception as e:
        logger.error(f"❌ Error evaluating {model_name}: {e}")
        return None

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def show_final_results(simple_metrics, attention_metrics):
    """Display the final comparison results."""
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    if simple_metrics and attention_metrics:
        print("\n" + "=" * 60)
        print("TEST PERFORMANCE COMPARISON")
        print("=" * 60)
        
        # Simple LSTM results
        print("\nSIMPLE LSTM TEST PERFORMANCE:")
        print(f"  Total Loss: {simple_metrics.get('total_loss', 'N/A'):.4f}")
        print(f"  Magnitude Loss: {simple_metrics.get('magnitude_loss', 'N/A'):.4f}")
        print(f"  Frequency Loss: {simple_metrics.get('frequency_loss', 'N/A'):.4f}")
        print(f"  Magnitude MAE: {simple_metrics.get('magnitude_mae', 'N/A'):.4f}")
        print(f"  Frequency MAE: {simple_metrics.get('frequency_mae', 'N/A'):.4f}")
        print(f"  Magnitude Correlation: {simple_metrics.get('magnitude_corr', 'N/A'):.4f}")
        print(f"  Frequency Correlation: {simple_metrics.get('frequency_corr', 'N/A'):.4f}")
        
        # Attention LSTM results
        print("\nATTENTION LSTM TEST PERFORMANCE:")
        print(f"  Total Loss: {attention_metrics.get('total_loss', 'N/A'):.4f}")
        print(f"  Magnitude Loss: {attention_metrics.get('magnitude_loss', 'N/A'):.4f}")
        print(f"  Frequency Loss: {attention_metrics.get('frequency_loss', 'N/A'):.4f}")
        print(f"  Magnitude MAE: {attention_metrics.get('magnitude_mae', 'N/A'):.4f}")
        print(f"  Frequency MAE: {attention_metrics.get('frequency_mae', 'N/A'):.4f}")
        print(f"  Magnitude Correlation: {attention_metrics.get('magnitude_corr', 'N/A'):.4f}")
        print(f"  Frequency Correlation: {attention_metrics.get('frequency_corr', 'N/A'):.4f}")
        
        # Performance comparison
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Magnitude comparison
        simple_mag_mae = simple_metrics.get('magnitude_mae', float('inf'))
        attention_mag_mae = attention_metrics.get('magnitude_mae', float('inf'))
        if simple_mag_mae != float('inf') and attention_mag_mae != float('inf'):
            mag_improvement = ((simple_mag_mae - attention_mag_mae) / simple_mag_mae) * 100
            print(f"  Magnitude Prediction:")
            print(f"    Simple LSTM MAE: {simple_mag_mae:.4f}")
            print(f"    Attention LSTM MAE: {attention_mag_mae:.4f}")
            print(f"    {'Attention' if attention_mag_mae < simple_mag_mae else 'Simple'} LSTM is {abs(mag_improvement):.1f}% {'better' if attention_mag_mae < simple_mag_mae else 'worse'}")
        
        # Frequency comparison
        simple_freq_mae = simple_metrics.get('frequency_mae', float('inf'))
        attention_freq_mae = attention_metrics.get('frequency_mae', float('inf'))
        if simple_freq_mae != float('inf') and attention_freq_mae != float('inf'):
            freq_improvement = ((simple_freq_mae - attention_freq_mae) / simple_freq_mae) * 100
            print(f"  Frequency Prediction:")
            print(f"    Simple LSTM MAE: {simple_freq_mae:.4f}")
            print(f"    Attention LSTM MAE: {attention_freq_mae:.4f}")
            print(f"    {'Attention' if attention_freq_mae < simple_freq_mae else 'Simple'} LSTM is {abs(freq_improvement):.1f}% {'better' if attention_freq_mae < simple_freq_mae else 'worse'}")
        
        # Overall loss comparison
        simple_total_loss = simple_metrics.get('total_loss', float('inf'))
        attention_total_loss = attention_metrics.get('total_loss', float('inf'))
        if simple_total_loss != float('inf') and attention_total_loss != float('inf'):
            loss_improvement = ((simple_total_loss - attention_total_loss) / simple_total_loss) * 100
            print(f"  Overall Performance:")
            print(f"    Simple LSTM Total Loss: {simple_total_loss:.4f}")
            print(f"    Attention LSTM Total Loss: {attention_total_loss:.4f}")
            print(f"    {'Attention' if attention_total_loss < simple_total_loss else 'Simple'} LSTM is {abs(loss_improvement):.1f}% {'better' if attention_total_loss < simple_total_loss else 'worse'}")
        
        # Winner determination
        print("\n" + "=" * 60)
        print("WINNER DETERMINATION")
        print("=" * 60)
        
        if simple_total_loss != float('inf') and attention_total_loss != float('inf'):
            if attention_total_loss < simple_total_loss:
                print("🏆 ATTENTION LSTM is the WINNER!")
                print(f"   Overall improvement: {abs(loss_improvement):.1f}% better than Simple LSTM")
            else:
                print("🏆 SIMPLE LSTM is the WINNER!")
                print(f"   Overall improvement: {abs(loss_improvement):.1f}% better than Attention LSTM")
        else:
            print("❓ Cannot determine winner - missing metrics")
        
        # Save results
        results = {
            "simple_lstm_metrics": convert_numpy_types(simple_metrics),
            "attention_lstm_metrics": convert_numpy_types(attention_metrics),
            "hyperparameters": {
                "learning_rate": 4e-4,
                "weight_decay": 5e-5,
                "magnitude_weight": 1.5,
                "frequency_weight": 2.0,
                "correlation_weight": 0.0
            }
        }
        
        results_file = Path("data/results/comparison_metrics.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
    else:
        print("❌ Could not evaluate both models. Check the error messages above.")
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETERS USED")
    print("=" * 80)
    print("  Learning Rate: 4e-4")
    print("  Weight Decay: 5e-5")
    print("  Magnitude Weight: 1.5")
    print("  Frequency Weight: 2.0")
    print("  Correlation Weight: 0.0")
    print("  Early Stopping Patience: 25")
    print("  Max Epochs: 300")
    print("=" * 80)

def main():
    """Main function to evaluate trained models."""
    logger = setup_logging()
    
    print("=" * 80)
    print("EARTHQUAKE FORECASTING - MODEL EVALUATION")
    print("=" * 80)
    
    # Load trained models
    simple_model, attention_model = load_trained_models()
    if not simple_model or not attention_model:
        print("❌ Failed to load trained models. Exiting.")
        return
    
    # Create test datasets
    datasets = create_test_datasets()
    if not datasets:
        print("❌ Failed to create test datasets. Exiting.")
        return
    
    test_loader = datasets['test_loader']
    
    # Evaluate Simple LSTM
    simple_metrics = evaluate_model(simple_model, test_loader, "Simple LSTM", logger)
    
    # Evaluate Attention LSTM
    attention_metrics = evaluate_model(attention_model, test_loader, "Attention LSTM", logger)
    
    # Show final results
    show_final_results(simple_metrics, attention_metrics)
    
    print("\n🎉 Model evaluation completed!")
    print("Your earthquake forecasting models have been successfully evaluated.")
    print("Check the results directory for the saved comparison metrics.")

if __name__ == "__main__":
    main()
