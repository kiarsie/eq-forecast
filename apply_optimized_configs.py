#!/usr/bin/env python3
"""
Apply Optimized Hyperparameter Configurations

This script applies the best hyperparameter configurations from your tuning results
to your main training script and models.
"""

import json
import os
from pathlib import Path
import torch

def load_config(config_file: str) -> dict:
    """Load a configuration file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def apply_config_to_model(model, config: dict):
    """Apply hyperparameters to a model instance."""
    print(f"Applying {config['name']} configuration to model...")
    
    # Apply frequency scaling parameters if they exist
    if hasattr(model, 'frequency_scale'):
        model.frequency_scale.data.fill_(config['frequency_scaling']['frequency_scale_init'])
        print(f"  Set frequency_scale to {config['frequency_scaling']['frequency_scale_init']}")
    
    if hasattr(model, 'frequency_bias'):
        model.frequency_bias.data.fill_(config['frequency_scaling']['frequency_bias_init'])
        print(f"  Set frequency_bias to {config['frequency_scaling']['frequency_bias_init']}")
    
    print(f"  Model configured for {config['name']}")

def create_optimized_model(model_class, config: dict, **kwargs):
    """Create a model with optimized hyperparameters."""
    print(f"Creating {config['name']} model...")
    
    # Extract model architecture parameters
    model_params = {
        'input_seq_features': config['model_architecture']['input_seq_features'],
        'metadata_features': config['model_architecture']['metadata_features'],
        'lookback_years': config['model_architecture']['lookback_years'],
        'lstm_hidden_1': config['model_architecture']['lstm_hidden_1'],
        'lstm_hidden_2': config['model_architecture']['lstm_hidden_2'],
        'dense_hidden': config['model_architecture']['dense_hidden'],
        'dropout_rate': config['model_architecture']['dropout_rate'],
        'freq_head_type': config['model_architecture']['freq_head_type']
    }
    
    # Add any additional kwargs
    model_params.update(kwargs)
    
    # Create model
    model = model_class(**model_params)
    
    # Apply frequency scaling
    apply_config_to_model(model, config)
    
    return model

def get_training_params(config: dict) -> dict:
    """Extract basic training parameters from config (for run_model_comparison)."""
    return {
        'learning_rate': config['training_parameters']['learning_rate'],
        'weight_decay': config['training_parameters']['weight_decay'],
        'num_epochs': config['training_parameters']['num_epochs'],
        'patience': config['training_parameters']['patience'],
        'magnitude_weight': config['loss_weights']['magnitude_weight'],
        'frequency_weight': config['loss_weights']['frequency_weight'],
        'correlation_weight': config['loss_weights']['correlation_weight']
    }

def get_all_training_params(config: dict) -> dict:
    """Extract all training parameters from config (for individual model training)."""
    return {
        'learning_rate': config['training_parameters']['learning_rate'],
        'weight_decay': config['training_parameters']['weight_decay'],
        'num_epochs': config['training_parameters']['num_epochs'],
        'patience': config['training_parameters']['patience'],
        'batch_size': config['training_parameters']['batch_size'],
        'gradient_clip': config['training_parameters']['gradient_clip'],
        'scheduler_T0': config['training_parameters']['scheduler_T0'],
        'scheduler_T_mult': config['training_parameters']['scheduler_T_mult'],
        'magnitude_weight': config['loss_weights']['magnitude_weight'],
        'frequency_weight': config['loss_weights']['frequency_weight'],
        'correlation_weight': config['loss_weights']['correlation_weight']
    }

def main():
    """Main function to demonstrate usage."""
    print("🎯 Optimized Hyperparameter Configuration Loader")
    print("=" * 60)
    
    # Check if config files exist
    config_files = [
        'best_frequency_config.json',
        'best_magnitude_config.json', 
        'best_balanced_config.json',
        'anti_overfitting_config.json',
        'balanced_anti_overfitting_config.json',
        'enhanced_frequency_scaling_config.json',
        'high_performance_balanced_config.json'
    ]
    
    available_configs = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            config = load_config(config_file)
            available_configs[config['name']] = config
            print(f"Loaded: {config['name']}")
            
            # Handle different config formats
            if 'performance' in config:
                print(f"   Frequency range: {config['performance']['frequency_range']:.2f}")
                print(f"   Magnitude range: {config['performance']['magnitude_range']:.2f}")
            elif 'anti_overfitting_features' in config:
                print(f"   Type: Anti-overfitting configuration")
                print(f"   Dropout: {config['model_architecture']['dropout_rate']}")
                print(f"   Weight decay: {config['training_parameters']['weight_decay']}")
        else:
            print(f"Missing: {config_file}")
    
    if not available_configs:
        print("\n❌ No configuration files found!")
        print("Make sure to run the hyperparameter tuning notebook first.")
        return
    
    print(f"\n📊 Available Configurations: {len(available_configs)}")
    
    # Example usage
    print("\n🚀 Example Usage:")
    print("1. Load a configuration:")
    print("   config = load_config('best_frequency_config.json')")
    
    print("\n2. Create an optimized model:")
    print("   from src.models.shared_lstm_model import SharedLSTMModel")
    print("   model = create_optimized_model(SharedLSTMModel, config)")
    
    print("\n3. Get training parameters:")
    print("   train_params = get_training_params(config)")
    print("   # Use in your training loop")
    
    print("\n4. Apply to existing model:")
    print("   apply_config_to_model(existing_model, config)")
    
    # Show best configurations
    print("\n🏆 Performance Summary:")
    for name, config in available_configs.items():
        perf = config['performance']
        print(f"\n{name}:")
        print(f"  Frequency range: {perf['frequency_range']:.2f}")
        print(f"  Magnitude range: {perf['magnitude_range']:.2f}")
        if 'combined_score' in perf:
            print(f"  Combined score: {perf['combined_score']:.2f}")
    
    print("\nRecommendations:")
    print("  • Use 'Best Frequency Prediction' if frequency accuracy is priority")
    print("  • Use 'Best Magnitude Prediction' if magnitude accuracy is priority") 
    print("  • Use 'Best Balanced Performance' for production deployment")
    print("  • Use 'Anti-Overfitting Configuration' to prevent overfitting and ensure generalization")
    print("  • Use 'Balanced Anti-Overfitting Configuration' for balanced performance and capacity")
    print("  • Use 'Enhanced Frequency Scaling Configuration' for maximum range coverage")
    print("  • Use 'High Performance Balanced Configuration' for maximum overall performance")
    
    print("\n🎯 Next Steps:")
    print("  1. Import this script in your main training code")
    print("  2. Use create_optimized_model() to create models")
    print("  3. Use get_training_params() for training parameters")
    print("  4. Retrain with optimized configurations")
    print("  5. Compare performance improvements!")

if __name__ == "__main__":
    main()
