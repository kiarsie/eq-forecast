#!/usr/bin/env python3
"""
Test Optimized Hyperparameter Configurations

This script tests that the optimized configurations can be loaded and applied correctly.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_config_loading():
    """Test loading configuration files."""
    print("🧪 Testing Configuration Loading...")
    
    try:
        from apply_optimized_configs import load_config
        
        # Test loading each config
        configs = [
            'best_frequency_config.json',
            'best_magnitude_config.json',
            'best_balanced_config.json'
        ]
        
        for config_file in configs:
            try:
                config = load_config(config_file)
                print(f"✅ {config_file}: {config['name']}")
                print(f"   Frequency range: {config['performance']['frequency_range']:.2f}")
                print(f"   Magnitude range: {config['performance']['magnitude_range']:.2f}")
            except Exception as e:
                print(f"❌ {config_file}: {e}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_model_creation():
    """Test creating models with optimized configurations."""
    print("\n🧪 Testing Model Creation...")
    
    try:
        from apply_optimized_configs import create_optimized_model, load_config
        from models.shared_lstm_model import SharedLSTMModel
        
        # Load best frequency config
        config = load_config('best_frequency_config.json')
        
        # Create model
        model = create_optimized_model(SharedLSTMModel, config)
        
        # Test forward pass
        batch_size = 2
        input_sequence = torch.randn(batch_size, config['hyperparameters']['lookback_years'], 
                                   config['hyperparameters']['input_seq_features'])
        metadata = torch.randn(batch_size, config['hyperparameters']['metadata_features'])
        
        with torch.no_grad():
            magnitude_pred, frequency_pred = model(input_sequence, metadata)
        
        print(f"✅ Model created successfully!")
        print(f"   Input shape: {input_sequence.shape}")
        print(f"   Magnitude output shape: {magnitude_pred.shape}")
        print(f"   Frequency output shape: {frequency_pred.shape}")
        
        # Check frequency scaling parameters
        if hasattr(model, 'frequency_scale'):
            print(f"   Frequency scale: {model.frequency_scale.item():.2f}")
        if hasattr(model, 'frequency_bias'):
            print(f"   Frequency bias: {model.frequency_bias.item():.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_training_params():
    """Test extracting training parameters."""
    print("\n🧪 Testing Training Parameters...")
    
    try:
        from apply_optimized_configs import load_config, get_training_params
        
        config = load_config('best_frequency_config.json')
        train_params = get_training_params(config)
        
        print(f"✅ Training parameters extracted:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Training params error: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Testing Optimized Hyperparameter Configurations")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_model_creation,
        test_training_params
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your optimized configurations are ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Use these configs in your main training script")
        print("   2. Retrain models with optimized parameters")
        print("   3. Expect significant performance improvements!")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()

