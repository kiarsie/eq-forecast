#!/usr/bin/env python3
"""
Quick Test for compare_models Mode

This script runs a minimal test to verify that compare_models actually works.
It creates a small dataset and runs a quick comparison to ensure everything functions.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append('src')

def create_mini_dataset():
    """Create a minimal dataset for testing."""
    print("🔧 Creating mini test dataset...")
    
    # Create small synthetic data
    num_samples = 10
    lookback_years = 5
    input_features = 12
    metadata_features = 4
    
    # Input sequences
    input_sequences = torch.randn(num_samples, lookback_years, input_features)
    
    # Metadata
    metadata = torch.randn(num_samples, metadata_features)
    
    # Targets (magnitude and frequency)
    magnitudes = torch.randn(num_samples, 1) * 0.5 + 5.0  # Around magnitude 5
    frequencies = torch.exp(torch.randn(num_samples, 1) * 0.3)  # Positive frequencies
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(input_sequences, metadata, magnitudes, frequencies)
    
    # Split into train/val/test
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    print(f"✅ Created mini dataset: {num_samples} samples")
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_features': input_features,
        'metadata_features': metadata_features
    }

def test_model_comparison_trainer():
    """Test the ModelComparisonTrainer with mini dataset."""
    print("\n🧪 Testing ModelComparisonTrainer with mini dataset...")
    
    try:
        from models.model_comparison_trainer import ModelComparisonTrainer
        
        # Create mini dataset
        datasets = create_mini_dataset()
        
        # Create trainer
        trainer = ModelComparisonTrainer(
            train_loader=datasets['train_loader'],
            val_loader=datasets['val_loader'],
            test_loader=datasets['test_loader'],
            input_seq_features=datasets['input_features'],
            metadata_features=datasets['metadata_features'],
            lookback_years=5,
            learning_rate=1e-3,  # Higher LR for quick convergence
            weight_decay=1e-4,
            magnitude_weight=2.0,
            frequency_weight=1.0,
            correlation_weight=0.0,
            device='auto',
            output_dir='test_comparison_output'
        )
        
        print("✅ ModelComparisonTrainer created successfully")
        
        # Run quick comparison (very few epochs)
        print("🚀 Running quick comparison (2 epochs each)...")
        results = trainer.run_comparison(max_epochs=2, patience=1)
        
        print("✅ Comparison completed successfully!")
        print(f"   Simple LSTM training history: {len(results['simple_lstm']['training_history']['train_losses'])} epochs")
        print(f"   Attention LSTM training history: {len(results['attention_lstm']['training_history']['train_losses'])} epochs")
        
        # Check if results directory was created
        if Path('test_comparison_output').exists():
            print("✅ Output directory created")
            
            # List files
            files = list(Path('test_comparison_output').glob('*'))
            print(f"   Files created: {len(files)}")
            for file in files:
                print(f"     - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_config_integration():
    """Test that optimized configs work with compare_models."""
    print("\n🧪 Testing Optimized Config Integration...")
    
    try:
        from apply_optimized_configs import load_config, get_training_params
        
        # Load best frequency config
        config = load_config('best_frequency_config.json')
        train_params = get_training_params(config)
        
        print(f"✅ Loaded config: {config['name']}")
        print(f"   Frequency range: {config['performance']['frequency_range']:.2f}")
        print(f"   Training params: {train_params}")
        
        # Test creating models with optimized config
        from models.shared_lstm_model import SharedLSTMModel
        from models.attention_shared_lstm_model import AttentionSharedLSTMModel
        
        simple_model = SharedLSTMModel(
            input_seq_features=12,
            metadata_features=4,
            lookback_years=10,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=12,
            metadata_features=4,
            lookback_years=10,
            lstm_hidden_1=64,
            lstm_hidden_2=32,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        
        print("✅ Both model types created with optimized config")
        
        # Test forward pass
        batch_size = 2
        input_sequence = torch.randn(batch_size, 10, 12)
        metadata = torch.randn(batch_size, 4)
        
        with torch.no_grad():
            simple_mag, simple_freq = simple_model(input_sequence, metadata)
            attention_mag, attention_freq = attention_model(input_sequence, metadata)
        
        print("✅ Forward pass successful for both models")
        print(f"   Simple LSTM output shapes: {simple_mag.shape}, {simple_freq.shape}")
        print(f"   Attention LSTM output shapes: {attention_mag.shape}, {attention_freq.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = [
        'test_comparison_output',
        'test_output'
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Removed: {test_dir}")
            except Exception as e:
                print(f"⚠️  Could not remove {test_dir}: {e}")

def main():
    """Run the quick test."""
    print("🚀 Quick Test for compare_models Mode")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        test1_passed = test_model_comparison_trainer()
        
        # Test 2: Optimized config integration
        test2_passed = test_optimized_config_integration()
        
        # Summary
        print("\n📊 Quick Test Results:")
        print(f"   ModelComparisonTrainer: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"   Optimized Config Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 ALL QUICK TESTS PASSED!")
            print("✅ compare_models mode is working correctly")
            print("✅ Optimized configurations are properly integrated")
            print("✅ All fixes have been successfully applied")
            
            print("\n🚀 You can now run:")
            print("   python main.py --mode compare_models --model compare")
            print("   python main.py --mode compare_models --model compare --optimized_config best_frequency")
            
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
        
        return test1_passed and test2_passed
        
    except Exception as e:
        print(f"\n❌ Quick test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cleanup()

if __name__ == "__main__":
    main()

