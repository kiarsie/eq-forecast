#!/usr/bin/env python3
"""
Test script for the anti-overfitting configuration.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_anti_overfitting_config():
    """Test loading and applying the anti-overfitting configuration."""
    print("🧪 Testing Anti-Overfitting Configuration")
    print("=" * 50)
    
    # Test 1: Load configuration file
    print("\n1. Testing configuration file loading...")
    try:
        with open('anti_overfitting_config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration file loaded successfully")
        print(f"   Name: {config['name']}")
        print(f"   Version: {config['version']}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test 2: Validate structure
    print("\n2. Testing configuration structure...")
    required_sections = [
        'model_architecture', 'training_parameters', 'loss_weights', 
        'frequency_scaling', 'anti_overfitting_features'
    ]
    
    for section in required_sections:
        if section in config:
            print(f"✅ {section} section found")
        else:
            print(f"❌ Missing {section} section")
            return False
    
    # Test 3: Validate anti-overfitting features
    print("\n3. Testing anti-overfitting features...")
    anti_overfitting = config['anti_overfitting_features']
    
    expected_features = [
        'early_stopping', 'learning_rate_scheduling', 'gradient_clipping',
        'reduced_model_complexity', 'increased_dropout', 'aggressive_weight_decay',
        'smaller_batch_size', 'reduced_training_epochs'
    ]
    
    for feature in expected_features:
        if feature in anti_overfitting and anti_overfitting[feature]:
            print(f"✅ {feature}: Enabled")
        else:
            print(f"❌ {feature}: Missing or disabled")
            return False
    
    # Test 4: Validate model architecture
    print("\n4. Testing model architecture...")
    arch = config['model_architecture']
    
    # Check for reduced complexity
    if arch['lstm_hidden_1'] <= 32:
        print(f"✅ LSTM hidden 1: {arch['lstm_hidden_1']} (reduced)")
    else:
        print(f"❌ LSTM hidden 1: {arch['lstm_hidden_1']} (should be <= 32)")
        return False
    
    if arch['lstm_hidden_2'] <= 16:
        print(f"✅ LSTM hidden 2: {arch['lstm_hidden_2']} (reduced)")
    else:
        print(f"❌ LSTM hidden 2: {arch['lstm_hidden_2']} (should be <= 16)")
        return False
    
    if arch['dropout_rate'] >= 0.5:
        print(f"✅ Dropout rate: {arch['dropout_rate']} (increased)")
    else:
        print(f"❌ Dropout rate: {arch['dropout_rate']} (should be >= 0.5)")
        return False
    
    # Test 5: Validate training parameters
    print("\n5. Testing training parameters...")
    train = config['training_parameters']
    
    if train['weight_decay'] >= 0.001:
        print(f"✅ Weight decay: {train['weight_decay']} (aggressive)")
    else:
        print(f"❌ Weight decay: {train['weight_decay']} (should be >= 0.001)")
        return False
    
    if train['batch_size'] <= 16:
        print(f"✅ Batch size: {train['batch_size']} (smaller)")
    else:
        print(f"❌ Batch size: {train['batch_size']} (should be <= 16)")
        return False
    
    if train['num_epochs'] <= 80:
        print(f"✅ Max epochs: {train['num_epochs']} (reduced)")
    else:
        print(f"❌ Max epochs: {train['num_epochs']} (should be <= 80)")
        return False
    
    # Test 6: Validate loss weights
    print("\n6. Testing loss weights...")
    weights = config['loss_weights']
    
    if weights['magnitude_weight'] == 1.0 and weights['frequency_weight'] == 1.0:
        print("✅ Loss weights: Balanced (1.0, 1.0)")
    else:
        print(f"❌ Loss weights: {weights['magnitude_weight']}, {weights['frequency_weight']} (should be 1.0, 1.0)")
        return False
    
    # Test 7: Test apply_optimized_configs integration
    print("\n7. Testing apply_optimized_configs integration...")
    try:
        from apply_optimized_configs import load_config, create_optimized_model, get_training_params
        
        # Test loading
        loaded_config = load_config('anti_overfitting_config.json')
        print("✅ Configuration loaded via apply_optimized_configs")
        
        # Test training params extraction
        train_params = get_training_params(loaded_config)
        expected_params = ['learning_rate', 'weight_decay', 'magnitude_weight', 'frequency_weight', 'correlation_weight']
        for param in expected_params:
            if param in train_params:
                print(f"✅ Training param '{param}' extracted")
            else:
                print(f"❌ Training param '{param}' missing")
                return False
        
        print("✅ All training parameters extracted successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import apply_optimized_configs: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test integration: {e}")
        return False
    
    print("\n🎉 All tests passed! Anti-overfitting configuration is ready to use.")
    print("\nUsage:")
    print("   python main.py --mode train --model simple --optimized_config anti_overfitting")
    print("   python main.py --mode train --model attention --optimized_config anti_overfitting")
    print("   python main.py --mode compare_models --model compare --optimized_config anti_overfitting")
    
    return True

if __name__ == "__main__":
    success = test_anti_overfitting_config()
    sys.exit(0 if success else 1)

