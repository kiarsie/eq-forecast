#!/usr/bin/env python3
"""
Test script to verify model creation and identify parameter mismatches.
"""

import json
import sys
import os

# Add src to path
sys.path.append('src')

def test_model_creation():
    """Test creating models with different parameter combinations."""
    print("🧪 Testing Model Creation")
    print("=" * 50)
    
    try:
        # Test 1: Import models
        print("1. Testing imports...")
        from src.models.shared_lstm_model import SharedLSTMModel
        from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
        print("   ✅ Models imported successfully")
        
        # Test 2: Check SharedLSTMModel constructor
        print("\n2. Testing SharedLSTMModel constructor...")
        print("   Expected parameters:")
        import inspect
        sig = inspect.signature(SharedLSTMModel.__init__)
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                print(f"     {param_name}: {param.default if param.default != param.empty else 'required'}")
        
        # Test 3: Check AttentionSharedLSTMModel constructor
        print("\n3. Testing AttentionSharedLSTMModel constructor...")
        sig = inspect.signature(AttentionSharedLSTMModel.__init__)
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                print(f"     {param_name}: {param.default if param.default != param.empty else 'required'}")
        
        # Test 4: Load config
        print("\n4. Testing config loading...")
        if os.path.exists('hybrid_balanced_config.json'):
            with open('hybrid_balanced_config.json', 'r') as f:
                config = json.load(f)
            print(f"   ✅ Config loaded: {config['name']}")
            print(f"   Model architecture keys: {list(config['model_architecture'].keys())}")
        else:
            print("   ❌ Config file not found")
            return
        
        # Test 5: Test model creation with config parameters only
        print("\n5. Testing model creation with config parameters only...")
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
        
        print(f"   Model params: {model_params}")
        
        # Test SharedLSTMModel
        try:
            simple_model = SharedLSTMModel(**model_params)
            print("   ✅ SharedLSTMModel created successfully")
        except Exception as e:
            print(f"   ❌ SharedLSTMModel creation failed: {e}")
        
        # Test AttentionSharedLSTMModel
        try:
            attention_model = AttentionSharedLSTMModel(**model_params)
            print("   ✅ AttentionSharedLSTMModel created successfully")
        except Exception as e:
            print(f"   ❌ AttentionSharedLSTMModel creation failed: {e}")
        
        # Test 6: Test with additional parameters that might be passed
        print("\n6. Testing with additional parameters...")
        extra_params = {
            'input_seq_features': 12,
            'metadata_features': 4,
            'lookback_years': 10,
            'lstm_hidden_1': 48,
            'lstm_hidden_2': 24,
            'dense_hidden': 24,
            'dropout_rate': 0.4,
            'freq_head_type': 'linear',
            'logger': 'test_logger',  # This should cause an error
            'lookback_years': 10      # This should be fine
        }
        
        print(f"   Extra params: {extra_params}")
        
        try:
            simple_model = SharedLSTMModel(**extra_params)
            print("   ✅ SharedLSTMModel with extra params created successfully")
        except Exception as e:
            print(f"   ❌ SharedLSTMModel with extra params failed: {e}")
        
        # Test 7: Test the create_optimized_model function
        print("\n7. Testing create_optimized_model function...")
        try:
            from apply_optimized_configs import create_optimized_model
            
            # Test without extra kwargs
            simple_model = create_optimized_model(SharedLSTMModel, config)
            print("   ✅ create_optimized_model without extra params works")
            
            # Test with extra kwargs (this should fail)
            try:
                simple_model = create_optimized_model(
                    SharedLSTMModel, 
                    config,
                    logger='test_logger'  # This should cause an error
                )
                print("   ✅ create_optimized_model with logger param works (unexpected)")
            except Exception as e:
                print(f"   ❌ create_optimized_model with logger param failed as expected: {e}")
                
        except Exception as e:
            print(f"   ❌ create_optimized_model import/usage failed: {e}")
        
        print("\n" + "=" * 50)
        print("🧪 Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_creation()
