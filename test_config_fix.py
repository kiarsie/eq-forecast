#!/usr/bin/env python3
"""
Test script to verify that the anti-overfitting configuration fix is working.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_config_fix():
    """Test that the anti-overfitting configuration fix is working."""
    print("🧪 Testing Anti-Overfitting Configuration Fix")
    print("=" * 60)
    
    # Test 1: Load configuration
    print("\n1. Loading anti-overfitting configuration...")
    try:
        with open('anti_overfitting_config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test 2: Test get_training_params function (for run_model_comparison)
    print("\n2. Testing get_training_params function (for run_model_comparison)...")
    try:
        from apply_optimized_configs import get_training_params
        
        train_params = get_training_params(config)
        print("✅ Training parameters extracted successfully")
        
        # Check that num_epochs is included
        if 'num_epochs' in train_params:
            print(f"✅ num_epochs: {train_params['num_epochs']} (should be 80)")
            if train_params['num_epochs'] == 80:
                print("✅ Correct value: 80")
            else:
                print(f"❌ Wrong value: {train_params['num_epochs']} (should be 80)")
                return False
        else:
            print("❌ num_epochs missing from training parameters")
            return False
        
        # Check that patience is included
        if 'patience' in train_params:
            print(f"✅ patience: {train_params['patience']} (should be 8)")
            if train_params['patience'] == 8:
                print("✅ Correct value: 8")
            else:
                print(f"❌ Wrong value: {train_params['patience']} (should be 8)")
                return False
        else:
            print("❌ patience missing from training parameters")
            return False
        
        # Check that batch_size is NOT included (for run_model_comparison compatibility)
        if 'batch_size' not in train_params:
            print("✅ batch_size NOT included (correct for run_model_comparison)")
        else:
            print("❌ batch_size should NOT be included for run_model_comparison")
            return False
        
        # Check all required parameters for run_model_comparison
        required_params = [
            'learning_rate', 'weight_decay', 'num_epochs', 'patience',
            'magnitude_weight', 'frequency_weight', 'correlation_weight'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in train_params:
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ Missing parameters: {missing_params}")
            return False
        else:
            print("✅ All required parameters present for run_model_comparison")
        
        # Display all parameters
        print("\n📊 Training Parameters for run_model_comparison:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
    except ImportError as e:
        print(f"❌ Failed to import apply_optimized_configs: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test training parameters: {e}")
        return False
    
    # Test 3: Test get_all_training_params function (for individual model training)
    print("\n3. Testing get_all_training_params function (for individual model training)...")
    try:
        from apply_optimized_configs import get_all_training_params
        
        all_train_params = get_all_training_params(config)
        print("✅ All training parameters extracted successfully")
        
        # Check that batch_size IS included
        if 'batch_size' in all_train_params:
            print(f"✅ batch_size: {all_train_params['batch_size']} (should be 16)")
            if all_train_params['batch_size'] == 16:
                print("✅ Correct value: 16")
            else:
                print(f"❌ Wrong value: {all_train_params['batch_size']} (should be 16)")
                return False
        else:
            print("❌ batch_size missing from all training parameters")
            return False
        
        # Check all required parameters for individual model training
        required_all_params = [
            'learning_rate', 'weight_decay', 'num_epochs', 'patience', 'batch_size',
            'gradient_clip', 'scheduler_T0', 'scheduler_T_mult',
            'magnitude_weight', 'frequency_weight', 'correlation_weight'
        ]
        
        missing_all_params = []
        for param in required_all_params:
            if param not in all_train_params:
                missing_all_params.append(param)
        
        if missing_all_params:
            print(f"❌ Missing parameters: {missing_all_params}")
            return False
        else:
            print("✅ All required parameters present for individual model training")
        
        # Display all parameters
        print("\n📊 All Training Parameters for individual model training:")
        for key, value in all_train_params.items():
            print(f"   {key}: {value}")
        
    except ImportError as e:
        print(f"❌ Failed to import get_all_training_params: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to test all training parameters: {e}")
        return False
    
    # Test 4: Test main.py integration
    print("\n4. Testing main.py integration...")
    try:
        # Check if main.py has the fix
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # Check that the fix is applied
        if 'num_epochs=args.num_epochs' in main_content:
            print("❌ main.py still has the old num_epochs=args.num_epochs")
            print("   This means the fix wasn't applied correctly")
            return False
        else:
            print("✅ main.py fix applied - num_epochs=args.num_epochs removed")
        
        # Check that optimized configs are supported
        if 'anti_overfitting' in main_content:
            print("✅ anti_overfitting configuration supported in main.py")
        else:
            print("❌ anti_overfitting configuration not found in main.py")
            return False
        
        # Check that get_all_training_params is imported
        if 'get_all_training_params' in main_content:
            print("✅ get_all_training_params imported in main.py")
        else:
            print("❌ get_all_training_params not imported in main.py")
            return False
        
    except Exception as e:
        print(f"❌ Failed to test main.py integration: {e}")
        return False
    
    print("\n🎉 All tests passed! The anti-overfitting configuration fix is working correctly.")
    print("\n✅ What was fixed:")
    print("   1. get_training_params() now includes only basic parameters (for run_model_comparison)")
    print("   2. get_all_training_params() includes all parameters (for individual model training)")
    print("   3. main.py no longer overrides config num_epochs with args.num_epochs")
    print("   4. Anti-overfitting config will now respect the 80 epoch limit")
    print("   5. batch_size error in run_model_comparison is fixed")
    
    print("\n🚀 Ready to use:")
    print("   python main.py --mode train --model simple --optimized_config anti_overfitting")
    print("   python main.py --mode compare_models --model compare --optimized_config anti_overfitting")
    
    return True

if __name__ == "__main__":
    success = test_config_fix()
    sys.exit(0 if success else 1)
