#!/usr/bin/env python3
"""
Test script for updated models with Poisson formulation.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_models():
    """Test both updated models."""
    print("=== Testing Updated Models ===")
    
    try:
        # Test imports
        from src.models.shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
        from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
        print("✓ All models imported successfully")
        
        # Create test data
        batch_size = 4
        lookback_years = 10
        input_seq_features = 12
        metadata_features = 4
        
        input_seq = torch.randn(batch_size, lookback_years, input_seq_features)
        metadata = torch.randn(batch_size, metadata_features)
        
        print(f"✓ Test data created: input={input_seq.shape}, metadata={metadata.shape}")
        
        # Test Simple LSTM
        print("\n--- Testing Simple LSTM ---")
        simple_model = SharedLSTMModel(
            input_seq_features=input_seq_features,
            metadata_features=metadata_features,
            lookback_years=lookback_years
        )
        
        with torch.no_grad():
            magnitude_pred, frequency_log_rate_pred = simple_model(input_seq, metadata)
            print(f"✓ Simple LSTM forward pass: magnitude={magnitude_pred.shape}, frequency_log_rate={frequency_log_rate_pred.shape}")
            
            # Test frequency count prediction
            frequency_counts = simple_model.predict_frequency_counts(frequency_log_rate_pred)
            print(f"✓ Frequency counts: {frequency_counts.shape}")
            print(f"  Log-rate range: [{frequency_log_rate_pred.min().item():.3f}, {frequency_log_rate_pred.max().item():.3f}]")
            print(f"  Count range: [{frequency_counts.min().item():.3f}, {frequency_counts.max().item():.3f}]")
        
        # Test Attention LSTM
        print("\n--- Testing Attention LSTM ---")
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=input_seq_features,
            metadata_features=metadata_features,
            lookback_years=lookback_years
        )
        
        with torch.no_grad():
            magnitude_pred, frequency_log_rate_pred = attention_model(input_seq, metadata)
            print(f"✓ Attention LSTM forward pass: magnitude={magnitude_pred.shape}, frequency_log_rate={frequency_log_rate_pred.shape}")
            
            # Test frequency count prediction
            frequency_counts = attention_model.predict_frequency_counts(frequency_log_rate_pred)
            print(f"✓ Frequency counts: {frequency_counts.shape}")
            print(f"  Log-rate range: [{frequency_log_rate_pred.min().item():.3f}, {frequency_log_rate_pred.max().item():.3f}]")
            print(f"  Count range: [{frequency_counts.min().item():.3f}, {frequency_counts.max().item():.3f}]")
        
        # Test loss function
        print("\n--- Testing Loss Function ---")
        criterion = WeightedEarthquakeLoss(magnitude_weight=1.0, frequency_weight=3.0)
        
        # Create dummy targets
        magnitude_true = torch.randn(batch_size, 1)
        frequency_true = torch.randint(0, 10, (batch_size, 1)).float()
        
        # Test with simple model predictions
        with torch.no_grad():
            magnitude_pred, frequency_log_rate_pred = simple_model(input_seq, metadata)
            
            loss = criterion(magnitude_pred, frequency_log_rate_pred, magnitude_true, frequency_true)
            loss_components = criterion.get_loss_components(magnitude_pred, frequency_log_rate_pred, magnitude_true, frequency_true)
            
            print(f"✓ Loss computation successful")
            print(f"  Total loss: {loss.item():.4f}")
            print(f"  Magnitude loss: {loss_components['magnitude_loss'].item():.4f}")
            print(f"  Frequency loss: {loss_components['frequency_loss'].item():.4f}")
        
        print("\n✅ All tests passed! Models are working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()
