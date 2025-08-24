#!/usr/bin/env python3
"""
Test script to verify the attention shared LSTM model works with the same configurations.
"""

import torch
import logging
from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
from src.models.shared_lstm_model import SharedLSTMModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test that both models can be created with the same parameters."""
    
    # Model parameters (matching the successful training)
    input_seq_features = 12
    metadata_features = 4
    lookback_years = 10
    lstm_hidden_1 = 64
    lstm_hidden_2 = 32  # This was the key fix
    dense_hidden = 32
    dropout_rate = 0.25
    freq_head_type = "linear"
    
    logger.info("Testing model creation with identical parameters...")
    
    # Create regular shared LSTM model
    regular_model = SharedLSTMModel(
        input_seq_features=input_seq_features,
        metadata_features=metadata_features,
        lookback_years=lookback_years,
        lstm_hidden_1=lstm_hidden_1,
        lstm_hidden_2=lstm_hidden_2,
        dense_hidden=dense_hidden,
        dropout_rate=dropout_rate,
        freq_head_type=freq_head_type
    )
    
    # Create attention shared LSTM model
    attention_model = AttentionSharedLSTMModel(
        input_seq_features=input_seq_features,
        metadata_features=metadata_features,
        lookback_years=lookback_years,
        lstm_hidden_1=lstm_hidden_1,
        lstm_hidden_2=lstm_hidden_2,
        dense_hidden=dense_hidden,
        dropout_rate=dropout_rate,
        freq_head_type=freq_head_type
    )
    
    logger.info(f"Regular model parameters: {sum(p.numel() for p in regular_model.parameters()):,}")
    logger.info(f"Attention model parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    return regular_model, attention_model

def test_forward_pass(regular_model, attention_model):
    """Test that both models can perform forward pass with the same input."""
    
    # Create dummy input data
    batch_size = 4
    lookback_years = 10
    input_seq_features = 12
    metadata_features = 4
    
    input_sequence = torch.randn(batch_size, lookback_years, input_seq_features)
    metadata = torch.randn(batch_size, metadata_features)
    
    logger.info("Testing forward pass...")
    
    # Test regular model
    with torch.no_grad():
        regular_mag, regular_freq = regular_model(input_sequence, metadata)
        logger.info(f"Regular model output shapes: mag={regular_mag.shape}, freq={regular_freq.shape}")
        logger.info(f"Regular model output ranges: mag=[{regular_mag.min():.4f}, {regular_mag.max():.4f}], freq=[{regular_freq.min():.4f}, {regular_freq.max():.4f}]")
    
    # Test attention model
    with torch.no_grad():
        attention_mag, attention_freq = attention_model(input_sequence, metadata)
        logger.info(f"Attention model output shapes: mag={attention_mag.shape}, freq={attention_freq.shape}")
        logger.info(f"Attention model output ranges: mag=[{attention_mag.min():.4f}, {attention_mag.max():.4f}], freq=[{attention_freq.min():.4f}, {attention_freq.max():.4f}]")
    
    return True

def test_frequency_conversion(regular_model, attention_model):
    """Test that both models can convert frequency predictions to counts."""
    
    # Create dummy frequency predictions
    batch_size = 4
    frequency_pred = torch.randn(batch_size, 1) * 2 + 1  # Random values around 1
    
    logger.info("Testing frequency conversion...")
    
    # Test regular model
    regular_counts = regular_model.predict_frequency_counts(frequency_pred)
    logger.info(f"Regular model count range: [{regular_counts.min():.4f}, {regular_counts.max():.4f}]")
    
    # Test attention model
    attention_counts = attention_model.predict_frequency_counts(frequency_pred)
    logger.info(f"Attention model count range: [{attention_counts.min():.4f}, {attention_counts.max():.4f}]")
    
    return True

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TESTING ATTENTION SHARED LSTM MODEL")
    logger.info("=" * 60)
    
    try:
        # Test 1: Model creation
        regular_model, attention_model = test_model_creation()
        logger.info("✅ Model creation test passed")
        
        # Test 2: Forward pass
        test_forward_pass(regular_model, attention_model)
        logger.info("✅ Forward pass test passed")
        
        # Test 3: Frequency conversion
        test_frequency_conversion(regular_model, attention_model)
        logger.info("✅ Frequency conversion test passed")
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! 🎉")
        logger.info("The attention model is ready to use with the same configurations.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()

