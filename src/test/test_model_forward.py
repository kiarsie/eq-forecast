import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.attention_lstm import AttentionLSTM, SimpleLSTM, AttentionLayer


class TestAttentionLayer(unittest.TestCase):
    """Test cases for AttentionLayer class."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_size = 64
        
        # Create dummy LSTM output
        self.lstm_output = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_attention_layer_initialization(self):
        """Test attention layer initialization."""
        attention_layer = AttentionLayer(hidden_size=self.hidden_size)
        
        self.assertIsInstance(attention_layer, AttentionLayer)
        self.assertEqual(attention_layer.hidden_size, self.hidden_size)
        self.assertIsInstance(attention_layer.attention, nn.Linear)
    
    def test_attention_forward_pass(self):
        """Test attention layer forward pass."""
        attention_layer = AttentionLayer(hidden_size=self.hidden_size)
        
        context_vector, attention_weights = attention_layer(self.lstm_output)
        
        # Check output shapes
        self.assertEqual(context_vector.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len, 1))
        
        # Check that attention weights sum to 1 for each batch
        attention_sums = torch.sum(attention_weights, dim=1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones(self.batch_size, 1), atol=1e-6))
        
        # Check that attention weights are non-negative
        self.assertTrue(torch.all(attention_weights >= 0))
    
    def test_attention_weights_normalization(self):
        """Test that attention weights are properly normalized."""
        attention_layer = AttentionLayer(hidden_size=self.hidden_size)
        
        context_vector, attention_weights = attention_layer(self.lstm_output)
        
        # Check that attention weights are probabilities (sum to 1)
        attention_sums = torch.sum(attention_weights, dim=1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones(self.batch_size, 1), atol=1e-6))
        
        # Check that all weights are between 0 and 1
        self.assertTrue(torch.all(attention_weights >= 0))
        self.assertTrue(torch.all(attention_weights <= 1))


class TestAttentionLSTM(unittest.TestCase):
    """Test cases for AttentionLSTM class."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 10
        self.input_size = 7  # magnitude, depth, lat, lon, month, year, day_of_year
        
        # Create dummy input
        self.input_features = torch.randn(self.batch_size, self.seq_len, self.input_size)
    
    def test_attention_lstm_initialization(self):
        """Test AttentionLSTM initialization."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
        
        self.assertIsInstance(model, AttentionLSTM)
        self.assertEqual(model.hidden_size, 128)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.dropout, 0.2)
        self.assertTrue(model.bidirectional)
        self.assertEqual(model.num_directions, 2)
    
    def test_attention_lstm_forward_pass(self):
        """Test AttentionLSTM forward pass."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
        
        # Check output ranges
        # Magnitude should be between 0 and 1 (sigmoid)
        self.assertTrue(torch.all(magnitude_pred >= 0))
        self.assertTrue(torch.all(magnitude_pred <= 1))
        
        # Events should be non-negative (ReLU)
        self.assertTrue(torch.all(events_pred >= 0))
    
    def test_attention_lstm_with_attention_weights(self):
        """Test AttentionLSTM forward pass with attention weights."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Forward pass with attention weights
        magnitude_pred, events_pred, attention_weights = model(
            self.input_features, return_attention=True
        )
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len, 1))
        
        # Check attention weights normalization
        attention_sums = torch.sum(attention_weights, dim=1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones(self.batch_size, 1), atol=1e-6))
    
    def test_attention_lstm_bidirectional(self):
        """Test bidirectional AttentionLSTM."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=True
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
    
    def test_attention_lstm_multiple_layers(self):
        """Test multi-layer AttentionLSTM."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=3,
            dropout=0.1,
            bidirectional=False
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
    
    def test_attention_lstm_get_attention_weights(self):
        """Test getting attention weights separately."""
        model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Get attention weights
        attention_weights = model.get_attention_weights(self.input_features)
        
        # Check shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len, 1))
        
        # Check normalization
        attention_sums = torch.sum(attention_weights, dim=1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones(self.batch_size, 1), atol=1e-6))


class TestSimpleLSTM(unittest.TestCase):
    """Test cases for SimpleLSTM class."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 10
        self.input_size = 7
        
        # Create dummy input
        self.input_features = torch.randn(self.batch_size, self.seq_len, self.input_size)
    
    def test_simple_lstm_initialization(self):
        """Test SimpleLSTM initialization."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
        
        self.assertIsInstance(model, SimpleLSTM)
        self.assertEqual(model.hidden_size, 128)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.dropout, 0.2)
        self.assertTrue(model.bidirectional)
        self.assertEqual(model.num_directions, 2)
    
    def test_simple_lstm_forward_pass(self):
        """Test SimpleLSTM forward pass."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
        
        # Check output ranges
        # Magnitude should be between 0 and 1 (sigmoid)
        self.assertTrue(torch.all(magnitude_pred >= 0))
        self.assertTrue(torch.all(magnitude_pred <= 1))
        
        # Events should be non-negative (ReLU)
        self.assertTrue(torch.all(events_pred >= 0))
    
    def test_simple_lstm_bidirectional(self):
        """Test bidirectional SimpleLSTM."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=True
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))
    
    def test_simple_lstm_multiple_layers(self):
        """Test multi-layer SimpleLSTM."""
        model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=3,
            dropout=0.1,
            bidirectional=False
        )
        
        # Forward pass
        magnitude_pred, events_pred = model(self.input_features)
        
        # Check output shapes
        self.assertEqual(magnitude_pred.shape, (self.batch_size, 1))
        self.assertEqual(events_pred.shape, (self.batch_size, 1))


class TestModelComparison(unittest.TestCase):
    """Test cases for comparing AttentionLSTM vs SimpleLSTM."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.seq_len = 10
        self.input_size = 7
        
        # Create dummy input
        self.input_features = torch.randn(self.batch_size, self.seq_len, self.input_size)
    
    def test_model_output_consistency(self):
        """Test that both models produce consistent output shapes."""
        attention_model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        simple_model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Forward pass
        att_magnitude, att_events = attention_model(self.input_features)
        sim_magnitude, sim_events = simple_model(self.input_features)
        
        # Check that outputs have the same shapes
        self.assertEqual(att_magnitude.shape, sim_magnitude.shape)
        self.assertEqual(att_events.shape, sim_events.shape)
        
        # Check that outputs have the same batch size
        self.assertEqual(att_magnitude.shape[0], self.batch_size)
        self.assertEqual(att_events.shape[0], self.batch_size)
    
    def test_model_parameter_count(self):
        """Test that both models have similar parameter counts."""
        attention_model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        simple_model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Count parameters
        att_params = sum(p.numel() for p in attention_model.parameters())
        sim_params = sum(p.numel() for p in simple_model.parameters())
        
        # Attention model should have more parameters due to attention layer
        self.assertGreater(att_params, sim_params)
        
        # But not too many more (just the attention mechanism)
        param_ratio = att_params / sim_params
        self.assertLess(param_ratio, 1.5)  # Should be reasonable
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through both models."""
        attention_model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        simple_model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Test gradient flow for AttentionLSTM
        attention_model.zero_grad()
        att_magnitude, att_events = attention_model(self.input_features)
        att_loss = att_magnitude.mean() + att_events.mean()
        att_loss.backward()
        
        # Check that gradients exist
        att_has_gradients = any(p.grad is not None for p in attention_model.parameters())
        self.assertTrue(att_has_gradients)
        
        # Test gradient flow for SimpleLSTM
        simple_model.zero_grad()
        sim_magnitude, sim_events = simple_model(self.input_features)
        sim_loss = sim_magnitude.mean() + sim_events.mean()
        sim_loss.backward()
        
        # Check that gradients exist
        sim_has_gradients = any(p.grad is not None for p in simple_model.parameters())
        self.assertTrue(sim_has_gradients)


if __name__ == '__main__':
    unittest.main()
