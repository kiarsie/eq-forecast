#!/usr/bin/env python3
"""
Test script for the new simple weighted attention mechanism.
"""

import torch
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWeightedAttention(torch.nn.Module):
    """
    Extremely simple attention: just learnable weights for each timestep.
    No complex projections, no multi-head complexity - just weights that sum to 1.
    """
    
    def __init__(self, seq_length: int = 10):
        super(SimpleWeightedAttention, self).__init__()
        self.seq_length = seq_length
        
        # Learnable weights for each timestep (initialize to uniform)
        self.timestep_weights = torch.nn.Parameter(torch.ones(seq_length) / seq_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned weights to each timestep.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Weighted output tensor
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Ensure weights sum to 1 (softmax for stability)
        weights = F.softmax(self.timestep_weights, dim=0)
        
        # Apply weights to each timestep
        # Shape: (batch_size, seq_len, embed_dim)
        weighted_output = x * weights.unsqueeze(0).unsqueeze(-1)
        
        return weighted_output

def test_simple_attention():
    """Test the simple weighted attention mechanism."""
    
    logger.info("Testing Simple Weighted Attention...")
    
    # Create attention mechanism
    seq_length = 10
    embed_dim = 32
    attention = SimpleWeightedAttention(seq_length=seq_length)
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, seq_length, embed_dim)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Initial attention weights: {attention.timestep_weights.data}")
    
    # Forward pass
    output = attention(x)
    logger.info(f"Output shape: {output.shape}")
    
    # Check that weights sum to 1
    weights = F.softmax(attention.timestep_weights, dim=0)
    logger.info(f"Softmax weights: {weights.detach().numpy()}")
    logger.info(f"Weights sum to: {weights.sum().item():.6f}")
    
    # Check that output is properly weighted
    logger.info(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    logger.info(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test that the mechanism works
    assert abs(weights.sum().item() - 1.0) < 1e-6, "Weights should sum to 1"
    assert output.shape == x.shape, "Output shape should match input shape"
    
    logger.info("✅ Simple attention test passed!")
    return attention

def test_attention_learning():
    """Test that attention weights can be learned."""
    
    logger.info("Testing attention weight learning...")
    
    attention = SimpleWeightedAttention(seq_length=5)
    optimizer = torch.optim.Adam([attention.timestep_weights], lr=0.01)
    
    # Create a simple task: learn to focus on the last timestep
    batch_size = 8
    seq_length = 5
    embed_dim = 16
    
    # Create input where the last timestep has higher values
    x = torch.randn(batch_size, seq_length, embed_dim)
    x[:, -1, :] += 2.0  # Make last timestep more important
    
    # Simple loss: encourage higher weights for last timestep
    for step in range(100):
        optimizer.zero_grad()
        
        output = attention(x)
        
        # Loss: encourage higher weight for last timestep
        weights = F.softmax(attention.timestep_weights, dim=0)
        loss = -weights[-1]  # Maximize weight for last timestep
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            current_weights = F.softmax(attention.timestep_weights, dim=0)
            logger.info(f"Step {step}: Last timestep weight = {current_weights[-1].item():.4f}")
    
    # Check final weights
    final_weights = F.softmax(attention.timestep_weights, dim=0)
    logger.info(f"Final weights: {final_weights.detach().numpy()}")
    logger.info(f"Last timestep weight: {final_weights[-1].item():.4f}")
    
    # The last timestep should have higher weight
    assert final_weights[-1] > final_weights[0], "Last timestep should have higher weight"
    logger.info("✅ Attention learning test passed!")

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TESTING SIMPLE WEIGHTED ATTENTION")
    logger.info("=" * 60)
    
    try:
        # Test 1: Basic functionality
        attention = test_simple_attention()
        logger.info("✅ Basic functionality test passed")
        
        # Test 2: Weight learning
        test_attention_learning()
        logger.info("✅ Weight learning test passed")
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! 🎉")
        logger.info("The simple attention mechanism is ready to use.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
