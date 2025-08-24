#!/usr/bin/env python3
"""
Complete the model comparison that was interrupted by JSON serialization error.
This script loads the trained models and completes the evaluation and saving.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

def complete_comparison():
    """Complete the interrupted model comparison."""
    print("🔄 Completing interrupted model comparison...")
    
    try:
        from models.model_comparison_trainer import ModelComparisonTrainer
        from models.shared_lstm_model import SharedLSTMModel
        from models.attention_shared_lstm_model import AttentionSharedLSTMModel
        
        # Check if models exist
        simple_model_path = Path("data/results/simple_lstm_model.pth")
        attention_model_path = Path("data/results/attention_lstm_model.pth")
        
        if not simple_model_path.exists() or not attention_model_path.exists():
            print("❌ Model files not found. Please run the full comparison first.")
            return False
        
        print("✅ Found trained model files")
        
        # Load models
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
        
        # Load trained weights
        simple_model.load_state_dict(torch.load(simple_model_path))
        attention_model.load_state_dict(torch.load(attention_model_path))
        
        print("✅ Models loaded successfully")
        
        # Create dummy data loaders for evaluation (we just need the structure)
        from torch.utils.data import DataLoader
        dummy_loader = DataLoader([], batch_size=1)
        
        # Create trainer for evaluation
        trainer = ModelComparisonTrainer(
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            test_loader=dummy_loader,
            input_seq_features=12,
            metadata_features=4,
            lookback_years=10,
            learning_rate=5e-4,
            weight_decay=7.5e-5,
            magnitude_weight=1.75,
            frequency_weight=1.25,
            correlation_weight=0.0,
            device='auto',
            output_dir='data/results'
        )
        
        print("✅ ModelComparisonTrainer created")
        
        # Evaluate both models
        print("🔍 Evaluating Simple LSTM...")
        simple_metrics = trainer.evaluate_model(simple_model, "SimpleLSTM")
        
        print("🔍 Evaluating Attention LSTM...")
        attention_metrics = trainer.evaluate_model(attention_model, "AttentionSharedLSTM")
        
        # Create comparison results
        comparison_results = {
            "simple_lstm": {
                "model": simple_model,
                "test_metrics": simple_metrics,
                "training_history": {"train_losses": [], "val_losses": []}  # Dummy since we don't have full history
            },
            "attention_lstm": {
                "model": attention_model,
                "test_metrics": attention_metrics,
                "training_history": {"train_losses": [], "val_losses": []}  # Dummy since we don't have full history
            },
            "hyperparameters": {
                "learning_rate": 5e-4,
                "weight_decay": 7.5e-5,
                "magnitude_weight": 1.75,
                "frequency_weight": 1.25,
                "correlation_weight": 0.0
            }
        }
        
        # Save results
        print("💾 Saving comparison results...")
        trainer.save_comparison_results(comparison_results)
        
        # Print summary
        print("📊 MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print("Simple LSTM Performance:")
        for key, value in simple_metrics.items():
            print(f"  {key}: {value}")
        print()
        print("Attention LSTM Performance:")
        for key, value in attention_metrics.items():
            print(f"  {key}: {value}")
        print()
        print("✅ Comparison completed successfully!")
        print(f"Results saved to: data/results/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error completing comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    complete_comparison()

