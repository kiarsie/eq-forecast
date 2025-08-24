#!/usr/bin/env python3
"""
Generate a summary table of past training runs from available results.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_training_stats(training_history):
    """Calculate key training statistics from training history."""
    if not training_history:
        return {}
    
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    
    if not train_losses or not val_losses:
        return {}
    
    # Find best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    
    # Calculate convergence metrics
    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None
    
    # Calculate training duration
    training_time = training_history.get('training_time', None)
    
    # Calculate epochs to convergence (if val loss < 0.3)
    convergence_threshold = 0.3
    convergence_epoch = None
    for i, val_loss in enumerate(val_losses):
        if val_loss < convergence_threshold:
            convergence_epoch = i + 1
            break
    
    return {
        'total_epochs': len(train_losses),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'training_time_seconds': training_time,
        'convergence_epoch': convergence_epoch,
        'convergence_threshold': convergence_threshold
    }

def main():
    """Generate the summary table."""
    results_dir = Path("data/results")
    
    # Load all available data
    print("Loading training results...")
    
    # Shared LSTM results
    shared_lstm_metrics = load_json_file(results_dir / "test_metrics.json")
    shared_lstm_history = load_json_file(results_dir / "training_history.json")
    
    # Attention LSTM results
    attention_metrics = load_json_file(results_dir / "attention_test_metrics.json")
    attention_history = load_json_file(results_dir / "attention_training_history.json")
    
    # Calculate training statistics
    shared_lstm_stats = calculate_training_stats(shared_lstm_history)
    attention_stats = calculate_training_stats(attention_history)
    
    # Create summary table
    summary_data = []
    
    # Shared LSTM Model
    if shared_lstm_metrics and shared_lstm_stats:
        summary_data.append({
            'Model': 'Shared LSTM',
            'Total Loss': f"{shared_lstm_metrics.get('total_loss', 'N/A'):.4f}",
            'Magnitude Loss': f"{shared_lstm_metrics.get('magnitude_loss', 'N/A'):.4f}",
            'Frequency Loss': f"{shared_lstm_metrics.get('frequency_loss', 'N/A'):.4f}",
            'Magnitude MAE': f"{shared_lstm_metrics.get('magnitude_mae', 'N/A'):.4f}",
            'Frequency MAE': f"{shared_lstm_metrics.get('frequency_mae', 'N/A'):.4f}",
            'Magnitude Correlation': f"{shared_lstm_metrics.get('magnitude_corr', 'N/A'):.4f}",
            'Frequency Correlation': f"{shared_lstm_metrics.get('frequency_corr', 'N/A'):.4f}",
            'Magnitude Accuracy (±0.3)': f"{shared_lstm_metrics.get('magnitude_accuracy', 'N/A'):.3f}",
            'Frequency Accuracy (±1)': f"{shared_lstm_metrics.get('frequency_accuracy', 'N/A'):.3f}",
            'Total Epochs': shared_lstm_stats.get('total_epochs', 'N/A'),
            'Best Epoch': shared_lstm_stats.get('best_epoch', 'N/A'),
            'Best Val Loss': f"{shared_lstm_stats.get('best_val_loss', 'N/A'):.4f}",
            'Training Time (s)': f"{shared_lstm_stats.get('training_time_seconds', 'N/A'):.1f}" if shared_lstm_stats.get('training_time_seconds') else 'N/A',
            'Convergence Epoch': shared_lstm_stats.get('convergence_epoch', 'N/A')
        })
    
    # Attention LSTM Model
    if attention_metrics and attention_stats:
        summary_data.append({
            'Model': 'Attention Shared LSTM',
            'Total Loss': f"{attention_metrics.get('total_loss', 'N/A'):.4f}",
            'Magnitude Loss': f"{attention_metrics.get('magnitude_loss', 'N/A'):.4f}",
            'Frequency Loss': f"{attention_metrics.get('frequency_loss', 'N/A'):.4f}",
            'Magnitude MAE': f"{attention_metrics.get('magnitude_mae', 'N/A'):.4f}",
            'Frequency MAE': f"{attention_metrics.get('frequency_mae', 'N/A'):.4f}",
            'Magnitude Correlation': f"{attention_metrics.get('magnitude_corr', 'N/A'):.4f}",
            'Frequency Correlation': f"{attention_metrics.get('frequency_corr', 'N/A'):.4f}",
            'Magnitude Accuracy (±0.3)': f"{attention_metrics.get('magnitude_accuracy', 'N/A'):.3f}",
            'Frequency Accuracy (±1)': f"{attention_metrics.get('frequency_accuracy', 'N/A'):.3f}",
            'Total Epochs': attention_stats.get('total_epochs', 'N/A'),
            'Best Epoch': attention_stats.get('best_epoch', 'N/A'),
            'Best Val Loss': f"{attention_stats.get('best_val_loss', 'N/A'):.4f}",
            'Training Time (s)': f"{attention_stats.get('training_time_seconds', 'N/A'):.1f}" if attention_stats.get('training_time_seconds') else 'N/A',
            'Convergence Epoch': attention_stats.get('convergence_epoch', 'N/A')
        })
    
    if not summary_data:
        print("No training results found!")
        return
    
    # Create DataFrame and display
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*120)
    print("EARTHQUAKE FORECASTING MODEL TRAINING SUMMARY")
    print("="*120)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = results_dir / "training_runs_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    if len(summary_data) == 2:
        shared = summary_data[0]
        attention = summary_data[1]
        
        print(f"Best Total Loss: {'Shared LSTM' if float(shared['Total Loss']) < float(attention['Total Loss']) else 'Attention LSTM'}")
        print(f"Best Magnitude Performance: {'Shared LSTM' if float(shared['Magnitude Correlation']) > float(attention['Magnitude Correlation']) else 'Attention LSTM'}")
        print(f"Best Frequency Performance: {'Shared LSTM' if float(shared['Frequency Correlation']) > float(attention['Frequency Correlation']) else 'Attention LSTM'}")
        print(f"Fastest Training: {'Shared LSTM' if float(shared['Training Time (s)']) < float(attention['Training Time (s)']) else 'Attention LSTM'}")
        
        # Check for issues
        print(f"\nPOTENTIAL ISSUES:")
        if float(attention['Frequency Correlation']) < 0:
            print("- Attention model shows negative frequency correlation (worse than random)")
        if float(attention['Total Loss']) > float(shared['Total Loss']):
            print("- Attention model has higher total loss than baseline")
        if float(attention['Training Time (s)']) > float(shared['Training Time (s)']) * 1.5:
            print("- Attention model training time significantly longer than baseline")

if __name__ == "__main__":
    main()
