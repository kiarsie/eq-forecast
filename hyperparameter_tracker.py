#!/usr/bin/env python3
"""
Hyperparameter Tracker for Earthquake Forecasting Models

This script helps track different hyperparameter configurations and their results
for systematic comparison of model performance.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class HyperparameterTracker:
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.history_file = self.results_dir / "hyperparameter_history.json"
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        """Load existing hyperparameter history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return {"runs": []}
        return {"runs": []}
    
    def add_run(self, 
                model_name: str,
                hyperparameters: Dict,
                results: Dict,
                training_time: float,
                notes: str = "") -> None:
        """Add a new training run to the history."""
        run_id = f"run_{len(self.history['runs']) + 1:03d}"
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "results": results,
            "training_time": training_time,
            "notes": notes
        }
        
        self.history["runs"].append(run_data)
        self._save_history()
        
        print(f"Added run {run_id} to hyperparameter history")
    
    def _save_history(self) -> None:
        """Save hyperparameter history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all runs."""
        if not self.history["runs"]:
            return pd.DataFrame()
        
        comparison_data = []
        
        for run in self.history["runs"]:
            row = {
                "Run ID": run["run_id"],
                "Model": run["model_name"],
                "Timestamp": run["timestamp"][:19],  # Truncate to seconds
                "Learning Rate": run["hyperparameters"].get("learning_rate", "N/A"),
                "Weight Decay": run["hyperparameters"].get("weight_decay", "N/A"),
                "Alpha (Mag)": run["hyperparameters"].get("magnitude_weight", "N/A"),
                "Beta (Freq)": run["hyperparameters"].get("frequency_weight", "N/A"),
                "Gamma (Corr)": run["hyperparameters"].get("correlation_weight", "N/A"),
                "Total Loss": run["results"].get("total_loss", "N/A"),
                "Magnitude Loss": run["results"].get("magnitude_loss", "N/A"),
                "Frequency Loss": run["results"].get("frequency_loss", "N/A"),
                "Magnitude MAE": run["results"].get("magnitude_mae", "N/A"),
                "Frequency MAE": run["results"].get("frequency_mae", "N/A"),
                "Magnitude Corr": run["results"].get("magnitude_corr", "N/A"),
                "Frequency Corr": run["results"].get("frequency_corr", "N/A"),
                "Training Time (s)": run["training_time"],
                "Notes": run["notes"]
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_hyperparameter_comparison(self, save_path: Optional[str] = None) -> None:
        """Create visualization comparing different hyperparameter configurations."""
        if not self.history["runs"]:
            print("No runs to compare!")
            return
        
        df = self.get_comparison_table()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Comparison Across Training Runs', fontsize=16)
        
        # Learning Rate vs Total Loss
        axes[0, 0].scatter(df['Learning Rate'], df['Total Loss'], alpha=0.7)
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Learning Rate vs Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Alpha vs Magnitude Loss
        axes[0, 1].scatter(df['Alpha (Mag)'], df['Magnitude Loss'], alpha=0.7)
        axes[0, 1].set_xlabel('Alpha (Magnitude Weight)')
        axes[0, 1].set_ylabel('Magnitude Loss')
        axes[0, 1].set_title('Alpha vs Magnitude Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Beta vs Frequency Loss
        axes[0, 2].scatter(df['Beta (Freq)'], df['Frequency Loss'], alpha=0.7)
        axes[0, 2].set_xlabel('Beta (Frequency Weight)')
        axes[0, 2].set_ylabel('Frequency Loss')
        axes[0, 2].set_title('Beta vs Frequency Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training Time vs Total Loss
        axes[1, 0].scatter(df['Training Time (s)'], df['Total Loss'], alpha=0.7)
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].set_title('Training Time vs Total Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Magnitude Correlation vs Frequency Correlation
        axes[1, 1].scatter(df['Magnitude Corr'], df['Frequency Corr'], alpha=0.7)
        axes[1, 1].set_xlabel('Magnitude Correlation')
        axes[1, 1].set_ylabel('Frequency Correlation')
        axes[1, 1].set_title('Magnitude vs Frequency Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Model comparison
        model_counts = df['Model'].value_counts()
        axes[1, 2].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Distribution of Model Types')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter comparison plot saved to: {save_path}")
        
        plt.show()
    
    def export_to_csv(self, output_path: Optional[str] = None) -> None:
        """Export comparison table to CSV."""
        df = self.get_comparison_table()
        
        if output_path is None:
            output_path = self.results_dir / "hyperparameter_comparison.csv"
        
        df.to_csv(output_path, index=False)
        print(f"Hyperparameter comparison exported to: {output_path}")
    
    def get_best_run(self, metric: str = "total_loss") -> Optional[Dict]:
        """Get the best run based on a specific metric."""
        if not self.history["runs"]:
            return None
        
        best_run = None
        best_value = float('inf')
        
        for run in self.history["runs"]:
            value = run["results"].get(metric, float('inf'))
            if value != "N/A" and value < best_value:
                best_value = value
                best_run = run
        
        return best_run

def main():
    """Example usage of the hyperparameter tracker."""
    tracker = HyperparameterTracker()
    
    print("Hyperparameter Tracker for Earthquake Forecasting Models")
    print("=" * 60)
    
    # Check if there are any existing runs
    if tracker.history["runs"]:
        print(f"Found {len(tracker.history['runs'])} existing runs")
        
        # Show comparison table
        df = tracker.get_comparison_table()
        print("\nHyperparameter Comparison Table:")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Export to CSV
        tracker.export_to_csv()
        
        # Show best run
        best_run = tracker.get_best_run("total_loss")
        if best_run:
            print(f"\nBest run (lowest total loss): {best_run['run_id']}")
            print(f"Model: {best_run['model_name']}")
            print(f"Total Loss: {best_run['results']['total_loss']}")
            print(f"Hyperparameters: {best_run['hyperparameters']}")
        
        # Create visualization
        tracker.plot_hyperparameter_comparison()
        
    else:
        print("No existing runs found. Here's how to use the tracker:")
        print("\n1. After training, add a run:")
        print("   tracker.add_run(")
        print("       model_name='Shared LSTM',")
        print("       hyperparameters={")
        print("           'learning_rate': 5e-4,")
        print("           'weight_decay': 1e-4,")
        print("           'magnitude_weight': 2.0,")
        print("           'frequency_weight': 0.5,")
        print("           'correlation_weight': 0.0")
        print("       },")
        print("       results=test_metrics,")
        print("       training_time=training_duration,")
        print("       notes='Baseline configuration'")
        print("   )")
        
        print("\n2. Generate comparison table:")
        print("   df = tracker.get_comparison_table()")
        
        print("\n3. Create visualizations:")
        print("   tracker.plot_hyperparameter_comparison()")
        
        print("\n4. Export to CSV:")
        print("   tracker.export_to_csv()")

if __name__ == "__main__":
    main()
