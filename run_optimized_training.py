#!/usr/bin/env python3
"""
Run Optimized Training Examples

This script demonstrates how to use the optimized hyperparameter configurations
with your main training script.
"""

import subprocess
import sys
from pathlib import Path

def run_training_with_config(model_type: str, config_name: str, mode: str = "train"):
    """
    Run training with a specific optimized configuration.
    
    Args:
        model_type: 'simple', 'attention', or 'compare'
        config_name: 'best_frequency', 'best_magnitude', or 'best_balanced'
        mode: 'train', 'full_pipeline', or 'compare_models'
    """
    
    print(f"🚀 Running {model_type} model training with {config_name} configuration...")
    print(f"Mode: {mode}")
    print("=" * 60)
    
    # Build command
    cmd = [
        "python", "main.py",
        "--mode", mode,
        "--model", model_type,
        "--optimized_config", config_name,
        "--num_epochs", "100",  # Reduced for faster testing
        "--output_dir", f"results_optimized_{config_name}_{model_type}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Training completed successfully!")
        print("\n📊 Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  Warnings/Info:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\n📊 Output:")
        print(e.stdout)
        print("\n❌ Errors:")
        print(e.stderr)
        
    except FileNotFoundError:
        print("❌ Error: main.py not found. Make sure you're in the project root directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function to demonstrate different training configurations."""
    
    print("🎯 Optimized Training Examples")
    print("=" * 60)
    print("This script demonstrates how to use your optimized hyperparameter configurations.")
    print()
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ Error: main.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check if optimized configs exist
    config_files = [
        "best_frequency_config.json",
        "best_magnitude_config.json", 
        "best_balanced_config.json"
    ]
    
    missing_configs = [f for f in config_files if not Path(f).exists()]
    if missing_configs:
        print(f"⚠️  Missing configuration files: {missing_configs}")
        print("Make sure to run the hyperparameter tuning notebook first.")
        print()
    
    print("📋 Available Training Options:")
    print("1. Simple LSTM with Best Frequency Configuration")
    print("2. Simple LSTM with Best Magnitude Configuration") 
    print("3. Simple LSTM with Best Balanced Configuration")
    print("4. Attention LSTM with Best Frequency Configuration")
    print("5. Attention LSTM with Best Magnitude Configuration")
    print("6. Attention LSTM with Best Balanced Configuration")
    print("7. Compare both models with Best Balanced Configuration")
    print()
    
    # Example 1: Simple LSTM with Best Frequency (highest frequency range)
    print("🎯 Example 1: Simple LSTM with Best Frequency Configuration")
    print("   Expected: Frequency range ~49.39, Magnitude range ~0.35")
    print("   Best for: When frequency prediction accuracy is priority")
    print()
    
    # Example 2: Simple LSTM with Best Magnitude (highest magnitude range)  
    print("🎯 Example 2: Simple LSTM with Best Magnitude Configuration")
    print("   Expected: Frequency range ~1.02, Magnitude range ~1.52")
    print("   Best for: When magnitude prediction accuracy is priority")
    print()
    
    # Example 3: Simple LSTM with Best Balanced (good both)
    print("🎯 Example 3: Simple LSTM with Best Balanced Configuration")
    print("   Expected: Frequency range ~25.0, Magnitude range ~0.9")
    print("   Best for: Production deployment with balanced performance")
    print()
    
    print("🚀 To run training with optimized configurations:")
    print()
    print("   # Best frequency prediction (49.39 range)")
    print("   python main.py --mode train --model simple --optimized_config best_frequency")
    print()
    print("   # Best magnitude prediction (1.52 range)")
    print("   python main.py --mode train --model simple --optimized_config best_magnitude")
    print()
    print("   # Best balanced performance")
    print("   python main.py --mode train --model simple --optimized_config best_balanced")
    print()
    print("   # Attention model with best frequency config")
    print("   python main.py --mode train --model attention --optimized_config best_frequency")
    print()
    print("   # Compare both models with balanced config")
    print("   python main.py --mode compare_models --model compare --optimized_config best_balanced")
    print()
    
    # Ask user if they want to run an example
    try:
        choice = input("Would you like to run an example training? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("\n🎯 Choose a configuration:")
            print("1. best_frequency (49.39 frequency range)")
            print("2. best_magnitude (1.52 magnitude range)")
            print("3. best_balanced (balanced performance)")
            
            config_choice = input("Enter config number (1-3): ").strip()
            config_map = {"1": "best_frequency", "2": "best_magnitude", "3": "best_balanced"}
            
            if config_choice in config_map:
                config_name = config_map[config_choice]
                print(f"\n🎯 Running training with {config_name} configuration...")
                run_training_with_config("simple", config_name, "train")
            else:
                print("❌ Invalid choice. Exiting.")
        else:
            print("✅ No training run. You can use the commands above when ready!")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user. Exiting.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

