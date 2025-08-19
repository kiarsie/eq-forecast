#!/usr/bin/env python3
"""
Test script to check if all required modules can be imported.
"""

import sys
import os
from pathlib import Path

print("Testing imports...")

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
print(f"Added {Path(__file__).parent / 'src'} to Python path")

try:
    print("Testing torch import...")
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    print("Testing pandas import...")
    import pandas as pd
    print("✓ Pandas imported successfully")
    
    print("Testing numpy import...")
    import numpy as np
    print("✓ NumPy imported successfully")
    
    print("Testing matplotlib import...")
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
    
    print("Testing tqdm import...")
    from tqdm import tqdm
    print("✓ TQDM imported successfully")
    
    print("Testing sklearn import...")
    from sklearn.preprocessing import RobustScaler
    print("✓ Scikit-learn imported successfully")
    
    print("\nTesting custom module imports...")
    
    print("Testing enhanced_shared_processor import...")
    from models.enhanced_shared_processor import EnhancedSharedDataset
    print("✓ EnhancedSharedDataset imported successfully")
    
    print("Testing shared_lstm_model import...")
    from models.shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
    print("✓ SharedLSTMModel and WeightedEarthquakeLoss imported successfully")
    
    print("Testing shared_lstm_trainer import...")
    from models.shared_lstm_trainer import SharedLSTMTrainer
    print("✓ SharedLSTMTrainer imported successfully")
    
    print("\n✅ All imports successful! The training script should work now.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
