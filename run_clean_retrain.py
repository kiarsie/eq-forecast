#!/usr/bin/env python3
"""
Clean Retrain Script with Anti-Overfitting

This script runs a complete clean retrain with anti-overfitting measures.
"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_clean_retrain():
    """Run clean retrain with anti-overfitting measures."""
    logger.info("🚀 Starting clean retrain with anti-overfitting measures...")
    
    # Step 1: Clean existing models
    logger.info("Step 1: Cleaning existing models...")
    subprocess.run([sys.executable, "clean_retrain_anti_overfitting.py"], check=True)
    
    # Step 2: Train Simple LSTM with anti-overfitting
    logger.info("Step 2: Training Simple LSTM with anti-overfitting...")
    subprocess.run([
        sys.executable, "train_anti_overfitting.py", 
        "--model_type", "simple"
    ], check=True)
    
    # Step 3: Train Attention LSTM with anti-overfitting
    logger.info("Step 3: Training Attention LSTM with anti-overfitting...")
    subprocess.run([
        sys.executable, "train_anti_overfitting.py", 
        "--model_type", "attention"
    ], check=True)
    
    # Step 4: Run model comparison
    logger.info("Step 4: Running model comparison...")
    subprocess.run([
        sys.executable, "main.py", 
        "--mode", "compare_models",
        "--model", "compare",
        "--anti_overfitting"
    ], check=True)
    
    logger.info("✅ Clean retrain completed successfully!")


if __name__ == "__main__":
    run_clean_retrain()
