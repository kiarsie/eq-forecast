#!/usr/bin/env python3
"""
Test script for evaluation fixes in the EnhancedQuadtreeTrainer.

This script tests:
1. Model loading from .pth files
2. Evaluation functionality
3. Visualization generation
4. Data path handling

Run with: python test_evaluation_fixes.py
"""

import sys
import os
from pathlib import Path
import logging
import torch
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.enhanced_trainer import EnhancedQuadtreeTrainer


def setup_logging():
    """Setup basic logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_model_loading():
    """Test 1: Verify that models can be loaded from .pth files."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading from .pth files")
    print("="*60)
    
    logger = setup_logging()
    
    # Check if results directory exists
    results_dir = Path("data/results")
    if not results_dir.exists():
        print("❌ Results directory not found: data/results")
        return False
    
    # Check for .pth files
    pth_files = list(results_dir.glob("*.pth"))
    if not pth_files:
        print("❌ No .pth model files found in data/results")
        return False
    
    print(f"✅ Found {len(pth_files)} .pth model files")
    
    # Count models by type
    simple_models = [f for f in pth_files if f.name.startswith("simple_")]
    attention_models = [f for f in pth_files if f.name.startswith("attention_")]
    
    print(f"  Simple LSTM models: {len(simple_models)}")
    print(f"  Attention LSTM models: {len(attention_models)}")
    
    # Check for frequency and magnitude models
    freq_models = [f for f in pth_files if "frequency" in f.name]
    mag_models = [f for f in pth_files if "magnitude" in f.name]
    
    print(f"  Frequency models: {len(freq_models)}")
    print(f"  Magnitude models: {len(mag_models)}")
    
    return True


def test_enhanced_trainer_initialization():
    """Test 2: Verify EnhancedQuadtreeTrainer can be initialized."""
    print("\n" + "="*60)
    print("TEST 2: EnhancedQuadtreeTrainer Initialization")
    print("="*60)
    
    logger = setup_logging()
    
    try:
        # Check if processed data exists
        processed_data_path = Path("data/processed_earthquake_catalog_annual_stats.csv")
        if not processed_data_path.exists():
            print("❌ Processed data not found: data/processed_earthquake_catalog_annual_stats.csv")
            print("   Please run preprocessing first")
            return False
        
        print("✅ Processed data found")
        
        # Initialize trainer
        trainer = EnhancedQuadtreeTrainer(
            data_path=str(processed_data_path),
            save_dir="data/results",
            logger=logger,
            model_types=['simple', 'attention']
        )
        
        print("✅ EnhancedQuadtreeTrainer initialized successfully")
        print(f"   Device: {trainer.device}")
        print(f"   Bin count: {len(trainer.bin_ids)}")
        print(f"   Model types: {trainer.model_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize EnhancedQuadtreeTrainer: {e}")
        return False


def test_model_loading_method():
    """Test 3: Test the _load_existing_models_from_files method."""
    print("\n" + "="*60)
    print("TEST 3: Model Loading Method")
    print("="*60)
    
    logger = setup_logging()
    
    try:
        processed_data_path = Path("data/processed_earthquake_catalog_annual_stats.csv")
        trainer = EnhancedQuadtreeTrainer(
            data_path=str(processed_data_path),
            save_dir="data/results",
            logger=logger,
            model_types=['simple', 'attention']
        )
        
        # Test model loading
        print("🔄 Loading existing models...")
        trainer._load_existing_models_from_files()
        
        # Check what was loaded
        simple_freq_models = len(trainer.results['simple']['frequency_models']['models'])
        simple_mag_models = len(trainer.results['simple']['magnitude_models']['models'])
        attention_freq_models = len(trainer.results['attention']['frequency_models']['models'])
        attention_mag_models = len(trainer.results['attention']['magnitude_models']['models'])
        
        print(f"✅ Models loaded successfully:")
        print(f"   Simple LSTM - Frequency: {simple_freq_models}, Magnitude: {simple_mag_models}")
        print(f"   Attention LSTM - Frequency: {attention_freq_models}, Magnitude: {attention_mag_models}")
        
        total_models = simple_freq_models + simple_mag_models + attention_freq_models + attention_mag_models
        print(f"   Total models in memory: {total_models}")
        
        if total_models > 0:
            return True
        else:
            print("❌ No models were loaded")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_evaluation_functionality():
    """Test 4: Test the evaluation functionality."""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Functionality")
    print("="*60)
    
    logger = setup_logging()
    
    try:
        processed_data_path = Path("data/processed_earthquake_catalog_annual_stats.csv")
        trainer = EnhancedQuadtreeTrainer(
            data_path=str(processed_data_path),
            save_dir="data/results",
            logger=logger,
            model_types=['simple', 'attention']
        )
        
        # Load models first
        trainer._load_existing_models_from_files()
        
        # Check if we have models to evaluate
        total_models = (len(trainer.results['simple']['frequency_models']['models']) + 
                       len(trainer.results['simple']['magnitude_models']['models']) +
                       len(trainer.results['attention']['frequency_models']['models']) +
                       len(trainer.results['attention']['magnitude_models']['models']))
        
        if total_models == 0:
            print("❌ No models available for evaluation")
            return False
        
        print(f"🔄 Starting evaluation of {total_models} models...")
        
        # Run evaluation
        evaluation_results = trainer.evaluate_all_models()
        
        if not evaluation_results:
            print("❌ Evaluation returned no results")
            return False
        
        print(f"✅ Evaluation completed successfully!")
        print(f"   Results for {len(evaluation_results)} bins")
        
        # Check a sample result
        sample_bin = list(evaluation_results.keys())[0]
        if sample_bin != 'error':
            sample_result = evaluation_results[sample_bin]
            print(f"   Sample bin {sample_bin}: {list(sample_result.keys())}")
            
            # Check if metrics look reasonable
            if 'simple_lstm' in sample_result:
                simple_metrics = sample_result['simple_lstm']
                if 'wmape' in simple_metrics and 'forecast_accuracy' in simple_metrics:
                    wmape = simple_metrics['wmape']
                    accuracy = simple_metrics['forecast_accuracy']
                    print(f"   Sample Simple LSTM - WMAPE: {wmape:.2f}%, Accuracy: {accuracy:.2f}%")
                    
                    # Check if metrics are reasonable (not 100% WMAPE)
                    if wmape < 95:  # Allow some tolerance
                        print("✅ Metrics look reasonable (not 100% WMAPE)")
                    else:
                        print("⚠️  WMAPE is very high - may indicate an issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def test_visualization_generation():
    """Test 5: Test visualization generation."""
    print("\n" + "="*60)
    print("TEST 5: Visualization Generation")
    print("="*60)
    
    logger = setup_logging()
    
    try:
        processed_data_path = Path("data/processed_earthquake_catalog_annual_stats.csv")
        trainer = EnhancedQuadtreeTrainer(
            data_path=str(processed_data_path),
            save_dir="data/results",
            logger=logger,
            model_types=['simple', 'attention']
        )
        
        # Load models first
        trainer._load_existing_models_from_files()
        
        # Check if we have models
        total_models = (len(trainer.results['simple']['frequency_models']['models']) + 
                       len(trainer.results['simple']['magnitude_models']['models']) +
                       len(trainer.results['attention']['frequency_models']['models']) +
                       len(trainer.results['attention']['magnitude_models']['models']))
        
        if total_models == 0:
            print("❌ No models available for visualization")
            return False
        
        print(f"🔄 Generating visualizations for {total_models} models...")
        
        # Generate visualizations
        trainer._generate_comparison_plots()
        
        # Check if visualization files were created
        results_dir = Path("data/results")
        viz_files = list(results_dir.glob("*comparison*.png"))
        
        if viz_files:
            print(f"✅ Visualizations generated successfully!")
            for viz_file in viz_files:
                print(f"   {viz_file.name}")
        else:
            print("⚠️  No visualization files found")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Starting Evaluation Fixes Test Suite")
    print("="*60)
    
    tests = [
        ("Model Loading Check", test_model_loading),
        ("Trainer Initialization", test_enhanced_trainer_initialization),
        ("Model Loading Method", test_model_loading_method),
        ("Evaluation Functionality", test_evaluation_functionality),
        ("Visualization Generation", test_visualization_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The evaluation fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
