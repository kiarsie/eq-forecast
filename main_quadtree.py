#!/usr/bin/env python3
"""
Quadtree-based Earthquake Forecasting System

Implements the methodology from the paper:
1. Pre-processing: Filter shallow earthquakes (<70km), classify into quadtree bins
2. LSTM Training: Separate networks for each bin (Simple LSTM vs Attention LSTM)
3. Forecasting: Predict frequency and maximum magnitude (no depth)
4. Evaluation: WMAPE and Forecast Accuracy metrics
5. Visualization: Comprehensive results analysis

Architecture:
- 4 hidden layers: 120, 90, 30, 30 neurons
- 10-year lookback period
- Sigmoid output activation
- 48 LSTM networks (one per quadtree bin)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing.earthquake_processor import EarthquakeProcessor
from src.models.quadtree_trainer import QuadtreeModelTrainer
from src.models.enhanced_trainer import train_enhanced_quadtree_models, EnhancedQuadtreeTrainer


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_file)
        logger.addHandler(file_handler)
    
    return logger


def preprocess_earthquake_data(input_path: str, output_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Preprocess earthquake catalog data following the paper's methodology.
    
    Args:
        input_path: Path to raw earthquake catalog
        output_path: Path to save processed data
        logger: Logger instance
        
    Returns:
        Processed earthquake catalog DataFrame
    """
    logger.info("Starting earthquake data preprocessing")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Load raw earthquake catalog
        logger.info("Loading raw earthquake catalog...")
        
        # Use the existing load_catalog function to get properly formatted data
        from src.preprocessing.load_catalog import load_catalog
        
        # Load and get the DataFrame (ignore the CSEP catalog)
        raw_df, _ = load_catalog(input_path)
        logger.info(f"Loaded {len(raw_df)} earthquake records")
        logger.info(f"Columns: {list(raw_df.columns)}")
        
        # Initialize processor
        processor = EarthquakeProcessor(min_depth=70.0)
        
        # Process catalog
        logger.info("Processing earthquake catalog...")
        processed_catalog, annual_stats = processor.process_catalog(
            df=raw_df,
            save_path=output_path
        )
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Processed catalog: {len(processed_catalog)} records")
        logger.info(f"Annual statistics: {len(annual_stats)} year-bin combinations")
        
        # Display summary statistics
        logger.info("\n=== Preprocessing Summary ===")
        logger.info(f"Original earthquakes: {len(raw_df)}")
        logger.info(f"Shallow earthquakes (<70km): {len(processed_catalog)}")
        logger.info(f"Quadtree bins created: {processed_catalog['bin_id'].nunique()}")
        logger.info(f"Years covered: {annual_stats['year'].min()} - {annual_stats['year'].max()}")
        
        # Display bin statistics
        bin_summary = processor.get_bin_statistics(annual_stats)
        logger.info(f"\nBin statistics summary:")
        logger.info(f"Average earthquakes per bin per year: {bin_summary['frequency_mean'].mean():.2f}")
        logger.info(f"Average max magnitude per bin: {bin_summary['max_magnitude_mean'].mean():.2f}")
        
        return processed_catalog, annual_stats
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def train_quadtree_models(data_path: str, 
                          save_dir: str, 
                          logger: logging.Logger,
                          model_types: List[str] = ['simple', 'attention'],
                          num_epochs: int = 100,
                          patience: int = 20,
                          clear_existing: bool = False) -> dict:
    """
    Train quadtree-based LSTM models with enhanced capabilities.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save results
        logger: Logger instance
        model_types: List of model types to train ('simple', 'attention')
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        
    Returns:
        Training results
    """
    logger.info("Starting enhanced quadtree-based model training")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Model types: {model_types}")
    
    try:
        # Use enhanced trainer for better model comparison
        if len(model_types) > 1 or 'both' in model_types:
            # Use enhanced trainer for multiple models
            if clear_existing:
                action = "Clearing existing results"
                logger.info(f"{action} before training...")
                # Create a temporary trainer to clear results
                temp_trainer = EnhancedQuadtreeTrainer(
                    data_path=data_path,
                    save_dir=save_dir,
                    logger=logger,
                    model_types=model_types
                )
                temp_trainer.clear_results()
                logger.info(f"{action} completed successfully!")
            
            results = train_enhanced_quadtree_models(
                data_path=data_path,
                save_dir=save_dir,
                logger=logger,
                model_types=model_types,
                num_epochs=num_epochs,
                patience=patience
            )
        else:
            # Use original trainer for single model type
            trainer = QuadtreeModelTrainer(
                data_path=data_path,
                save_dir=save_dir,
                lookback_years=10,  # As per paper
                target_horizon=1,
                batch_size=32,
                learning_rate=0.001
            )
            
            # Train models for all bins
            logger.info("Training models for all quadtree bins...")
            results = trainer.train_all_bins(
                num_epochs=num_epochs,
                patience=patience
            )
        
        logger.info("Training completed successfully!")
        
        # Display training summary
        if isinstance(results, dict) and 'comparison' in results:
            # Enhanced trainer results
            logger.info(f"\n=== Enhanced Training Summary ===")
            for model_type in model_types:
                if model_type in results:
                    freq_models = len(results[model_type].get('frequency_models', {}).get('models', {}))
                    mag_models = len(results[model_type].get('magnitude_models', {}).get('models', {}))
                    logger.info(f"{model_type.title()} LSTM - Frequency models: {freq_models}, Magnitude models: {mag_models}")
        else:
            # Original trainer results
            successful_bins = sum(1 for r in results.values() if 'error' not in r)
            logger.info(f"\n=== Training Summary ===")
            logger.info(f"Total bins: {len(results)}")
            logger.info(f"Successfully trained: {successful_bins}")
            logger.info(f"Failed: {len(results) - successful_bins}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_quadtree_models(data_path: str, 
                            save_dir: str, 
                            logger: logging.Logger) -> dict:
    """
    Evaluate trained quadtree-based models.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Evaluation results
    """
    logger.info("Starting model evaluation")
    
    try:
        # Initialize trainer
        trainer = QuadtreeModelTrainer(
            data_path=data_path,
            save_dir=save_dir,
            lookback_years=10,
            target_horizon=1,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Evaluate models for all bins
        logger.info("Evaluating models for all quadtree bins...")
        evaluation_results = trainer.evaluate_all_bins()
        
        logger.info("Evaluation completed successfully!")
        
        # Display evaluation summary
        successful_bins = sum(1 for r in evaluation_results.values() if 'error' not in r)
        logger.info(f"\n=== Evaluation Summary ===")
        logger.info(f"Total bins: {len(evaluation_results)}")
        logger.info(f"Successfully evaluated: {successful_bins}")
        logger.info(f"Failed: {len(evaluation_results) - successful_bins}")
        
        if successful_bins > 0:
            # Calculate average metrics
            simple_accuracies = []
            attention_accuracies = []
            simple_wmape = []
            attention_wmape = []
            
            for bin_results in evaluation_results.values():
                if 'error' not in bin_results:
                    if 'simple_lstm' in bin_results:
                        simple_accuracies.append(bin_results['simple_lstm']['forecast_accuracy'])
                        simple_wmape.append(bin_results['simple_lstm']['wmape'])
                    if 'attention_lstm' in bin_results:
                        attention_accuracies.append(bin_results['attention_lstm']['forecast_accuracy'])
                        attention_wmape.append(bin_results['attention_lstm']['wmape'])
            
            if simple_accuracies:
                logger.info(f"\nSimple LSTM Performance:")
                logger.info(f"  Average Forecast Accuracy: {np.mean(simple_accuracies):.2f}%")
                logger.info(f"  Average WMAPE: {np.mean(simple_wmape):.2f}%")
            
            if attention_accuracies:
                logger.info(f"\nAttention LSTM Performance:")
                logger.info(f"  Average Forecast Accuracy: {np.mean(attention_accuracies):.2f}%")
                logger.info(f"  Average WMAPE: {np.mean(attention_wmape):.2f}%")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def test_quadtree_models(data_path: str, save_dir: str, logger: logging.Logger):
    """
    Test trained quadtree models on the test set.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save test results
        logger: Logger instance
        
    Returns:
        Test results dictionary
    """
    logger.info("Starting quadtree model testing")
    
    try:
        # Initialize trainer
        trainer = QuadtreeModelTrainer(
            data_path=data_path,
            save_dir=save_dir,
            lookback_years=10,
            target_horizon=1,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Test models for all bins
        logger.info("Testing models for all quadtree bins...")
        test_results = trainer.test_all_bins()
        
        logger.info("Testing completed successfully!")
        
        # Display test summary
        successful_bins = sum(1 for r in test_results.values() if 'error' not in r)
        logger.info(f"\n=== Testing Summary ===")
        logger.info(f"Total bins: {len(test_results)}")
        logger.info(f"Successfully tested: {successful_bins}")
        logger.info(f"Failed: {len(test_results) - successful_bins}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


def validate_quadtree_models(data_path: str, save_dir: str, logger: logging.Logger):
    """
    Validate trained quadtree models on the validation set.
    
    Args:
        data_path: Path to processed earthquake data
        save_dir: Directory to save validation results
        logger: Logger instance
        
    Returns:
        Validation results dictionary
    """
    logger.info("Starting quadtree model validation")
    
    try:
        # Initialize trainer
        trainer = QuadtreeModelTrainer(
            data_path=data_path,
            save_dir=save_dir,
            lookback_years=10,
            target_horizon=1,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Validate models for all bins
        logger.info("Validating models for all quadtree bins...")
        validation_results = trainer.validate_all_bins()
        
        logger.info("Validation completed successfully!")
        
        # Display validation summary
        successful_bins = sum(1 for r in validation_results.values() if 'error' not in r)
        logger.info(f"\n=== Validation Summary ===")
        logger.info(f"Total bins: {len(validation_results)}")
        logger.info(f"Successfully validated: {successful_bins}")
        logger.info(f"Failed: {len(validation_results) - successful_bins}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise


def generate_visualizations(save_dir: str, logger: logging.Logger):
    """
    Generate comprehensive visualizations of results.
    
    Args:
        save_dir: Directory containing results
        logger: Logger instance
    """
    logger.info("Generating visualizations")
    
    try:
        # Load evaluation results
        results_path = Path(save_dir) / "evaluation_results.json"
        if not results_path.exists():
            logger.warning("Evaluation results not found, skipping visualizations")
            return
        
        with open(results_path, 'r') as f:
            import json
            results = json.load(f)
        
        # Initialize trainer to generate visualizations
        trainer = QuadtreeModelTrainer(
            data_path="",  # Not needed for visualization
            save_dir=save_dir,
            lookback_years=10,
            target_horizon=1
        )
        
        # Generate visualizations
        trainer.generate_visualizations(results)
        
        logger.info("Visualizations generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def main():
    """Main function."""
    
    def get_data_path(mode, output_dir, input_data):
        """Helper function to get the appropriate data path based on mode and available data."""
        if mode == 'full_pipeline':
            # For full pipeline, we expect the processed data to exist
            processed_data_path = output_dir / "processed_earthquake_catalog.csv"
            annual_stats_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
            if annual_stats_path.exists():
                return str(annual_stats_path)
            elif processed_data_path.exists():
                return str(processed_data_path)
            else:
                return input_data
        else:
            # Check if preprocessed data exists
            annual_stats_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
            if annual_stats_path.exists():
                # Verify it has the required columns
                try:
                    import pandas as pd
                    df = pd.read_csv(annual_stats_path)
                    required_columns = ['bin_id', 'year', 'frequency', 'max_magnitude']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        logger.warning(f"Preprocessed data missing required columns: {missing_columns}")
                        return input_data
                    return str(annual_stats_path)
                except Exception as e:
                    logger.warning(f"Error reading preprocessed data: {e}")
                    return input_data
            else:
                return input_data
    
    parser = argparse.ArgumentParser(
        description="Quadtree-based Earthquake Forecasting System"
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default='data/eq_catalog.csv',
        help='Path to raw earthquake catalog CSV file (default: data/eq_catalog.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Directory to save processed data and results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['preprocess', 'train', 'test', 'validate', 'evaluate', 'full_pipeline'],
        default='full_pipeline',
        help='Mode: preprocess, train, test, validate, evaluate, or full_pipeline'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['simple', 'attention', 'both'],
        default='both',
        help='LSTM model type: simple, attention, or both for comparison'
    )
    parser.add_argument(
        '--compare_models',
        action='store_true',
        help='Compare Simple LSTM vs Attention LSTM performance'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--clear_existing',
        action='store_true',
        help='Clear existing results and models before training'
    )
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List existing models without training'
    )
    parser.add_argument(
        '--force_retrain',
        action='store_true',
        help='Force retrain all models (equivalent to clear_existing)'
    )
    parser.add_argument(
        '--show_summary',
        action='store_true',
        help='Show training summary without training'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'earthquake_forecasting.log'
    logger = setup_logging(args.log_level, str(log_file))
    
    logger.info("=" * 80)
    logger.info("Quadtree-based Earthquake Forecasting System")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Compare models: {args.compare_models}")
    logger.info(f"Input data: {args.input_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Training epochs: {args.num_epochs}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Clear existing: {args.clear_existing}")
    logger.info(f"Force retrain: {args.force_retrain}")
    logger.info(f"List models: {args.list_models}")
    logger.info(f"Show summary: {args.show_summary}")
    
    try:
        # Handle standalone operations first (these should not trigger the pipeline)
        if args.list_models:
            # List existing models without training
            logger.info("\n" + "="*50)
            logger.info("LISTING EXISTING MODELS")
            logger.info("="*50)
            
            results_dir = output_dir / "results"
            if results_dir.exists():
                # Create a temporary trainer to list models
                # Use a dummy data path that exists to avoid initialization errors
                dummy_data_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
                if not dummy_data_path.exists():
                    # If the processed data doesn't exist, use the raw data
                    dummy_data_path = Path(args.input_data)
                
                temp_trainer = EnhancedQuadtreeTrainer(
                    data_path=str(dummy_data_path),
                    save_dir=str(results_dir),
                    logger=logger,
                    model_types=['simple', 'attention']
                )
                temp_trainer.check_save_directory()
            else:
                logger.info("No results directory found.")
            
            # Exit after listing models - don't continue with pipeline
            logger.info("Model listing completed. Exiting.")
            return
        
        if args.show_summary:
            # Show training summary without training
            logger.info("\n" + "="*50)
            logger.info("TRAINING SUMMARY")
            logger.info("="*50)
            
            results_dir = output_dir / "results"
            if results_dir.exists():
                # Create a temporary trainer to show summary
                # Use a dummy data path that exists to avoid initialization errors
                dummy_data_path = output_dir / "processed_earthquake_catalog_annual_stats.csv"
                if not dummy_data_path.exists():
                    # If the processed data doesn't exist, use the raw data
                    dummy_data_path = Path(args.input_data)
                
                temp_trainer = EnhancedQuadtreeTrainer(
                    data_path=str(dummy_data_path),
                    save_dir=str(results_dir),
                    logger=logger,
                    model_types=['simple', 'attention']
                )
                summary = temp_trainer.get_training_summary()
                logger.info(f"Training Status: {summary['status']}")
                logger.info(f"Total Models: {summary['total_models']}")
                logger.info("Models by Type:")
                for model_type, targets in summary['models_by_type'].items():
                    logger.info(f"  {model_type}: {targets['frequency']} frequency, {targets['magnitude']} magnitude")
                if summary['missing_models']:
                    logger.info("Missing Models:")
                    for missing in summary['missing_models']:
                        logger.info(f"  - {missing}")
            else:
                logger.info("No results directory found.")
            
            # Exit after showing summary - don't continue with pipeline
            logger.info("Summary display completed. Exiting.")
            return
        
        # Only proceed with pipeline operations if no standalone operations were requested
        if args.mode == 'preprocess' or args.mode == 'full_pipeline':
            # Step 1: Preprocessing
            logger.info("\n" + "="*50)
            logger.info("STEP 1: EARTHQUAKE DATA PREPROCESSING")
            logger.info("="*50)
            
            processed_data_path = output_dir / "processed_earthquake_catalog.csv"
            processed_catalog, annual_stats = preprocess_earthquake_data(
                input_path=args.input_data,
                output_path=str(processed_data_path),
                logger=logger
            )
            
            logger.info(f"Preprocessed data saved to: {processed_data_path}")
        
        if args.mode == 'train' or args.mode == 'full_pipeline':
            # Step 2: Model Training
            logger.info("\n" + "="*50)
            logger.info("STEP 2: LSTM MODEL TRAINING")
            logger.info("="*50)
            
            # Get appropriate data path
            data_path = get_data_path(args.mode, output_dir, args.input_data)
            if data_path != args.input_data:
                logger.info(f"Using preprocessed data: {data_path}")
                # Verify the file exists
                if not Path(data_path).exists():
                    logger.error(f"Preprocessed data file not found: {data_path}")
                    logger.error("Please run preprocessing first or use --mode full_pipeline")
                    sys.exit(1)
            else:
                logger.warning("No preprocessed data found. Please run preprocessing first or use --mode full_pipeline")
                logger.error("Cannot proceed without preprocessed data")
                sys.exit(1)
            
            results_dir = output_dir / "results"
            
            # Determine model types to train
            if args.model_type == 'both':
                model_types = ['simple', 'attention']
            else:
                model_types = [args.model_type]
            
            # Handle force retrain if requested
            if args.force_retrain:
                logger.info("Force retraining all models...")
                # Create a temporary trainer to force retrain
                temp_trainer = EnhancedQuadtreeTrainer(
                    data_path=data_path,
                    save_dir=str(results_dir),
                    logger=logger,
                    model_types=model_types
                )
                temp_trainer.force_retrain()
                logger.info("Force retrain completed - all existing models cleared!")
            
            training_results = train_quadtree_models(
                data_path=data_path,
                save_dir=str(results_dir),
                logger=logger,
                model_types=model_types,
                num_epochs=args.num_epochs,
                patience=args.patience,
                clear_existing=args.clear_existing
            )
            
            logger.info(f"Training results saved to: {results_dir}")
        
        if args.mode == 'test' or args.mode == 'full_pipeline':
            # Step 2.5: Model Testing (on test set)
            logger.info("\n" + "="*50)
            logger.info("STEP 2.5: LSTM MODEL TESTING")
            logger.info("="*50)
            
            # Get appropriate data path
            data_path = get_data_path(args.mode, output_dir, args.input_data)
            
            results_dir = output_dir / "results"
            test_results = test_quadtree_models(
                data_path=data_path,
                save_dir=str(results_dir),
                logger=logger
            )
            
            logger.info(f"Testing results saved to: {results_dir}")
        
        if args.mode == 'validate' or args.mode == 'full_pipeline':
            # Step 2.75: Model Validation (on validation set)
            logger.info("\n" + "="*50)
            logger.info("STEP 2.75: LSTM MODEL VALIDATION")
            logger.info("="*50)
            
            # Get appropriate data path
            data_path = get_data_path(args.mode, output_dir, args.input_data)
            
            results_dir = output_dir / "results"
            validation_results = validate_quadtree_models(
                data_path=data_path,
                save_dir=str(results_dir),
                logger=logger
            )
            
            logger.info(f"Validation results saved to: {results_dir}")
        
        if args.mode == 'evaluate' or args.mode == 'full_pipeline':
            # Step 3: Model Evaluation
            logger.info("\n" + "="*50)
            logger.info("STEP 3: MODEL EVALUATION")
            logger.info("="*50)
            
            # Get appropriate data path
            data_path = get_data_path(args.mode, output_dir, args.input_data)
            
            results_dir = output_dir / "results"
            evaluation_results = evaluate_quadtree_models(
                data_path=data_path,
                save_dir=str(results_dir),
                logger=logger
            )
            
            logger.info(f"Evaluation results saved to: {results_dir}")
            
            # Step 4: Generate Visualizations
            logger.info("\n" + "="*50)
            logger.info("STEP 4: GENERATING VISUALIZATIONS")
            logger.info("="*50)
            
            generate_visualizations(str(results_dir), logger)
        
        logger.info("\n" + "="*80)
        logger.info("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Final summary
        logger.info("\n=== FINAL SUMMARY ===")
        logger.info(f"Processed data: {output_dir / 'processed_earthquake_catalog.csv'}")
        logger.info(f"Results: {output_dir / 'results'}")
        logger.info(f"Log file: {log_file}")
        
        if args.mode == 'full_pipeline':
            logger.info("\nThe system has completed the full pipeline:")
            logger.info("1. ✅ Preprocessed earthquake data (filtered shallow, quadtree binned)")
            logger.info("2. ✅ Trained LSTM models for each quadtree bin")
            logger.info("3. ✅ Evaluated model performance using WMAPE and Forecast Accuracy")
            logger.info("4. ✅ Generated comprehensive visualizations")
            logger.info("\nYou can now analyze the results in the output directory!")
        
        return
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
