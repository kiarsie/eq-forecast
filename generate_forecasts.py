#!/usr/bin/env python3
"""
Earthquake Forecasting Script

This script loads trained models and generates forecasts for earthquake magnitude and frequency
for each spatial bin from 1920-2025. It uses:
- Simple LSTM for magnitude predictions
- Attention LSTM for frequency predictions

Output: JSON file with year, bin_id, predicted_max_magnitude, predicted_frequency
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.shared_lstm_model import SharedLSTMModel
from src.models.attention_shared_lstm_model import AttentionSharedLSTMModel
from src.models.enhanced_shared_processor import EnhancedSharedDataset


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_trained_models(
    simple_lstm_path: str,
    attention_lstm_path: str,
    logger: logging.Logger
) -> Tuple[SharedLSTMModel, AttentionSharedLSTMModel]:
    """
    Load both trained models for forecasting.
    
    Args:
        simple_lstm_path: Path to simple LSTM model (for magnitude)
        attention_lstm_path: Path to attention LSTM model (for frequency)
        logger: Logger instance
        
    Returns:
        Tuple of (simple_lstm_model, attention_lstm_model)
    """
    logger.info("Loading trained models...")
    
    try:
        # Load simple LSTM model for magnitude
        logger.info(f"Loading Simple LSTM from: {simple_lstm_path}")
        simple_checkpoint = torch.load(simple_lstm_path, map_location='cpu')
        
        # Extract model architecture from checkpoint
        model_state = simple_checkpoint['model_state_dict']
        lstm2_hidden_size = model_state['lstm2.weight_ih_l0'].shape[0] // 4
        lstm1_hidden_size = model_state['lstm1.weight_ih_l0'].shape[0] // 4
        dense1_input_size = model_state['dense1.weight'].shape[1]
        metadata_features = dense1_input_size - lstm2_hidden_size
        input_features = model_state['lstm1.weight_ih_l0'].shape[1]
        
        logger.info(f"Simple LSTM architecture: input={input_features}, metadata={metadata_features}, lstm1={lstm1_hidden_size}, lstm2={lstm2_hidden_size}")
        
        simple_model = SharedLSTMModel(
            input_seq_features=input_features,
            metadata_features=metadata_features,
            lookback_years=10,
            lstm_hidden_1=lstm1_hidden_size,
            lstm_hidden_2=lstm2_hidden_size,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        simple_model.load_state_dict(model_state)
        simple_model.eval()
        
        # Load attention LSTM model for frequency
        logger.info(f"Loading Attention LSTM from: {attention_lstm_path}")
        attention_checkpoint = torch.load(attention_lstm_path, map_location='cpu')
        
        # Extract model architecture from checkpoint
        model_state = attention_checkpoint['model_state_dict']
        lstm2_hidden_size = model_state['lstm2.weight_ih_l0'].shape[0] // 4
        lstm1_hidden_size = model_state['lstm1.weight_ih_l0'].shape[0] // 4
        dense1_input_size = model_state['dense1.weight'].shape[1]
        metadata_features = dense1_input_size - lstm2_hidden_size
        input_features = model_state['lstm1.weight_ih_l0'].shape[1]
        
        logger.info(f"Attention LSTM architecture: input={input_features}, metadata={metadata_features}, lstm1={lstm1_hidden_size}, lstm2={lstm2_hidden_size}")
        
        attention_model = AttentionSharedLSTMModel(
            input_seq_features=input_features,
            metadata_features=metadata_features,
            lookback_years=10,
            lstm_hidden_1=lstm1_hidden_size,
            lstm_hidden_2=lstm2_hidden_size,
            dense_hidden=32,
            dropout_rate=0.25,
            freq_head_type="linear"
        )
        attention_model.load_state_dict(model_state)
        attention_model.eval()
        
        logger.info("Both models loaded successfully!")
        return simple_model, attention_model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def create_forecasting_dataset(data_path: str, logger: logging.Logger) -> EnhancedSharedDataset:
    """
    Create dataset for forecasting using the same processed data as training.
    
    Args:
        data_path: Path to processed earthquake data
        logger: Logger instance
        
    Returns:
        EnhancedSharedDataset for forecasting
    """
    logger.info("Creating forecasting dataset...")
    
    try:
        # Create dataset with same parameters as training
        dataset = EnhancedSharedDataset(
            data_path=data_path,
            lookback_years=10,
            target_horizon=1,
            normalize=True,
            rolling_windows=[3, 5, 10],
            train_end_year=2009,
            val_end_year=2017,
            test_end_year=2025
        )
        
        logger.info(f"Dataset created with {len(dataset.sequences)} sequences")
        logger.info(f"Feature dimensions: Input={dataset.get_feature_dimensions()[0]}, Target={dataset.get_feature_dimensions()[1]}, Metadata={dataset.get_feature_dimensions()[2]}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def generate_forecasts(
    simple_model: SharedLSTMModel,
    attention_model: AttentionSharedLSTMModel,
    dataset: EnhancedSharedDataset,
    logger: logging.Logger
) -> List[Dict]:
    """
    Generate forecasts for all bins from 1920-2025.
    
    Args:
        simple_model: Trained simple LSTM model for magnitude
        attention_model: Trained attention LSTM model for frequency
        dataset: Dataset for forecasting
        logger: Logger instance
        
    Returns:
        List of forecast dictionaries
    """
    logger.info("Generating forecasts for all bins from 1920-2025...")
    
    forecasts = []
    
    try:
        # Extract unique years and bin IDs from the dataset sequences
        all_years = set()
        all_bin_ids = set()
        
        for sequence in dataset.sequences:
            # Get years from input sequence
            for _, row in sequence['input_sequence'].iterrows():
                all_years.add(int(row['year']))
            # Get target years
            for _, row in sequence['target_sequence'].iterrows():
                all_years.add(int(row['year']))
            # Get bin ID
            all_bin_ids.add(sequence['bin_id'])
        
        years = sorted(all_years)
        bin_ids = sorted(all_bin_ids)
        
        logger.info(f"Forecasting for years: {min(years)}-{max(years)}")
        logger.info(f"Forecasting for bins: {len(bin_ids)} bins")
        
        # Generate forecasts for each sequence in the dataset
        for idx, sequence in enumerate(dataset.sequences):
            try:
                # Get the target year (what we're predicting)
                target_year = int(sequence['target_sequence'].iloc[0]['year'])
                bin_id = sequence['bin_id']
                
                # Skip if not in our target range (1920-2025)
                if target_year < 1920 or target_year > 2025:
                    continue
                
                # Get input data for this sequence
                input_seq, target_seq, metadata, metadata_dict = dataset[idx]
                
                # Convert to tensors with batch dimension
                input_tensor = input_seq.unsqueeze(0)
                metadata_tensor = metadata.unsqueeze(0)
                
                # Generate predictions
                with torch.no_grad():
                    # Magnitude prediction from simple LSTM
                    magnitude_pred, _ = simple_model(input_tensor, metadata_tensor)
                    predicted_magnitude = magnitude_pred.squeeze().item()
                    
                    # Frequency prediction from attention LSTM
                    _, frequency_log_rate_pred = attention_model(input_tensor, metadata_tensor)
                    frequency_count_pred = attention_model.predict_frequency_counts(frequency_log_rate_pred.squeeze())
                    predicted_frequency = frequency_count_pred.item()
                
                # Denormalize predictions
                denorm_magnitude = dataset.denormalize_single_feature(predicted_magnitude, 'magnitude')
                denorm_frequency = dataset.denormalize_single_feature(predicted_frequency, 'frequency')
                
                # Create forecast entry
                forecast = {
                    "year": target_year,
                    "bin_id": int(float(bin_id)) if bin_id != '-1' else 0,
                    "predicted_max_magnitude": round(float(denorm_magnitude), 2),
                    "predicted_frequency": int(round(float(denorm_frequency)))
                }
                
                forecasts.append(forecast)
                
                if len(forecasts) % 100 == 0:
                    logger.info(f"Generated {len(forecasts)} forecasts...")
                
            except Exception as e:
                logger.warning(f"Error forecasting for sequence {idx}: {e}")
                continue
        
        logger.info(f"Forecasting completed! Generated {len(forecasts)} predictions")
        return forecasts
        
    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        raise


def save_forecasts(forecasts: List[Dict], output_path: str, logger: logging.Logger):
    """
    Save forecasts to JSON file.
    
    Args:
        forecasts: List of forecast dictionaries
        output_path: Path to save the JSON file
        logger: Logger instance
    """
    logger.info(f"Saving forecasts to: {output_path}")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(forecasts, f, indent=2)
        
        logger.info(f"Forecasts saved successfully! Total predictions: {len(forecasts)}")
        
        # Display sample of forecasts
        logger.info("Sample forecasts:")
        for i, forecast in enumerate(forecasts[:5]):
            logger.info(f"  {forecast}")
        
    except Exception as e:
        logger.error(f"Error saving forecasts: {e}")
        raise


def main():
    """Main function to run the forecasting pipeline."""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Earthquake Forecasts')
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed_earthquake_catalog_annual_stats.csv',
        help='Path to processed earthquake data (default: data/processed_earthquake_catalog_annual_stats.csv)'
    )
    parser.add_argument(
        '--simple_lstm_path',
        type=str,
        default='data/results/optimized_comparison/simplelstm/simplelstm_best_model.pth',
        help='Path to trained Simple LSTM model for magnitude prediction'
    )
    parser.add_argument(
        '--attention_lstm_path',
        type=str,
        default='data/results/optimized_comparison/attentionsharedlstm/attentionsharedlstm_best_model.pth',
        help='Path to trained Attention LSTM model for frequency prediction'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/forecasted/earthquake_forecasts_1920_2025.json',
        help='Path to save forecast results'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Earthquake Forecasting Pipeline")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Simple LSTM path: {args.simple_lstm_path}")
    logger.info(f"Attention LSTM path: {args.attention_lstm_path}")
    logger.info(f"Output path: {args.output_path}")
    
    try:
        # Check if input files exist
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            sys.exit(1)
        
        if not Path(args.simple_lstm_path).exists():
            logger.error(f"Simple LSTM model not found: {args.simple_lstm_path}")
            sys.exit(1)
        
        if not Path(args.attention_lstm_path).exists():
            logger.error(f"Attention LSTM model not found: {args.attention_lstm_path}")
            sys.exit(1)
        
        # Step 1: Load trained models
        logger.info("\n" + "="*50)
        logger.info("STEP 1: LOADING TRAINED MODELS")
        logger.info("="*50)
        
        simple_model, attention_model = load_trained_models(
            args.simple_lstm_path,
            args.attention_lstm_path,
            logger
        )
        
        # Step 2: Create forecasting dataset
        logger.info("\n" + "="*50)
        logger.info("STEP 2: CREATING FORECASTING DATASET")
        logger.info("="*50)
        
        dataset = create_forecasting_dataset(args.data_path, logger)
        
        # Step 3: Generate forecasts
        logger.info("\n" + "="*50)
        logger.info("STEP 3: GENERATING FORECASTS")
        logger.info("="*50)
        
        forecasts = generate_forecasts(simple_model, attention_model, dataset, logger)
        
        # Step 4: Save forecasts
        logger.info("\n" + "="*50)
        logger.info("STEP 4: SAVING FORECASTS")
        logger.info("="*50)
        
        save_forecasts(forecasts, args.output_path, logger)
        
        logger.info("\n" + "="*80)
        logger.info("FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Total forecasts generated: {len(forecasts)}")
        logger.info(f"Forecasts saved to: {args.output_path}")
        logger.info("You can now use this data for your Supabase integration!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
