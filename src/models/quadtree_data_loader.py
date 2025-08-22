import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from torch.utils.data import Dataset, DataLoader


class QuadtreeDataLoader:
    """
    Data loader for quadtree-based earthquake forecasting.
    """
    
    def __init__(self, 
                 data_path: str,
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 batch_size: int = 32):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to processed earthquake data
            lookback_years: Number of years to look back
            target_horizon: Number of years to predict ahead
            batch_size: Batch size for training
        """
        self.data_path = Path(data_path)
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load earthquake data from the specified path."""
        try:
            # Try to load the data file
            if self.data_path.suffix == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == '.parquet':
                self.data = pd.read_parquet(self.data_path)
            else:
                # Try to find any data file in the directory
                data_files = list(self.data_path.glob('*.csv')) + list(self.data_path.glob('*.parquet'))
                if data_files:
                    data_file = data_files[0]
                    if data_file.suffix == '.csv':
                        self.data = pd.read_csv(data_file)
                    else:
                        self.data = pd.read_parquet(data_file)
                else:
                    # Create dummy data for testing
                    self.logger.warning("No data files found, creating dummy data for testing")
                    self._create_dummy_data()
                    return
                    
            self.logger.info(f"Loaded data with shape: {self.data.shape}")
            
        except Exception as e:
            self.logger.warning(f"Could not load data from {self.data_path}: {e}")
            self.logger.info("Creating dummy data for testing")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing purposes."""
        # Create dummy earthquake data
        np.random.seed(42)
        
        # Generate dummy data with realistic structure
        n_samples = 1000
        n_years = 20
        
        # Create time series data
        years = np.arange(2000, 2000 + n_years)
        
        # Generate dummy sequences
        sequences = []
        for i in range(n_samples):
            # Generate random features for each year
            sequence_data = []
            for year in years:
                row = {
                    'year': year,
                    'max_magnitude': np.random.uniform(3.0, 8.0),
                    'frequency': np.random.poisson(5),
                    'center_lat': np.random.uniform(30.0, 50.0),
                    'center_lon': np.random.uniform(-120.0, -70.0),
                    'bin_area': np.random.uniform(1000, 10000),
                    'bin_id': i % 10  # 10 bins
                }
                sequence_data.append(row)
            
            sequences.append(pd.DataFrame(sequence_data))
        
        # Combine all sequences
        self.data = pd.concat(sequences, ignore_index=True)
        self.logger.info(f"Created dummy data with shape: {self.data.shape}")
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """
        Get input and output feature dimensions.
        
        Returns:
            Tuple of (input_size, output_size)
        """
        # Calculate feature dimensions based on available columns
        input_features = []
        
        # Basic features
        if 'max_magnitude' in self.data.columns:
            input_features.append('max_magnitude')
        if 'frequency' in self.data.columns:
            input_features.append('frequency')
        
        # Add rolling features if they exist
        rolling_cols = [col for col in self.data.columns if col.startswith('rolling_')]
        input_features.extend(rolling_cols)
        
        # Add year feature if it exists
        if 'year' in self.data.columns or 'year_normalized' in self.data.columns:
            input_features.append('year_normalized' if 'year_normalized' in self.data.columns else 'year')
        
        # Calculate input size
        input_size = len(input_features)
        
        # Output size is typically 2 (magnitude and frequency)
        output_size = 2
        
        self.logger.info(f"Feature dimensions - Input: {input_size}, Output: {output_size}")
        return input_size, output_size
    
    def get_bin_count(self) -> int:
        """
        Get the number of quadtree bins.
        
        Returns:
            Number of bins
        """
        if 'bin_id' in self.data.columns:
            bin_count = self.data['bin_id'].nunique()
        else:
            # Default to 10 bins if bin_id not available
            bin_count = 10
            
        self.logger.info(f"Number of quadtree bins: {bin_count}")
        return bin_count
    
    def get_data_for_bin(self, bin_id: int) -> pd.DataFrame:
        """
        Get data for a specific quadtree bin.
        
        Args:
            bin_id: ID of the quadtree bin
            
        Returns:
            DataFrame containing data for the specified bin
        """
        if 'bin_id' in self.data.columns:
            return self.data[self.data['bin_id'] == bin_id].copy()
        else:
            # If no bin_id, return all data
            return self.data.copy()
    
    def create_sequences(self, bin_id: int, split: str = 'train') -> List[Dict]:
        """
        Create sequences for training/validation/testing.
        
        Args:
            bin_id: ID of the quadtree bin
            split: Data split ('train', 'val', 'test')
            
        Returns:
            List of sequence dictionaries
        """
        bin_data = self.get_data_for_bin(bin_id)
        
        if bin_data.empty:
            self.logger.warning(f"No data found for bin {bin_id}")
            return []
        
        # Sort by year
        if 'year' in bin_data.columns:
            bin_data = bin_data.sort_values('year')
        elif 'year_normalized' in bin_data.columns:
            bin_data = bin_data.sort_values('year_normalized')
        
        sequences = []
        
        # Create sequences with lookback and target horizon
        for i in range(len(bin_data) - self.lookback_years - self.target_horizon + 1):
            # Input sequence
            input_start = i
            input_end = i + self.lookback_years
            input_sequence = bin_data.iloc[input_start:input_end]
            
            # Target sequence
            target_start = input_end
            target_end = target_start + self.target_horizon
            target_sequence = bin_data.iloc[target_start:target_end]
            
            sequence = {
                'bin_id': bin_id,
                'split': split,
                'input_sequence': input_sequence,
                'target_sequence': target_sequence,
                'input_years': input_sequence['year'].tolist() if 'year' in input_sequence.columns else [],
                'target_years': target_sequence['year'].tolist() if 'year' in target_sequence.columns else []
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def get_train_val_test_split(self, bin_id: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            bin_id: ID of the quadtree bin
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Dictionary with train, val, and test sequences
        """
        all_sequences = self.create_sequences(bin_id)
        
        if not all_sequences:
            return {'train': [], 'val': [], 'test': []}
        
        # Shuffle sequences
        np.random.shuffle(all_sequences)
        
        n_sequences = len(all_sequences)
        n_train = int(n_sequences * train_ratio)
        n_val = int(n_sequences * val_ratio)
        
        train_sequences = all_sequences[:n_train]
        val_sequences = all_sequences[n_train:n_train + n_val]
        test_sequences = all_sequences[n_train + n_val:]
        
        return {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences
        }
