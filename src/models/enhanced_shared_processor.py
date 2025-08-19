#!/usr/bin/env python3
"""
Enhanced Data Processor for Shared LSTM Earthquake Forecasting

Handles:
- Annual aggregation of earthquake data (1910 → March 2025)
- Rolling features (3, 5, 10 years) computed causally
- Time-based data splitting (train ≤2009, val 2010–2017, test 2018–2025)
- Feature engineering for shared LSTM model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class EnhancedSharedDataset(Dataset):
    """
    Enhanced dataset for shared LSTM earthquake forecasting.
    
    Features:
    - Annual counts, max magnitude
    - Rolling features (3, 5, 10 years) computed causally
    - Bin metadata (lat, lon, area, etc.)
    - Time-based splitting to prevent data leakage
    """
    
    def __init__(self, 
                 data_path: str,
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 normalize: bool = True,
                 rolling_windows: List[int] = [3, 5, 10],
                 train_end_year: int = 2009,
                 val_end_year: int = 2017,
                 test_end_year: int = 2025):
        """
        Initialize the enhanced shared dataset.
        
        Args:
            data_path: Path to the earthquake catalog CSV
            lookback_years: Number of years to look back (default: 10)
            target_horizon: Number of years to predict ahead (default: 1)
            normalize: Whether to normalize the data
            rolling_windows: List of rolling window sizes for features
            train_end_year: Last year for training data
            val_end_year: Last year for validation data
            test_end_year: Last year for test data
        """
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.normalize = normalize
        self.rolling_windows = rolling_windows
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.test_end_year = test_end_year
        
        # Load and preprocess data
        self.raw_data = pd.read_csv(data_path)
        self.logger = logging.getLogger(__name__)
        
        # Process data into annual aggregates
        self.annual_data = self._create_annual_aggregates()
        
        # Prepare sequences with rolling features
        self.sequences = self._prepare_sequences()
        
        # Setup normalization
        if self.normalize:
            self._setup_normalization()
        
        self.logger.info(f"EnhancedSharedDataset initialized with {len(self.sequences)} sequences")
        self.logger.info(f"Lookback: {lookback_years} years, Target horizon: {target_horizon} years")
        self.logger.info(f"Rolling windows: {rolling_windows}")
        self.logger.info(f"Time split: Train<={train_end_year}, Val{val_end_year-2009}-{val_end_year}, Test{val_end_year+1}-{test_end_year}")
    
    def _create_annual_aggregates(self) -> pd.DataFrame:
        """
        Create annual aggregates from earthquake catalog data.
        
        Returns:
            DataFrame with annual earthquake statistics per bin
        """
        # Filter for shallow events (<70 km depth)
        shallow_data = self.raw_data[self.raw_data['Depth'] < 70].copy()
        
        # Convert date to year
        if 'Date' in shallow_data.columns:
            shallow_data['year'] = pd.to_datetime(shallow_data['Date']).dt.year
        elif 'Year' in shallow_data.columns:
            shallow_data['year'] = shallow_data['Year']
        else:
            # Assume first column is date-like
            shallow_data['year'] = pd.to_datetime(shallow_data.iloc[:, 0]).dt.year
        
        # Filter for valid years (1910-2025)
        shallow_data = shallow_data[
            (shallow_data['year'] >= 1910) & 
            (shallow_data['year'] <= 2025)
        ].copy()
        
        # Create spatial bins (simplified grid for now)
        # You can replace this with your quadtree binning logic
        lat_bins = np.linspace(shallow_data['N_Lat'].min(), shallow_data['N_Lat'].max(), 8)
        lon_bins = np.linspace(shallow_data['E_Long'].min(), shallow_data['E_Long'].max(), 8)
        
        shallow_data['lat_bin'] = pd.cut(shallow_data['N_Lat'], bins=lat_bins, labels=False)
        shallow_data['lon_bin'] = pd.cut(shallow_data['E_Long'], bins=lon_bins, labels=False)
        shallow_data['bin_id'] = shallow_data['lat_bin'].astype(str) + '_' + shallow_data['lon_bin'].astype(str)
        
        # Annual aggregation per bin
        annual_stats = []
        
        for bin_id in shallow_data['bin_id'].unique():
            bin_data = shallow_data[shallow_data['bin_id'] == bin_id]
            
            # Get bin metadata (center coordinates, area)
            bin_center_lat = bin_data['N_Lat'].mean()
            bin_center_lon = bin_data['E_Long'].mean()
            bin_area = (lat_bins[1] - lat_bins[0]) * (lon_bins[1] - lon_bins[0])
            
            # Group by year and aggregate
            yearly = bin_data.groupby('year').agg({
                'Mag': ['count', 'max'],
                'Depth': 'mean'
            }).reset_index()
            
            yearly.columns = ['year', 'frequency', 'max_magnitude', 'avg_depth']
            
            # Add bin metadata
            yearly['bin_id'] = bin_id
            yearly['center_lat'] = bin_center_lat
            yearly['center_lon'] = bin_center_lon
            yearly['bin_area'] = bin_area
            
            annual_stats.append(yearly)
        
        annual_data = pd.concat(annual_stats, ignore_index=True)
        
        # Fill missing years with zeros (no earthquakes)
        complete_years = pd.DataFrame({'year': range(1910, 2026)})
        complete_data = []
        
        for bin_id in annual_data['bin_id'].unique():
            bin_data = annual_data[annual_data['bin_id'] == bin_id].copy()
            bin_metadata = bin_data.iloc[0][['bin_id', 'center_lat', 'center_lon', 'bin_area']]
            
            # Merge with complete years
            complete_bin = complete_years.merge(bin_data, on='year', how='left')
            complete_bin = complete_bin.fillna({
                'frequency': 0,
                'max_magnitude': 0,
                'avg_depth': 0
            })
            
            # Fill metadata
            for col in ['bin_id', 'center_lat', 'center_lon', 'bin_area']:
                complete_bin[col] = bin_metadata[col]
            
            complete_data.append(complete_bin)
        
        final_data = pd.concat(complete_data, ignore_index=True)
        final_data = final_data.sort_values(['bin_id', 'year']).reset_index(drop=True)
        
        self.logger.info(f"Created annual aggregates: {len(final_data)} records, {final_data['bin_id'].nunique()} bins")
        self.logger.info(f"Year range: {final_data['year'].min()} - {final_data['year'].max()}")
        
        return final_data
    
    def _add_rolling_features(self, bin_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling features computed causally (past-only, no leakage).
        
        Args:
            bin_data: DataFrame for a specific bin, sorted by year
            
        Returns:
            DataFrame with added rolling features
        """
        bin_data = bin_data.copy()
        
        # Rolling features computed causally (only using past data)
        for window in self.rolling_windows:
            # Rolling count (frequency)
            bin_data[f'rolling_count_{window}'] = bin_data['frequency'].rolling(
                window=window, min_periods=1, center=False
            ).mean()
            
            # Rolling max magnitude
            bin_data[f'rolling_max_mag_{window}'] = bin_data['max_magnitude'].rolling(
                window=window, min_periods=1, center=False
            ).max()
            
            # Rolling std of magnitude
            bin_data[f'rolling_std_mag_{window}'] = bin_data['max_magnitude'].rolling(
                window=window, min_periods=1, center=False
            ).std().fillna(0)
        
        # Additional features
        bin_data['year_normalized'] = (bin_data['year'] - 1910) / (2025 - 1910)  # Normalize years
        
        return bin_data
    
    def _prepare_sequences(self) -> List[Dict]:
        """
        Prepare sequences with rolling features and time-based splitting.
        
        Returns:
            List of sequence dictionaries with split information
        """
        sequences = []
        
        # Group by bin_id
        for bin_id in self.annual_data['bin_id'].unique():
            bin_data = self.annual_data[self.annual_data['bin_id'] == bin_id].copy()
            bin_data = bin_data.sort_values('year').reset_index(drop=True)
            
            # Add rolling features
            bin_data = self._add_rolling_features(bin_data)
            
            # Create sliding windows
            for i in range(len(bin_data) - self.lookback_years - self.target_horizon + 1):
                # Input sequence (lookback years)
                input_start = i
                input_end = i + self.lookback_years
                input_sequence = bin_data.iloc[input_start:input_end]
                
                # Target sequence (next target_horizon years)
                target_start = input_end
                target_end = target_start + self.target_horizon
                target_sequence = bin_data.iloc[target_start:target_end]
                
                # Determine data split based on target year
                target_year = target_sequence['year'].iloc[0]
                
                if target_year <= self.train_end_year:
                    split = 'train'
                elif target_year <= self.val_end_year:
                    split = 'val'
                else:
                    split = 'test'
                
                # Only include if we have complete sequences
                if len(input_sequence) == self.lookback_years and len(target_sequence) == self.target_horizon:
                    sequences.append({
                        'bin_id': bin_id,
                        'split': split,
                        'input_sequence': input_sequence,
                        'target_sequence': target_sequence,
                        'input_years': input_sequence['year'].tolist(),
                        'target_years': target_sequence['year'].tolist()
                    })
        
        return sequences
    
    def _setup_normalization(self):
        """Setup normalization parameters for features."""
        # Get all values for normalization
        all_max_magnitudes = self.annual_data['max_magnitude'].values
        all_frequencies = self.annual_data['frequency'].values
        
        # Compute statistics
        self.max_magnitude_mean = np.mean(all_max_magnitudes)
        self.max_magnitude_std = np.std(all_max_magnitudes)
        self.frequency_mean = np.mean(all_frequencies)
        self.frequency_std = np.std(all_frequencies)
        
        # Handle zero standard deviation
        if self.max_magnitude_std == 0:
            self.max_magnitude_std = 1.0
        if self.frequency_std == 0:
            self.frequency_std = 1.0
        
        self.logger.info(f"Normalization parameters:")
        self.logger.info(f"  Max Magnitude: mean={self.max_magnitude_mean:.3f}, std={self.max_magnitude_std:.3f}")
        self.logger.info(f"  Frequency: mean={self.frequency_mean:.3f}, std={self.frequency_std:.3f}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if not self.normalize:
            return features
        
        normalized = features.copy()
        normalized[:, 0] = (normalized[:, 0] - self.max_magnitude_mean) / self.max_magnitude_std
        normalized[:, 1] = (normalized[:, 1] - self.frequency_mean) / self.frequency_std
        
        return normalized
    
    def _denormalize_features(self, features: np.ndarray) -> np.ndarray:
        """Denormalize features back to original scale."""
        if not self.normalize:
            return features
        
        denormalized = features.copy()
        
        # Handle different feature structures
        if features.shape[1] == 2:  # [magnitude, frequency]
            denormalized[:, 0] = denormalized[:, 0] * self.max_magnitude_std + self.max_magnitude_mean
            denormalized[:, 1] = denormalized[:, 1] * self.frequency_std + self.frequency_mean
        elif features.shape[1] == 1:  # Single feature (e.g., just frequency)
            # Assume it's frequency if only one column
            denormalized[:, 0] = denormalized[:, 0] * self.frequency_std + self.frequency_mean
        else:
            # Handle extended features (with rolling features)
            denormalized[:, 0] = denormalized[:, 0] * self.max_magnitude_std + self.max_magnitude_mean
            denormalized[:, 1] = denormalized[:, 1] * self.frequency_std + self.frequency_mean
            
            # Handle rolling features if they exist
            if hasattr(self, 'rolling_count_3_std'):
                feature_idx = 2
                for window in self.rolling_windows:
                    if hasattr(self, f'rolling_count_{window}_std'):
                        std_attr = getattr(self, f'rolling_count_{window}_std')
                        mean_attr = getattr(self, f'rolling_count_{window}_mean')
                        if feature_idx < features.shape[1]:
                            denormalized[:, feature_idx] = denormalized[:, feature_idx] * std_attr + mean_attr
                            feature_idx += 1
                    
                    if hasattr(self, f'rolling_max_mag_{window}_std'):
                        std_attr = getattr(self, f'rolling_max_mag_{window}_std')
                        mean_attr = getattr(self, f'rolling_max_mag_{window}_mean')
                        if feature_idx < features.shape[1]:
                            denormalized[:, feature_idx] = denormalized[:, feature_idx] * std_attr + mean_attr
                            feature_idx += 1
                    
                    if hasattr(self, f'rolling_std_mag_{window}_std'):
                        std_attr = getattr(self, f'rolling_std_mag_{window}_std')
                        mean_attr = getattr(self, f'rolling_std_mag_{window}_mean')
                        if feature_idx < features.shape[1]:
                            denormalized[:, feature_idx] = denormalized[:, feature_idx] * std_attr + mean_attr
                            feature_idx += 1
                
                # Handle year feature
                if feature_idx < features.shape[1]:
                    denormalized[:, feature_idx] = denormalized[:, feature_idx] * self.year_std + self.year_mean
        
        return denormalized
    
    def denormalize_single_feature(self, feature_value: float, feature_type: str) -> float:
        """
        Denormalize a single feature value.
        
        Args:
            feature_value: Normalized feature value
            feature_type: Type of feature ('magnitude' or 'frequency')
            
        Returns:
            Denormalized feature value
        """
        if not self.normalize:
            return feature_value
        
        if feature_type == 'magnitude':
            return feature_value * self.max_magnitude_std + self.max_magnitude_mean
        elif feature_type == 'frequency':
            return feature_value * self.frequency_std + self.frequency_mean
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_features, target_features, metadata, metadata_dict)
        """
        sequence = self.sequences[idx]
        
        # Extract sequential features from input sequence
        input_features = []
        for _, row in sequence['input_sequence'].iterrows():
            features = [
                row['max_magnitude'],
                row['frequency']
            ]
            
            # Add rolling features
            for window in self.rolling_windows:
                features.extend([
                    row[f'rolling_count_{window}'],
                    row[f'rolling_max_mag_{window}'],
                    row[f'rolling_std_mag_{window}']
                ])
            
            # Add year feature
            features.append(row['year_normalized'])
            
            input_features.append(features)
        
        # Extract target features
        target_features = []
        for _, row in sequence['target_sequence'].iterrows():
            features = [
                row['max_magnitude'],
                row['frequency']
            ]
            target_features.append(features)
        
        # Extract metadata features
        metadata_row = sequence['input_sequence'].iloc[0]
        metadata_features = [
            metadata_row['center_lat'],
            metadata_row['center_lon'],
            metadata_row['bin_area'],
            metadata_row['year_normalized']
        ]
        
        # Convert to numpy arrays
        input_features = np.array(input_features, dtype=np.float32)
        target_features = np.array(target_features, dtype=np.float32)
        metadata_features = np.array(metadata_features, dtype=np.float32)
        
        # Normalize features
        input_features = self._normalize_features(input_features)
        target_features = self._normalize_features(target_features)
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_features)
        target_tensor = torch.FloatTensor(target_features)
        metadata_tensor = torch.FloatTensor(metadata_features)
        
        # Metadata dictionary
        metadata_dict = {
            'bin_id': sequence['bin_id'],
            'split': sequence['split'],
            'input_years': sequence['input_years'],
            'target_years': sequence['target_years']
        }
        
        return input_tensor, target_tensor, metadata_tensor, metadata_dict
    
    def get_split_sequences(self, split: str) -> List[Dict]:
        """Get sequences for a specific split (train/val/test)."""
        return [seq for seq in self.sequences if seq['split'] == split]
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """Get input, target, and metadata feature dimensions."""
        sample_input, sample_target, sample_metadata, _ = self[0]
        return sample_input.shape[1], sample_target.shape[1], sample_metadata.shape[0]
