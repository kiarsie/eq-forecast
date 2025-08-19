#!/usr/bin/env python3
"""
Enhanced Data Processor for Earthquake Forecasting

Handles sparse earthquake data with:
- Improved normalization for zero-inflated data
- Feature engineering (rolling averages, seasonal patterns)
- Data augmentation for sparse sequences
- Better temporal representation
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


class EnhancedEarthquakeDataset(Dataset):
    """
    Enhanced dataset for earthquake forecasting with better feature engineering.
    
    Features:
    - Rolling averages for temporal smoothing
    - Seasonal patterns (if applicable)
    - Better normalization for sparse data
    - Enhanced temporal representation
    """
    
    def __init__(self, 
                 data_path: str,
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 normalize: bool = True,
                 add_rolling_features: bool = True,
                 rolling_windows: List[int] = [3, 5, 7]):
        """
        Initialize the enhanced dataset.
        
        Args:
            data_path: Path to the processed earthquake data CSV
            lookback_years: Number of years to look back (default: 10)
            target_horizon: Number of years to predict ahead (default: 1)
            normalize: Whether to normalize the data
            add_rolling_features: Whether to add rolling average features
            rolling_windows: List of rolling window sizes for features
        """
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.normalize = normalize
        self.add_rolling_features = add_rolling_features
        self.rolling_windows = rolling_windows
        
        # Load data
        self.data = pd.read_csv(data_path)
        self.logger = logging.getLogger(__name__)
        
        # Prepare sequences with enhanced features
        self.sequences = self._prepare_enhanced_sequences()
        
        # Setup enhanced normalization
        if self.normalize:
            self._setup_enhanced_normalization()
        
        self.logger.info(f"Enhanced dataset initialized with {len(self.sequences)} sequences")
        self.logger.info(f"Lookback: {lookback_years} years, Target horizon: {target_horizon} years")
        self.logger.info(f"Rolling features: {add_rolling_features}, Windows: {rolling_windows}")
    
    def _prepare_enhanced_sequences(self) -> List[Dict]:
        """
        Prepare enhanced sequences with rolling features and better temporal representation.
        
        Returns:
            List of enhanced sequence dictionaries
        """
        sequences = []
        
        # Group by bin_id
        for bin_id in self.data['bin_id'].unique():
            bin_data = self.data[self.data['bin_id'] == bin_id].copy()
            bin_data = bin_data.sort_values('year').reset_index(drop=True)
            
            # Add rolling features if enabled
            if self.add_rolling_features:
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
                
                # Only include if we have complete sequences
                if len(input_sequence) == self.lookback_years and len(target_sequence) == self.target_horizon:
                    sequences.append({
                        'bin_id': bin_id,
                        'input_sequence': input_sequence,
                        'target_sequence': target_sequence,
                        'input_years': input_sequence['year'].tolist(),
                        'target_years': target_sequence['year'].tolist()
                    })
        
        return sequences
    
    def _add_rolling_features(self, bin_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling average features to handle sparse data better.
        
        Args:
            bin_data: DataFrame for a specific bin
            
        Returns:
            DataFrame with additional rolling features
        """
        enhanced_data = bin_data.copy()
        
        # Add rolling averages for magnitude and frequency
        for window in self.rolling_windows:
            if len(bin_data) >= window:
                # Rolling average of max magnitude (handle zeros)
                enhanced_data[f'mag_rolling_{window}'] = (
                    bin_data['max_magnitude'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling average of frequency
                enhanced_data[f'freq_rolling_{window}'] = (
                    bin_data['frequency'].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std for variability
                enhanced_data[f'mag_std_{window}'] = (
                    bin_data['max_magnitude'].rolling(window=window, min_periods=1).std().fillna(0)
                )
                enhanced_data[f'freq_std_{window}'] = (
                    bin_data['frequency'].rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        # Add trend features (simple linear trend over lookback period)
        if len(bin_data) >= self.lookback_years:
            enhanced_data['mag_trend'] = 0.0
            enhanced_data['freq_trend'] = 0.0
            
            for i in range(len(bin_data) - self.lookback_years + 1):
                # Calculate trend for the next lookback_years
                mag_window = bin_data.iloc[i:i+self.lookback_years]['max_magnitude'].values
                freq_window = bin_data.iloc[i:i+self.lookback_years]['frequency'].values
                
                if len(mag_window) > 1:
                    # Simple linear trend (slope)
                    x = np.arange(len(mag_window))
                    mag_trend = np.polyfit(x, mag_window, 1)[0] if len(mag_window) > 1 else 0
                    freq_trend = np.polyfit(x, freq_window, 1)[0] if len(freq_window) > 1 else 0
                    
                    enhanced_data.iloc[i, enhanced_data.columns.get_loc('mag_trend')] = mag_trend
                    enhanced_data.iloc[i, enhanced_data.columns.get_loc('freq_trend')] = freq_trend
        
        # Fill NaN values
        enhanced_data = enhanced_data.fillna(0.0)
        
        return enhanced_data
    
    def _setup_enhanced_normalization(self):
        """Setup enhanced normalization parameters for all features."""
        # Get all feature columns (excluding metadata)
        feature_columns = [col for col in self.data.columns 
                          if col not in ['bin_id', 'year', 'target_year', 'target_max_magnitude', 'target_frequency']]
        
        # Initialize scalers for each feature
        self.scalers = {}
        
        for col in feature_columns:
            values = self.data[col].values
            
            # 🔧 IMPROVEMENT: Different normalization strategies based on feature type
            if col.startswith('max_magnitude') or col.startswith('mag_'):
                # 🔧 IMPROVEMENT: Use MinMaxScaler [0,1] for magnitudes instead of z-score
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                # For magnitude features, use log transformation for non-zero values
                non_zero_mask = values > 0
                if np.any(non_zero_mask):
                    # Log transform non-zero values, keep zeros as zeros
                    log_values = np.log10(values[non_zero_mask])
                    scaler.fit(log_values.reshape(-1, 1))
                else:
                    # All zeros, use default scaling
                    scaler.fit(values.reshape(-1, 1))
                    
            elif col.startswith('frequency') or col.startswith('freq_'):
                # 🔧 IMPROVEMENT: Apply log(1 + count) transform before normalizing frequency
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
                # Apply log(1 + count) transformation to handle zero-inflated data
                log1p_values = np.log1p(values)  # log(1 + frequency)
                scaler.fit(log1p_values.reshape(-1, 1))
                
            else:
                # For other features, use standard scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(values.reshape(-1, 1))
            
            self.scalers[col] = scaler
            
            self.logger.info(f"Scaler for {col}: {scaler}")
            if col.startswith('frequency') or col.startswith('freq_'):
                self.logger.info(f"  Applied log(1+count) transform before normalization")
            elif col.startswith('max_magnitude') or col.startswith('mag_'):
                self.logger.info(f"  Applied log10 transform + MinMax scaling [0,1]")
    
    def _normalize_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Normalize features using enhanced normalization."""
        if not self.normalize:
            return features
        
        normalized = features.copy()
        
        for i, col in enumerate(feature_names):
            if col in self.scalers:
                # 🔧 IMPROVEMENT: Apply appropriate transformation based on feature type
                if col.startswith('max_magnitude') or col.startswith('mag_'):
                    # For magnitude features, apply log transformation first, then MinMax scaling
                    non_zero_mask = normalized[:, i] > 0
                    if np.any(non_zero_mask):
                        normalized[non_zero_mask, i] = np.log10(normalized[non_zero_mask, i])
                
                elif col.startswith('frequency') or col.startswith('freq_'):
                    # 🔧 IMPROVEMENT: Apply log(1 + count) transformation before normalization
                    normalized[:, i] = np.log1p(normalized[:, i])
                
                # Apply scaling
                normalized[:, i] = self.scalers[col].transform(normalized[:, i].reshape(-1, 1)).flatten()
        
        return normalized
    
    def _denormalize_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Denormalize features back to original scale."""
        if not self.normalize:
            return features
        
        denormalized = features.copy()
        
        for i, col in enumerate(feature_names):
            if col in self.scalers:
                # Apply inverse scaling first
                denormalized[:, i] = self.scalers[col].inverse_transform(denormalized[:, i].reshape(-1, 1)).flatten()
                
                # 🔧 IMPROVEMENT: Apply inverse transformations based on feature type
                if col.startswith('max_magnitude') or col.startswith('mag_'):
                    # Inverse log transformation for magnitude features
                    denormalized[:, i] = 10 ** denormalized[:, i]
                    
                elif col.startswith('frequency') or col.startswith('freq_'):
                    # 🔧 IMPROVEMENT: Inverse log(1 + count) transformation for frequency
                    denormalized[:, i] = np.expm1(denormalized[:, i])  # exp(x) - 1
        
        return denormalized
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get an enhanced sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_features, target_features, metadata)
        """
        sequence = self.sequences[idx]
        
        # Extract features from input sequence
        input_features = []
        feature_names = []
        
        for _, row in sequence['input_sequence'].iterrows():
            features = []
            names = []
            
            # Basic features
            features.extend([row['max_magnitude'], row['frequency']])
            names.extend(['max_magnitude', 'frequency'])
            
            # Rolling features
            if self.add_rolling_features:
                for window in self.rolling_windows:
                    if f'mag_rolling_{window}' in row:
                        features.append(row[f'mag_rolling_{window}'])
                        names.append(f'mag_rolling_{window}')
                    if f'freq_rolling_{window}' in row:
                        features.append(row[f'freq_rolling_{window}'])
                        names.append(f'freq_rolling_{window}')
                    if f'mag_std_{window}' in row:
                        features.append(row[f'mag_std_{window}'])
                        names.append(f'mag_std_{window}')
                    if f'freq_std_{window}' in row:
                        features.append(row[f'freq_std_{window}'])
                        names.append(f'freq_std_{window}')
                
                # Trend features
                if 'mag_trend' in row:
                    features.append(row['mag_trend'])
                    names.append('mag_trend')
                if 'freq_trend' in row:
                    features.append(row['freq_trend'])
                    names.append('freq_trend')
            
            input_features.append(features)
            if not feature_names:  # Only set once
                feature_names = names
        
        # Extract features from target sequence (basic features only)
        target_features = []
        for _, row in sequence['target_sequence'].iterrows():
            features = [row['max_magnitude'], row['frequency']]
            target_features.append(features)
        
        # Convert to numpy arrays
        input_features = np.array(input_features, dtype=np.float32)
        target_features = np.array(target_features, dtype=np.float32)
        
        # Normalize features
        input_features = self._normalize_features(input_features, feature_names)
        target_features = self._normalize_features(target_features, ['max_magnitude', 'frequency'])
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_features)
        target_tensor = torch.FloatTensor(target_features)
        
        # Metadata
        metadata = {
            'bin_id': sequence['bin_id'],
            'input_years': sequence['input_years'],
            'target_years': sequence['target_years'],
            'feature_names': feature_names
        }
        
        return input_tensor, target_tensor, metadata
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Get input and output feature dimensions."""
        sample_input, sample_target, _ = self[0]
        return sample_input.shape[1], sample_target.shape[1]
    
    def get_feature_names(self) -> List[str]:
        """Get the names of input features."""
        if self.sequences:
            sample_input, _, metadata = self[0]
            return metadata.get('feature_names', [])
        return []


class EnhancedQuadtreeDataLoader:
    """
    Enhanced data loader for quadtree-based earthquake forecasting.
    
    Features:
    - Better data splitting strategy
    - Enhanced feature engineering
    - Improved normalization
    """
    
    def __init__(self, 
                 data_path: str,
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 batch_size: int = 32,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 normalize: bool = True,
                 shuffle: bool = True,
                 add_rolling_features: bool = True,
                 rolling_windows: List[int] = [3, 5, 7]):
        """
        Initialize the enhanced data loader.
        
        Args:
            data_path: Path to processed earthquake data
            lookback_years: Number of years to look back
            target_horizon: Number of years to predict ahead
            batch_size: Batch size for training
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            normalize: Whether to normalize features
            shuffle: Whether to shuffle data
            add_rolling_features: Whether to add rolling features
            rolling_windows: List of rolling window sizes
        """
        self.data_path = data_path
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.shuffle = shuffle
        self.add_rolling_features = add_rolling_features
        self.rolling_windows = rolling_windows
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Create enhanced dataset
        self.dataset = EnhancedEarthquakeDataset(
            data_path=data_path,
            lookback_years=lookback_years,
            target_horizon=target_horizon,
            normalize=normalize,
            add_rolling_features=add_rolling_features,
            rolling_windows=rolling_windows
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced QuadtreeDataLoader initialized successfully")
        self.logger.info(f"Input features: {self.dataset.get_feature_dimensions()[0]}")
        self.logger.info(f"Output features: {self.dataset.get_feature_dimensions()[1]}")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders with temporal splitting."""
        # Calculate split indices
        total_size = len(self.dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create indices
        indices = list(range(total_size))
        if self.shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Get input and output feature dimensions."""
        return self.dataset.get_feature_dimensions()
    
    def get_feature_names(self) -> List[str]:
        """Get the names of input features."""
        return self.dataset.get_feature_names()

