#!/usr/bin/env python3
"""
Quadtree-based Data Loader for Earthquake Forecasting

Implements the methodology from the paper:
- 10-year lookback period for LSTM training
- Sliding window approach for each quadtree bin
- Separate LSTM networks for each bin (48 total)
- Features: frequency and maximum magnitude (no depth)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path


class QuadtreeEarthquakeDataset(Dataset):
    """
    Dataset for earthquake forecasting using quadtree bins.
    
    Each sample contains:
    - Input: 10-year sequence of earthquake variables per bin
    - Target: Next year's earthquake variables for that bin
    """
    
    def __init__(self, 
                 data_path: str,
                 lookback_years: int = 10,
                 target_horizon: int = 1,
                 normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed earthquake data CSV
            lookback_years: Number of years to look back (default: 10)
            target_horizon: Number of years to predict ahead (default: 1)
            normalize: Whether to normalize the data
        """
        self.lookback_years = lookback_years
        self.target_horizon = target_horizon
        self.normalize = normalize
        
        # Load data
        self.data = pd.read_csv(data_path)
        self.logger = logging.getLogger(__name__)
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
        # Setup normalization
        if self.normalize:
            self._setup_normalization()
        
        self.logger.info(f"Dataset initialized with {len(self.sequences)} sequences")
        self.logger.info(f"Lookback: {lookback_years} years, Target horizon: {target_horizon} years")
    
    def _prepare_sequences(self) -> List[Dict]:
        """
        Prepare sliding window sequences for each quadtree bin.
        
        Returns:
            List of sequence dictionaries
        """
        sequences = []
        
        # Group by bin_id
        for bin_id in self.data['bin_id'].unique():
            bin_data = self.data[self.data['bin_id'] == bin_id].copy()
            bin_data = bin_data.sort_values('year').reset_index(drop=True)
            
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
    
    def _setup_normalization(self):
        """Setup normalization parameters for features."""
        # Get all values for normalization
        all_max_magnitudes = self.data['max_magnitude'].values
        all_frequencies = self.data['frequency'].values
        
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
        denormalized[:, 0] = denormalized[:, 0] * self.max_magnitude_std + self.max_magnitude_mean
        denormalized[:, 1] = denormalized[:, 1] * self.frequency_std + self.frequency_mean
        
        return denormalized
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_features, target_features, metadata)
        """
        sequence = self.sequences[idx]
        
        # Extract features from input sequence
        input_features = []
        for _, row in sequence['input_sequence'].iterrows():
            features = [
                row['max_magnitude'],
                row['frequency']
            ]
            input_features.append(features)
        
        # Extract features from target sequence
        target_features = []
        for _, row in sequence['target_sequence'].iterrows():
            features = [
                row['max_magnitude'],
                row['frequency']
            ]
            target_features.append(features)
        
        # Convert to numpy arrays
        input_features = np.array(input_features, dtype=np.float32)
        target_features = np.array(target_features, dtype=np.float32)
        
        # Normalize features
        input_features = self._normalize_features(input_features)
        target_features = self._normalize_features(target_features)
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_features)
        target_tensor = torch.FloatTensor(target_features)
        
        # Metadata
        metadata = {
            'bin_id': sequence['bin_id'],
            'input_years': sequence['input_years'],
            'target_years': sequence['target_years']
        }
        
        return input_tensor, target_tensor, metadata
    
    def get_bin_statistics(self) -> pd.DataFrame:
        """Get statistics for each quadtree bin."""
        return self.data.groupby('bin_id').agg({
            'max_magnitude': ['count', 'mean', 'std', 'min', 'max'],
            'frequency': ['mean', 'std', 'min', 'max'],
            'year': ['min', 'max']
        }).round(3)
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Get input and output feature dimensions."""
        sample_input, sample_target, _ = self[0]
        return sample_input.shape[1], sample_target.shape[1]


class QuadtreeDataLoader:
    """
    Data loader for quadtree-based earthquake forecasting.
    
    Creates separate data loaders for each quadtree bin.
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
                 shuffle: bool = True):
        """
        Initialize the data loader.
        
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
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Create dataset
        self.dataset = QuadtreeEarthquakeDataset(
            data_path=data_path,
            lookback_years=lookback_years,
            target_horizon=target_horizon,
            normalize=normalize
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("QuadtreeDataLoader initialized successfully")
    
    @property
    def bin_ids(self):
        """Get the list of unique quadtree bin IDs."""
        return sorted(self.dataset.data['bin_id'].unique())
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
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
    
    def get_bin_loaders(self) -> Dict[int, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get separate data loaders for each quadtree bin.
        
        Returns:
            Dictionary mapping bin_id to (train_loader, val_loader, test_loader)
        """
        # Use cached loaders if already created
        if hasattr(self, '_cached_bin_loaders') and self._cached_bin_loaders:
            self.logger.info("Using cached bin loaders")
            return self._cached_bin_loaders
        
        bin_loaders = {}
        
        # Get unique bin IDs
        unique_bins = self.dataset.data['bin_id'].unique()
        self.logger.info(f"Found {len(unique_bins)} unique bins: {sorted(unique_bins)}")
        
        # Pre-filter sequences by bin_id for efficiency
        sequences_by_bin = {}
        for seq in self.dataset.sequences:
            bin_id = seq['bin_id']
            if bin_id not in sequences_by_bin:
                sequences_by_bin[bin_id] = []
            sequences_by_bin[bin_id].append(seq)
        
        for bin_id in unique_bins:
            # Filter dataset for this bin
            bin_data = self.dataset.data[self.dataset.data['bin_id'] == bin_id]
            
            # Check if we have enough data points for at least one sequence
            # We need at least lookback_years + target_horizon data points
            if len(bin_data) < self.lookback_years + self.target_horizon:
                self.logger.warning(f"Bin {bin_id} has insufficient data ({len(bin_data)} points), skipping")
                continue
            
            # Additional check: ensure we have enough years span for meaningful sequences
            year_span = bin_data['year'].max() - bin_data['year'].min() + 1
            if year_span < self.lookback_years + self.target_horizon:
                self.logger.warning(f"Bin {bin_id} has insufficient year span ({year_span} years), skipping")
                continue
            
            # Create bin-specific dataset efficiently
            bin_dataset = QuadtreeEarthquakeDataset(
                data_path=self.data_path,
                lookback_years=self.lookback_years,
                target_horizon=self.target_horizon,
                normalize=self.normalize
            )
            
            # Use pre-filtered sequences instead of filtering again
            if bin_id in sequences_by_bin:
                bin_dataset.sequences = sequences_by_bin[bin_id]
            else:
                bin_dataset.sequences = []
            
            # Create data loaders for this bin
            bin_train_loader, bin_val_loader, bin_test_loader = self._create_bin_loaders(bin_dataset)
            
            bin_loaders[bin_id] = (bin_train_loader, bin_val_loader, bin_test_loader)
            self.logger.info(f"✅ Bin {bin_id}: {len(bin_data)} data points, year span {year_span}")
        
        self.logger.info(f"Created separate loaders for {len(bin_loaders)} quadtree bins")
        self.logger.info(f"Available bins for training: {sorted(bin_loaders.keys())}")
        
        # Cache the results to avoid recreating them
        self._cached_bin_loaders = bin_loaders
        
        return bin_loaders
    
    def _create_bin_loaders(self, bin_dataset: QuadtreeEarthquakeDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for a specific bin."""
        total_size = len(bin_dataset)
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
            bin_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            bin_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            bin_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Get input and output feature dimensions."""
        return self.dataset.get_feature_dimensions()
    
    def get_bin_count(self) -> int:
        """Get the number of unique quadtree bins."""
        return self.dataset.data['bin_id'].nunique()
    
    def get_bin_ids(self) -> List[int]:
        """Get the list of unique quadtree bin IDs."""
        return sorted(self.dataset.data['bin_id'].unique())
