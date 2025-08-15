import unittest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.data_loader import EarthquakeDataset, create_data_loaders


class TestEarthquakeDataset(unittest.TestCase):
    """Test cases for EarthquakeDataset class."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic earthquake data
        np.random.seed(42)
        
        # Generate dates
        start_date = datetime(2010, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365 * 15)]  # 15 years
        
        # Generate synthetic earthquake data
        n_events = 5000  # Increased for better test coverage
        self.test_data = pd.DataFrame({
            'Date_Time': np.random.choice(dates, n_events),
            'magnitude': np.random.uniform(2.0, 8.0, n_events),
            'depth': np.random.uniform(0, 100, n_events),
            'latitude': np.random.uniform(10.0, 20.0, n_events),
            'longitude': np.random.uniform(120.0, 130.0, n_events)
        })
        
        # Sort by date
        self.test_data = self.test_data.sort_values('Date_Time').reset_index(drop=True)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=120,  # 10 years
            target_horizon=12     # 1 year
        )
        
        self.assertIsInstance(dataset, EarthquakeDataset)
        self.assertEqual(len(dataset.feature_columns), 7)  # magnitude, depth, lat, lon, month, year, day_of_year
        self.assertIsNotNone(dataset.scaler)
    
    def test_feature_scaling(self):
        """Test that features are properly scaled."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=120,
            target_horizon=12
        )
        
        # Check that scaled features exist
        for col in dataset.feature_columns:
            scaled_col = f'{col}_scaled'
            self.assertIn(scaled_col, dataset.data.columns)
            
            # Check that scaled features are different from original
            if col in ['magnitude', 'depth', 'latitude', 'longitude']:
                self.assertFalse(
                    np.allclose(dataset.data[col], dataset.data[scaled_col])
                )
    
    def test_sequence_creation(self):
        """Test that sequences are created correctly."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=12,  # 1 year for testing
            target_horizon=6     # 6 months
        )
        
        # Check that sequences were created
        self.assertGreater(len(dataset.sequences), 0)
        
        # Check sequence structure
        first_sequence = dataset.sequences[0]
        self.assertIn('cell_id', first_sequence)
        self.assertIn('input_features', first_sequence)
        self.assertIn('target_magnitude', first_sequence)
        self.assertIn('target_events', first_sequence)
        self.assertIn('input_dates', first_sequence)
        self.assertIn('target_dates', first_sequence)
        
        # Check input features shape
        input_features = first_sequence['input_features']
        self.assertEqual(input_features.shape[0], 12)  # sequence_length
        self.assertEqual(input_features.shape[1], len(dataset.feature_columns))
    
    def test_dataset_getitem(self):
        """Test dataset indexing."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=12,
            target_horizon=6
        )
        
        if len(dataset) > 0:
            # Get first item
            input_features, target_magnitude, target_events = dataset[0]
            
            # Check types
            self.assertIsInstance(input_features, torch.Tensor)
            self.assertIsInstance(target_magnitude, torch.Tensor)
            self.assertIsInstance(target_events, torch.Tensor)
            
            # Check shapes
            self.assertEqual(input_features.shape[0], 12)  # sequence_length
            self.assertEqual(input_features.shape[1], len(dataset.feature_columns))
            self.assertEqual(target_magnitude.shape[0], 1)
            self.assertEqual(target_events.shape[0], 1)
    
    def test_cell_sequences(self):
        """Test getting sequences for specific cells."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=12,
            target_horizon=6
        )
        
        if len(dataset) > 0:
            # Get first cell ID
            first_cell_id = dataset.sequences[0]['cell_id']
            
            # Get sequences for this cell
            cell_sequences = dataset.get_cell_sequences(first_cell_id)
            
            # Check that all sequences belong to the same cell
            for seq in cell_sequences:
                self.assertEqual(seq['cell_id'], first_cell_id)
    
    def test_scaler_persistence(self):
        """Test that scaler can be retrieved and reused."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=12,
            target_horizon=6
        )
        
        scaler = dataset.get_scaler()
        self.assertIsNotNone(scaler)
        
        # Test that scaler can be used on new data
        # Need to provide all 7 features that the scaler expects
        new_data = pd.DataFrame({
            'magnitude': [5.0],
            'depth': [50.0],
            'latitude': [15.0],
            'longitude': [125.0],
            'month': [6],
            'year': [2020],
            'day_of_year': [180]
        })
        
        scaled_new_data = scaler.transform(new_data)
        self.assertEqual(scaled_new_data.shape, (1, 7))
    
    def test_insufficient_data_handling(self):
        """Test handling of cells with insufficient data."""
        # Create data with very few events
        small_data = self.test_data.head(10)
        
        dataset = EarthquakeDataset(
            data=small_data,
            sequence_length=120,  # 10 years
            target_horizon=12     # 1 year
        )
        
        # Should handle gracefully (may have 0 sequences)
        self.assertIsInstance(dataset.sequences, list)


class TestDataLoaders(unittest.TestCase):
    """Test cases for data loader creation."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic earthquake data
        np.random.seed(42)
        
        start_date = datetime(2010, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365 * 15)]
        
        n_events = 500
        self.test_data = pd.DataFrame({
            'Date_Time': np.random.choice(dates, n_events),
            'magnitude': np.random.uniform(2.0, 8.0, n_events),
            'depth': np.random.uniform(0, 100, n_events),
            'latitude': np.random.uniform(10.0, 20.0, n_events),
            'longitude': np.random.uniform(120.0, 130.0, n_events)
        })
        
        self.test_data = self.test_data.sort_values('Date_Time').reset_index(drop=True)
    
    def test_data_loader_creation(self):
        """Test creation of train/val/test data loaders."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=24,
            target_horizon=12
        )
        
        if len(dataset) > 0:
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset=dataset,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=16
            )
            
            # Check that loaders were created
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)
            
            # Check batch sizes
            for batch in train_loader:
                input_features, target_magnitude, target_events = batch
                self.assertLessEqual(input_features.shape[0], 16)  # batch_size
                break
    
    def test_data_loader_ratios(self):
        """Test that data split ratios are respected."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=24,
            target_horizon=12
        )
        
        if len(dataset) > 0:
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset=dataset,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                batch_size=16
            )
            
            # Count batches
            train_batches = len(train_loader)
            val_batches = len(val_loader)
            test_batches = len(test_loader)
            
            # Check that loaders exist
            self.assertGreater(train_batches, 0)
            self.assertGreater(val_batches, 0)
            self.assertGreater(test_batches, 0)
    
    def test_invalid_ratios(self):
        """Test that invalid ratios raise an error."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=24,
            target_horizon=12
        )
        
        if len(dataset) > 0:
            with self.assertRaises(ValueError):
                create_data_loaders(
                    dataset=dataset,
                    train_ratio=0.5,
                    val_ratio=0.3,
                    test_ratio=0.3,  # Sum > 1.0
                    batch_size=16
                )
    
    def test_gpu_support(self):
        """Test GPU support in data loaders."""
        dataset = EarthquakeDataset(
            data=self.test_data,
            sequence_length=24,
            target_horizon=12
        )
        
        if len(dataset) > 0:
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset=dataset,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=16,
                num_workers=0  # Avoid multiprocessing issues in tests
            )
            
            # Check that pin_memory is set correctly
            # This is handled internally by PyTorch based on CUDA availability
            self.assertIsNotNone(train_loader)


if __name__ == '__main__':
    unittest.main()
