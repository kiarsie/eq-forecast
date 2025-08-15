"""
Models module for earthquake forecasting.

This module contains:
- AttentionLSTM: LSTM with attention mechanism
- SimpleLSTM: Baseline LSTM without attention
- QuadtreeEarthquakeDataset: Dataset for quadtree-based approach
- QuadtreeModelTrainer: Training and evaluation utilities
"""

from .simple_lstm import SimpleLSTM
from .attention_lstm import AttentionLSTM
from .quadtree_data_loader import QuadtreeEarthquakeDataset, QuadtreeDataLoader
from .quadtree_trainer import QuadtreeModelTrainer

__all__ = [
    'SimpleLSTM',
    'AttentionLSTM',
    'QuadtreeEarthquakeDataset',
    'QuadtreeDataLoader',
    'QuadtreeModelTrainer'
]
