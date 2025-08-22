"""
Models module for earthquake forecasting.

This module contains:
- SharedLSTMModel: Shared LSTM model for earthquake forecasting
- SharedLSTMTrainer: Training utilities for shared LSTM
- EnhancedSharedDataset: Enhanced dataset processor
- QuadtreeModelTrainer: Training and evaluation utilities
- ModelComparisonTrainer: Model comparison utilities
"""

from .shared_lstm_model import SharedLSTMModel, WeightedEarthquakeLoss
from .shared_lstm_trainer import SharedLSTMTrainer
from .enhanced_shared_processor import EnhancedSharedDataset
from .quadtree_trainer import QuadtreeModelTrainer
from .model_comparison_trainer import ModelComparisonTrainer

__all__ = [
    'SharedLSTMModel',
    'WeightedEarthquakeLoss',
    'SharedLSTMTrainer',
    'EnhancedSharedDataset',
    'QuadtreeModelTrainer',
    'ModelComparisonTrainer'
]
