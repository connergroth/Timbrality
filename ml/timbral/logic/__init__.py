"""
Training and embedding logic for Timbral.

This package contains training scripts, embedding builders, and
model training utilities for the music recommendation system.
"""

from .trainer import ModelTrainer
from .embedding_builder import EmbeddingBuilder
from .data_processor import DataProcessor

__all__ = ["ModelTrainer", "EmbeddingBuilder", "DataProcessor"] 