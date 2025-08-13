"""
Timbral - A hybrid NMF + BERT-based music recommendation system.

This package provides a comprehensive music recommendation system that combines
Non-negative Matrix Factorization (NMF) with BERT-based content encoding for
accurate and explainable music recommendations.
"""

__version__ = "0.1.0"
__author__ = "Conner Groth"

from .config import settings
from .utils.redis_connector import RedisConnector

__all__ = ["settings", "RedisConnector"] 