"""
Global settings and configuration for Timbral.

This module defines all configuration settings, environment variables,
and constants used throughout the music recommendation system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    DEBUG: bool = False
    
    # Redis Configuration (Upstash)
    REDIS_URL: str
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Model Configuration
    NMF_N_COMPONENTS: int = 100
    NMF_RANDOM_STATE: int = 42
    BERT_MODEL_NAME: str = "bert-base-uncased"
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Data Configuration
    DATA_BRONZE_PATH: str = "data/bronze"
    DATA_SILVER_PATH: str = "data/silver"
    DATA_GOLD_PATH: str = "data/gold"
    
    # Model Storage
    MODEL_CACHE_DIR: str = "models"
    EMBEDDING_CACHE_DIR: str = "embeddings"
    
    # Training Configuration
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 10
    
    # Recommendation Configuration
    TOP_K_RECOMMENDATIONS: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.1
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Global constants
class Constants:
    """Global constants used throughout the application."""
    
    # Data processing
    MIN_PLAYS_THRESHOLD = 5
    MAX_PLAYS_THRESHOLD = 10000
    
    # Model parameters
    NMF_MAX_ITER = 200
    NMF_TOL = 1e-4
    
    # Cache keys
    USER_EMBEDDINGS_KEY = "user_embeddings"
    ITEM_EMBEDDINGS_KEY = "item_embeddings"
    RECOMMENDATIONS_KEY = "recommendations"
    
    # File extensions
    SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".flac", ".m4a"]
    SUPPORTED_DATA_FORMATS = [".csv", ".parquet", ".json"] 