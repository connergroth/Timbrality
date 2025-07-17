"""
Settings Configuration for Tensoe Backend

Comprehensive configuration management for all backend services including
ingestion, ML, database, and external API integrations.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    database_url: str = Field("sqlite:///./test.db", env="DATABASE_URL")
    database_test_url: Optional[str] = Field(None, env="DATABASE_TEST_URL")
    supabase_url: Optional[str] = Field(None, env="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(None, env="SUPABASE_ANON_KEY")
    
    # Connection pool settings
    pool_size: int = Field(5, env="DB_POOL_SIZE")
    max_overflow: int = Field(10, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    
    class Config:
        extra = "allow"


class SpotifySettings(BaseSettings):
    """Spotify API configuration"""
    client_id: Optional[str] = Field(None, env="SPOTIFY_CLIENT_ID")
    client_secret: Optional[str] = Field(None, env="SPOTIFY_CLIENT_SECRET")
    redirect_uri: str = Field("http://localhost:3000", env="SPOTIFY_REDIRECT_URI")
    
    # API settings
    request_timeout: int = Field(30, env="SPOTIFY_TIMEOUT")
    max_retries: int = Field(3, env="SPOTIFY_MAX_RETRIES")
    rate_limit_delay: float = Field(0.1, env="SPOTIFY_RATE_LIMIT_DELAY")
    
    class Config:
        extra = "allow"


class LastFMSettings(BaseSettings):
    """Last.fm API configuration"""
    api_key: Optional[str] = Field(None, env="LASTFM_API_KEY")
    api_secret: Optional[str] = Field(None, env="LASTFM_API_SECRET")
    api_url: str = Field("http://ws.audioscrobbler.com/2.0/", env="LASTFM_API_URL")
    
    # API settings
    request_timeout: int = Field(30, env="LASTFM_TIMEOUT")
    max_retries: int = Field(3, env="LASTFM_MAX_RETRIES")
    rate_limit_delay: float = Field(0.2, env="LASTFM_RATE_LIMIT_DELAY")
    
    class Config:
        extra = "allow"


class AOTYSettings(BaseSettings):
    """Album of the Year (AOTY) API configuration"""
    api_url: Optional[str] = Field(None, env="AOTY_API_URL")
    
    # Scraping settings
    request_timeout: int = Field(30, env="AOTY_TIMEOUT")
    max_retries: int = Field(3, env="AOTY_MAX_RETRIES")
    rate_limit_delay: float = Field(1.0, env="AOTY_RATE_LIMIT_DELAY")
    
    class Config:
        extra = "allow"


class CacheSettings(BaseSettings):
    """Cache configuration"""
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    redis_rest_token: Optional[str] = Field(None, env="REDIS_REST_TOKEN")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Cache TTL settings (in seconds)
    default_ttl: int = Field(3600, env="CACHE_DEFAULT_TTL")  # 1 hour
    album_ttl: int = Field(7200, env="CACHE_ALBUM_TTL")      # 2 hours
    user_ttl: int = Field(1800, env="CACHE_USER_TTL")        # 30 minutes
    track_ttl: int = Field(3600, env="CACHE_TRACK_TTL")      # 1 hour
    
    class Config:
        extra = "allow"


class IngestionSettings(BaseSettings):
    """Data ingestion configuration"""
    batch_size: int = Field(50, env="INGESTION_BATCH_SIZE")
    max_concurrent: int = Field(10, env="INGESTION_MAX_CONCURRENT")
    retry_attempts: int = Field(3, env="INGESTION_RETRY_ATTEMPTS")
    retry_delay: float = Field(1.0, env="INGESTION_RETRY_DELAY")
    
    # Processing limits
    max_tracks_per_album: int = Field(100, env="MAX_TRACKS_PER_ALBUM")
    max_genres_per_track: int = Field(10, env="MAX_GENRES_PER_TRACK")
    max_moods_per_track: int = Field(15, env="MAX_MOODS_PER_TRACK")
    
    # Data quality thresholds
    min_track_duration_ms: int = Field(10000, env="MIN_TRACK_DURATION_MS")  # 10 seconds
    max_track_duration_ms: int = Field(1800000, env="MAX_TRACK_DURATION_MS")  # 30 minutes
    
    class Config:
        extra = "allow"


class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    training_data_limit: int = Field(50000, env="ML_TRAINING_DATA_LIMIT")
    feature_matrix_limit: int = Field(20000, env="ML_FEATURE_MATRIX_LIMIT")
    min_features_per_track: int = Field(3, env="ML_MIN_FEATURES_PER_TRACK")
    
    # Model training settings
    validation_split: float = Field(0.2, env="ML_VALIDATION_SPLIT")
    test_split: float = Field(0.1, env="ML_TEST_SPLIT")
    random_seed: int = Field(42, env="ML_RANDOM_SEED")
    
    # Export settings
    export_directory: str = Field("./exports", env="ML_EXPORT_DIRECTORY")
    
    class Config:
        extra = "allow"


class APISettings(BaseSettings):
    """API server configuration"""
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Rate limiting
    rate_limit_requests: int = Field(30, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: str = Field("1/minute", env="RATE_LIMIT_PERIOD")
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    
    # Security
    api_key: Optional[str] = Field(None, env="API_KEY")
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        extra = "allow"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # File logging
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    max_file_size: int = Field(10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    class Config:
        extra = "allow"


class ScrapingSettings(BaseSettings):
    """Web scraping configuration"""
    browser_timeout: int = Field(30000, env="BROWSER_TIMEOUT")  # 30 seconds
    page_timeout: int = Field(60000, env="PAGE_TIMEOUT")        # 60 seconds
    max_browser_instances: int = Field(3, env="MAX_BROWSER_INSTANCES")
    
    # User agent settings
    user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="USER_AGENT"
    )
    
    # Rate limiting for scraping
    scraping_delay: float = Field(1.0, env="SCRAPING_DELAY")
    max_scraping_retries: int = Field(3, env="MAX_SCRAPING_RETRIES")
    
    class Config:
        extra = "allow"


class Settings(BaseSettings):
    """Main settings class combining all configuration sections"""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    spotify: SpotifySettings = SpotifySettings()
    lastfm: LastFMSettings = LastFMSettings()
    aoty: AOTYSettings = AOTYSettings()
    cache: CacheSettings = CacheSettings()
    ingestion: IngestionSettings = IngestionSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    scraping: ScrapingSettings = ScrapingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() in ["development", "dev"]
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ["production", "prod"]
    
    @property
    def is_testing(self) -> bool:
        return self.environment.lower() in ["testing", "test"]
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.cache.redis_url:
            return self.cache.redis_url
        
        auth = f":{self.cache.redis_password}@" if self.cache.redis_password else ""
        return f"redis://{auth}{self.cache.redis_host}:{self.cache.redis_port}/{self.cache.redis_db}"
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy"""
        return {
            "pool_size": self.database.pool_size,
            "max_overflow": self.database.max_overflow,
            "pool_timeout": self.database.pool_timeout,
            "pool_recycle": self.database.pool_recycle,
            "echo": self.is_development,
        }
    
    def validate_required_settings(self):
        """Validate that all required settings are present"""
        warnings = []
        
        # Check Spotify
        if not self.spotify.client_id:
            warnings.append("SPOTIFY_CLIENT_ID not set - ingestion will not work")
        if not self.spotify.client_secret:
            warnings.append("SPOTIFY_CLIENT_SECRET not set - ingestion will not work")
        
        # Check Last.fm
        if not self.lastfm.api_key:
            warnings.append("LASTFM_API_KEY not set - mood/genre enrichment will not work")
        
        # Check Supabase (if using)
        if self.database.supabase_url and not self.database.supabase_anon_key:
            warnings.append("SUPABASE_ANON_KEY is required when SUPABASE_URL is set")
        
        # Print warnings but don't fail
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
            print("💡 Server will start but some features may not work without API keys")
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information for documentation"""
        return {
            "title": "Tensoe Backend API",
            "description": "Complete music discovery and ML platform",
            "version": "1.0.0",
            "environment": self.environment,
            "debug": self.api.debug,
        }


# Global settings instance
try:
    settings = Settings()
    settings.validate_required_settings()
except Exception as e:
    print(f"⚠️  Configuration warning: {e}")
    print("💡 Using default settings - server will start but some features may not work")
    # Create a minimal settings instance with defaults
    try:
        settings = Settings()
    except Exception:
        # If even defaults fail, create a basic settings object
        import types
        settings = types.SimpleNamespace()
        settings.database = types.SimpleNamespace()
        settings.database.database_url = "sqlite:///./test.db"
        settings.is_development = True
        settings.is_production = False
        settings.is_testing = False


# Helper functions for easy access
def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def get_database_url() -> str:
    """Get database connection URL"""
    return settings.database.database_url


def get_redis_url() -> str:
    """Get Redis connection URL"""
    return settings.get_redis_url()


def is_development() -> bool:
    """Check if running in development mode"""
    return settings.is_development


def is_production() -> bool:
    """Check if running in production mode"""
    return settings.is_production 