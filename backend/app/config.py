"""
Enhanced Configuration for Timbre Ingestion Pipeline
Pydantic-based settings with environment variable support
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Main configuration settings"""
    
    # Last.fm API Configuration
    lastfm_api_key: str = Field(..., env="LASTFM_API_KEY")
    lastfm_username: Optional[str] = Field(None, env="LASTFM_USERNAME")
    lastfm_api_secret: Optional[str] = Field(None, env="LASTFM_API_SECRET")
    lastfm_api_url: Optional[str] = Field(None, env="LASTFM_API_URL")
    
    # Spotify API Configuration
    spotify_client_id: str = Field(..., env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str = Field(..., env="SPOTIFY_CLIENT_SECRET")
    spotify_redirect_uri: Optional[str] = Field(None, env="SPOTIFY_REDIRECT_URI")
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_role_key: Optional[str] = Field(None, env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_anon_key: Optional[str] = Field(None, env="SUPABASE_ANON_KEY")
    
    # Database Configuration
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    upstash_redis_rest_url: Optional[str] = Field(None, env="UPSTASH_REDIS_REST_URL")
    upstash_redis_rest_token: Optional[str] = Field(None, env="UPSTASH_REDIS_REST_TOKEN")
    
    # AOTY Scraping Configuration
    aoty_proxy_url: Optional[str] = Field(None, env="AOTY_PROXY_URL")
    aoty_api_url: Optional[str] = Field(None, env="AOTY_API_URL")
    
    # AI/ML Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    perplexity_api_key: Optional[str] = Field(None, env="PERPLEXITY_API_KEY")
    scrape_concurrency: int = Field(1, env="SCRAPE_CONCURRENCY")     # Reduced to 1 (sequential)
    scrape_delay_sec: float = Field(5.0, env="SCRAPE_DELAY_SEC")     # Increased to 5 seconds
    
    # Pipeline Configuration
    max_songs: int = Field(10_000, env="MAX_SONGS")
    
    # Rate Limiting (requests per minute) - Reduced for better stability
    spotify_rate_limit: int = Field(60, env="SPOTIFY_RATE_LIMIT")  # Reduced from 100
    lastfm_rate_limit: int = Field(120, env="LASTFM_RATE_LIMIT")   # Reduced from 200
    aoty_rate_limit: int = Field(20, env="AOTY_RATE_LIMIT")        # Reduced from 30
    
    # Processing Configuration - Reduced concurrency
    batch_size: int = Field(25, env="BATCH_SIZE")                  # Reduced from 50
    max_concurrent_requests: int = Field(5, env="MAX_CONCURRENT_REQUESTS")  # Reduced from 10
    request_timeout: int = Field(45, env="REQUEST_TIMEOUT")        # Increased timeout
    
    # Database Configuration
    db_batch_size: int = Field(2000, env="DB_BATCH_SIZE")
    db_retry_attempts: int = Field(3, env="DB_RETRY_ATTEMPTS")
    db_retry_delay: float = Field(1.0, env="DB_RETRY_DELAY")
    
    # Data Quality Configuration
    max_genres_per_track: int = Field(10, env="MAX_GENRES_PER_TRACK")
    max_moods_per_track: int = Field(15, env="MAX_MOODS_PER_TRACK")
    min_aoty_rating_count: int = Field(5, env="MIN_AOTY_RATING_COUNT")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # File Paths
    data_dir: str = Field("./data", env="DATA_DIR")
    logs_dir: str = Field("./logs", env="LOGS_DIR")
    exports_dir: str = Field("./exports", env="EXPORTS_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()

# Create directories if they don't exist
for directory in [settings.data_dir, settings.logs_dir, settings.exports_dir]:
    os.makedirs(directory, exist_ok=True)