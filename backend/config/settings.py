"""
Simple Settings Configuration for Timbre Backend

Uses straightforward environment variable loading with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Main application settings loaded from environment variables."""
    
    def __init__(self):
        # Database
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./test.db")
        
        # External APIs
        self.spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.spotify_redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:3000")
        
        self.lastfm_api_key = os.getenv("LASTFM_API_KEY")
        self.lastfm_api_secret = os.getenv("LASTFM_API_SECRET")
        self.lastfm_api_url = os.getenv("LASTFM_API_URL", "http://ws.audioscrobbler.com/2.0/")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Supabase
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        # Redis Cache
        self.upstash_redis_rest_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.upstash_redis_rest_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        
        # API Server
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        
        # AOTY
        self.aoty_api_url = os.getenv("AOTY_API_URL")
        
        # Print warnings for missing keys
        self._validate_settings()
    
    def _validate_settings(self):
        """Print warnings for missing critical settings."""
        warnings = []
        
        if not self.spotify_client_id:
            warnings.append("SPOTIFY_CLIENT_ID not set - Spotify integration will not work")
        if not self.spotify_client_secret:
            warnings.append("SPOTIFY_CLIENT_SECRET not set - Spotify integration will not work")
        if not self.lastfm_api_key:
            warnings.append("LASTFM_API_KEY not set - Last.fm integration will not work")
        if not self.openai_api_key:
            warnings.append("OPENAI_API_KEY not set - LLM features will use fallback")
        
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
            print("Server will start but some features may not work without API keys")


# Global settings instance
settings = Settings()