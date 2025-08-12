"""
Enhanced Pydantic Data Models for Timbre Ingestion Pipeline
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import hashlib
import uuid


class SongCore(BaseModel):
    """Core song data from any source"""
    artist: str
    title: str
    isrc: Optional[str] = None
    spotify_id: Optional[str] = None
    playcount: Optional[int] = None
    source: str = "unknown"
    canonical_id: Optional[str] = None
    
    @validator('artist', 'title', pre=True)
    def normalize_text(cls, v):
        """Normalize text fields - lowercase, strip, remove accents"""
        if v:
            import unicodedata
            # Remove accents and normalize
            normalized = unicodedata.normalize('NFD', v.lower().strip())
            return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return v


class SpotifyAttrs(BaseModel):
    """Spotify-specific track attributes (basic metadata only)"""
    track_id: Optional[str] = None  # Spotify track ID
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    album_id: Optional[str] = None
    artist_id: Optional[str] = None
    album_name: Optional[str] = None
    release_date: Optional[str] = None
    explicit: Optional[bool] = None
    track_number: Optional[int] = None


class AotyAttrs(BaseModel):
    """Album of the Year attributes"""
    user_score: Optional[float] = None
    rating_count: Optional[int] = None
    tags: Optional[List[str]] = None
    genres: Optional[List[str]] = None
    album_url: Optional[str] = None
    album_title: Optional[str] = None
    
    @validator('tags', 'genres', pre=True)
    def limit_list_size(cls, v):
        """Limit tags and genres to reasonable size"""
        if v and isinstance(v, list):
            return v[:10]  # Max 10 items
        return v


class LastfmStats(BaseModel):
    """Last.fm user statistics"""
    playcount: int
    user_loved: Optional[bool] = None
    tags: Optional[List[str]] = None
    
    @validator('tags', pre=True)
    def limit_tags(cls, v):
        """Limit tags to reasonable size"""
        if v and isinstance(v, list):
            return v[:15]  # Max 15 tags
        return v


class EnhancedSong(BaseModel):
    """Complete song data with all enrichments"""
    # Core data
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    artist: str
    title: str
    canonical_id: str
    
    # Identifiers
    isrc: Optional[str] = None
    spotify_id: Optional[str] = None
    
    # Enriched data
    spotify_attrs: Optional[SpotifyAttrs] = None
    aoty_attrs: Optional[AotyAttrs] = None
    lastfm_stats: Optional[LastfmStats] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# Database table models for PostgreSQL schema

class DBSong(BaseModel):
    """Database model for songs table"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    artist: str
    title: str
    canonical_id: str
    isrc: Optional[str] = None
    spotify_id: Optional[str] = None


class DBSpotifyAttrs(BaseModel):
    """Database model for spotify_attrs table"""
    song_id: str
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    album_id: Optional[str] = None
    artist_id: Optional[str] = None
    album_name: Optional[str] = None
    release_date: Optional[str] = None
    explicit: Optional[bool] = None
    track_number: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


class DBLastfmStats(BaseModel):
    """Database model for lastfm_stats table"""
    song_id: str
    playcount: int
    user_loved: Optional[bool] = None
    tags: Optional[Dict[str, Any]] = None  # JSON field
    pulled_at: datetime = Field(default_factory=datetime.now)


class DBAotyAttrs(BaseModel):
    """Database model for aoty_attrs table"""
    song_id: str
    user_score: Optional[float] = None
    rating_count: Optional[int] = None
    tags: Optional[Dict[str, Any]] = None  # JSON field
    genres: Optional[Dict[str, Any]] = None  # JSON field
    album_url: Optional[str] = None
    album_title: Optional[str] = None
    pulled_at: datetime = Field(default_factory=datetime.now)


# Utility functions for canonical ID generation

def make_canonical_id(song: SongCore, spotify: Optional[SpotifyAttrs] = None) -> str:
    """
    Generate canonical ID using hierarchy: isrc > spotify_id > hash(artist::title)
    
    Args:
        song: Core song data
        spotify: Optional Spotify attributes
        
    Returns:
        Canonical ID string
    """
    # Priority 1: ISRC (International Standard Recording Code)
    if song.isrc:
        return f"isrc:{song.isrc}"
    
    # Priority 2: Spotify ID
    if song.spotify_id:
        return f"spotify:{song.spotify_id}"
    elif spotify and spotify.artist_id:
        return f"spotify:{spotify.artist_id}"
    
    # Priority 3: Hash of normalized artist::title
    normalized_key = f"{song.artist}::{song.title}".lower().strip()
    hash_value = hashlib.sha256(normalized_key.encode()).hexdigest()[:16]
    return f"hash:{hash_value}"


# Validation schemas

class IngestionRequest(BaseModel):
    """Request model for ingestion pipeline"""
    user_id: Optional[str] = None
    max_tracks: int = Field(default=10000, ge=1, le=50000)
    include_recent: bool = Field(default=True)
    skip_aoty: bool = Field(default=False)
    dry_run: bool = Field(default=False)


class IngestionResult(BaseModel):
    """Result model for ingestion pipeline"""
    success: bool
    total_processed: int
    successful_inserts: int
    failed_inserts: int
    processing_time_seconds: float
    errors: List[str] = []
    coverage_stats: Dict[str, float] = {}


class PipelineStats(BaseModel):
    """Statistics for pipeline execution"""
    lastfm_tracks_pulled: int = 0
    spotify_enrichments: int = 0
    aoty_enrichments: int = 0
    songs_deduplicated: int = 0
    final_songs_count: int = 0
    spotify_coverage_pct: float = 0.0
    aoty_coverage_pct: float = 0.0
    processing_time_seconds: float = 0.0