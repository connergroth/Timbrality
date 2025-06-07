"""
Enhanced Models for Tensoe Ingestion Pipeline

These models extend the base models with additional fields needed for ML training
and include comprehensive metadata from Spotify, Last.fm, and AOTY.
"""

from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, Float, func
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from sqlalchemy.orm import relationship
from .database import Base
from typing import Dict, List, Optional
from pydantic import BaseModel


class EnhancedTrack(Base):
    """Enhanced track model for ML training with rich metadata"""
    __tablename__ = "tracks"

    # Core identification
    id = Column(String, primary_key=True, index=True)               # Spotify track ID
    title = Column(String, nullable=False, index=True)              # Track title
    artist = Column(String, nullable=False, index=True)             # Artist name
    album = Column(String, nullable=True, index=True)               # Album name
    
    # Legacy compatibility
    genre = Column(String, nullable=True)                           # Single genre (backward compatibility)
    
    # Enhanced genre and mood classification
    genres = Column(ARRAY(String), nullable=True, default=[])       # Multiple genres from AOTY
    moods = Column(ARRAY(String), nullable=True, default=[])        # Mood tags from Last.fm
    
    # Audio characteristics
    duration_ms = Column(Integer, nullable=True)                    # Track duration in milliseconds
    popularity = Column(Integer, nullable=True)                     # Spotify popularity (0-100)
    explicit = Column(Boolean, default=False)                       # Explicit content flag
    track_number = Column(Integer, nullable=True)                   # Track number in album
    album_total_tracks = Column(Integer, nullable=True)             # Total tracks in album
    
    # Release information
    release_date = Column(Date, nullable=True, index=True)          # Release date
    
    # Rating and scoring
    aoty_score = Column(Integer, nullable=True)                     # AOTY rating (0-100)
    
    # Audio features from Spotify
    audio_features = Column(JSON, nullable=False, default={})       # Spotify audio analysis
    
    # URLs and media
    spotify_url = Column(String, nullable=True)                     # Spotify track URL
    cover_url = Column(String, nullable=True)                       # Album cover image URL
    
    # Metadata timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class EnhancedAlbum(Base):
    """Enhanced album model with comprehensive metadata"""
    __tablename__ = "albums"

    # Core identification
    id = Column(String, primary_key=True, index=True)               # Spotify album ID
    title = Column(String, nullable=False, index=True)              # Album title
    artist = Column(String, nullable=False, index=True)             # Artist name
    
    # Legacy compatibility
    genre = Column(String, nullable=True)                           # Single genre (backward compatibility)
    
    # Enhanced classification
    genres = Column(ARRAY(String), nullable=True, default=[])       # Multiple genres from AOTY
    
    # Album metadata
    release_date = Column(Date, nullable=True, index=True)          # Release date
    duration_ms = Column(Integer, nullable=True)                    # Total album duration
    total_tracks = Column(Integer, nullable=True)                   # Number of tracks
    explicit = Column(Boolean, default=False)                       # Has explicit content
    
    # Rating and scoring
    aoty_score = Column(Integer, nullable=True)                     # AOTY album rating
    
    # URLs and media
    spotify_url = Column(String, nullable=True)                     # Spotify album URL
    cover_url = Column(String, nullable=True)                       # Album cover image URL
    
    # Metadata timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class EnhancedArtist(Base):
    """Enhanced artist model with genre classification and metadata"""
    __tablename__ = "artists"

    # Core identification
    id = Column(String, primary_key=True, index=True)               # Spotify artist ID
    name = Column(String, nullable=False, index=True)               # Artist name
    
    # Legacy compatibility
    genre = Column(String, nullable=True)                           # Single genre (backward compatibility)
    
    # Enhanced classification
    genres = Column(ARRAY(String), nullable=True, default=[])       # Multiple genres from Last.fm/AOTY
    
    # Artist metrics
    popularity = Column(Integer, nullable=True)                     # Spotify popularity
    aoty_score = Column(Integer, nullable=True)                     # AOTY artist rating
    
    # URLs and media
    spotify_url = Column(String, nullable=True)                     # Spotify artist URL
    image_url = Column(String, nullable=True)                       # Artist profile picture
    
    # Metadata timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# Pydantic models for API serialization
class TrackResponse(BaseModel):
    """Pydantic model for track API responses"""
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    genre: Optional[str] = None
    genres: List[str] = []
    moods: List[str] = []
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    explicit: bool = False
    track_number: Optional[int] = None
    album_total_tracks: Optional[int] = None
    release_date: Optional[str] = None
    aoty_score: Optional[int] = None
    audio_features: Dict = {}
    spotify_url: Optional[str] = None
    cover_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class AlbumResponse(BaseModel):
    """Pydantic model for album API responses"""
    id: str
    title: str
    artist: str
    genre: Optional[str] = None
    genres: List[str] = []
    release_date: Optional[str] = None
    duration_ms: Optional[int] = None
    total_tracks: Optional[int] = None
    explicit: bool = False
    aoty_score: Optional[int] = None
    spotify_url: Optional[str] = None
    cover_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class ArtistResponse(BaseModel):
    """Pydantic model for artist API responses"""
    id: str
    name: str
    genre: Optional[str] = None
    genres: List[str] = []
    popularity: Optional[int] = None
    aoty_score: Optional[int] = None
    spotify_url: Optional[str] = None
    image_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class IngestionStats(BaseModel):
    """Model for tracking ingestion statistics"""
    total_tracks: int = 0
    total_albums: int = 0
    total_artists: int = 0
    tracks_with_genres: int = 0
    tracks_with_moods: int = 0
    tracks_with_aoty_scores: int = 0
    tracks_with_audio_features: int = 0
    average_genres_per_track: float = 0.0
    average_moods_per_track: float = 0.0
    latest_ingestion: Optional[str] = None


class MLTrainingData(BaseModel):
    """Structured data format for ML training"""
    # Core features
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None
    
    # Categorical features
    genres: List[str] = []
    moods: List[str] = []
    explicit: bool = False
    
    # Numerical features
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    track_number: Optional[int] = None
    album_total_tracks: Optional[int] = None
    aoty_score: Optional[int] = None
    release_year: Optional[int] = None
    
    # Audio features (if available)
    audio_features: Dict = {}
    
    # Target variables (for supervised learning)
    target_score: Optional[float] = None  # Could be aoty_score or derived metric


def get_ml_training_data(tracks: List[EnhancedTrack]) -> List[MLTrainingData]:
    """Convert database tracks to ML training format"""
    training_data = []
    
    for track in tracks:
        # Extract release year from date
        release_year = None
        if track.release_date:
            release_year = track.release_date.year
        
        # Use aoty_score as target if available
        target_score = float(track.aoty_score) if track.aoty_score else None
        
        ml_data = MLTrainingData(
            track_id=track.id,
            title=track.title,
            artist=track.artist,
            album=track.album,
            genres=track.genres or [],
            moods=track.moods or [],
            explicit=track.explicit,
            duration_ms=track.duration_ms,
            popularity=track.popularity,
            track_number=track.track_number,
            album_total_tracks=track.album_total_tracks,
            aoty_score=track.aoty_score,
            release_year=release_year,
            audio_features=track.audio_features or {},
            target_score=target_score
        )
        
        training_data.append(ml_data)
    
    return training_data 