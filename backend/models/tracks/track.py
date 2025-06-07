from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, func
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from ..database import Base

class Track(Base):
    __tablename__ = "tracks"

    # Core identification
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    artist = Column(String, nullable=False)  # Keep as string for compatibility
    album = Column(String, nullable=True)    # Keep as string for compatibility
    
    # Legacy single genre (for backward compatibility)
    genre = Column(String, nullable=True)
    
    # Enhanced multi-genre and mood support
    genres = Column(ARRAY(String), nullable=True, default=[])    # Multiple genres
    moods = Column(ARRAY(String), nullable=True, default=[])     # Mood tags from Last.fm
    
    # Audio metadata
    duration_ms = Column(Integer, nullable=True)
    popularity = Column(Integer, nullable=True)
    explicit = Column(Boolean, default=False)
    track_number = Column(Integer, nullable=True)
    album_total_tracks = Column(Integer, nullable=True)
    
    # Release information
    release_date = Column(Date, nullable=True)
    
    # Scoring and features
    aoty_score = Column(Integer, nullable=True)  # Keep as integer for compatibility
    audio_features = Column(JSON, nullable=False, default={})  # Spotify audio features
    
    # URLs and metadata
    spotify_url = Column(String, nullable=True)
    cover_url = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

