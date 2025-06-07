from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, func
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from .database import Base

class Album(Base):
    __tablename__ = "albums"

    # Core identification
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    artist = Column(String, nullable=False)  # Keep as string for compatibility
    
    # Legacy single genre (for backward compatibility)
    genre = Column(String, nullable=True)
    
    # Enhanced multi-genre support
    genres = Column(ARRAY(String), nullable=True, default=[])    # Multiple genres from AOTY
    
    # Album metadata
    release_date = Column(Date, nullable=True)
    duration_ms = Column(Integer, nullable=True)        # Total album duration
    total_tracks = Column(Integer, nullable=True)
    explicit = Column(Boolean, default=False)           # Has explicit content
    
    # Scoring
    aoty_score = Column(Integer, nullable=True)  # Keep as integer for compatibility
    
    # URLs and metadata
    spotify_url = Column(String, nullable=True)
    cover_url = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
