from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.dialects.postgresql import ARRAY
from .database import Base

class Artist(Base):
    __tablename__ = "artists"

    # Core identification
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    
    # Legacy single genre (for backward compatibility)
    genre = Column(String, nullable=True)
    
    # Enhanced multi-genre support
    genres = Column(ARRAY(String), nullable=True, default=[])    # Multiple genres from Last.fm/AOTY
    
    # Artist metrics
    popularity = Column(Integer, nullable=True)         # Spotify popularity score
    aoty_score = Column(Integer, nullable=True)         # AOTY score (keep as integer)
    
    # URLs and metadata
    image_url = Column(String, nullable=True)           # Artist profile picture
    spotify_url = Column(String, nullable=True)         # Spotify artist URL
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
