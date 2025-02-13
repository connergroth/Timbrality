from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy.dialects.postgresql import JSON


class Track(Base):
    __tablename__ = "tracks"
    id = Column(String, primary_key=True, index=True) # Unique track ID
    title = Column(String, nullable=False) # Track title
    artist_id = Column(String, ForeignKey("artists.id"), nullable=False)  # Artist reference
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)  # Album reference
    duration_ms = Column(Integer, nullable=False) # Duration of each track
    popularity = Column(Integer, nullable=True) # Popularity value
    aoty_score = Column(Integer, nullable=True) # AOTY User Score
    play_count = Column(Integer, nullable=True) # Last.fm playcount 
    audio_features = Column(JSON, nullable=False) # JSON field for audio features (tempo, energy, etc.)
    preview_url = Column(String, nullable=True)  # Spotify preview URL
    cover_url = Column(String, nullable=True) # URL to song/album cover image
    source = Column(String, nullable=False) # Spotify/Last.fm
    


    
