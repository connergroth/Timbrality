from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.dialects.postgresql import JSON
from app.models.database import Base

class Track(Base):
    __tablename__ = "tracks"

    id = Column(String, primary_key=True, index=True)  # Unique track ID
    title = Column(String, nullable=False)  # Track title
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), nullable=False)  # Artist reference
    album_id = Column(String, ForeignKey("albums.id", ondelete="CASCADE"), nullable=False)  # Album reference
    tags = Column(JSON, nullable=True)  # Stores track-level tags like ["synthwave", "lo-fi"]
    duration_ms = Column(Integer, nullable=True)  # Allow NULL for missing duration
    popularity = Column(Integer, nullable=True)  # Popularity value
    aoty_score = Column(Float, nullable=True)  # AOTY User Score (supports decimals)
    play_count = Column(Integer, nullable=True, default=0)  # Play count
    audio_features = Column(JSON, nullable=False)  # JSON field for audio features (tempo, energy, etc.)
    preview_url = Column(String, nullable=True)  # Spotify preview URL
    cover_url = Column(String, nullable=True)  # URL to song/album cover image
    source = Column(String, nullable=False)  # Spotify/Last.fm
