from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.database import Base


class Album(Base):
    __tablename__ = "albums"
    id = Column(String, primary_key=True, index=False) # Unique album ID
    title = Column(String, nullable=False) # Album title
    artist_id = Column(String, ForeignKey("artists.id"), nullable=False)  # Artist reference
    release_date = Column(DateTime, nullable=True) # Release date
    total_tracks = Column(Integer, nullable=True) # Total number of tracks
    aoty_score = Column(Integer, nullable=True) # AOTY User Score
    play_count = Column(Integer, nullable=True) # Last.fm playcount 
    cover_url = Column(String, nullable=True) # URL to album cover image
    source = Column(String, nullable=False) # Spotify/Last.fm
