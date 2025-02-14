from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date, JSON
from app.models.database import Base

class Album(Base):
    __tablename__ = "albums"

    id = Column(String, primary_key=True, index=False)  # Unique album ID
    title = Column(String, nullable=False)  # Album title
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), nullable=False)  # Artist reference
    tags = Column(JSON, nullable=True)  # Stores album-level tags like ["progressive rock", "experimental"]
    release_date = Column(Date, nullable=True)  # Release date (only Date, not DateTime)
    total_tracks = Column(Integer, nullable=True)  # Total number of tracks
    aoty_score = Column(Float, nullable=True)  # AOTY User Score (supports decimals)
    play_count = Column(Integer, nullable=True, default=0)  # Play count
    cover_url = Column(String, nullable=True)  # URL to album cover image
