from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy.dialects.postgresql import JSON


class Playlist(Base):
    __tablename__ = "playlists"
    id = Column(String, primary_key=True, index=True) # Unique playlist ID
    name = Column(String, nullable=False)  # Playlist name
    track_ids = Column(JSON, nullable=False)  # List of track IDs in this playlist (JSON field)
    cover_url = Column(String, nullable=True) # URL to playlist cover image
