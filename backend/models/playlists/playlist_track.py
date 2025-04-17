from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.database import Base

class PlaylistTrack(Base):
    __tablename__ = "playlist_tracks"

    playlist_id = Column(String, ForeignKey("playlists.id", ondelete="CASCADE"), primary_key=True)
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    added_at = Column(DateTime, server_default=func.now())  # Timestamp when track was added
