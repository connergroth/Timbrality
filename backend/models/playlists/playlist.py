from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.database import Base

class Playlist(Base):
    __tablename__ = "playlists"

    id = Column(String, primary_key=True, index=True)  # Unique playlist ID
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)  # Playlist owner
    name = Column(String, nullable=False)  # Playlist name
    cover_url = Column(String, nullable=True)  # URL to playlist cover image
    created_at = Column(DateTime, server_default=func.now())  # Timestamp when playlist was created
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())  # Last modified timestamp
