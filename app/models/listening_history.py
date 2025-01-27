from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class ListeningHistory(Base):
    __tablename__ = "listening_histories"
    id = Column(Integer, primary_key=True, index=True) # Unique listener ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to a user
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False) # Links to a track
    album = Column(Integer, ForeignKey("albums.id"), nullable=False) # Links to an album
    play_count = Column(Integer, nullable=False) # Number of times the user has listened to the track
    timestamp = Column(DateTime, server_default=func.now()) # Timestamp of the last time the user listened to the track





    
