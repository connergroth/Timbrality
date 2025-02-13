from sqlalchemy import Column, Integer, String, UniqueConstraint
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class TrackListeningHistory(Base):
    __tablename__ = "track_listening_histories"
    id = Column(String, primary_key=True, index=True) # Unique listener ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to a user
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False) # Links to a track
    play_count = Column(Integer, nullable=False) # Number of times the user has listened to the track

    __table_args__ = (
        UniqueConstraint("user_id", "track_id", name="uq_user_track"),
    )


    
