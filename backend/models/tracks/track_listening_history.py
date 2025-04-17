from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, UniqueConstraint, func
from app.models.database import Base

class TrackListeningHistory(Base):
    __tablename__ = "track_listening_histories"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    play_count = Column(Integer, nullable=False, default=1)  # Tracks how many times a song was played
    skipped = Column(Boolean, nullable=False, default=False)  # Tracks if the song was skipped
    liked = Column(Boolean, nullable=False, default=False)  # Tracks if the user liked the song
    last_played = Column(DateTime, server_default=func.now())  # Last time the user played this track

    __table_args__ = (
        UniqueConstraint("user_id", "track_id", name="uq_user_track"),
    )
