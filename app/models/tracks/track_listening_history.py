from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from app.models.database import Base

class TrackListeningHistory(Base):
    __tablename__ = "track_listening_histories"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    play_count = Column(Integer, nullable=False, default=1)  # Defaults to 1 on first listen

    __table_args__ = (
        UniqueConstraint("user_id", "track_id", name="uq_user_track"),
    )
