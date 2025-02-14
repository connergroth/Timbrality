from sqlalchemy import Column, Integer, String, ForeignKey, CheckConstraint
from app.models.database import Base

class UserTrack(Base):
    __tablename__ = "user_tracks"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    play_count = Column(Integer, nullable=True, default=0) 
    rating = Column(Float, nullable=True)  
    source = Column(String, nullable=False)  # 'spotify' or 'lastfm'

    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 100 OR rating IS NULL", name="check_track_rating_range"),
    )
