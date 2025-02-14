from sqlalchemy import Column, Integer, String, ForeignKey, CheckConstraint
from app.models.database import Base

class UserArtist(Base):
    __tablename__ = "user_artists"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), primary_key=True)
    play_count = Column(Integer, nullable=True, default=0)  
    source = Column(String, nullable=False)  # 'spotify' or 'lastfm'

    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 100 OR rating IS NULL", name="check_artist_rating_range"),
    )
