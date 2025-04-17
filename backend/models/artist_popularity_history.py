from sqlalchemy import String, Integer, DateTime, ForeignKey
from app.models.database import Base

class ArtistPopularityHistory(Base):
    __tablename__ = "artist_popularity_histories"

    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), primary_key=True)  # Tracks when the popularity was recorded
    popularity_score = Column(Integer, nullable=False)
