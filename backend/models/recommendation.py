from sqlalchemy import Column, String, Integer, ForeignKey, Float, DateTime, CheckConstraint
from sqlalchemy.sql import func
from app.models.database import Base

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)  # Unique Recommendation ID
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)  # Links to user
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False)  # Links to track
    album_id = Column(String, ForeignKey("albums.id", ondelete="CASCADE"), nullable=True)  # Links to album
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), nullable=False)  # Links to artist
    recommendation_score = Column(Float, nullable=False, index=True)  # Use Float for accuracy
    method = Column(String, nullable=False, default="collaborative")  # Method used to generate recommendation
    source = Column(String, nullable=False)  # Source of the recommendation (Spotify, Last.fm, Hybrid)
    created_at = Column(DateTime, server_default=func.now())  # Timestamp when recommendation was made

    __table_args__ = (
        CheckConstraint("method IN ('collaborative', 'content-based', 'hybrid')", name="check_recommendation_method"),
        CheckConstraint("source IN ('spotify', 'lastfm', 'hybrid')", name="check_recommendation_source"),
    )
