from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True) # Unique recommendation ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to a user
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False) # Links to a track
    album = Column(Integer, ForeignKey("albums.id"), nullable=False) # Links to an album
    recommendation_score = Column(Integer, nullable=False) # Recommendation score