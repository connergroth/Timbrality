from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class TrackCompatibility(Base):
    __tablename__ = "track_compatibilities"
    id = Column(String, primary_key=True, index=True)  # Track Compatibility ID
    user_id_1 = Column(Integer, ForeignKey("users.id"), nullable=False)  # Links to first user
    user_id_2 = Column(Integer, ForeignKey("users.id"), nullable=False)  # Links to second user
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False)  # Links to a track
    compatibility_score = Column(Integer, nullable=False)  # Compatibility score for the track

