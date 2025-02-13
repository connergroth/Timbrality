from sqlalchemy import Column, String, Integer, ForeignKey
from app.models.database import Base

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True) # Unique Recommendation ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to user
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False)  # Links to track
    album_id = Column(String, ForeignKey("albums.id"), nullable=True) # Links to album
    artist_id = Column(String, ForeignKey("artists.id"), nullable=False) # Links to artist
    recommendation_score = Column(Integer, nullable=False, index=True) # Recommendation score
    method = Column(String, nullable=False, default="collaborative")  # Stores recommendation method used
    source = Column(String, nullable=False)  # 'spotify', 'lastfm' or 'hybrid'
