from sqlalchemy import Column, Integer, String, JSON, Float, CheckConstraint
from app.models.database import Base

class Artist(Base):
    __tablename__ = "artists"

    id = Column(String, primary_key=True, index=True)  # Unique artist ID
    name = Column(String, nullable=False)  # Artist name
    tags = Column(JSON, nullable=True)  # Stores both genres & descriptive tags
    popularity = Column(Integer, nullable=True)  # Popularity score
    followers = Column(Integer, nullable=True)  # Artist's follower count
    aoty_score = Column(Float, nullable=True)  # AOTY User Score 
    play_count = Column(Integer, nullable=True, default=0)  # Play count
    image_url = Column(String, nullable=True)  # Artist profile picture
    source = Column(String, nullable=False)  # 'spotify' or 'lastfm'

    __table_args__ = (
        CheckConstraint("source IN ('spotify', 'lastfm')", name="check_artist_source"),
    )
