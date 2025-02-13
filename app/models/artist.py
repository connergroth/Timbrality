from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from app.models.database import Base


class Artist(Base):
    __tablename__ = "artists"
    id = Column(String, primary_key=True, index=True)  # Unique artist ID
    name = Column(String, nullable=False)  # Artist name
    genres = Column(JSON, nullable=True)  # Main genre of the artist
    popularity = Column(Integer, nullable=True)  # Popularity score
    followers = Column(Integer, nullable=True) # Artist's follower count
    aoty_score = Column(Integer, nullable=True) # AOTY User Score
    play_count = Column(Integer, nullable=True) # Last.fm playcount 
    profile_picture = Column(String, nullable=True) # Artist profile picture
    source = Column(String, nullable=False) # Spotify/Last.fm