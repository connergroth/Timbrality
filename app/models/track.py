from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy.dialects.postgresql import JSON


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True, index=True) # Unique track ID
    title = Column(String, nullable=False) # Track title
    artist = Column(String, nullable=False) # Artist name
    album = Column(String, nullable=True) # Album name
    genre = Column(String, nullable=True) # Genre
    popularity = Column(Integer, nullable=True) # Popularity value
    aoty_score = Column(Integer, nullable=True) # AOTY User Score
    audio_features = Column(JSON, nullable=False) # JSON field for audio features (tempo, energy, etc.)
    

    
