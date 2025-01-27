from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base


class Artist(Base):
    __tablename__ = "artists"
    id = Column(Integer, primary_key=True, index=True)  # Unique artist ID
    name = Column(String, nullable=False)  # Artist name
    genre = Column(String, nullable=True)  # Main genre of the artist
    popularity = Column(Integer, nullable=True)  # Popularity score
    aoty_score = Column(Integer, nullable=True) # AOTY User Score