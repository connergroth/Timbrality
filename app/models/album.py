from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base


class Album(Base):
    __tablename__ = "albums"
    id = Column(Integer, primary_key=True, index=False) # Unique album ID
    title = Column(String, nullable=False) # Album title
    artist = Column(String, nullable=False) # Artist name
    release_date = Column(DateTime, nullable=True) # Release date
    genre = Column(String, nullable=True) # Genre
    aoty_score = Column(Integer, nullable=True) # AOTY User Score

