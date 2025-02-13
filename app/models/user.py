from sqlalchemy import Column, String, Integer, JSON, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy.ext.declarative import declarative_base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True) # Unique user ID
    spotify_username = Column(String, unique=True, nullable=True)  # Store Spotify username
    lastfm_username = Column(String, unique=True, nullable=True)  # Store Spotify username
    email = Column(String, unique=True, nullable=False) # Unique email
    password_hash = Column(String, nullable=False) # Hashed password
    created_at = Column(DateTime, server_default=func.now()) # Account created timestamp
    scrobbles = Column(Integer, nullable=True) # Last.fm scrobbles
    top_artists = Column(JSON, nullable=True) # User's top artists
    top_albums = Column(JSON, nullable=True) # User's top albums
    top_tracks = Column(JSON, nullable=True) # User's top tracks
    source = Column(String, nullable=False) # Spotify/Last.fm