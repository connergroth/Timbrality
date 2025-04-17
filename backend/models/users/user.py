from sqlalchemy import Column, String, Integer, JSON, DateTime
from sqlalchemy.sql import func
from app.models.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique user ID
    spotify_username = Column(String, unique=True, nullable=True)  # Store Spotify username
    lastfm_username = Column(String, unique=True, nullable=True)  # Store Last.fm username
    aoty_username = Column(String, unique=True, nullable=True)  # Store AOTY username
    email = Column(String, unique=True, nullable=False)  # Unique email
    password_hash = Column(String, nullable=False)  # Hashed password
    created_at = Column(DateTime, server_default=func.now())  # Account created timestamp
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())  # Last time user was active
    scrobbles = Column(Integer, nullable=True)  # Last.fm scrobbles
    profile_image_url = Column(String, nullable=True)  # Profile picture from Spotify/Last.fm/AOTY
    preferences = Column(JSON, nullable=True)  # Store user preferences (theme, genre prefs, etc.)
