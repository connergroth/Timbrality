from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy.ext.declarative import declarative_base


class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, unique=True, nullable=False)  # Use Last.fm username
    email = Column(String, unique=True, nullable=False) # Unique email
    password_hash = Column(String, nullable=False) # Hashed password
    created_at = Column(DateTime, server_default=func.now()) # Account created timestamp
