from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class AlbumCompatability(Base):
    __tablename__ = "album_compatibilities"
    id = Column(String, primary_key=True, index=True) # Album compatibility ID
    user_id_1 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to first user
    user_id_2 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to second user
    album_id = Column(String, ForeignKey("albums.id"), nullable=False) # Links to an album
    compatibility_score = Column(Integer, nullable=False) # Compatibility score for a given album

