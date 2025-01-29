from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class ArtistCompatibility(Base):
    __tablename__ = "artist_compatibilities"
    id = Column(Integer, primary_key=True, index=True) # Artist compatibility ID
    user_id_1 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to first user
    user_id_2 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to second user
    artist_id = Column(Integer, ForeignKey("artists.id"), nullable=False) # Links to an artist
    compatibility_score = Column(Integer, nullable=False) # Compatibility score for a given artist

