from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.dialects.postgresql import JSON
from app.models.database import Base

class Track(Base):
    __tablename__ = "tracks"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), nullable=False)
    album_id = Column(String, ForeignKey("albums.id", ondelete="CASCADE"), nullable=False)
    tags = Column(JSON, nullable=True)  # Genre tags
    duration_ms = Column(Integer, nullable=True)
    popularity = Column(Integer, nullable=True)
    aoty_score = Column(Float, nullable=True)
    play_count = Column(Integer, nullable=True, default=0)

    cover_url = Column(String, nullable=True)

