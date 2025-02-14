from sqlalchemy import Column, Integer, String, ForeignKey, CheckConstraint
from app.models.database import Base

class UserAlbum(Base):
    __tablename__ = "user_albums"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    album_id = Column(String, ForeignKey("albums.id", ondelete="CASCADE"), primary_key=True)
    rating = Column(Float, nullable=True)  
    source = Column(String, nullable=False)  # 'spotify' or 'lastfm'

    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 100", name="check_rating_range"),
    )
