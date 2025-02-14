from sqlalchemy import Column, Integer, String, ForeignKey, Float, UniqueConstraint
from app.models.database import Base

class AlbumCompatibility(Base):
    __tablename__ = "album_compatibilities"

    user_id_1 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user_id_2 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    album_id = Column(String, ForeignKey("albums.id", ondelete="CASCADE"), primary_key=True)
    compatibility_score = Column(Float, nullable=False)  # Use Float for more precision

    __table_args__ = (
        UniqueConstraint("user_id_1", "user_id_2", "album_id", name="uq_album_compatibility"),
    )
