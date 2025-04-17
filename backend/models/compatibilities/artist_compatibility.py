from sqlalchemy import Column, Integer, String, ForeignKey, Float, UniqueConstraint
from app.models.database import Base

class ArtistCompatibility(Base):
    __tablename__ = "artist_compatibilities"

    user_id_1 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user_id_2 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    artist_id = Column(String, ForeignKey("artists.id", ondelete="CASCADE"), primary_key=True)
    compatibility_score = Column(Float, nullable=False)  # Use Float for more precision

    __table_args__ = (
        UniqueConstraint("user_id_1", "user_id_2", "artist_id", name="uq_artist_compatibility"),
    )
