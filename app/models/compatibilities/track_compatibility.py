from sqlalchemy import Column, Integer, String, ForeignKey, Float, UniqueConstraint
from app.models.database import Base

class TrackCompatibility(Base):
    __tablename__ = "track_compatibilities"

    user_id_1 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user_id_2 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    track_id = Column(String, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    compatibility_score = Column(Float, nullable=False)  # Use Float for more precision

    __table_args__ = (
        UniqueConstraint("user_id_1", "user_id_2", "track_id", name="uq_track_compatibility"),
    )
