from sqlalchemy import Column, Integer, ForeignKey, Float, UniqueConstraint
from app.models.database import Base

class Compatibility(Base):
    __tablename__ = "compatibilities"

    user_id_1 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user_id_2 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    compatibility_score = Column(Float, nullable=False)  # Use Float for more precision

    __table_args__ = (
        UniqueConstraint("user_id_1", "user_id_2", name="uq_user_compatibility"),
    )
