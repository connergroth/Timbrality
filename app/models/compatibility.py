from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.database import Base
from sqlalchemy import ForeignKey


class Compatibility(Base):
    __tablename__ = "compatibilities"
    id = Column(Integer, primary_key=True, index=True) # Unique compatibility ID
    user_id_1 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to first user
    user_id_2 = Column(Integer, ForeignKey("users.id"), nullable=False) # Links to second user
    compatibility_score = Column(Integer, nullable=False) # Compatibility score for the two users for a given track