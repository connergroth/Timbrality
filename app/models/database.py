from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgronner34@localhost:5432/sonance_test")

# Create the database engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,         # Number of connections in the pool
    max_overflow=20,      # Connections beyond pool_size that can be created
    pool_timeout=30,      # Timeout in seconds for connections
    pool_recycle=1800     # Recycle connections every 30 minutes
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
class Base(DeclarativeBase):
    pass
    