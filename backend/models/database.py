from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..config.settings import get_settings

# Get settings
settings = get_settings()

# Create the database engine with configuration from settings
engine = create_engine(
    settings.database.database_url,
    **settings.get_database_config()
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
class Base(DeclarativeBase):
    pass


# Dependency for getting database sessions
def get_db():
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    