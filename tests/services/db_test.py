import pytest
from sqlalchemy.orm import Session
from app.models.database import Base, engine, SessionLocal
from app.models.album import Album

# Fixture to provide a clean test database
@pytest.fixture(scope="function")
def test_db():
    # Create the test database schema
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()
    # Drop the test database schema
    Base.metadata.drop_all(bind=engine)

# Test inserting an album into the database
def test_insert_album(test_db: Session):
    album = Album(
        title="Test Album",
        artist="Test Artist",
        release_date="2023-01-01",
        genre="Pop",
        aoty_score=85,
        cover_url="http://example.com/cover.jpg"
    )

    test_db.add(album)
    test_db.commit()
    test_db.refresh(album)

    # Verify the album was added
    assert album.id is not None
    assert album.title == "Test Album"
    assert album.artist == "Test Artist"
    assert album.release_date == "2023-01-01"
