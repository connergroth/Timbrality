import pytest
from sqlalchemy.orm import Session
from app.models.database import Base, engine, SessionLocal
from app.models.track_listening_history import TrackListeningHistory
from app.models.user import User
from app.utils.database_utils import insert_data_to_db
from app.services.lastfm_service import fetch_user_songs  # Use the real function

@pytest.fixture(scope="function")
def test_db():
    """Fixture for setting up and tearing down the test database."""
    Base.metadata.create_all(bind=engine)  # Create tables
    db = SessionLocal()
    yield db  # Provide the database session to the test
    db.close()
    Base.metadata.drop_all(bind=engine)  # Drop tables after the test


def test_fetch_and_insert_real_lastfm_data(test_db: Session):
    """Test fetching real Last.fm data and inserting it into the database."""
    import uuid

    # Step 1: Insert a mock user with a unique username
    unique_username = f"testuser_{uuid.uuid4().hex[:8]}"
    unique_email = f"testuser@example.com_{uuid.uuid4().hex[:8]}"
    mock_user = User(username=unique_username, email=unique_email, password_hash="hashedpassword")
    test_db.add(mock_user)
    test_db.commit()

    # Step 2: Fetch real data from Last.fm
    username = "connergroth"  # Replace with your real Last.fm username
    top_songs = fetch_user_songs(username)

    # Debug: Print the fetched data
    import pandas as pd
    print(top_songs)

    # Ensure data is returned from the API
    assert len(top_songs) > 0, "No data returned from Last.fm API"

    # Convert the fetched data to a DataFrame
    top_songs_df = pd.DataFrame(top_songs)

    # Check if the DataFrame is empty
    if not top_songs_df.empty:
        # Rename columns to match the database schema
        top_songs_df.rename(columns={"playcount": "play_count"}, inplace=True)

        # Ensure the data is formatted correctly
        assert "track_id" in top_songs_df.columns, "Expected column 'track_id' is missing"
        assert "play_count" in top_songs_df.columns, "Expected column 'play_count' is missing"

        # Add the user_id field for foreign key
        top_songs_df["user_id"] = mock_user.id

        # Insert the fetched data into the database
        insert_data_to_db(top_songs_df, "track_listening_histories")  # Adjust table name if needed

        # Verify data in the database
        results = test_db.query(TrackListeningHistory).all()
        assert len(results) > 0, "No rows were inserted into the track_listening_histories table"
        print(f"Inserted {len(results)} rows into track_listening_histories")
    else:
        pytest.fail("Fetched data is empty; no rows to insert.")
