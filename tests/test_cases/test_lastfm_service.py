import pytest
from app.services.lastfm_service import fetch_user_data

def test_fetch_user_data():
    user = "connergroth" 
    user_data = fetch_user_data(user)

    # Ensure all keys are present
    assert "top_songs" in user_data
    assert "top_artists" in user_data
    assert "top_albums" in user_data
    assert "recently_played" in user_data
    assert "scrobbles" in user_data

    # Ensure each key contains data
    assert not user_data["top_songs"].empty
    assert not user_data["top_artists"].empty
    assert not user_data["top_albums"].empty
