import os
import sys
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.services.spotify_service import SpotifyService

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enable user authentication for testing user-specific features
spotify_service = SpotifyService(user_auth=True)

def test_fetch_user_saved_tracks():
    """Test fetching user's saved tracks."""
    print("\n Testing fetch_user_saved_tracks()...")
    saved_tracks_df = spotify_service.fetch_user_saved_tracks(user_id="test_user")
    print(saved_tracks_df.head())

def test_check_saved_tracks():
    """Test checking if specific tracks are saved."""
    print("\n Testing check_saved_tracks()...")
    track_ids = ["0VA5FzFlysCcSG1IdOkhUb", "7dS5EaCoMnN7DzlpT6aRn2"]  # Replace with actual track IDs
    status = spotify_service.check_saved_tracks(track_ids)
    print(status)

def test_fetch_user_top_tracks():
    """Test fetching user's top tracks."""
    print("\n Testing fetch_user_top_tracks()...")
    top_tracks_df = spotify_service.fetch_user_top_tracks(limit=5, time_range="short_term")
    print(top_tracks_df.head())

def test_fetch_user_top_artists():
    """Test fetching user's top artists."""
    print("\n Testing fetch_user_top_artists()...")
    top_artists_df = spotify_service.fetch_user_top_artists(limit=5, time_range="short_term")
    print(top_artists_df.head())

def test_fetch_followed_artists():
    """Test fetching followed artists."""
    print("\n Testing fetch_followed_artists()...")
    followed_artists_df = spotify_service.fetch_followed_artists(limit=5)
    print(followed_artists_df.head())

def test_check_artist_followed():
    """Test checking if specific artists are followed."""
    print("\n Testing check_artist_followed()...")
    artist_ids = ["36QJpDe2go2KgaRleHCDTp"]  # Replace with actual artist IDs
    status = spotify_service.check_artist_followed(artist_ids)
    print(status)

def test_fetch_artist_info():
    """Test fetching artist info."""
    print("\n Testing fetch_artist_info()...")
    artist_info = spotify_service.fetch_artist_info("Radiohead")
    print(artist_info)

def test_fetch_artist_top_tracks():
    """Test fetching artist's top tracks."""
    print("\n Testing fetch_artist_top_tracks()...")
    top_tracks_df = spotify_service.fetch_artist_top_tracks(artist_id="2YZyLoL8N0Wb9xBt1NhZWg", country="US")
    print(top_tracks_df.head())

# def test_fetch_related_artists():
#     """Test fetching related artists."""
#     print("\n Testing fetch_related_artists()...")
#     related_artists_df = spotify_service.fetch_related_artists(artist_id="2YZyLoL8N0Wb9xBt1NhZWg")
#     print(related_artists_df.head())

def test_fetch_album_data():
    """Test fetching album data."""
    print("\n Testing fetch_album_data()...")
    albums_df = spotify_service.fetch_album_data(artist_id="2YZyLoL8N0Wb9xBt1NhZWg")
    print(albums_df.head())

def test_fetch_user_saved_albums():
    """Test fetching user's saved albums."""
    print("\n Testing fetch_user_saved_albums()...")
    saved_albums_df = spotify_service.fetch_user_saved_albums(limit=5)
    print(saved_albums_df.head())

def test_check_saved_albums():
    """Test checking if specific albums are saved."""
    print("\n Testing check_saved_albums()...")
    album_ids = ["6mUdeDZCsExyJLMdAfDuwh"]  # Replace with actual album IDs
    status = spotify_service.check_saved_albums(album_ids)
    print(status)

def test_fetch_track_data():
    """Test fetching track data."""
    print("\n Testing fetch_track_data()...")
    track_info = spotify_service.fetch_track_data(track_id="0j2T0R9dR9qdJYsB7ciXhf")
    print(track_info)

def test_fetch_recommended_tracks():
    """Test fetching recommended tracks."""
    print("\n Testing fetch_recommended_tracks()...")
    recommended_tracks_df = spotify_service.fetch_recommended_tracks(
        seed_track_ids=["3n3Ppam7vgaVa1iaRUc9Lp"]
    )

    if recommended_tracks_df.empty:
        print("No recommendations found.")
    else:
        print(recommended_tracks_df.head())

def test_save_and_remove_tracks():
    """Test saving and removing tracks."""
    print("\n Testing save_tracks() and remove_tracks()...")
    track_ids = ["3n3Ppam7vgaVa1iaRUc9Lp"]
    
    # Save
    save_status = spotify_service.save_tracks(track_ids)
    print("Save Status:", save_status)

    # Remove
    remove_status = spotify_service.remove_tracks(track_ids)
    print("Remove Status:", remove_status)

def test_save_and_unfollow_artists():
    """Test following and unfollowing artists."""
    print("\n Testing follow_artists() and unfollow_artists()...")
    artist_ids = ["36QJpDe2go2KgaRleHCDTp"]

    # Follow
    follow_status = spotify_service.follow_artists(artist_ids)
    print("Follow Status:", follow_status)

    # Unfollow
    unfollow_status = spotify_service.unfollow_artists(artist_ids)
    print("Unfollow Status:", unfollow_status)

def test_fetch_multiple_track_features():
    """Test fetching multiple tracks' audio features."""
    print("\n Testing fetch_multiple_track_features()...")
    track_ids = ["5JjnoGJyOxfSZUZtk2rRwZ", "7dS5EaCoMnN7DzlpT6aRn2"]
    track_features_df = spotify_service.fetch_multiple_track_features(track_ids)
    print(track_features_df.head())

def test_fetch_track_audio_analysis():
    """Test fetching track audio analysis."""
    print("\n Testing fetch_track_audio_analysis()...")
    audio_analysis = spotify_service.fetch_track_audio_analysis(track_id="3n3Ppam7vgaVa1iaRUc9Lp")
    print(audio_analysis)

### Run Tests ###
if __name__ == "__main__":
    # test_fetch_user_saved_tracks()
    # test_check_saved_tracks()
    # test_fetch_user_top_tracks()
    # test_fetch_user_top_artists()
    # test_fetch_followed_artists()
    # test_check_artist_followed()
    # test_fetch_artist_info()
    # test_fetch_artist_top_tracks()
    # # test_fetch_related_artists()
    # test_fetch_album_data()
    # test_fetch_user_saved_albums()
    # test_check_saved_albums()
    test_fetch_track_data()
    # test_fetch_recommended_tracks()
    # test_save_and_remove_tracks()
    test_fetch_multiple_track_features()
    test_fetch_track_audio_analysis()
