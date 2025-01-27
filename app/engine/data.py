import requests
import pandas as pd
import random
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.models.listening_history import ListeningHistory
from app.models.playlist import Playlist
from app.models.recommendation import Recommendation
  
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")  # Last.fm API key from .env file
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

# Spotify API Setup
scope = "user-library-read user-read-recently-played playlist-read-private"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope
))


# Function to fetch user liked songs from Spotify
def fetch_user_liked_songs(sp):
    """Fetch liked songs for a Spotify user."""
    results = sp.current_user_saved_tracks(limit=50)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    if not tracks:
        print("No liked songs found.")
        return pd.DataFrame()

    # Assuming the track's `id` is the unique identifier for the track
    return pd.DataFrame([{
        "name": track['track']['name'],
        "track_id": track['track']['id'],  # Added track_id here
        "playcount": random.randint(1, 30)  # Placeholder for playcount
    } for track in tracks])


# Function to fetch user playlists from Spotify
def fetch_user_playlists(sp):
    playlists = sp.current_user_playlists(limit=50)
    all_tracks = []
    for playlist in playlists['items']:
        tracks = sp.playlist_tracks(playlist['id'])
        all_tracks.extend([{
            "name": item['track']['name'],
            "artist": item['track']['artists'][0]['name'],
            "track_id": item['track']['id'],  # Add track_id here
            "playcount": 1  # Placeholder
        } for item in tracks['items']])
    return pd.DataFrame(all_tracks)


# Function to fetch recently played tracks from Spotify
def fetch_recently_played(sp):
    results = sp.current_user_recently_played(limit=50)
    tracks = results['items']
    return pd.DataFrame([{
        "name": track['track']['name'],
        "artist": track['track']['artists'][0]['name'],
        "track_id": track['track']['id'],  # Add track_id here
        "playcount": 1  # Placeholder
    } for track in tracks])


# Function to fetch user songs from Last.fm
def fetch_user_songs(user: str):
    """Fetch top songs for a user from Last.fm."""
    params = {
        "method": "user.getTopTracks",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1000,
    }
    data = make_request(params)
    tracks = data.get("toptracks", {}).get("track", [])
    return pd.DataFrame([{
        "name": track["name"],
        "track_id": track["url"],  # Last.fm provides a URL, but here we're using `track_id`
        "playcount": int(track["playcount"])
    } for track in tracks])


# Helper function to make API requests for Last.fm
def make_request(params):
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}


# Function to combine Spotify and Last.fm data
def create_combined_dataset(spotify_data, lastfm_data):
    """Combine Spotify and Last.fm data into a single dataset."""
    if spotify_data.empty:
        return lastfm_data
    if lastfm_data.empty:
        return spotify_data

    combined_data = pd.concat([spotify_data, lastfm_data], ignore_index=True)
    combined_data["playcount"] = (combined_data["playcount"] - combined_data["playcount"].min()) / \
                                 (combined_data["playcount"].max() - combined_data["playcount"].min())
    return combined_data


# Function to insert data into the database
def insert_data_to_db(data, table):
    """Insert fetched data into the PostgreSQL database."""
    DATABASE_URL = "postgresql://postgres:postgronner34@localhost:5432/Sonance"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    if table == 'listening_histories':
        for index, row in data.iterrows():
            session.execute(
                Listening_History.__table__.insert(),
                {'user_id': 1, 'track_id': row['track_id'], 'album': 1, 'play_count': row['playcount'], 'timestamp': func.now()}
            )
    elif table == 'playlists':
        for index, row in data.iterrows():
            session.execute(
                Playlist.__table__.insert(),
                {'name': row['name'], 'track_ids': row['track_ids']}  # Assuming 'track_ids' is a list of tracks
            )
    elif table == 'recommendations':
        for index, row in data.iterrows():
            session.execute(
                Recommendation.__table__.insert(),
                {'user_id': 1, 'track_id': row['track_id'], 'album': 1, 'recommendation_score': row['playcount']}
            )

    session.commit()
    session.close()


# Test function to fetch and insert data
def test_insert_data():
    """Test function to fetch data from Spotify and Last.fm, combine, and insert into the database."""
    
    # Fetch data from Spotify
    print("Fetching Spotify data...")
    spotify_liked_songs = fetch_user_liked_songs(sp)
    spotify_playlists = fetch_user_playlists(sp)
    spotify_recently_played = fetch_recently_played(sp)
    
    # Fetch data from Last.fm
    print("Fetching Last.fm data...")
    lastfm_songs = fetch_user_songs("lastfm_username")  # Replace with an actual username

    # Combine Spotify and Last.fm data into a single dataset
    combined_data = create_combined_dataset(spotify_liked_songs, lastfm_songs)

    # Insert combined data into 'listening_histories' table
    print("Inserting combined data into 'listening_histories' table...")
    insert_data_to_db(combined_data, 'listening_histories')

    # Insert playlists data into 'playlists' table (optional)
    print("Inserting playlists data into 'playlists' table...")
    insert_data_to_db(spotify_playlists, 'playlists')

    # Insert recently played data into 'listening_histories' table (optional)
    print("Inserting recently played data into 'listening_histories' table...")
    insert_data_to_db(spotify_recently_played, 'listening_histories')

    print("Data insertion completed successfully.")


# Run the test function
test_insert_data()
