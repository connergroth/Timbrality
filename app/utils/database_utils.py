import sys
import os
import json
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSON
from app.services.spotify_service import get_top_tracks

# Fix Unicode issue in Windows Terminal
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRCECT_URI")

DATABASE_URL = "postgresql://postgres:postgronner34@localhost:5432/Sonance"

create_engine(DATABASE_URL)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
                redirect_uri=SPOTIPY_REDIRECT_URI,
                scope="user-library-read user-library-modify user-top-read user-follow-read"

def insert_top_artists:
    df = get_top_tracks()

    artist_records = df[["artist_id", "artist_name"]].drop_duplicates().to_dict(orient="records")

    conn.execute(text("""
        INSERT INTO artists (id, name, source)  
        VALUES (:artist_id, :artist_name, 'spotify')
        ON CONFLICT (id) DO NOTHING
    """), artist_records)