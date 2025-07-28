import sys
import os
import json
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSON

# Fix Unicode issue in Windows Terminal
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRCECT_URI")

DATABASE_URL = "postgresql://postgres:postgronner34@localhost:5432/Sonance"

# Only initialize Spotify if credentials are available
sp = None
if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id=SPOTIPY_CLIENT_ID,
                    client_secret=SPOTIPY_CLIENT_SECRET,
                    redirect_uri=SPOTIPY_REDIRECT_URI,
                    scope="user-library-read user-library-modify user-top-read user-follow-read"
        ))
    except Exception:
        # If Spotify initialization fails, continue without it
        pass

def insert_top_artists():
    # This function would need to be implemented properly
    # For now, just a placeholder
    pass

def get_database_connection():
    # This function would need to be implemented properly
    # For now, just a placeholder
    return None

