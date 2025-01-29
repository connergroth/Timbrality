# services/spotify_service.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

    return pd.DataFrame([{
        "name": track['track']['name'],
        "track_id": track['track']['id'],
        "playcount": 1  # Placeholder for playcount
    } for track in tracks])

def fetch_user_playlists(sp):
    """Fetch playlists for a user from Spotify."""
    playlists = sp.current_user_playlists(limit=50)
    all_tracks = []
    for playlist in playlists['items']:
        tracks = sp.playlist_tracks(playlist['id'])
        all_tracks.extend([{
            "name": item['track']['name'],
            "artist": item['track']['artists'][0]['name'],
            "track_id": item['track']['id'],
            "playcount": 1  # Placeholder
        } for item in tracks['items']])
    return pd.DataFrame(all_tracks)

def fetch_recently_played(sp):
    """Fetch recently played tracks for a user."""
    results = sp.current_user_recently_played(limit=50)
    tracks = results['items']
    return pd.DataFrame([{
        "name": track['track']['name'],
        "artist": track['track']['artists'][0]['name'],
        "track_id": track['track']['id'],
        "playcount": 1  # Placeholder
    } for track in tracks])


