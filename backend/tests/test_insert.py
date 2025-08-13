import os
import sys
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSON
import json

# Fix Unicode issue in Windows Terminal
sys.stdout.reconfigure(encoding='utf-8')

# 🔹 Define your Spotify API credentials
load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRCECT_URI")

# 🔹 Define your database connection string
DATABASE_URL = "postgresql://postgres:postgronner34@localhost:5432/Sonance"

# 🔹 Create the engine
engine = create_engine(DATABASE_URL)

# 🔹 Authenticate and create Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-top-read user-library-read"  
))

# 🔹 Fetch the user's top tracks
def get_top_tracks(limit=20):
    results = sp.current_user_top_tracks(limit=limit, time_range="medium_term")
    return results["items"]

# 🔹 Fetch and print top tracks
top_tracks = get_top_tracks()
for idx, track in enumerate(top_tracks):
    print(f"{idx + 1}. {track['name']} - {track['artists'][0]['name']}")

# 🔹 Insert Spotify Data into PostgreSQL
with engine.connect() as conn:
    for track in top_tracks:
        # Fetch additional metadata
        track_image = track["album"]["images"][0]["url"] if track["album"]["images"] else None  # Album cover image

        conn.execute(text("""
            INSERT INTO artists (id, name)
            VALUES (:id, :name)
            ON CONFLICT (id) DO NOTHING
        """), {
            "id": track["artists"][0]["id"],
            "name": track["artists"][0]["name"]
        })

        # Check if album exists before inserting a track
        album_id = track["album"]["id"]

        conn.execute(text("""
            INSERT INTO albums (id, title, artist_id)
            VALUES (:id, :title, :artist_id)
            ON CONFLICT (id) DO NOTHING
        """), {
            "id": album_id,
            "title": track["album"]["name"],
            "artist_id": track["artists"][0]["id"]
        })

        conn.execute(text("""
            INSERT INTO tracks (
                id, title, artist_id, album_id, duration_ms, popularity, 
                tempo, energy, valence, danceability, loudness, 
                speechiness, acousticness, instrumentalness, 
                preview_url, cover_url
            ) 
            VALUES (
                :id, :title, :artist_id, :album_id, :duration_ms, :popularity, 
                :tempo, :energy, :valence, :danceability, :loudness, 
                :speechiness, :acousticness, :instrumentalness, 
                :preview_url, :cover_url
            )
            ON CONFLICT (id) DO NOTHING
        """), {
            "id": track["id"],
            "title": track["name"],
            "artist_id": track["artists"][0]["id"],
            "album_id": track["album"]["id"],
            "duration_ms": track["duration_ms"],
            "popularity": track.get("popularity"),  # Use .get() to avoid errors if missing
            
            # ✅ Check if 'audio_features' exists before accessing it
            "tempo": track.get("audio_features", {}).get("tempo"),
            "energy": track.get("audio_features", {}).get("energy"),
            "valence": track.get("audio_features", {}).get("valence"),
            "danceability": track.get("audio_features", {}).get("danceability"),
            "loudness": track.get("audio_features", {}).get("loudness"),
            "speechiness": track.get("audio_features", {}).get("speechiness"),
            "acousticness": track.get("audio_features", {}).get("acousticness"),
            "instrumentalness": track.get("audio_features", {}).get("instrumentalness"),
            
            "preview_url": track.get("preview_url"),
            "cover_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None
        })

    conn.commit()

print("✅ Spotify tracks inserted successfully!")
