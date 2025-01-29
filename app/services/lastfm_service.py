import requests
import os
import pandas as pd
from dotenv import load_dotenv
from app.cache.redis import cache_user_profile
from app.utils.database_utils import insert_data_to_db

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")  # Last.fm API key

def make_request(params):
    try:
        response = requests.get("http://ws.audioscrobbler.com/2.0/", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}

# Fetch user data from Last.fm
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
        "track_id": track["url"],
        "playcount": int(track["playcount"])
    } for track in tracks])

def fetch_user_artists(user: str):
    """Fetch top artists for a user from Last.fm."""
    params = {
        "method": "user.getTopArtists",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1000,
    }
    data = make_request(params)
    artists = data.get("topartists", {}).get("artist", [])
    return pd.DataFrame([{
        "name": artist["name"],
        "artist_id": artist["url"],
        "playcount": int(artist["playcount"])
    } for artist in artists])

def fetch_user_albums(user: str):
    """Fetch top albums for a user from Last.fm."""
    params = {
        "method": "user.getTopAlbums",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1000,
    }
    data = make_request(params)
    albums = data.get("topalbums", {}).get("album", [])
    return pd.DataFrame([{
        "name": album["name"],
        "album_id": album["url"],
        "playcount": int(album["playcount"])
    } for album in albums])

def fetch_recently_played(user: str):
    """Fetch recently played tracks for a user from Last.fm."""
    params = {
        "method": "user.getRecentTracks",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1000,
    }
    data = make_request(params)
    tracks = data.get("recenttracks", {}).get("track", [])
    return pd.DataFrame([{
        "name": track["name"],
        "artist": track["artist"]["#text"],
        "track_id": track["url"],
        "playcount": int(track["playcount"]) if track.get("playcount") else 0
    } for track in tracks])

def fetch_user_scrobbles(user: str, limit=500):
    """Fetch scrobbled tracks for a user from Last.fm."""
    params = {
        "method": "user.getRecentTracks",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": limit,
    }
    data = make_request(params)
    tracks = data.get("recenttracks", {}).get("track", [])
    return pd.DataFrame([{
        "name": track["name"],
        "artist": track["artist"]["#text"],
        "track_id": track["url"],
        "timestamp": track["date"]["uts"]
    } for track in tracks if "date" in track])

def fetch_user_friends(user: str):
    """Fetch a user's friends from Last.fm."""
    params = {
        "method": "user.getFriends",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 1000,
    }
    data = make_request(params)
    friends = data.get("friends", {}).get("user", [])
    return pd.DataFrame([{
        "friend_name": friend["name"],
        "friend_url": friend["url"]
    } for friend in friends])

def fetch_track_metadata(track_id: str):
    """Fetch metadata for a specific track."""
    params = {
        "method": "track.getInfo",
        "track": track_id,
        "api_key": API_KEY,
        "format": "json",
    }
    data = make_request(params)
    if "track" in data:
        track = data["track"]
        return {
            "name": track["name"],
            "artist": track["artist"]["name"],
            "album": track["album"]["title"],
            "playcount": int(track["playcount"]),
            "tags": [tag["name"] for tag in track["toptags"]["tag"]]
        }
    return {}

def fetch_track_tags(track_id: str):
    """Fetch top tags for a specific track."""
    params = {
        "method": "track.getTopTags",
        "track": track_id,
        "api_key": API_KEY,
        "format": "json",
    }
    data = make_request(params)
    tags = data.get("toptags", {}).get("tag", [])
    return [tag["name"] for tag in tags]

# New function for fetching and storing all relevant data
async def fetch_and_store_lastfm_data(user: str):
    # Fetch all relevant data for the user
    user_data = fetch_user_data(user)

    # Cache the fetched data (useful for frequent data access)
    await cache_user_data(user, user_data)

    # Insert the data into the PostgreSQL database
    insert_data_to_db(user_data['top_songs'], 'listening_histories')
    insert_data_to_db(user_data['top_artists'], 'user_artists')
    insert_data_to_db(user_data['top_albums'], 'user_albums')

    return {"message": "Last.fm data fetched and stored successfully"}

# New helper function to fetch multiple data types
def fetch_user_data(user: str):
    """Fetch multiple user data from Last.fm."""
    top_songs = fetch_user_songs(user)
    top_artists = fetch_user_artists(user)
    top_albums = fetch_user_albums(user)
    recently_played = fetch_recently_played(user)
    scrobbles = fetch_user_scrobbles(user)

    # Combine all the data into a dictionary
    return {
        "top_songs": top_songs,
        "top_artists": top_artists,
        "top_albums": top_albums,
        "recently_played": recently_played,
        "scrobbles": scrobbles,
    }
