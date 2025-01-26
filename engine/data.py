import requests
import pandas as pd

API_KEY = "e1e432a42418c811876ed33474eda642"
API_URL = "http://ws.audioscrobbler.com/2.0/"

def make_request(params):
    """Make an API request and handle errors."""
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}

def fetch_user_artists(user: str):
    params = {
        "method": "user.getTopArtists",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 100,
    }
    data = make_request(params)
    artists = data.get("topartists", {}).get("artist", [])
    return pd.DataFrame([{"name": artist["name"], "playcount": int(artist["playcount"])} for artist in artists])


def fetch_user_songs(user: str):
    """Fetch top songs for a user using the Last.fm API."""
    params = {
        "method": "user.getTopTracks",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 100,
    }
    data = make_request(params)
    tracks = data.get("toptracks", {}).get("track", [])
    if not tracks:
        print(f"No songs found for user '{user}'.")
        return pd.DataFrame()
    return pd.DataFrame([{"name": track["name"], "playcount": int(track["playcount"])} for track in tracks])


def fetch_user_albums(user: str):
    # Implement similar logic for albums
    pass

def fetch_multiple_users_data(usernames: list):
    user_data = []
    for username in usernames:
        user_data_df = fetch_user_songs(username)
        if not user_data_df.empty:
            user_data_df["user"] = username
            user_data.append(user_data_df)
    return pd.concat(user_data, ignore_index=True)