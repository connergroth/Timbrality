import requests
import pandas as pd
import random

# Spotify API
import spotipy 
from spotipy.oauth2 import SpotifyOAuth

scope = "user-library-read user-read-recently-played playlist-read-private"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope
))

# Spotify
def fetch_user_liked_songs(sp):
    """Fetch liked songs for a Spotify user."""
    print("Fetching liked songs...")
    results = sp.current_user_saved_tracks(limit=50)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    if not tracks:
        print("No liked songs found.")
        return pd.DataFrame()

    # Process track data into a DataFrame
    return pd.DataFrame([{
        "name": track['track']['name'],
        "playcount": random.randint(1,30)  # Assuming playcount is 1 for saved tracks
    } for track in tracks])

def fetch_user_playlists(sp):
    playlists = sp.current_user_playlists(limit=50)
    all_tracks = []
    for playlist in playlists['items']:
        tracks = sp.playlist_tracks(playlist['id'])
        all_tracks.extend([{
            "name": item['track']['name'],
            "artist": item['track']['artists'][0]['name'],
            "playcount": 1  # Use 1 as a placeholder for playcount
        } for item in tracks['items']])
    return pd.DataFrame(all_tracks)

def fetch_recently_played(sp):
    results = sp.current_user_recently_played(limit=50)
    tracks = results['items']
    return pd.DataFrame([{
        "name": track['track']['name'],
        "artist": track['track']['artists'][0]['name'],
        "playcount": 1  # Use 1 as a placeholder for playcount
    } for track in tracks])


def create_combined_dataset(spotify_data, lastfm_data):
    """Combine Spotify and Last.fm data into a single dataset."""
    if spotify_data.empty:
        print("No Spotify data found. Using Last.fm data only.")
        return lastfm_data

    if lastfm_data.empty:
        print("No Last.fm data found. Using Spotify data only.")
        return spotify_data

    # Combine both datasets
    combined_data = pd.concat([spotify_data, lastfm_data], ignore_index=True)

    # Normalize playcount across sources
    combined_data["playcount"] = (combined_data["playcount"] - combined_data["playcount"].min()) / \
                                 (combined_data["playcount"].max() - combined_data["playcount"].min())

    return combined_data

# Last.FM

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
        "limit": 1000,
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



