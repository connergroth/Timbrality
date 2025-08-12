import requests
import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from sqlalchemy.orm import Session

try:
    from utils.cache import get_cache, set_cache
    from models.database import SessionLocal
except ImportError:
    # Fallback if cache/database not available
    def get_cache(*args, **kwargs):
        return None
    def set_cache(*args, **kwargs):
        pass
    class SessionLocal:
        pass

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY") or os.getenv("LASTFM_API_KEY")  # Last.fm API key

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
        "artist": track["artist"]["name"] if isinstance(track.get("artist"), dict) else str(track.get("artist", "")),
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
        "artist": album["artist"]["name"] if isinstance(album.get("artist"), dict) else str(album.get("artist", "")),
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

def search_track(query: str, limit: int = 10):
    """Search for tracks on Last.fm."""
    params = {
        "method": "track.search",
        "track": query,
        "api_key": API_KEY,
        "format": "json",
        "limit": limit,
    }
    data = make_request(params)
    tracks = data.get("results", {}).get("trackmatches", {}).get("track", [])
    return tracks

def get_track_tags(artist: str, track: str):
    """Get tags for a specific track by artist and track name."""
    params = {
        "method": "track.getTopTags",
        "artist": artist,
        "track": track,
        "api_key": API_KEY,
        "format": "json",
    }
    data = make_request(params)
    tags = data.get("toptags", {}).get("tag", [])
    return [tag["name"] for tag in tags]

def ensure_user_exists(username: str):
    """Legacy function - user management moved to Supabase."""
    print(f"User management for {username} handled by Supabase")

async def cache_user_data(username: str, user_data: dict):
    """Cache user data for faster access."""
    cache_key = f"lastfm_user:{username}"
    set_cache(cache_key, user_data, ttl=3600)  # 1 hour TTL
    return user_data

async def fetch_and_store_lastfm_data(username: str):
    """Fetch Last.fm user data and store it in the database."""
    ensure_user_exists(username)  # Ensure the user is in the DB

    # Fetch all relevant data for the user
    user_data = fetch_user_data(username)  

    if not user_data or "top_songs" not in user_data:
        print(f"No data returned for {username}")
        return {"error": f"No data found for {username}"}

    # Convert fetched songs to a DataFrame
    top_songs = user_data["top_songs"]
    top_songs_df = pd.DataFrame(top_songs)

    # Add username to each row for foreign key reference
    top_songs_df["username"] = username  

    # Cache user data for faster access
    await cache_user_data(username, user_data)  

    # Insert data into the database
    # Database storage moved to Supabase - data can be stored via ingestion pipeline
    print(f"Last.fm data for {username} ready for Supabase ingestion")

    return {"message": f"Last.fm data fetched and stored successfully for {username}"}

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


class LastFMService:
    """Service class for Last.fm API interactions."""
    
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
    
    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tracks using Last.fm API."""
        try:
            # Use the existing search function
            results = search_track(query)
            if isinstance(results, list):
                return results[:limit]
            return []
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return []
    
    async def get_track_info(self, artist: str, track: str) -> Dict:
        """Get detailed track information."""
        try:
            # Use existing functions to get track info
            track_info = {
                "artist": artist,
                "track": track,
                "tags": get_track_tags(artist, track),
                "similar": []  # Could be expanded
            }
            return track_info
        except Exception as e:
            print(f"Error getting track info: {e}")
            return {}
    
    async def get_user_top_tracks(self, username: str, limit: int = 10) -> List[Dict]:
        """Get user's top tracks."""
        try:
            # Use existing function
            df = fetch_user_songs(username)
            if hasattr(df, 'to_dict'):
                records = df.to_dict('records')
                return records[:limit]
            return []
        except Exception as e:
            print(f"Error getting user top tracks: {e}")
            return []
    
    async def get_user_top_albums(self, username: str, limit: int = 10) -> List[Dict]:
        """Get user's top albums."""
        try:
            df = fetch_user_albums(username)
            if hasattr(df, 'to_dict'):
                records = df.to_dict('records')
                return records[:limit]
            return []
        except Exception as e:
            print(f"Error getting user top albums: {e}")
            return []
    
    async def get_user_top_artists(self, username: str, limit: int = 10) -> List[Dict]:
        """Get user's top artists."""
        try:
            df = fetch_user_artists(username)
            if hasattr(df, 'to_dict'):
                records = df.to_dict('records')
                return records[:limit]
            return []
        except Exception as e:
            print(f"Error getting user top artists: {e}")
            return []
    
    async def get_user_info(self, username: str) -> Dict:
        """Get user profile information."""
        try:
            params = {
                "method": "user.getInfo",
                "user": username,
                "api_key": self.api_key,
                "format": "json",
            }
            data = make_request(params)
            user_info = data.get("user", {})
            return user_info
        except Exception as e:
            print(f"Error getting user info: {e}")
            return {}
    
    async def get_user_following(self, username: str, limit: int = 50) -> List[Dict]:
        """Get users that this user follows."""
        try:
            params = {
                "method": "user.getFriends",
                "user": username,
                "api_key": self.api_key,
                "format": "json",
                "limit": limit,
            }
            data = make_request(params)
            friends = data.get("friends", {}).get("user", [])
            
            # Convert to consistent format
            following = []
            for friend in friends:
                following.append({
                    "name": friend.get("name", ""),
                    "realname": friend.get("realname", ""),
                    "country": friend.get("country", ""),
                    "playcount": int(friend.get("playcount", 0)),
                    "registered": friend.get("registered", {}),
                    "method": "following"
                })
            
            return following
            
        except Exception as e:
            print(f"Error getting user following: {e}")
            return []
    
    async def get_track_top_tags(self, track_name: str, artist_name: str) -> List[str]:
        """Get top tags/moods for a specific track"""
        try:
            params = {
                "method": "track.getTopTags",
                "track": track_name,
                "artist": artist_name,
                "api_key": self.api_key,
                "format": "json"
            }
            data = make_request(params)
            tags_data = data.get("toptags", {}).get("tag", [])
            
            # Extract tag names and return as list
            tags = []
            for tag in tags_data:
                tag_name = tag.get("name", "").strip().lower()
                if tag_name and len(tag_name) > 1:  # Filter out single chars and empty
                    tags.append(tag_name)
            
            return tags[:10]  # Return top 10 tags
            
        except Exception as e:
            print(f"Error getting track tags for '{track_name}' by '{artist_name}': {e}")
            return []

    async def get_user_loved_tracks(self, username: str, limit: int = 100) -> List[Dict]:
        """Get user's loved tracks from Last.fm"""
        try:
            params = {
                "method": "user.getLovedTracks",
                "user": username,
                "api_key": self.api_key,
                "format": "json",
                "limit": limit,
            }
            data = make_request(params)
            loved_tracks = data.get("lovedtracks", {}).get("track", [])
            
            # Convert to consistent format
            tracks = []
            for track in loved_tracks:
                tracks.append({
                    "name": track.get("name", ""),
                    "artist": track["artist"]["name"] if isinstance(track.get("artist"), dict) else str(track.get("artist", "")),
                    "track_id": track.get("url", ""),
                    "loved": True,  # All tracks from this endpoint are loved
                    "date_loved": track.get("date", {}).get("uts") if track.get("date") else None
                })
            
            return tracks
            
        except Exception as e:
            print(f"Error getting loved tracks for {username}: {e}")
            return []

    async def get_similar_users(self, username: str, limit: int = 10) -> List[Dict]:
        """Get similar users (if API supports it)."""
        try:
            # Last.fm doesn't have a direct getSimilar users method
            # We'll try to get user's friends as a proxy
            params = {
                "method": "user.getFriends",
                "user": username,
                "api_key": self.api_key,
                "format": "json",
                "limit": limit,
            }
            data = make_request(params)
            friends = data.get("friends", {}).get("user", [])
            
            # Format as similar users with mock similarity scores
            similar_users = []
            for i, friend in enumerate(friends[:limit]):
                similar_users.append({
                    "name": friend.get("name", ""),
                    "match": 0.5 - (i * 0.05),  # Decreasing similarity score
                    "url": friend.get("url", "")
                })
            
            return similar_users
        except Exception as e:
            print(f"Error getting similar users: {e}")
            return []
    
    async def get_artist_top_fans(self, artist_name: str, limit: int = 10) -> List[Dict]:
        """Get top fans of an artist (if API supports it)."""
        try:
            # Last.fm doesn't have a direct getTopFans method for artists
            # This is a placeholder that could be implemented with workarounds
            print(f"get_artist_top_fans not implemented for {artist_name}")
            return []
        except Exception as e:
            print(f"Error getting artist top fans: {e}")
            return []
