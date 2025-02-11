# services/spotify_service.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

class SpotifyService:
    def __init__(self, user_auth=False):
        """Initialize Spotify API with user auth or client credentials."""
        if user_auth:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
                redirect_uri=SPOTIPY_REDIRECT_URI,
                scope="user-library-read user-library-modify user-top-read user-follow-read"
            ))
        else:
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET
            ))

    ### USER DATA

    ## Fetch User's Liked Tracks
    def fetch_user_saved_tracks(self, user_id, limit=50):
        """Fetch all saved tracks for a user with pagination."""
        saved_tracks = []
        offset = 0  # Start at first track

        while True:
            results = self.sp.current_user_saved_tracks(limit=limit, offset=offset)

            for item in results["items"]:
                track = item["track"]
                saved_tracks.append({
                    "user_id": user_id,
                    "track_id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album": track["album"]["name"],
                    "popularity": track["popularity"],
                    "duration_ms": track["duration_ms"],
                    "preview_url": track.get("preview_url", None),
                    "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                    "added_at": item["added_at"]
                })

            offset += limit  # Increase offset for next batch
            if not results["next"]:
                break  # Stop pagination when there are no more tracks

        return pd.DataFrame(saved_tracks) if saved_tracks else pd.DataFrame(columns=["track_id", "name", "artist", "album", "popularity", "duration_ms", "preview_url", "cover_art", "added_at"])

    ## Fetch User’s Top Tracks
    def fetch_user_top_tracks(self, limit=20, time_range="medium_term"):
        """Fetch the current user's top tracks."""
        top_tracks = []
        offset = 0

        while True:
            results = self.sp.current_user_top_tracks(limit=limit, offset=offset, time_range=time_range)["items"]
            if not results:
                break  # Stop if no more results

            for track in results:
                top_tracks.append({
                    "track_id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "popularity": track["popularity"],
                    "preview_url": track.get("preview_url", None),
                    "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None
                })

            offset += limit  # Increase offset for next batch

        return pd.DataFrame(top_tracks) if top_tracks else pd.DataFrame(columns=["track_id", "name", "artist", "popularity", "preview_url", "cover_art"])
    
    ## Fetch User’s Top Artists
    def fetch_user_top_artists(self, limit=20, time_range="medium_term"):
        """Fetch the current user's top artists with pagination."""
        top_artists = []
        offset = 0

        while True:
            results = self.sp.current_user_top_artists(limit=limit, offset=offset, time_range=time_range)["items"]
            if not results:
                break  # Stop if no more results

            for artist in results:
                top_artists.append({
                    "artist_id": artist["id"],
                    "name": artist["name"],
                    "genres": artist["genres"],
                    "popularity": artist["popularity"],
                    "followers": artist["followers"]["total"],
                    "profile_picture": artist["images"][0]["url"] if artist["images"] else None
                })

            offset += limit  # Increase offset for next batch

        return pd.DataFrame(top_artists) if top_artists else pd.DataFrame(columns=["artist_id", "name", "genres", "popularity", "followers", "profile_picture"])

    ## Fetch Followed Artists
    def fetch_followed_artists(self, limit=20):
        """Fetch artists that the user follows."""
        followed_artists = []
        after = None  # Pagination cursor

        while True:
            results = self.sp.current_user_followed_artists(limit=limit, after=after)["artists"]
            
            for artist in results["items"]:
                followed_artists.append({
                    "artist_id": artist["id"],
                    "name": artist["name"],
                    "genres": artist["genres"],
                    "popularity": artist["popularity"],
                    "followers": artist["followers"]["total"],
                    "profile_picture": artist["images"][0]["url"] if artist["images"] else None
                })

            after = results.get("cursors", {}).get("after")
            if not after:
                break  # Stop pagination when there are no more results

        return pd.DataFrame(followed_artists) if followed_artists else pd.DataFrame(columns=["artist_id", "name", "genres", "popularity", "followers", "profile_picture"])

    ## Check If Artist Is Followed
    def check_artist_followed(self, artist_ids):
        """Check if the user follows specific artists."""
        followed_status = self.sp.current_user_following_artists(ids=artist_ids)
        return dict(zip(artist_ids, followed_status))  # Returns {artist_id: True/False}

    ## Check If Tracks Are Saved
    def check_saved_tracks(self, track_ids):
        """Check if one or more tracks are saved in the user's library."""
        saved_status = self.sp.current_user_saved_tracks_contains(track_ids)
        return dict(zip(track_ids, saved_status))  # Returns {track_id: True/False}

    ### ARTIST
    
    ## Fetch Artist Info
    def fetch_artist_info(self, artist_name):
        """Fetch detailed artist information."""
        results = self.sp.search(q=artist_name, type="artist")["artists"]["items"]
        if not results:
            return None  # Return None if artist is not found

        artist = results[0]
        return {
            "artist_id": artist["id"],
            "name": artist["name"],
            "genres": artist["genres"],
            "popularity": artist["popularity"],
            "followers": artist["followers"]["total"]
        }

    ## Fetch Artist's Top Tracks
    def fetch_artist_top_tracks(self, artist_id, country="US"):
        """Fetch top tracks for an artist."""
        try:
            tracks = self.sp.artist_top_tracks(artist_id, country=country)["tracks"]

            return pd.DataFrame([{
                "track_id": track["id"],
                "name": track["name"],
                "preview_url": track.get("preview_url", None),  # Track snippet
                "popularity": track["popularity"],
                "album_name": track["album"]["name"],
                "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None
            } for track in tracks])

        except Exception as e:
            return {"error": str(e), "message": "Failed to fetch artist's top tracks."}

    ## Fetch Related Artists
    def fetch_related_artists(self, artist_id):
        """Fetch related artists for a given artist."""
        related = self.sp.artist_related_artists(artist_id)["artists"]
        return pd.DataFrame([{
            "artist_id": artist["id"],
            "name": artist["name"],
            "genres": artist["genres"],
            "popularity": artist["popularity"],
            "followers": artist["followers"]["total"]
        } for artist in related])

    ### ALBUM

    ## Fetch Album Data
    def fetch_album_data(self, artist_id):
        """Fetch albums by an artist."""
        albums = self.sp.artist_albums(artist_id, album_type="album")["items"]
        return pd.DataFrame([{
            "album_id": album["id"],
            "name": album["name"],
            "release_date": album["release_date"],
            "total_tracks": album["total_tracks"],
            "cover_art": album["images"][0]["url"] if album["images"] else None  
        } for album in albums])

    ## Fetch User's Saved Albums
    def fetch_user_saved_albums(self, limit=50):
        """Fetch all albums saved by the user with pagination."""
        saved_albums = []
        offset = 0  # Start at first album

        while True:
            results = self.sp.current_user_saved_albums(limit=limit, offset=offset)

            for item in results["items"]:
                album = item["album"]
                saved_albums.append({
                    "album_id": album["id"],
                    "name": album["name"],
                    "artist": album["artists"][0]["name"],
                    "release_date": album["release_date"],
                    "total_tracks": album["total_tracks"],
                    "cover_art": album["images"][0]["url"] if album["images"] else None
                })

            offset += limit  # Increase offset for next batch
            if not results["next"]:
                break  # Stop pagination when there are no more albums

        return pd.DataFrame(saved_albums) if saved_albums else pd.DataFrame(columns=["album_id", "name", "artist", "release_date", "total_tracks", "cover_art"])

    ## Check If Albums Are Saved
    def check_saved_albums(self, album_ids):
        """Check if one or more albums are saved in the user's library."""
        saved_status = self.sp.current_user_saved_albums_contains(album_ids)
        return dict(zip(album_ids, saved_status))  # Returns {album_id: True/False}

    ### TRACK

    ## Fetch Track Data (spotipy function broken)
    def fetch_track_data(self, track_id): 
        """Fetch track details and manually fetch audio features."""
        track = self.sp.track(track_id)

        # Manually call the correct Spotify API endpoint for a single track
        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = {"Authorization": f"Bearer {self.sp.auth_manager.get_access_token(as_dict=False)}"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            features = response.json()
        else:
            features = {"error": "Failed to fetch audio features", "status_code": response.status_code}

        return {
            "track_id": track["id"],
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "popularity": track["popularity"],
            "duration_ms": track["duration_ms"],
            "preview_url": track["preview_url"],
            "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
            "acousticness": features.get("acousticness", None),
            "danceability": features.get("danceability", None),
            "energy": features.get("energy", None),
            "valence": features.get("valence", None),
            "tempo": features.get("tempo", None)
        }

    ## Fetch Multiple Tracks Audio Features
    def fetch_multiple_track_features(self, track_ids):
        """Fetch audio features for multiple tracks by reusing fetch_track_data()."""

        if not track_ids:
            return pd.DataFrame(columns=["track_id", "name", "artist", "popularity", 
                                        "duration_ms", "preview_url", "cover_art",
                                        "acousticness", "danceability", "energy", 
                                        "valence", "tempo"])

        track_features = [self.fetch_track_data(track_id) for track_id in track_ids]

        return pd.DataFrame(track_features)

    ## Fetch Track Audio Analysis
    def fetch_track_audio_analysis(self, track_id):
        """Fetch detailed audio analysis for a track."""
        try:
            return self.sp.audio_analysis(track_id)
        except Exception as e:
            return {"error": str(e), "message": "Audio analysis unavailable for this track"}

    ## Fetch Recommended Tracks (spotipy function broken)
    def fetch_recommended_tracks(self, seed_track_ids=None, seed_artist_ids=None, seed_genres=None, limit=10):
        """Fetch Spotify track recommendations using a manually constructed request."""
        
        # Ensure at least one seed is provided
        if not (seed_track_ids or seed_artist_ids or seed_genres):
            return pd.DataFrame(columns=["track_id", "name", "artist", "preview_url", "cover_art"])  # Return empty DataFrame

        # Construct query parameters
        params = {
            "limit": limit,
            "seed_tracks": ",".join(seed_track_ids) if seed_track_ids else None,
            "seed_artists": ",".join(seed_artist_ids) if seed_artist_ids else None,
            "seed_genres": ",".join(seed_genres) if seed_genres else None
        }

        # Remove None values from params
        params = {k: v for k, v in params.items() if v}

        # Construct URL
        url = "https://api.spotify.com/v1/recommendations"
        headers = {"Authorization": f"Bearer {self.sp.auth_manager.get_access_token(as_dict=False)}"}

        # Send request
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            recommendations = response.json()["tracks"]
            return pd.DataFrame([{
                "track_id": track["id"],
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "preview_url": track.get("preview_url", None),
                "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None
            } for track in recommendations])
        else:
            print(f"Error fetching recommendations: {response.status_code}")
            return pd.DataFrame(columns=["track_id", "name", "artist", "preview_url", "cover_art"])  # Return empty DataFrame

    ## Save Tracks to User Library
    def save_tracks(self, track_ids):
        """Save tracks to the user's library."""
        try:
            self.sp.current_user_saved_tracks_add(track_ids)
            return {"status": "success", "message": f"{len(track_ids)} tracks saved successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    ## Remove Tracks from User Library
    def remove_tracks(self, track_ids):
        """Remove tracks from the user's library."""
        try:
            self.sp.current_user_saved_tracks_delete(track_ids)
            return {"status": "success", "message": f"{len(track_ids)} tracks removed successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
