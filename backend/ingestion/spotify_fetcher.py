"""
Spotify Fetcher - Get track/album metadata from Spotify API
"""
import os
import logging
from typing import List, Dict, Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def get_spotify_client() -> spotipy.Spotify:
    """Initialize and return Spotify client"""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
    
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_album_tracks(album_name: str, artist_name: str) -> List[Dict]:
    """
    Get all tracks from a specific album via Spotify API
    
    Args:
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        List of track dictionaries with metadata
    """
    try:
        sp = get_spotify_client()
        
        # Search for the album
        search_query = f"album:{album_name} artist:{artist_name}"
        results = sp.search(q=search_query, type='album', limit=1)
        
        if not results['albums']['items']:
            logger.warning(f"Album not found: {album_name} by {artist_name}")
            return []
        
        album = results['albums']['items'][0]
        album_id = album['id']
        
        # Get tracks from the album
        tracks_result = sp.album_tracks(album_id)
        tracks = []
        
        for track in tracks_result['items']:
            track_data = {
                'id': f"spotify:{track['id']}",
                'name': track['name'],
                'artist': artist_name,
                'album': album_name,
                'album_id': album['id'],
                'track_number': track['track_number'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'preview_url': track['preview_url'],
                'spotify_url': track['external_urls']['spotify'],
                'release_date': album['release_date'],
                'total_tracks': album['total_tracks'],
                'popularity': None  # Will be fetched separately for individual tracks
            }
            
            # Get individual track details for popularity
            try:
                track_details = sp.track(track['id'])
                track_data['popularity'] = track_details['popularity']
            except Exception as e:
                logger.warning(f"Could not fetch popularity for track {track['name']}: {e}")
                track_data['popularity'] = 0
                
            tracks.append(track_data)
            
        logger.info(f"Successfully fetched {len(tracks)} tracks from album '{album_name}' by {artist_name}")
        return tracks
        
    except Exception as e:
        logger.error(f"Error fetching album tracks: {e}")
        return []


def get_track_metadata(track_name: str, artist_name: str) -> Optional[Dict]:
    """
    Get metadata for a specific track
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        
    Returns:
        Track metadata dictionary or None if not found
    """
    try:
        sp = get_spotify_client()
        
        # Search for the track
        search_query = f"track:{track_name} artist:{artist_name}"
        results = sp.search(q=search_query, type='track', limit=1)
        
        if not results['tracks']['items']:
            logger.warning(f"Track not found: {track_name} by {artist_name}")
            return None
            
        track = results['tracks']['items'][0]
        album = track['album']
        
        track_data = {
            'id': f"spotify:{track['id']}",
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': album['name'],
            'album_id': album['id'],
            'track_number': track['track_number'],
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'popularity': track['popularity'],
            'preview_url': track['preview_url'],
            'spotify_url': track['external_urls']['spotify'],
            'release_date': album['release_date'],
            'total_tracks': album['total_tracks']
        }
        
        logger.info(f"Successfully fetched metadata for track '{track_name}' by {artist_name}")
        return track_data
        
    except Exception as e:
        logger.error(f"Error fetching track metadata: {e}")
        return None


def search_albums(query: str, limit: int = 10) -> List[Dict]:
    """
    Search for albums on Spotify
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of album dictionaries
    """
    try:
        sp = get_spotify_client()
        
        results = sp.search(q=query, type='album', limit=limit)
        albums = []
        
        for album in results['albums']['items']:
            album_data = {
                'id': album['id'],
                'name': album['name'],
                'artist': ', '.join([artist['name'] for artist in album['artists']]),
                'release_date': album['release_date'],
                'total_tracks': album['total_tracks'],
                'spotify_url': album['external_urls']['spotify'],
                'images': album['images'],
                'album_type': album['album_type']
            }
            albums.append(album_data)
            
        logger.info(f"Found {len(albums)} albums for query: {query}")
        return albums
        
    except Exception as e:
        logger.error(f"Error searching albums: {e}")
        return []


def get_artist_top_tracks(artist_name: str, country: str = 'US') -> List[Dict]:
    """
    Get top tracks for an artist
    
    Args:
        artist_name: Name of the artist
        country: Country code (default: US)
        
    Returns:
        List of top track dictionaries
    """
    try:
        sp = get_spotify_client()
        
        # Search for the artist
        results = sp.search(q=artist_name, type='artist', limit=1)
        
        if not results['artists']['items']:
            logger.warning(f"Artist not found: {artist_name}")
            return []
            
        artist = results['artists']['items'][0]
        artist_id = artist['id']
        
        # Get top tracks
        top_tracks = sp.artist_top_tracks(artist_id, country=country)
        tracks = []
        
        for track in top_tracks['tracks']:
            track_data = {
                'id': f"spotify:{track['id']}",
                'name': track['name'],
                'artist': artist_name,
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'preview_url': track['preview_url'],
                'spotify_url': track['external_urls']['spotify']
            }
            tracks.append(track_data)
            
        logger.info(f"Found {len(tracks)} top tracks for artist: {artist_name}")
        return tracks
        
    except Exception as e:
        logger.error(f"Error fetching artist top tracks: {e}")
        return [] 