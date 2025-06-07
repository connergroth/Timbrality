"""
Last.fm Fetcher - Enrich tracks with tags and moods from Last.fm API
"""
import os
import logging
from typing import List, Dict, Optional
import pylast
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Last.fm API credentials
LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')
LASTFM_API_SECRET = os.getenv('LASTFM_API_SECRET')

def get_lastfm_network() -> pylast.LastFMNetwork:
    """Initialize and return Last.fm network client"""
    if not LASTFM_API_KEY or not LASTFM_API_SECRET:
        raise ValueError("Last.fm credentials not found. Please set LASTFM_API_KEY and LASTFM_API_SECRET environment variables.")
    
    return pylast.LastFMNetwork(
        api_key=LASTFM_API_KEY,
        api_secret=LASTFM_API_SECRET
    )


def enrich_with_tags(track_name: str, artist_name: str, max_tags: int = 10) -> List[str]:
    """
    Get tags/moods for a specific track from Last.fm
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        max_tags: Maximum number of tags to return
        
    Returns:
        List of tag strings
    """
    try:
        network = get_lastfm_network()
        
        # Try to get track-specific tags first
        try:
            track = network.get_track(artist_name, track_name)
            track_tags = track.get_top_tags(limit=max_tags)
            
            if track_tags:
                tags = [tag.item.name.lower() for tag in track_tags]
                logger.info(f"Found {len(tags)} track tags for '{track_name}' by {artist_name}")
                return tags
                
        except Exception as e:
            logger.debug(f"Could not fetch track tags for '{track_name}' by {artist_name}: {e}")
        
        # Fallback to artist tags if track tags are not available
        try:
            artist = network.get_artist(artist_name)
            artist_tags = artist.get_top_tags(limit=max_tags)
            
            if artist_tags:
                tags = [tag.item.name.lower() for tag in artist_tags]
                logger.info(f"Found {len(tags)} artist tags for {artist_name}")
                return tags
                
        except Exception as e:
            logger.debug(f"Could not fetch artist tags for {artist_name}: {e}")
            
        logger.warning(f"No tags found for '{track_name}' by {artist_name}")
        return []
        
    except Exception as e:
        logger.error(f"Error enriching with tags: {e}")
        return []


def get_artist_tags(artist_name: str, max_tags: int = 15) -> List[str]:
    """
    Get tags for a specific artist from Last.fm
    
    Args:
        artist_name: Name of the artist
        max_tags: Maximum number of tags to return
        
    Returns:
        List of tag strings
    """
    try:
        network = get_lastfm_network()
        
        artist = network.get_artist(artist_name)
        artist_tags = artist.get_top_tags(limit=max_tags)
        
        if artist_tags:
            tags = [tag.item.name.lower() for tag in artist_tags]
            logger.info(f"Found {len(tags)} tags for artist: {artist_name}")
            return tags
        else:
            logger.warning(f"No tags found for artist: {artist_name}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching artist tags: {e}")
        return []


def get_track_info(track_name: str, artist_name: str) -> Optional[Dict]:
    """
    Get additional track information from Last.fm
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        
    Returns:
        Dictionary with track information or None if not found
    """
    try:
        network = get_lastfm_network()
        
        track = network.get_track(artist_name, track_name)
        
        # Get basic track info
        track_info = {
            'name': track.get_name(),
            'artist': track.get_artist().get_name(),
            'listeners': track.get_listener_count(),
            'playcount': track.get_playcount(),
            'url': track.get_url(),
            'tags': []
        }
        
        # Get track tags
        try:
            tags = track.get_top_tags(limit=10)
            track_info['tags'] = [tag.item.name.lower() for tag in tags]
        except Exception as e:
            logger.debug(f"Could not fetch tags for track: {e}")
        
        # Get album info if available
        try:
            album = track.get_album()
            if album:
                track_info['album'] = album.get_name()
        except Exception as e:
            logger.debug(f"Could not fetch album info: {e}")
            
        logger.info(f"Successfully fetched track info for '{track_name}' by {artist_name}")
        return track_info
        
    except Exception as e:
        logger.error(f"Error fetching track info: {e}")
        return None


def get_similar_tracks(track_name: str, artist_name: str, limit: int = 10) -> List[Dict]:
    """
    Get similar tracks from Last.fm
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        limit: Maximum number of similar tracks to return
        
    Returns:
        List of similar track dictionaries
    """
    try:
        network = get_lastfm_network()
        
        track = network.get_track(artist_name, track_name)
        similar_tracks = track.get_similar(limit=limit)
        
        similar_list = []
        for similar_track in similar_tracks:
            similar_info = {
                'name': similar_track.item.get_name(),
                'artist': similar_track.item.get_artist().get_name(),
                'match_score': similar_track.match,
                'url': similar_track.item.get_url()
            }
            similar_list.append(similar_info)
            
        logger.info(f"Found {len(similar_list)} similar tracks for '{track_name}' by {artist_name}")
        return similar_list
        
    except Exception as e:
        logger.error(f"Error fetching similar tracks: {e}")
        return []


def get_artist_info(artist_name: str) -> Optional[Dict]:
    """
    Get artist information from Last.fm
    
    Args:
        artist_name: Name of the artist
        
    Returns:
        Dictionary with artist information or None if not found
    """
    try:
        network = get_lastfm_network()
        
        artist = network.get_artist(artist_name)
        
        artist_info = {
            'name': artist.get_name(),
            'listeners': artist.get_listener_count(),
            'playcount': artist.get_playcount(),
            'url': artist.get_url(),
            'tags': [],
            'bio': None
        }
        
        # Get artist tags
        try:
            tags = artist.get_top_tags(limit=15)
            artist_info['tags'] = [tag.item.name.lower() for tag in tags]
        except Exception as e:
            logger.debug(f"Could not fetch tags for artist: {e}")
        
        # Get artist bio
        try:
            bio = artist.get_bio_summary()
            if bio:
                artist_info['bio'] = bio
        except Exception as e:
            logger.debug(f"Could not fetch bio for artist: {e}")
            
        logger.info(f"Successfully fetched artist info for: {artist_name}")
        return artist_info
        
    except Exception as e:
        logger.error(f"Error fetching artist info: {e}")
        return None


def filter_relevant_tags(tags: List[str]) -> List[str]:
    """
    Filter tags to keep only music-relevant ones and remove generic tags
    
    Args:
        tags: List of raw tags from Last.fm
        
    Returns:
        List of filtered, relevant tags
    """
    # Generic tags to filter out
    generic_tags = {
        'seen live', 'favorites', 'favourite', 'love', 'loved', 'awesome', 
        'great', 'good', 'cool', 'nice', 'beautiful', 'amazing', 'perfect',
        'best', 'classic', 'legendary', 'epic', 'masterpiece', 'genius',
        'under 2000 listeners', 'recommended', 'discover', 'new', 'fresh'
    }
    
    # Keep only relevant music tags
    relevant_tags = []
    for tag in tags:
        tag_lower = tag.lower().strip()
        if (len(tag_lower) > 2 and 
            tag_lower not in generic_tags and
            not tag_lower.isdigit() and
            len(tag_lower) < 25):  # Avoid overly long descriptive tags
            relevant_tags.append(tag_lower)
    
    return relevant_tags[:10]  # Limit to top 10 relevant tags


def get_mood_tags(track_name: str, artist_name: str) -> List[str]:
    """
    Get mood-specific tags for a track, focusing on emotional/atmospheric descriptors
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        
    Returns:
        List of mood-related tags
    """
    # Common mood-related keywords
    mood_keywords = {
        'happy', 'sad', 'melancholy', 'upbeat', 'energetic', 'calm', 'relaxing',
        'dreamy', 'dark', 'bright', 'atmospheric', 'ambient', 'chill', 'intense',
        'aggressive', 'peaceful', 'nostalgic', 'romantic', 'dramatic', 'moody',
        'euphoric', 'hypnotic', 'meditative', 'uplifting', 'depressing', 'soothing',
        'haunting', 'ethereal', 'groovy', 'funky', 'smooth', 'rough', 'raw'
    }
    
    all_tags = enrich_with_tags(track_name, artist_name, max_tags=20)
    mood_tags = []
    
    for tag in all_tags:
        if any(mood_word in tag.lower() for mood_word in mood_keywords):
            mood_tags.append(tag)
    
    return mood_tags[:5]  # Return top 5 mood tags 