"""
Normalizer - Merge & align data from multiple sources into unified format
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class TrackData:
    """Unified track data structure for ML training"""
    track_id: str
    title: str
    artist: str
    album: str
    release_date: Optional[str] = None
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    genres: List[str] = None
    moods: List[str] = None
    aoty_score: Optional[float] = None
    spotify_url: Optional[str] = None
    explicit: Optional[bool] = None
    track_number: Optional[int] = None
    album_total_tracks: Optional[int] = None
    
    def __post_init__(self):
        if self.genres is None:
            self.genres = []
        if self.moods is None:
            self.moods = []


def clean_string(text: str) -> str:
    """Clean and normalize text strings"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.\,\&\(\)\[\]]', '', text)
    
    return text


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date string to YYYY-MM-DD format"""
    if not date_str:
        return None
    
    try:
        # Handle various date formats
        date_str = date_str.strip()
        
        # YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # YYYY-MM format
        if re.match(r'^\d{4}-\d{2}$', date_str):
            return f"{date_str}-01"
        
        # YYYY format
        if re.match(r'^\d{4}$', date_str):
            return f"{date_str}-01-01"
        
        # Try to parse with datetime
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
        
        try:
            dt = datetime.strptime(date_str, '%Y-%m')
            return dt.strftime('%Y-%m-01')
        except ValueError:
            pass
            
        try:
            dt = datetime.strptime(date_str, '%Y')
            return dt.strftime('%Y-01-01')
        except ValueError:
            pass
        
        logger.warning(f"Could not normalize date: {date_str}")
        return None
        
    except Exception as e:
        logger.error(f"Error normalizing date '{date_str}': {e}")
        return None


def normalize_genres(genres: List[str]) -> List[str]:
    """Normalize and clean genre tags"""
    if not genres:
        return []
    
    normalized = []
    for genre in genres:
        if not genre:
            continue
            
        # Clean and normalize genre
        genre = clean_string(genre.lower())
        
        # Skip empty or very short genres
        if len(genre) < 2:
            continue
        
        # Common genre normalizations
        genre_mappings = {
            'alternative rock': 'alternative',
            'indie rock': 'indie',
            'electronic music': 'electronic',
            'hip hop': 'hip-hop',
            'rhythm and blues': 'r&b',
            'country music': 'country',
            'classic rock': 'rock',
            'hard rock': 'rock',
            'soft rock': 'rock',
            'pop rock': 'pop',
            'dance music': 'dance',
            'experimental music': 'experimental'
        }
        
        # Apply genre mappings
        for original, normalized_genre in genre_mappings.items():
            if original in genre:
                genre = normalized_genre
                break
        
        if genre not in normalized:
            normalized.append(genre)
    
    return normalized[:5]  # Limit to top 5 genres


def normalize_moods(moods: List[str]) -> List[str]:
    """Normalize and clean mood tags"""
    if not moods:
        return []
    
    normalized = []
    for mood in moods:
        if not mood:
            continue
            
        # Clean and normalize mood
        mood = clean_string(mood.lower())
        
        # Skip empty or very short moods
        if len(mood) < 3:
            continue
        
        # Common mood normalizations
        mood_mappings = {
            'melancholic': 'melancholy',
            'energetic': 'energetic',
            'calm': 'calm',
            'relaxed': 'relaxing',
            'happy': 'upbeat',
            'sad': 'melancholy',
            'aggressive': 'intense',
            'peaceful': 'calm',
            'nostalgic': 'nostalgic',
            'romantic': 'romantic',
            'dark': 'dark',
            'bright': 'uplifting',
            'atmospheric': 'atmospheric',
            'dreamy': 'dreamy',
            'groovy': 'groovy'
        }
        
        # Apply mood mappings
        for original, normalized_mood in mood_mappings.items():
            if original in mood:
                mood = normalized_mood
                break
        
        if mood not in normalized:
            normalized.append(mood)
    
    return normalized[:5]  # Limit to top 5 moods


def normalize_track(spotify_track: Dict, lastfm_tags: List[str], aoty_data: Dict) -> TrackData:
    """
    Normalize track data from multiple sources into unified format
    
    Args:
        spotify_track: Track data from Spotify
        lastfm_tags: Tags from Last.fm
        aoty_data: Album/track data from AOTY
        
    Returns:
        TrackData object with normalized data
    """
    try:
        # Extract basic track info
        track_id = spotify_track.get('id', '')
        title = clean_string(spotify_track.get('name', ''))
        artist = clean_string(spotify_track.get('artist', ''))
        album = clean_string(spotify_track.get('album', ''))
        
        # Normalize release date
        release_date = normalize_date(spotify_track.get('release_date', ''))
        
        # Extract numerical values
        duration_ms = spotify_track.get('duration_ms')
        popularity = spotify_track.get('popularity')
        track_number = spotify_track.get('track_number')
        album_total_tracks = spotify_track.get('total_tracks')
        
        # Extract boolean values
        explicit = spotify_track.get('explicit', False)
        
        # Extract URLs
        spotify_url = spotify_track.get('spotify_url')
        
        # Normalize genres from AOTY data
        genres = []
        if aoty_data and 'genres' in aoty_data:
            genres = normalize_genres(aoty_data['genres'])
        
        # Normalize moods from Last.fm tags
        moods = normalize_moods(lastfm_tags)
        
        # Extract AOTY score
        aoty_score = None
        if aoty_data and 'track_scores' in aoty_data:
            # Try to find the track score
            track_scores = aoty_data['track_scores']
            if title in track_scores:
                aoty_score = track_scores[title]
            else:
                # Try case-insensitive match
                for aoty_title, score in track_scores.items():
                    if title.lower() == aoty_title.lower():
                        aoty_score = score
                        break
        
        # If no track score, use album score
        if aoty_score is None and aoty_data and 'album_score' in aoty_data:
            aoty_score = aoty_data['album_score']
        
        # Create normalized track data
        normalized_track = TrackData(
            track_id=track_id,
            title=title,
            artist=artist,
            album=album,
            release_date=release_date,
            duration_ms=duration_ms,
            popularity=popularity,
            genres=genres,
            moods=moods,
            aoty_score=aoty_score,
            spotify_url=spotify_url,
            explicit=explicit,
            track_number=track_number,
            album_total_tracks=album_total_tracks
        )
        
        logger.info(f"Successfully normalized track: '{title}' by {artist}")
        return normalized_track
        
    except Exception as e:
        logger.error(f"Error normalizing track data: {e}")
        # Return a minimal track data object
        return TrackData(
            track_id=spotify_track.get('id', ''),
            title=clean_string(spotify_track.get('name', '')),
            artist=clean_string(spotify_track.get('artist', '')),
            album=clean_string(spotify_track.get('album', ''))
        )


def normalize_batch(tracks_data: List[tuple]) -> List[TrackData]:
    """
    Normalize a batch of tracks
    
    Args:
        tracks_data: List of (spotify_track, lastfm_tags, aoty_data) tuples
        
    Returns:
        List of normalized TrackData objects
    """
    normalized_tracks = []
    
    for spotify_track, lastfm_tags, aoty_data in tracks_data:
        try:
            normalized_track = normalize_track(spotify_track, lastfm_tags, aoty_data)
            normalized_tracks.append(normalized_track)
        except Exception as e:
            logger.error(f"Error normalizing track in batch: {e}")
            continue
    
    logger.info(f"Successfully normalized {len(normalized_tracks)} tracks in batch")
    return normalized_tracks


def to_dict(track: TrackData) -> Dict[str, Any]:
    """Convert TrackData to dictionary for database insertion"""
    return {
        'id': track.track_id,
        'title': track.title,
        'artist': track.artist,
        'album': track.album,
        'genre': track.genres[0] if track.genres else None,  # Use first genre for existing column
        'release_date': track.release_date,
        'duration_ms': track.duration_ms,
        'popularity': track.popularity,
        'genres': track.genres,  # New array column
        'moods': track.moods,    # New array column
        'aoty_score': int(track.aoty_score) if track.aoty_score else None,  # Keep as integer for existing schema
        'audio_features': {},    # Empty for now - can be enhanced later
        'cover_url': None,       # Can be added later
        'spotify_url': track.spotify_url,
        'explicit': track.explicit,
        'track_number': track.track_number,
        'album_total_tracks': track.album_total_tracks
    }


def validate_track_data(track: TrackData) -> bool:
    """Validate that track data meets minimum requirements"""
    if not track.track_id:
        logger.warning("Track missing ID")
        return False
    
    if not track.title:
        logger.warning(f"Track {track.track_id} missing title")
        return False
    
    if not track.artist:
        logger.warning(f"Track {track.track_id} missing artist")
        return False
    
    return True


def deduplicate_tracks(tracks: List[TrackData]) -> List[TrackData]:
    """Remove duplicate tracks based on track_id"""
    seen_ids = set()
    deduplicated = []
    
    for track in tracks:
        if track.track_id not in seen_ids:
            seen_ids.add(track.track_id)
            deduplicated.append(track)
        else:
            logger.debug(f"Skipping duplicate track: {track.track_id}")
    
    logger.info(f"Deduplicated {len(tracks)} tracks to {len(deduplicated)} unique tracks")
    return deduplicated 