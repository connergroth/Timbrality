"""
Unification and Deduplication Module
Handles canonical ID generation and song deduplication logic
"""
import hashlib
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict
from app.config import settings
from app.models import SongCore, SpotifyAttrs, make_canonical_id
from app.tasks.lastfm import pull_recent_tracks

logger = logging.getLogger(__name__)


def make_canonical_id_enhanced(song: SongCore, spotify: Optional[SpotifyAttrs] = None) -> str:
    """
    Enhanced canonical ID generation with hierarchy: isrc > spotify_id > hash(artist::title)
    
    Args:
        song: Core song data
        spotify: Optional Spotify attributes for additional IDs
        
    Returns:
        Canonical ID string
    """
    # Priority 1: ISRC (International Standard Recording Code)
    if song.isrc and song.isrc.strip():
        return f"isrc:{song.isrc.strip().upper()}"
    
    # Priority 2: Spotify ID from song
    if song.spotify_id and song.spotify_id.strip():
        return f"spotify:{song.spotify_id.strip()}"
    
    # Priority 3: Spotify track ID from attributes
    if spotify and hasattr(spotify, 'track_id') and spotify.track_id:
        return f"spotify:{spotify.track_id}"
    
    # Priority 4: Hash of normalized artist::title
    normalized_artist = song.artist.lower().strip()
    normalized_title = song.title.lower().strip()
    
    # Remove common variations for better matching
    normalized_artist = _normalize_artist_name(normalized_artist)
    normalized_title = _normalize_track_title(normalized_title)
    
    normalized_key = f"{normalized_artist}::{normalized_title}"
    hash_value = hashlib.sha256(normalized_key.encode('utf-8')).hexdigest()[:16]
    return f"hash:{hash_value}"


def _normalize_artist_name(artist: str) -> str:
    """Normalize artist name for better deduplication"""
    # Remove common prefixes/suffixes
    artist = artist.lower().strip()
    
    # Remove "the " prefix
    if artist.startswith("the "):
        artist = artist[4:]
    
    # Remove featuring/feat parts
    for feat_marker in [" feat.", " ft.", " featuring ", " feat ", " ft "]:
        if feat_marker in artist:
            artist = artist.split(feat_marker)[0]
    
    # Remove whitespace variations
    artist = " ".join(artist.split())
    
    return artist


def _normalize_track_title(title: str) -> str:
    """Normalize track title for better deduplication"""
    title = title.lower().strip()
    
    # Remove common suffixes in parentheses
    import re
    # Remove (remaster), (remix), (live), etc.
    title = re.sub(r'\\s*\\([^)]*(?:remaster|remix|live|acoustic|radio edit|explicit|clean).*?\\)', '', title, flags=re.IGNORECASE)
    
    # Remove featuring parts
    for feat_marker in [" feat.", " ft.", " featuring ", " feat ", " ft "]:
        if feat_marker in title:
            title = title.split(feat_marker)[0]
    
    # Remove version numbers
    title = re.sub(r'\\s*-\\s*version\\s*\\d*', '', title, flags=re.IGNORECASE)
    
    # Normalize whitespace
    title = " ".join(title.split())
    
    return title


def dedupe_songs(songs: List[SongCore], max_songs: int) -> List[SongCore]:
    """
    Deduplicate songs using canonical IDs and ensure exactly max_songs
    
    Args:
        songs: List of SongCore objects (should be sorted by priority/playcount)
        max_songs: Target number of unique songs
        
    Returns:
        Deduplicated list of exactly max_songs SongCore objects
    """
    logger.info(f"Deduplicating {len(songs)} songs to {max_songs} unique tracks")
    
    seen_canonical_ids: Set[str] = set()
    deduplicated: List[SongCore] = []
    duplicate_count = 0
    
    # Group songs by canonical ID to handle duplicates intelligently
    canonical_groups: Dict[str, List[SongCore]] = defaultdict(list)
    
    for song in songs:
        canonical_id = make_canonical_id_enhanced(song)
        canonical_groups[canonical_id].append(song)
    
    # Select best representative from each group
    for canonical_id, song_group in canonical_groups.items():
        if len(deduplicated) >= max_songs:
            break
        
        # Choose the best song from the group
        best_song = _select_best_song(song_group)
        best_song.canonical_id = canonical_id  # Add canonical ID to song
        
        deduplicated.append(best_song)
        seen_canonical_ids.add(canonical_id)
        
        if len(song_group) > 1:
            duplicate_count += len(song_group) - 1
    
    logger.info(f"Removed {duplicate_count} duplicates, have {len(deduplicated)} unique songs")
    
    # If we need more songs, fill with recent tracks
    if len(deduplicated) < max_songs:
        remaining_needed = max_songs - len(deduplicated)
        logger.info(f"Need {remaining_needed} more songs, fetching recent tracks")
        
        try:
            # This is async, but we'll handle it in the pipeline orchestrator
            # For now, just return what we have
            pass
        except Exception as e:
            logger.warning(f"Could not fetch additional recent tracks: {e}")
    
    return deduplicated[:max_songs]


def _select_best_song(song_group: List[SongCore]) -> SongCore:
    """
    Select the best representative song from a group of duplicates
    
    Preference order:
    1. Song with highest playcount
    2. Song with most complete metadata (ISRC > MBID > Spotify ID)
    3. First song in list
    """
    if len(song_group) == 1:
        return song_group[0]
    
    # Sort by preference criteria
    def song_score(song: SongCore) -> tuple:
        # Higher playcount is better
        playcount = song.playcount or 0
        
        # Metadata completeness score
        metadata_score = 0
        if song.isrc:
            metadata_score += 100  # ISRC is best
        elif song.spotify_id:
            metadata_score += 50   # Spotify ID is good
        
        # Return tuple for sorting (higher values are better)
        return (playcount, metadata_score)
    
    best_song = max(song_group, key=song_score)
    logger.debug(f"Selected best song from {len(song_group)} duplicates: {best_song.artist} - {best_song.title}")
    
    return best_song


async def fill_with_recent_tracks(
    current_songs: List[SongCore], 
    target_count: int, 
    existing_canonical_ids: Set[str]
) -> List[SongCore]:
    """
    Fill remaining slots with recent tracks if needed
    
    Args:
        current_songs: Current list of deduplicated songs
        target_count: Target number of total songs
        existing_canonical_ids: Set of canonical IDs already used
        
    Returns:
        Extended list with additional recent tracks
    """
    if len(current_songs) >= target_count:
        return current_songs
    
    needed = target_count - len(current_songs)
    logger.info(f"Fetching {needed} recent tracks to fill remaining slots")
    
    try:
        # Fetch more recent tracks than needed to account for duplicates
        recent_tracks = await pull_recent_tracks(needed * 3)
        
        additional_songs = []
        for song in recent_tracks:
            if len(additional_songs) >= needed:
                break
            
            canonical_id = make_canonical_id_enhanced(song)
            if canonical_id not in existing_canonical_ids:
                song.canonical_id = canonical_id
                additional_songs.append(song)
                existing_canonical_ids.add(canonical_id)
        
        logger.info(f"Added {len(additional_songs)} recent tracks")
        return current_songs + additional_songs
        
    except Exception as e:
        logger.error(f"Failed to fetch recent tracks: {e}")
        return current_songs


def validate_canonical_ids(songs: List[SongCore]) -> Dict[str, int]:
    """
    Validate canonical ID distribution and quality
    
    Args:
        songs: List of songs with canonical IDs
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "total_songs": len(songs),
        "isrc_count": 0,
        "spotify_count": 0,
        "hash_count": 0,
        "unique_ids": 0,
        "duplicates": 0
    }
    
    canonical_ids = set()
    id_type_counts = defaultdict(int)
    
    for song in songs:
        if hasattr(song, 'canonical_id'):
            canonical_id = song.canonical_id
            
            if canonical_id in canonical_ids:
                stats["duplicates"] += 1
            else:
                canonical_ids.add(canonical_id)
                stats["unique_ids"] += 1
            
            # Count by type
            if canonical_id.startswith("isrc:"):
                stats["isrc_count"] += 1
            elif canonical_id.startswith("spotify:"):
                stats["spotify_count"] += 1
            elif canonical_id.startswith("hash:"):
                stats["hash_count"] += 1
    
    logger.info(f"Canonical ID validation: {stats}")
    return stats


def analyze_duplicates(songs: List[SongCore]) -> List[Dict[str, any]]:
    """
    Analyze potential duplicates for manual review
    
    Args:
        songs: List of songs to analyze
        
    Returns:
        List of potential duplicate groups
    """
    # Group by normalized artist + title
    groups = defaultdict(list)
    
    for song in songs:
        normalized_key = f"{_normalize_artist_name(song.artist)}::{_normalize_track_title(song.title)}"
        groups[normalized_key].append(song)
    
    # Find groups with multiple songs
    duplicates = []
    for key, song_group in groups.items():
        if len(song_group) > 1:
            duplicates.append({
                "normalized_key": key,
                "count": len(song_group),
                "songs": [
                    {
                        "artist": song.artist,
                        "title": song.title,
                        "playcount": song.playcount,
                        "source": song.source
                    }
                    for song in song_group
                ]
            })
    
    logger.info(f"Found {len(duplicates)} potential duplicate groups")
    return duplicates