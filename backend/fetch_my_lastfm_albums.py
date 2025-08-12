#!/usr/bin/env python3
"""
Fetch My Last.fm Top Albums Script

This script fetches your top albums from Last.fm (username: connergroth) and
then fetches all tracks for each album with full data enrichment:
- Last.fm: album info, play counts, tags
- AOTY: album ratings, reviews, complete track listings
- Spotify: track metadata and audio features
- Perfect matching since we have album context

This approach ensures much better AOTY matching rates since we're working with
complete albums rather than individual tracks.

Usage:
    python fetch_my_lastfm_albums.py <number_of_albums>
    python fetch_my_lastfm_albums.py 10  # Fetch top 10 albums
"""

import asyncio
import sys
import os
import logging
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import json

# Add the backend directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

try:
    import pylast
    PYLAST_AVAILABLE = True
except ImportError:
    PYLAST_AVAILABLE = False
    print("[FAIL] pylast not found. Install with: pip install pylast")

from services.music_matching_service import MusicMatchingService, LastfmTrack
from scraper.utils.scraper import get_album_url, scrape_album
from scraper.models import Album as AOTYAlbum
from ingestion.lastfm_fetcher import get_lastfm_network, filter_relevant_tags
from ingestion.spotify_fetcher import get_album_tracks, search_albums
from ingestion.insert_to_supabase import setup_database, get_track_count, insert_tracks
from ingestion.normalizer import TrackData
from utils.fuzzy_matcher import MusicFuzzyMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LASTFM_USERNAME = "connergroth"


@dataclass
class EnrichedAlbumTrack:
    """Complete track data from all sources with album context"""
    # Basic track info
    name: str
    artist: str
    album: str
    track_number: int = 0
    duration: str = ""
    
    # Last.fm data
    lastfm_playcount: int = 0
    lastfm_listeners: int = 0
    lastfm_tags: List[str] = None
    lastfm_url: str = ""
    
    # Album-level Last.fm data
    album_lastfm_playcount: int = 0
    album_lastfm_listeners: int = 0
    album_lastfm_tags: List[str] = None
    album_rank: int = 0  # User's album ranking
    
    # AOTY data
    aoty_match_found: bool = False
    aoty_confidence: str = "none"
    aoty_album_score: float = 0.0
    aoty_album_ratings: int = 0
    aoty_track_rating: Optional[int] = None
    aoty_track_length: str = ""
    aoty_is_must_hear: bool = False
    aoty_critic_reviews_count: int = 0
    aoty_user_reviews_count: int = 0
    
    # Spotify data
    spotify_match_found: bool = False
    spotify_track_id: str = ""
    spotify_preview_url: str = ""
    spotify_track_url: str = ""
    spotify_popularity: int = 0
    spotify_explicit: bool = False
    spotify_duration_ms: int = 0
    spotify_audio_features: Dict = None
    
    # Processing metadata
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.lastfm_tags is None:
            self.lastfm_tags = []
        if self.album_lastfm_tags is None:
            self.album_lastfm_tags = []
        if self.errors is None:
            self.errors = []
        if self.spotify_audio_features is None:
            self.spotify_audio_features = {}


@dataclass
class EnrichedAlbum:
    """Complete album data with all tracks"""
    # Basic album info
    name: str
    artist: str
    year: Optional[int] = None
    
    # Last.fm data
    lastfm_playcount: int = 0
    lastfm_listeners: int = 0
    lastfm_tags: List[str] = None
    lastfm_url: str = ""
    album_rank: int = 0
    
    # AOTY data
    aoty_match_found: bool = False
    aoty_url: str = ""
    aoty_score: float = 0.0
    aoty_num_ratings: int = 0
    aoty_critic_score: Optional[float] = None
    aoty_is_must_hear: bool = False
    aoty_genres: List[str] = None
    aoty_critic_reviews: int = 0
    aoty_user_reviews: int = 0
    
    # Spotify data
    spotify_match_found: bool = False
    spotify_album_id: str = ""
    spotify_total_tracks: int = 0
    spotify_release_date: str = ""
    spotify_album_url: str = ""
    spotify_cover_url: str = ""
    
    # Tracks
    tracks: List[EnrichedAlbumTrack] = None
    total_tracks: int = 0
    
    # Processing metadata
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.lastfm_tags is None:
            self.lastfm_tags = []
        if self.aoty_genres is None:
            self.aoty_genres = []
        if self.tracks is None:
            self.tracks = []
        if self.errors is None:
            self.errors = []


class LastfmTopAlbumsProcessor:
    """Process Last.fm top albums with complete track enrichment"""
    
    def __init__(self):
        self.matching_service = MusicMatchingService()
        self.fuzzy_matcher = MusicFuzzyMatcher()
        self.stats = {
            "albums_fetched": 0,
            "albums_enriched": 0,
            "aoty_albums_matched": 0,
            "spotify_albums_matched": 0,
            "total_tracks_found": 0,
            "tracks_with_aoty_data": 0,
            "tracks_with_spotify_data": 0,
            "processing_time": 0.0,
            "errors": []
        }
    
    def fetch_user_top_albums(self, username: str, limit: int = 20, period: str = "overall") -> List[Dict[str, Any]]:
        """Fetch top albums from Last.fm for a user"""
        if not PYLAST_AVAILABLE:
            raise ImportError("pylast is required. Install with: pip install pylast")
        
        logger.info(f"Fetching top {limit} albums for Last.fm user: {username}")
        
        try:
            network = get_lastfm_network()
            user = network.get_user(username)
            
            # Available periods: overall, 7day, 1month, 3month, 6month, 12month
            period_map = {
                "overall": pylast.PERIOD_OVERALL,
                "7day": pylast.PERIOD_7DAYS,
                "1month": pylast.PERIOD_1MONTH,
                "3month": pylast.PERIOD_3MONTHS,
                "6month": pylast.PERIOD_6MONTHS,
                "12month": pylast.PERIOD_12MONTHS
            }
            
            period_param = period_map.get(period, pylast.PERIOD_OVERALL)
            
            logger.info(f"Using time period: {period}")
            top_albums = user.get_top_albums(period=period_param, limit=limit)
            
            if not top_albums:
                logger.warning(f"No top albums found for user {username}")
                return []
            
            albums_data = []
            for i, album_item in enumerate(top_albums, 1):
                try:
                    album = album_item.item
                    playcount = int(album_item.weight) if album_item.weight else 0
                    
                    album_data = {
                        "name": album.get_name(),
                        "artist": album.get_artist().get_name(),
                        "playcount": playcount,
                        "rank": i,
                        "lastfm_url": album.get_url()
                    }
                    
                    # Get additional album info
                    try:
                        album_data["listeners"] = album.get_listener_count()
                    except Exception:
                        album_data["listeners"] = 0
                    
                    # Try to get release year
                    try:
                        album_info = album.get_wiki_summary()
                        # Extract year from wiki if available (basic extraction)
                        album_data["year"] = None  # Would need more sophisticated parsing
                    except Exception:
                        album_data["year"] = None
                    
                    albums_data.append(album_data)
                    logger.info(f"  {i:2d}. {album_data['artist']} - {album_data['name']} ({playcount} plays)")
                    
                except Exception as e:
                    logger.error(f"Error processing album {i}: {e}")
                    self.stats["errors"].append(f"Album {i} processing error: {e}")
                    continue
            
            self.stats["albums_fetched"] = len(albums_data)
            logger.info(f"[SUCCESS] Successfully fetched {len(albums_data)} top albums")
            return albums_data
            
        except Exception as e:
            logger.error(f"Error fetching top albums for user {username}: {e}")
            self.stats["errors"].append(f"User fetch error: {e}")
            return []
    
    async def enrich_album_with_all_sources(self, album_data: Dict[str, Any]) -> EnrichedAlbum:
        """Enrich a single album with data from all sources"""
        start_time = time.time()
        
        enriched = EnrichedAlbum(
            name=album_data["name"],
            artist=album_data["artist"],
            year=album_data.get("year"),
            lastfm_playcount=album_data.get("playcount", 0),
            lastfm_listeners=album_data.get("listeners", 0),
            lastfm_url=album_data.get("lastfm_url", ""),
            album_rank=album_data.get("rank", 0)
        )
        
        logger.info(f"[ALBUM] Enriching: {enriched.artist} - {enriched.name}")
        
        # 1. Enrich with Last.fm album tags
        await self.enrich_with_lastfm_album_data(enriched)
        
        # 2. Get AOTY album data (this is the key improvement!)
        await self.enrich_with_aoty_album_data(enriched)
        
        # 3. Get Spotify album data
        await self.enrich_with_spotify_album_data(enriched)
        
        # 4. Get all tracks for the album
        await self.fetch_album_tracks(enriched)
        
        enriched.processing_time = time.time() - start_time
        enriched.total_tracks = len(enriched.tracks)
        
        logger.info(f"[ALBUM] Completed: {enriched.artist} - {enriched.name} "
                   f"({enriched.total_tracks} tracks, AOTY: {'YES' if enriched.aoty_match_found else 'NO'})")
        
        self.stats["albums_enriched"] += 1
        self.stats["total_tracks_found"] += enriched.total_tracks
        
        if enriched.aoty_match_found:
            self.stats["aoty_albums_matched"] += 1
            self.stats["tracks_with_aoty_data"] += enriched.total_tracks
        
        if enriched.spotify_match_found:
            self.stats["spotify_albums_matched"] += 1
        
        return enriched
    
    async def enrich_with_lastfm_album_data(self, enriched: EnrichedAlbum):
        """Enrich with Last.fm album tags and metadata"""
        try:
            network = get_lastfm_network()
            album = network.get_album(enriched.artist, enriched.name)
            
            # Get album tags
            try:
                album_tags = album.get_top_tags(limit=15)
                raw_tags = [tag.item.name.lower() for tag in album_tags]
                enriched.lastfm_tags = filter_relevant_tags(raw_tags)
                
                if enriched.lastfm_tags:
                    logger.debug(f"  Last.fm album tags: {', '.join(enriched.lastfm_tags[:5])}")
                
            except Exception as e:
                logger.debug(f"  Could not fetch album tags: {e}")
                # Fallback to artist tags
                try:
                    artist = network.get_artist(enriched.artist)
                    artist_tags = artist.get_top_tags(limit=10)
                    raw_tags = [tag.item.name.lower() for tag in artist_tags]
                    enriched.lastfm_tags = filter_relevant_tags(raw_tags)
                except Exception as e2:
                    logger.debug(f"  Could not fetch artist tags either: {e2}")
            
        except Exception as e:
            error_msg = f"Last.fm album enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def enrich_with_aoty_album_data(self, enriched: EnrichedAlbum):
        """Get complete AOTY album data using CloudScraper"""
        try:
            logger.debug(f"  [AOTY] Searching for album: {enriched.artist} - {enriched.name}")
            
            # Use the new CloudScraper to find the album
            url_result = await get_album_url(enriched.artist, enriched.name)
            
            if not url_result:
                logger.debug(f"  [AOTY] No URL found for {enriched.artist} - {enriched.name}")
                return
            
            url, found_artist, found_title = url_result
            logger.debug(f"  [AOTY] Found: {found_artist} - {found_title}")
            
            # Scrape complete album data
            aoty_album = await scrape_album(url, found_artist, found_title)
            
            if aoty_album:
                enriched.aoty_match_found = True
                enriched.aoty_url = url
                enriched.aoty_score = aoty_album.user_score or 0.0
                enriched.aoty_num_ratings = aoty_album.num_ratings or 0
                enriched.aoty_critic_score = getattr(aoty_album, 'critic_score', None)
                enriched.aoty_is_must_hear = aoty_album.is_must_hear or False
                enriched.aoty_critic_reviews = len(aoty_album.critic_reviews) if aoty_album.critic_reviews else 0
                enriched.aoty_user_reviews = len(aoty_album.popular_reviews) if aoty_album.popular_reviews else 0
                
                # Extract genres from metadata if available
                if hasattr(aoty_album, 'metadata') and aoty_album.metadata and hasattr(aoty_album.metadata, 'genres'):
                    enriched.aoty_genres = aoty_album.metadata.genres
                
                logger.debug(f"  [AOTY] Album data: Score {enriched.aoty_score}, "
                           f"{enriched.aoty_num_ratings} ratings, {len(aoty_album.tracks)} tracks")
                
                # Store the complete AOTY album for track matching
                enriched._aoty_album = aoty_album
                
            else:
                logger.debug(f"  [AOTY] Failed to scrape album data")
                
        except Exception as e:
            error_msg = f"AOTY album enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def enrich_with_spotify_album_data(self, enriched: EnrichedAlbum):
        """Get Spotify album data with complete metadata"""
        try:
            # Search for albums on Spotify
            search_results = search_albums(f"{enriched.artist} {enriched.name}", limit=5)
            
            if search_results:
                # Find best match (this is a simplified approach)
                best_match = None
                for result in search_results:
                    if (enriched.artist.lower() in result.get('artist', '').lower() and
                        enriched.name.lower() in result.get('name', '').lower()):
                        best_match = result
                        break
                
                if not best_match and search_results:
                    best_match = search_results[0]  # Use first result as fallback
                
                if best_match:
                    enriched.spotify_match_found = True
                    enriched.spotify_album_id = best_match.get('id', '')
                    enriched.spotify_total_tracks = best_match.get('total_tracks', 0)
                    
                    # Extract additional Spotify metadata
                    enriched.spotify_release_date = best_match.get('release_date', '')
                    enriched.spotify_album_url = best_match.get('spotify_url', '')
                    
                    # Get cover image URL (use largest available)
                    images = best_match.get('images', [])
                    if images:
                        enriched.spotify_cover_url = images[0]['url']  # First image is usually largest
                    
                    logger.debug(f"  [SPOTIFY] Found album: {best_match.get('name')}")
                    logger.debug(f"  [SPOTIFY] Release date: {enriched.spotify_release_date}")
                    logger.debug(f"  [SPOTIFY] Cover URL: {enriched.spotify_cover_url[:50] if hasattr(enriched, 'spotify_cover_url') else 'None'}...")
            
        except Exception as e:
            error_msg = f"Spotify album enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def fetch_album_tracks(self, enriched: EnrichedAlbum):
        """Fetch all tracks for the album from various sources"""
        tracks = []
        
        # Strategy 1: If we have AOTY data, use that as the primary source
        if enriched.aoty_match_found and hasattr(enriched, '_aoty_album'):
            aoty_album = enriched._aoty_album
            
            logger.debug(f"  [TRACKS] Using AOTY as primary source ({len(aoty_album.tracks)} tracks)")
            
            for aoty_track in aoty_album.tracks:
                track = EnrichedAlbumTrack(
                    name=aoty_track.title,
                    artist=enriched.artist,
                    album=enriched.name,
                    track_number=aoty_track.number,
                    duration=aoty_track.length or "",
                    
                    # Album-level data
                    album_lastfm_playcount=enriched.lastfm_playcount,
                    album_lastfm_listeners=enriched.lastfm_listeners,
                    album_lastfm_tags=enriched.lastfm_tags.copy(),
                    album_rank=enriched.album_rank,
                    
                    # AOTY data
                    aoty_match_found=True,
                    aoty_confidence="high",  # High confidence since we have the complete album
                    aoty_album_score=enriched.aoty_score,
                    aoty_album_ratings=enriched.aoty_num_ratings,
                    aoty_track_rating=aoty_track.rating,
                    aoty_track_length=aoty_track.length or "",
                    aoty_is_must_hear=enriched.aoty_is_must_hear,
                    aoty_critic_reviews_count=enriched.aoty_critic_reviews,
                    aoty_user_reviews_count=enriched.aoty_user_reviews
                )
                
                # Try to enrich individual track with Last.fm data
                await self.enrich_track_with_lastfm(track)
                
                # Also try to enrich with Spotify track data for ID and explicit flag
                if enriched.spotify_match_found:
                    await self.enrich_track_with_spotify(track, enriched)
                
                tracks.append(track)
        
        # Strategy 2: Use Spotify if available and no AOTY
        elif enriched.spotify_match_found:
            logger.debug(f"  [TRACKS] Using Spotify as primary source")
            try:
                # Get tracks from Spotify album using the album ID
                spotify_tracks = get_album_tracks(enriched.name, enriched.artist)
                
                if spotify_tracks:
                    for spotify_track in spotify_tracks:
                        # Convert duration from ms to string format
                        duration_str = ""
                        if spotify_track.get('duration_ms'):
                            total_seconds = spotify_track['duration_ms'] // 1000
                            minutes = total_seconds // 60
                            seconds = total_seconds % 60
                            duration_str = f"{minutes}:{seconds:02d}"
                        
                        track = EnrichedAlbumTrack(
                            name=spotify_track.get('name', ''),
                            artist=spotify_track.get('artist', enriched.artist),
                            album=enriched.name,
                            track_number=spotify_track.get('track_number', 0),
                            duration=duration_str,
                            
                            # Album-level data
                            album_lastfm_playcount=enriched.lastfm_playcount,
                            album_lastfm_listeners=enriched.lastfm_listeners,
                            album_lastfm_tags=enriched.lastfm_tags.copy(),
                            album_rank=enriched.album_rank,
                            
                            # Spotify data
                            spotify_match_found=True,
                            spotify_track_id=spotify_track.get('id', ''),
                            spotify_preview_url=spotify_track.get('preview_url', ''),
                            spotify_track_url=spotify_track.get('spotify_url', ''),
                            spotify_popularity=spotify_track.get('popularity', 0),
                            spotify_explicit=spotify_track.get('explicit', False),
                            spotify_duration_ms=spotify_track.get('duration_ms', 0)
                        )
                        
                        # Try to enrich with Last.fm data
                        await self.enrich_track_with_lastfm(track)
                        
                        tracks.append(track)
                
            except Exception as e:
                logger.debug(f"  [TRACKS] Spotify track fetch failed: {e}")
                enriched.errors.append(f"Spotify track fetch error: {e}")
        
        # Strategy 3: Last.fm only (if no other sources available)
        if not tracks:
            logger.debug(f"  [TRACKS] Using Last.fm only (no album tracks available)")
            # Create a placeholder track representing the album
            track = EnrichedAlbumTrack(
                name=f"Album: {enriched.name}",
                artist=enriched.artist,
                album=enriched.name,
                track_number=1,
                
                # Album-level data
                album_lastfm_playcount=enriched.lastfm_playcount,
                album_lastfm_listeners=enriched.lastfm_listeners,
                album_lastfm_tags=enriched.lastfm_tags.copy(),
                album_rank=enriched.album_rank
            )
            tracks.append(track)
        
        enriched.tracks = tracks
        logger.debug(f"  [TRACKS] Final track count: {len(tracks)}")
    
    async def enrich_track_with_lastfm(self, track: EnrichedAlbumTrack):
        """Enrich individual track with Last.fm data"""
        try:
            network = get_lastfm_network()
            lastfm_track = network.get_track(track.artist, track.name)
            
            # Get track-specific data
            try:
                track.lastfm_playcount = lastfm_track.get_playcount() or 0
                track.lastfm_listeners = lastfm_track.get_listener_count() or 0
                track.lastfm_url = lastfm_track.get_url() or ""
                
                # Get track tags
                track_tags = lastfm_track.get_top_tags(limit=10)
                raw_tags = [tag.item.name.lower() for tag in track_tags]
                track.lastfm_tags = filter_relevant_tags(raw_tags)
                
            except Exception as e:
                logger.debug(f"    Could not fetch Last.fm data for track {track.name}: {e}")
                # Use album tags as fallback
                track.lastfm_tags = track.album_lastfm_tags[:5] if track.album_lastfm_tags else []
        
        except Exception as e:
            logger.debug(f"    Last.fm track enrichment failed: {e}")
    
    async def enrich_track_with_spotify(self, track: EnrichedAlbumTrack, enriched_album: EnrichedAlbum):
        """Enrich individual track with Spotify data (ID, explicit flag, etc.)"""
        try:
            # Get track metadata from Spotify
            from ingestion.spotify_fetcher import get_track_metadata
            
            spotify_track_data = get_track_metadata(track.name, track.artist)
            
            if spotify_track_data:
                logger.debug(f"    [SPOTIFY] Found track data for {track.name}")
                
                # Extract Spotify track ID (remove prefix if present)
                spotify_id = spotify_track_data.get('id', '')
                if spotify_id.startswith('spotify:'):
                    spotify_id = spotify_id.replace('spotify:', '').replace('track:', '')
                
                # Update track with Spotify data
                track.spotify_match_found = True
                track.spotify_track_id = spotify_id
                track.spotify_explicit = spotify_track_data.get('explicit', False)
                track.spotify_popularity = spotify_track_data.get('popularity', 0)
                track.spotify_track_url = spotify_track_data.get('spotify_url', '')
                track.spotify_preview_url = spotify_track_data.get('preview_url', '')
                track.spotify_duration_ms = spotify_track_data.get('duration_ms', 0)
                
                logger.debug(f"    [SPOTIFY] Track ID: {spotify_id}")
                logger.debug(f"    [SPOTIFY] Explicit: {track.spotify_explicit}")
            else:
                logger.debug(f"    [SPOTIFY] No track data found for {track.name}")
                
        except Exception as e:
            logger.debug(f"    Spotify track enrichment failed for {track.name}: {e}")
    
    async def process_albums_batch(self, albums_data: List[Dict], max_concurrent: int = 2) -> List[EnrichedAlbum]:
        """Process multiple albums with controlled concurrency"""
        logger.info(f"[PROCESS] Processing {len(albums_data)} albums with max {max_concurrent} concurrent requests")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_album(album_data: Dict, index: int) -> EnrichedAlbum:
            async with semaphore:
                try:
                    enriched = await self.enrich_album_with_all_sources(album_data)
                    logger.info(f"[{index+1:2d}/{len(albums_data)}] Completed: {enriched.artist} - {enriched.name} "
                               f"({enriched.total_tracks} tracks, AOTY: {'YES' if enriched.aoty_match_found else 'NO'}, "
                               f"{enriched.processing_time:.2f}s)")
                    return enriched
                except Exception as e:
                    logger.error(f"[{index+1:2d}/{len(albums_data)}] Failed: {album_data.get('artist', 'Unknown')} - {album_data.get('name', 'Unknown')}: {e}")
                    # Return minimal enriched album with error
                    enriched = EnrichedAlbum(
                        name=album_data.get("name", "Unknown"),
                        artist=album_data.get("artist", "Unknown"),
                        lastfm_playcount=album_data.get("playcount", 0),
                        album_rank=album_data.get("rank", 0),
                        errors=[f"Processing failed: {e}"]
                    )
                    return enriched
        
        # Process all albums
        start_time = time.time()
        tasks = [process_single_album(album_data, i) for i, album_data in enumerate(albums_data)]
        enriched_albums = await asyncio.gather(*tasks)
        
        self.stats["processing_time"] = time.time() - start_time
        
        logger.info(f"[SUCCESS] Batch processing completed in {self.stats['processing_time']:.2f} seconds")
        return enriched_albums
    
    def save_to_json(self, enriched_albums: List[EnrichedAlbum], filename: str = "lastfm_albums_enriched.json"):
        """Save enriched albums to JSON for inspection"""
        try:
            # Convert to serializable format
            albums_data = []
            for album in enriched_albums:
                album_dict = {
                    "name": album.name,
                    "artist": album.artist,
                    "year": album.year,
                    "lastfm_playcount": album.lastfm_playcount,
                    "album_rank": album.album_rank,
                    "lastfm_tags": album.lastfm_tags,
                    "aoty_match_found": album.aoty_match_found,
                    "aoty_score": album.aoty_score,
                    "aoty_num_ratings": album.aoty_num_ratings,
                    "aoty_genres": album.aoty_genres,
                    "spotify_match_found": album.spotify_match_found,
                    "total_tracks": album.total_tracks,
                    "processing_time": album.processing_time,
                    "errors": album.errors,
                    "tracks": [
                        {
                            "name": track.name,
                            "track_number": track.track_number,
                            "duration": track.duration,
                            "lastfm_tags": track.lastfm_tags,
                            "aoty_match_found": track.aoty_match_found,
                            "aoty_track_rating": track.aoty_track_rating,
                            "spotify_match_found": track.spotify_match_found
                        }
                        for track in album.tracks
                    ]
                }
                albums_data.append(album_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "username": LASTFM_USERNAME,
                    "total_albums": len(albums_data),
                    "total_tracks": sum(album["total_tracks"] for album in albums_data),
                    "statistics": self.stats,
                    "albums": albums_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[FILE] Saved enriched albums to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def convert_to_track_data(self, enriched_albums: List[EnrichedAlbum]) -> List[TrackData]:
        """Convert enriched albums to TrackData format for database insertion"""
        track_data_list = []
        
        for album in enriched_albums:
            for track in album.tracks:
                try:
                    # Generate unique track ID
                    track_id = f"lastfm_{album.artist}_{album.name}_{track.name}".replace(" ", "_").lower()
                    track_id = re.sub(r'[^a-z0-9_]', '', track_id)
                    
                    # Parse duration from string format (e.g., "3:42") to milliseconds
                    duration_ms = None
                    if track.duration:
                        try:
                            if ':' in track.duration:
                                minutes, seconds = track.duration.split(':')
                                duration_ms = (int(minutes) * 60 + int(seconds)) * 1000
                        except Exception:
                            logger.debug(f"Could not parse duration: {track.duration}")
                    
                    # Combine genres from Last.fm tags and AOTY genres
                    genres = []
                    if track.lastfm_tags:
                        genres.extend(track.lastfm_tags[:3])  # Top 3 from Last.fm
                    if album.lastfm_tags:
                        for tag in album.lastfm_tags[:3]:  # Add album tags
                            if tag not in genres:
                                genres.append(tag)
                    if album.aoty_genres:
                        for genre in album.aoty_genres[:2]:  # Add AOTY genres
                            if genre not in genres:
                                genres.append(genre)
                    
                    # Extract mood-related tags
                    mood_keywords = {
                        'happy', 'sad', 'melancholy', 'upbeat', 'energetic', 'calm', 'relaxing',
                        'dreamy', 'dark', 'bright', 'atmospheric', 'ambient', 'chill', 'intense',
                        'aggressive', 'peaceful', 'nostalgic', 'romantic', 'dramatic', 'moody'
                    }
                    moods = []
                    for tag in track.lastfm_tags + track.album_lastfm_tags:
                        if any(mood_word in tag.lower() for mood_word in mood_keywords):
                            if tag not in moods:
                                moods.append(tag)
                    
                    # Determine best release date source
                    release_date = None
                    if album.spotify_release_date:
                        # Parse Spotify release date format
                        try:
                            release_date = album.spotify_release_date
                            # Spotify sometimes gives just year, convert to full date
                            if len(release_date) == 4:  # Just year
                                release_date = f"{release_date}-01-01"
                            elif len(release_date) == 7:  # Year-month
                                release_date = f"{release_date}-01"
                        except Exception:
                            release_date = None
                    
                    # Use Spotify duration if available, otherwise parsed duration
                    final_duration_ms = duration_ms
                    if track.spotify_match_found and track.spotify_duration_ms:
                        final_duration_ms = track.spotify_duration_ms
                    
                    # Use Spotify popularity if available, otherwise derive from Last.fm playcount
                    final_popularity = None
                    if track.spotify_match_found and track.spotify_popularity:
                        final_popularity = track.spotify_popularity
                    elif track.album_lastfm_playcount:
                        final_popularity = min(100, max(0, int(track.album_lastfm_playcount / 10)))
                    
                    # Extract Spotify ID (remove 'spotify:' prefix if present)
                    spotify_id = None
                    if track.spotify_match_found and track.spotify_track_id:
                        spotify_id = track.spotify_track_id.replace('spotify:', '').replace('track:', '') if track.spotify_track_id.startswith('spotify:') else track.spotify_track_id
                    
                    # Create TrackData object
                    track_data = TrackData(
                        track_id=track_id,
                        title=track.name,
                        artist=track.artist,
                        album=track.album,
                        release_date=release_date,
                        duration_ms=final_duration_ms,
                        popularity=final_popularity,
                        genres=genres[:5],  # Limit to top 5
                        moods=moods[:5],    # Limit to top 5
                        aoty_score=track.aoty_album_score if track.aoty_match_found else None,
                        spotify_url=track.spotify_track_url if track.spotify_match_found else album.spotify_album_url,
                        spotify_id=spotify_id,
                        explicit=track.spotify_explicit if track.spotify_match_found else False,
                        track_number=track.track_number,
                        album_total_tracks=album.total_tracks,
                        cover_url=getattr(album, 'spotify_cover_url', None)  # Use album cover URL
                    )
                    
                    track_data_list.append(track_data)
                    
                except Exception as e:
                    logger.error(f"Error converting track to TrackData: {track.name}: {e}")
                    self.stats["errors"].append(f"Track conversion error: {e}")
        
        logger.info(f"Converted {len(track_data_list)} tracks to TrackData format")
        return track_data_list
    
    def insert_tracks_to_database(self, enriched_albums: List[EnrichedAlbum]) -> bool:
        """Insert enriched tracks into the database"""
        try:
            # Convert to TrackData format
            track_data_list = self.convert_to_track_data(enriched_albums)
            
            if not track_data_list:
                logger.error("No tracks to insert into database")
                return False
            
            # Insert tracks into database
            logger.info(f"Inserting {len(track_data_list)} tracks into database...")
            success = insert_tracks(track_data_list, batch_size=50)
            
            if success:
                self.stats["tracks_stored"] = len(track_data_list)
                logger.info(f"Successfully inserted {len(track_data_list)} tracks into database")
                return True
            else:
                logger.error("Failed to insert tracks into database")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting tracks to database: {e}")
            self.stats["errors"].append(f"Database insertion error: {e}")
            return False
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*80)
        print("LAST.FM TOP ALBUMS PROCESSING STATISTICS")
        print("="*80)
        print(f"Username: {LASTFM_USERNAME}")
        print(f"Albums fetched from Last.fm: {self.stats['albums_fetched']}")
        print(f"Albums enriched: {self.stats['albums_enriched']}")
        print(f"Albums matched with AOTY: {self.stats['aoty_albums_matched']}")
        print(f"Albums matched with Spotify: {self.stats['spotify_albums_matched']}")
        print(f"Total tracks found: {self.stats['total_tracks_found']}")
        print(f"Tracks with AOTY data: {self.stats['tracks_with_aoty_data']}")
        print(f"Tracks stored in database: {self.stats.get('tracks_stored', 0)}")
        print(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['albums_fetched'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['albums_fetched']
            aoty_match_rate = (self.stats['aoty_albums_matched'] / self.stats['albums_fetched']) * 100
            print(f"Average time per album: {avg_time:.2f} seconds")
            print(f"AOTY album match rate: {aoty_match_rate:.1f}%")
            
            if self.stats['total_tracks_found'] > 0:
                track_aoty_rate = (self.stats['tracks_with_aoty_data'] / self.stats['total_tracks_found']) * 100
                print(f"Tracks with AOTY data: {track_aoty_rate:.1f}%")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            print("Recent errors:")
            for error in self.stats['errors'][-5:]:
                print(f"  - {error}")
        
        # Show matching service stats
        service_stats = self.matching_service.get_match_statistics()
        print(f"\nMatching service statistics:")
        print(f"  Cache hit rate: {service_stats['cache_hit_rate_percent']}%")
        print(f"  AOTY requests made: {service_stats['aoty_requests_made']}")
        print("="*80)


async def main():
    """Main function"""
    print("[LASTFM] Top Albums Fetcher for connergroth")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python fetch_my_lastfm_albums.py <number_of_albums>")
        print("Example: python fetch_my_lastfm_albums.py 10")
        sys.exit(1)
    
    try:
        num_albums = int(sys.argv[1])
        if num_albums <= 0 or num_albums > 100:
            print("[FAIL] Number of albums must be between 1 and 100")
            sys.exit(1)
    except ValueError:
        print("[FAIL] Please provide a valid number")
        sys.exit(1)
    
    print(f"Fetching top {num_albums} albums for username: {LASTFM_USERNAME}")
    
    # Check if pylast is available
    if not PYLAST_AVAILABLE:
        print("[FAIL] pylast library is required. Install with: pip install pylast")
        sys.exit(1)
    
    # Setup database
    setup_database()
    initial_track_count = get_track_count()
    print(f"Current tracks in database: {initial_track_count}")
    
    # Initialize processor
    processor = LastfmTopAlbumsProcessor()
    
    try:
        # Step 1: Fetch top albums from Last.fm
        print(f"\n[FETCH] Step 1: Fetching top {num_albums} albums from Last.fm...")
        albums_data = processor.fetch_user_top_albums(LASTFM_USERNAME, limit=num_albums)
        
        if not albums_data:
            print("[FAIL] No albums found. Check username and Last.fm API credentials.")
            sys.exit(1)
        
        # Step 2: Enrich albums and fetch all tracks
        print(f"\n[PROCESS] Step 2: Enriching albums and fetching all tracks...")
        print(f"This will fetch complete track listings for each album using AOTY and Spotify...")
        enriched_albums = await processor.process_albums_batch(albums_data, max_concurrent=2)
        
        # Step 3: Save enriched data to JSON for inspection
        print(f"\n[SAVE] Step 3: Saving enriched data...")
        processor.save_to_json(enriched_albums, f"lastfm_top_{num_albums}_albums.json")
        
        # Step 4: Insert tracks into database
        print(f"\n[DATABASE] Step 4: Inserting tracks into database...")
        db_success = processor.insert_tracks_to_database(enriched_albums)
        
        if db_success:
            final_track_count = get_track_count()
            print(f"[SUCCESS] Database now contains {final_track_count} total tracks")
        else:
            print(f"[WARNING] Database insertion failed - data still saved to JSON")
        
        # Step 5: Count total tracks and summary
        total_tracks = sum(len(album.tracks) for album in enriched_albums)
        aoty_albums = sum(1 for album in enriched_albums if album.aoty_match_found)
        
        print(f"\n[SUMMARY] Processing Complete:")
        print(f"  Albums processed: {len(enriched_albums)}")
        print(f"  Total tracks found: {total_tracks}")
        print(f"  Albums with AOTY data: {aoty_albums}/{len(enriched_albums)}")
        print(f"  Average tracks per album: {total_tracks/len(enriched_albums):.1f}")
        if db_success:
            print(f"  Tracks inserted into database: {processor.stats.get('tracks_stored', 0)}")
        
        # Print final statistics
        processor.print_statistics()
        
        print(f"\n[SUCCESS] Successfully processed your top {num_albums} Last.fm albums!")
        print(f"[FILE] Data saved to: lastfm_top_{num_albums}_albums.json")
        if db_success:
            print(f"[DATABASE] Tracks successfully inserted into database for ML training")
        print(f"\nThis approach gives you much better AOTY matching since we have")
        print(f"complete album context for every track!")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
        processor.print_statistics()
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Error during processing: {e}")
        logger.exception("Fatal error in main process")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())