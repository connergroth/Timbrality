#!/usr/bin/env python3
"""
Fetch My Last.fm Top Tracks Script

This script fetches your top tracks from Last.fm (username: connergroth) and
inserts them into the database with full data enrichment from multiple sources:
- Last.fm: play counts, tags, track info
- AOTY: album ratings, reviews, track details
- Spotify: audio features, metadata
- Fuzzy matching to handle inconsistencies between platforms

Usage:
    python fetch_my_lastfm_tracks.py <number_of_tracks>
    python fetch_my_lastfm_tracks.py 50  # Fetch top 50 tracks
"""

import asyncio
import sys
import os
import logging
import time
from typing import List, Dict, Optional, Any
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
from ingestion.lastfm_fetcher import get_lastfm_network, filter_relevant_tags
from ingestion.spotify_fetcher import search_albums, get_track_metadata
from ingestion.insert_to_supabase import setup_database, get_track_count
from utils.fuzzy_matcher import MusicFuzzyMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LASTFM_USERNAME = "connergroth"


@dataclass
class EnrichedTrack:
    """Complete track data from all sources"""
    # Last.fm data
    name: str
    artist: str
    album: str = ""
    playcount: int = 0
    loved: bool = False
    lastfm_url: str = ""
    lastfm_tags: List[str] = None
    lastfm_listeners: int = 0
    lastfm_rank: int = 0
    
    # AOTY data (via matching service)
    aoty_match_found: bool = False
    aoty_confidence: str = "none"
    aoty_score: float = 0.0
    aoty_album_data: Dict = None
    aoty_track_data: Dict = None
    
    # Spotify data
    spotify_match_found: bool = False
    spotify_data: Dict = None
    
    # Processing metadata
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.lastfm_tags is None:
            self.lastfm_tags = []
        if self.errors is None:
            self.errors = []
        if self.aoty_album_data is None:
            self.aoty_album_data = {}
        if self.aoty_track_data is None:
            self.aoty_track_data = {}
        if self.spotify_data is None:
            self.spotify_data = {}


class LastfmTopTracksProcessor:
    """Process Last.fm top tracks with multi-source enrichment"""
    
    def __init__(self):
        self.matching_service = MusicMatchingService()
        self.fuzzy_matcher = MusicFuzzyMatcher()
        self.stats = {
            "tracks_fetched": 0,
            "lastfm_enriched": 0,
            "aoty_matched": 0,
            "spotify_matched": 0,
            "tracks_stored": 0,
            "processing_time": 0.0,
            "errors": []
        }
    
    def fetch_user_top_tracks(self, username: str, limit: int = 50, period: str = "overall") -> List[Dict[str, Any]]:
        """Fetch top tracks from Last.fm for a user"""
        if not PYLAST_AVAILABLE:
            raise ImportError("pylast is required. Install with: pip install pylast")
        
        logger.info(f"Fetching top {limit} tracks for Last.fm user: {username}")
        
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
            top_tracks = user.get_top_tracks(period=period_param, limit=limit)
            
            if not top_tracks:
                logger.warning(f"No top tracks found for user {username}")
                return []
            
            tracks_data = []
            for i, track_item in enumerate(top_tracks, 1):
                try:
                    track = track_item.item
                    playcount = int(track_item.weight) if track_item.weight else 0
                    
                    track_data = {
                        "name": track.get_name(),
                        "artist": track.get_artist().get_name(),
                        "playcount": playcount,
                        "rank": i,
                        "lastfm_url": track.get_url()
                    }
                    
                    # Try to get album info
                    try:
                        album = track.get_album()
                        if album:
                            track_data["album"] = album.get_name()
                        else:
                            track_data["album"] = ""
                    except Exception as e:
                        logger.debug(f"Could not fetch album for {track.get_name()}: {e}")
                        track_data["album"] = ""
                    
                    # Get additional track info
                    try:
                        track_data["listeners"] = track.get_listener_count()
                    except Exception:
                        track_data["listeners"] = 0
                    
                    tracks_data.append(track_data)
                    logger.info(f"  {i:2d}. {track_data['artist']} - {track_data['name']} ({playcount} plays)")
                    
                except Exception as e:
                    logger.error(f"Error processing track {i}: {e}")
                    self.stats["errors"].append(f"Track {i} processing error: {e}")
                    continue
            
            self.stats["tracks_fetched"] = len(tracks_data)
            logger.info(f"[SUCCESS] Successfully fetched {len(tracks_data)} top tracks")
            return tracks_data
            
        except Exception as e:
            logger.error(f"Error fetching top tracks for user {username}: {e}")
            self.stats["errors"].append(f"User fetch error: {e}")
            return []
    
    async def enrich_track_with_all_sources(self, track_data: Dict[str, Any]) -> EnrichedTrack:
        """Enrich a single track with data from all sources"""
        start_time = time.time()
        
        enriched = EnrichedTrack(
            name=track_data["name"],
            artist=track_data["artist"],
            album=track_data.get("album", ""),
            playcount=track_data.get("playcount", 0),
            lastfm_url=track_data.get("lastfm_url", ""),
            lastfm_listeners=track_data.get("listeners", 0),
            lastfm_rank=track_data.get("rank", 0)
        )
        
        logger.info(f"[TRACK] Enriching: {enriched.artist} - {enriched.name}")
        
        # 1. Enrich with Last.fm tags
        await self.enrich_with_lastfm_tags(enriched)
        
        # 2. Try to match with AOTY
        await self.enrich_with_aoty_data(enriched)
        
        # 3. Try to get Spotify data
        await self.enrich_with_spotify_data(enriched)
        
        enriched.processing_time = time.time() - start_time
        return enriched
    
    async def enrich_with_lastfm_tags(self, enriched: EnrichedTrack):
        """Enrich with Last.fm tags and additional metadata"""
        try:
            network = get_lastfm_network()
            track = network.get_track(enriched.artist, enriched.name)
            
            # Get track tags
            try:
                track_tags = track.get_top_tags(limit=15)
                raw_tags = [tag.item.name.lower() for tag in track_tags]
                enriched.lastfm_tags = filter_relevant_tags(raw_tags)
                
                if enriched.lastfm_tags:
                    logger.debug(f"  Last.fm tags: {', '.join(enriched.lastfm_tags[:5])}")
                
            except Exception as e:
                logger.debug(f"  Could not fetch track tags: {e}")
                # Fallback to artist tags
                try:
                    artist = network.get_artist(enriched.artist)
                    artist_tags = artist.get_top_tags(limit=10)
                    raw_tags = [tag.item.name.lower() for tag in artist_tags]
                    enriched.lastfm_tags = filter_relevant_tags(raw_tags)
                except Exception as e2:
                    logger.debug(f"  Could not fetch artist tags either: {e2}")
            
            # Check if track is loved (this requires user authentication, so skip for now)
            # enriched.loved = track.get_loved()  # Would need user auth
            
            self.stats["lastfm_enriched"] += 1
            
        except Exception as e:
            error_msg = f"Last.fm enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def enrich_with_aoty_data(self, enriched: EnrichedTrack):
        """Try to match track with AOTY data using the matching service"""
        try:
            if not enriched.album:
                logger.debug(f"  No album info, skipping AOTY matching")
                return
            
            # Create LastfmTrack object for the matching service
            lastfm_track = LastfmTrack(
                name=enriched.name,
                artist=enriched.artist,
                album=enriched.album,
                playcount=enriched.playcount,
                loved=enriched.loved
            )
            
            # Use the matching service to find AOTY data
            match_result = await self.matching_service.match_lastfm_track_to_aoty(lastfm_track)
            
            if match_result.match_type == "track_match":
                enriched.aoty_match_found = True
                enriched.aoty_confidence = match_result.confidence
                enriched.aoty_score = match_result.score
                
                if match_result.target_data:
                    enriched.aoty_album_data = match_result.target_data.get("album", {})
                    enriched.aoty_track_data = match_result.target_data.get("matched_track", {})
                    
                    album_score = enriched.aoty_album_data.get("user_score", 0)
                    logger.debug(f"  [SUCCESS] AOTY match found (confidence: {enriched.aoty_confidence}, album score: {album_score})")
                
                self.stats["aoty_matched"] += 1
                
            elif match_result.match_type == "album_only":
                enriched.aoty_match_found = True
                enriched.aoty_confidence = "album_only"
                enriched.aoty_score = match_result.score
                
                if match_result.target_data:
                    enriched.aoty_album_data = match_result.target_data.get("album", {})
                
                logger.debug(f"  [WARNING] AOTY album found but track not matched")
                
            else:
                logger.debug(f"  [FAIL] No AOTY match found")
                
        except Exception as e:
            error_msg = f"AOTY enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def enrich_with_spotify_data(self, enriched: EnrichedTrack):
        """Try to get Spotify data for the track"""
        try:
            # Search for the track on Spotify
            if enriched.album:
                search_query = f"track:{enriched.name} artist:{enriched.artist} album:{enriched.album}"
            else:
                search_query = f"track:{enriched.name} artist:{enriched.artist}"
            
            # Use Spotify search (this would need the Spotify fetcher to be updated)
            # For now, we'll implement a basic search
            logger.debug(f"  Spotify search skipped (not implemented in current fetcher)")
            
            # TODO: Implement Spotify track search and audio features
            # spotify_data = await search_spotify_track(enriched.name, enriched.artist)
            # if spotify_data:
            #     enriched.spotify_match_found = True
            #     enriched.spotify_data = spotify_data
            #     self.stats["spotify_matched"] += 1
            
        except Exception as e:
            error_msg = f"Spotify enrichment error: {e}"
            logger.debug(f"  {error_msg}")
            enriched.errors.append(error_msg)
    
    async def process_tracks_batch(self, tracks_data: List[Dict], max_concurrent: int = 3) -> List[EnrichedTrack]:
        """Process multiple tracks with controlled concurrency"""
        logger.info(f"[PROCESS] Processing {len(tracks_data)} tracks with max {max_concurrent} concurrent requests")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_track(track_data: Dict, index: int) -> EnrichedTrack:
            async with semaphore:
                try:
                    enriched = await self.enrich_track_with_all_sources(track_data)
                    logger.info(f"[{index+1:2d}/{len(tracks_data)}] Processed: {enriched.artist} - {enriched.name} "
                               f"(AOTY: {'[SUCCESS]' if enriched.aoty_match_found else '[FAIL]'}, "
                               f"{enriched.processing_time:.2f}s)")
                    return enriched
                except Exception as e:
                    logger.error(f"[{index+1:2d}/{len(tracks_data)}] Failed: {track_data.get('artist', 'Unknown')} - {track_data.get('name', 'Unknown')}: {e}")
                    # Return minimal enriched track with error
                    enriched = EnrichedTrack(
                        name=track_data.get("name", "Unknown"),
                        artist=track_data.get("artist", "Unknown"),
                        album=track_data.get("album", ""),
                        playcount=track_data.get("playcount", 0),
                        lastfm_rank=track_data.get("rank", 0),
                        errors=[f"Processing failed: {e}"]
                    )
                    return enriched
        
        # Process all tracks
        start_time = time.time()
        tasks = [process_single_track(track_data, i) for i, track_data in enumerate(tracks_data)]
        enriched_tracks = await asyncio.gather(*tasks)
        
        self.stats["processing_time"] = time.time() - start_time
        
        logger.info(f"[SUCCESS] Batch processing completed in {self.stats['processing_time']:.2f} seconds")
        return enriched_tracks
    
    def convert_to_db_format(self, enriched_tracks: List[EnrichedTrack]) -> List[Dict[str, Any]]:
        """Convert enriched tracks to database format"""
        db_tracks = []
        
        for track in enriched_tracks:
            try:
                # Create normalized track data
                db_track = {
                    # Basic track info
                    'title': track.name,
                    'artist': track.artist,
                    'album': track.album or "Unknown Album",
                    
                    # Last.fm data
                    'lastfm_playcount': track.playcount,
                    'lastfm_listeners': track.lastfm_listeners,
                    'lastfm_rank': track.lastfm_rank,
                    'lastfm_url': track.lastfm_url,
                    'lastfm_tags': track.lastfm_tags,
                    
                    # AOTY data
                    'aoty_match_found': track.aoty_match_found,
                    'aoty_confidence': track.aoty_confidence,
                    'aoty_match_score': track.aoty_score,
                    'aoty_album_score': track.aoty_album_data.get('user_score', 0) if track.aoty_album_data else 0,
                    'aoty_album_ratings': track.aoty_album_data.get('num_ratings', 0) if track.aoty_album_data else 0,
                    'aoty_is_must_hear': track.aoty_album_data.get('is_must_hear', False) if track.aoty_album_data else False,
                    'aoty_track_number': track.aoty_track_data.get('number', 0) if track.aoty_track_data else 0,
                    'aoty_track_length': track.aoty_track_data.get('length', '') if track.aoty_track_data else '',
                    
                    # Spotify data (placeholder for future implementation)
                    'spotify_match_found': track.spotify_match_found,
                    'spotify_data': track.spotify_data,
                    
                    # Metadata
                    'processing_time': track.processing_time,
                    'enrichment_errors': track.errors,
                    'data_sources': self.get_data_sources(track)
                }
                
                db_tracks.append(db_track)
                
            except Exception as e:
                logger.error(f"Error converting track to DB format: {track.name} by {track.artist}: {e}")
                self.stats["errors"].append(f"DB conversion error: {e}")
        
        return db_tracks
    
    def get_data_sources(self, track: EnrichedTrack) -> List[str]:
        """Get list of data sources used for this track"""
        sources = ["lastfm"]  # Always have Last.fm
        
        if track.aoty_match_found:
            sources.append("aoty")
        
        if track.spotify_match_found:
            sources.append("spotify")
        
        return sources
    
    def save_to_json(self, enriched_tracks: List[EnrichedTrack], filename: str = "lastfm_tracks_enriched.json"):
        """Save enriched tracks to JSON for inspection"""
        try:
            # Convert to serializable format
            tracks_data = []
            for track in enriched_tracks:
                track_dict = {
                    "name": track.name,
                    "artist": track.artist,
                    "album": track.album,
                    "playcount": track.playcount,
                    "lastfm_rank": track.lastfm_rank,
                    "lastfm_tags": track.lastfm_tags,
                    "aoty_match_found": track.aoty_match_found,
                    "aoty_confidence": track.aoty_confidence,
                    "aoty_score": track.aoty_score,
                    "spotify_match_found": track.spotify_match_found,
                    "processing_time": track.processing_time,
                    "errors": track.errors,
                    "data_sources": self.get_data_sources(track)
                }
                tracks_data.append(track_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "username": LASTFM_USERNAME,
                    "total_tracks": len(tracks_data),
                    "statistics": self.stats,
                    "tracks": tracks_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[FILE] Saved enriched tracks to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*80)
        print("LAST.FM TOP TRACKS PROCESSING STATISTICS")
        print("="*80)
        print(f"Username: {LASTFM_USERNAME}")
        print(f"Tracks fetched from Last.fm: {self.stats['tracks_fetched']}")
        print(f"Tracks enriched with Last.fm tags: {self.stats['lastfm_enriched']}")
        print(f"Tracks matched with AOTY: {self.stats['aoty_matched']}")
        print(f"Tracks matched with Spotify: {self.stats['spotify_matched']}")
        print(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['tracks_fetched'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['tracks_fetched']
            aoty_match_rate = (self.stats['aoty_matched'] / self.stats['tracks_fetched']) * 100
            print(f"Average time per track: {avg_time:.2f} seconds")
            print(f"AOTY match rate: {aoty_match_rate:.1f}%")
        
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
    print("[LASTFM] Top Tracks Fetcher for connergroth")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python fetch_my_lastfm_tracks.py <number_of_tracks>")
        print("Example: python fetch_my_lastfm_tracks.py 50")
        sys.exit(1)
    
    try:
        num_tracks = int(sys.argv[1])
        if num_tracks <= 0 or num_tracks > 1000:
            print("[FAIL] Number of tracks must be between 1 and 1000")
            sys.exit(1)
    except ValueError:
        print("[FAIL] Please provide a valid number")
        sys.exit(1)
    
    print(f"Fetching top {num_tracks} tracks for username: {LASTFM_USERNAME}")
    
    # Check if pylast is available
    if not PYLAST_AVAILABLE:
        print("[FAIL] pylast library is required. Install with: pip install pylast")
        sys.exit(1)
    
    # Setup database
    setup_database()
    initial_track_count = get_track_count()
    print(f"Current tracks in database: {initial_track_count}")
    
    # Initialize processor
    processor = LastfmTopTracksProcessor()
    
    try:
        # Step 1: Fetch top tracks from Last.fm
        print(f"\n[FETCH] Step 1: Fetching top {num_tracks} tracks from Last.fm...")
        tracks_data = processor.fetch_user_top_tracks(LASTFM_USERNAME, limit=num_tracks)
        
        if not tracks_data:
            print("[FAIL] No tracks found. Check username and Last.fm API credentials.")
            sys.exit(1)
        
        # Step 2: Enrich tracks with data from all sources
        print(f"\n[SEARCH] Step 2: Enriching tracks with data from multiple sources...")
        enriched_tracks = await processor.process_tracks_batch(tracks_data, max_concurrent=3)
        
        # Step 3: Save enriched data to JSON for inspection
        print(f"\n[SAVE] Step 3: Saving enriched data...")
        processor.save_to_json(enriched_tracks, f"lastfm_top_{num_tracks}_tracks.json")
        
        # Step 4: Convert to database format
        print(f"\n[DATABASE]  Step 4: Preparing data for database insertion...")
        db_tracks = processor.convert_to_db_format(enriched_tracks)
        
        # Step 5: Insert into database (placeholder - would need actual DB insertion logic)
        print(f"\n[FETCH] Step 5: Inserting {len(db_tracks)} tracks into database...")
        print("[WARNING]  Database insertion not yet implemented in this script")
        print("   Enriched data has been saved to JSON file for now")
        
        # TODO: Implement actual database insertion
        # success = insert_enriched_tracks(db_tracks)
        # if success:
        #     processor.stats["tracks_stored"] = len(db_tracks)
        #     final_track_count = get_track_count()
        #     print(f"[SUCCESS] Successfully inserted tracks. Database now has {final_track_count} tracks.")
        # else:
        #     print("[FAIL] Failed to insert tracks into database")
        
        # Print final statistics
        processor.print_statistics()
        
        print(f"\n[SUCCESS] Successfully processed your top {num_tracks} Last.fm tracks!")
        print(f"[FILE] Data saved to: lastfm_top_{num_tracks}_tracks.json")
        
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