"""
Updated Ingestion Runner - Uses new CloudScraper and fuzzy matching
"""
import asyncio
import logging
import csv
import sys
import os
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import time

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.music_matching_service import MusicMatchingService, LastfmTrack
from scraper.utils.scraper import get_album_url, scrape_album
from spotify_fetcher import get_album_tracks, get_track_metadata, search_albums
from lastfm_fetcher import enrich_with_tags, get_artist_tags, filter_relevant_tags
from normalizer import normalize_track, normalize_batch, TrackData, validate_track_data, deduplicate_tracks
from insert_to_supabase import insert_tracks, setup_database, get_track_count

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernIngestionRunner:
    """Modern ingestion runner using CloudScraper and fuzzy matching"""
    
    def __init__(self):
        self.matching_service = MusicMatchingService()
        self.stats = {
            "albums_processed": 0,
            "tracks_found": 0,
            "tracks_matched": 0,
            "tracks_stored": 0,
            "errors": []
        }
    
    async def ingest_album_modern(self, album_name: str, artist_name: str) -> bool:
        """
        Modern album ingestion using CloudScraper and fuzzy matching
        """
        logger.info(f"Starting modern ingestion for '{album_name}' by {artist_name}")
        
        try:
            # Step 1: Get album tracks from Spotify
            logger.info("Step 1: Fetching tracks from Spotify...")
            album_tracks = get_album_tracks(album_name, artist_name)
            
            if not album_tracks:
                logger.error(f"No tracks found for album '{album_name}' by {artist_name}")
                return False
            
            self.stats["tracks_found"] += len(album_tracks)
            logger.info(f"Found {len(album_tracks)} tracks from Spotify")
            
            # Step 2: Get AOTY data using modern CloudScraper
            logger.info("Step 2: Fetching AOTY data with CloudScraper...")
            aoty_album = await self.get_aoty_data_modern(album_name, artist_name)
            
            # Step 3: Match tracks with fuzzy matching
            logger.info("Step 3: Matching tracks with fuzzy logic...")
            matched_tracks = []
            
            if aoty_album:
                # Create LastfmTrack objects from Spotify data
                lastfm_tracks = []
                for track in album_tracks:
                    lastfm_track = LastfmTrack(
                        name=track['name'],
                        artist=track['artist'],
                        album=album_name,
                        playcount=0,  # No playcount from Spotify
                        loved=False
                    )
                    lastfm_tracks.append(lastfm_track)
                
                # Use fuzzy matching to match Spotify tracks to AOTY tracks
                for spotify_track, lastfm_track in zip(album_tracks, lastfm_tracks):
                    match_result = await self.matching_service.match_lastfm_track_to_aoty(lastfm_track)
                    
                    if match_result.match_type == "track_match":
                        self.stats["tracks_matched"] += 1
                        
                        # Combine Spotify and AOTY data
                        combined_track = self.combine_track_data(
                            spotify_track, 
                            match_result.target_data
                        )
                        matched_tracks.append(combined_track)
                        
                        logger.info(f"✅ Matched: {spotify_track['name']} -> {match_result.target_data['matched_track']['title']}")
                    else:
                        logger.info(f"⚠️  No AOTY match for: {spotify_track['name']}")
                        # Still include track with Spotify data only
                        combined_track = self.combine_track_data(spotify_track, None)
                        matched_tracks.append(combined_track)
            else:
                logger.warning("No AOTY data found, using Spotify data only")
                for track in album_tracks:
                    combined_track = self.combine_track_data(track, None)
                    matched_tracks.append(combined_track)
            
            # Step 4: Enrich with Last.fm tags
            logger.info("Step 4: Enriching with Last.fm data...")
            enriched_tracks = []
            
            for track in matched_tracks:
                try:
                    lastfm_tags = enrich_with_tags(track['name'], track['artist'])
                    lastfm_tags = filter_relevant_tags(lastfm_tags)
                    
                    # Add Last.fm tags to track data
                    track['lastfm_tags'] = lastfm_tags
                    enriched_tracks.append(track)
                    
                except Exception as e:
                    logger.warning(f"Could not fetch Last.fm tags for '{track['name']}': {e}")
                    track['lastfm_tags'] = []
                    enriched_tracks.append(track)
            
            # Step 5: Normalize and validate
            logger.info("Step 5: Normalizing and validating tracks...")
            normalized_tracks = []
            
            for track in enriched_tracks:
                try:
                    # Create a mock AOTY data structure for the normalizer
                    aoty_data = track.get('aoty_data', {})
                    lastfm_tags = track.get('lastfm_tags', [])
                    
                    normalized_track = normalize_track(track, lastfm_tags, aoty_data)
                    if validate_track_data(normalized_track):
                        normalized_tracks.append(normalized_track)
                        self.stats["tracks_stored"] += 1
                    else:
                        logger.warning(f"Track failed validation: {track['name']}")
                        
                except Exception as e:
                    logger.error(f"Error normalizing track '{track['name']}': {e}")
                    self.stats["errors"].append(f"Normalization error for {track['name']}: {e}")
            
            if not normalized_tracks:
                logger.error("No tracks were successfully normalized")
                return False
            
            # Step 6: Remove duplicates and insert
            normalized_tracks = deduplicate_tracks(normalized_tracks)
            
            logger.info(f"Step 6: Inserting {len(normalized_tracks)} tracks into database...")
            success = insert_tracks(normalized_tracks)
            
            if success:
                self.stats["albums_processed"] += 1
                logger.info(f"✅ Successfully completed modern ingestion for '{album_name}' by {artist_name}")
                return True
            else:
                logger.error(f"❌ Failed to insert tracks for '{album_name}' by {artist_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error in modern ingestion pipeline: {e}")
            self.stats["errors"].append(f"Pipeline error for {album_name}: {e}")
            return False
    
    async def get_aoty_data_modern(self, album_name: str, artist_name: str) -> Optional[Dict]:
        """Get AOTY data using the modern CloudScraper"""
        try:
            url_result = await get_album_url(artist_name, album_name)
            if not url_result:
                logger.warning(f"No AOTY URL found for {artist_name} - {album_name}")
                return None
            
            url, found_artist, found_title = url_result
            logger.info(f"Found AOTY URL: {found_artist} - {found_title}")
            
            album_data = await scrape_album(url, found_artist, found_title)
            return album_data.model_dump() if album_data else None
            
        except Exception as e:
            logger.error(f"Error fetching AOTY data: {e}")
            return None
    
    def combine_track_data(self, spotify_track: Dict, aoty_match_data: Optional[Dict]) -> Dict:
        """Combine Spotify track data with AOTY match data"""
        combined = spotify_track.copy()
        
        if aoty_match_data and aoty_match_data.get('matched_track'):
            aoty_track = aoty_match_data['matched_track']
            aoty_album = aoty_match_data['album']
            
            combined.update({
                'aoty_data': {
                    'track_number': aoty_track.get('number'),
                    'track_length': aoty_track.get('length'),
                    'track_rating': aoty_track.get('rating'),
                    'album_user_score': aoty_album.get('user_score'),
                    'album_num_ratings': aoty_album.get('num_ratings'),
                    'album_critic_reviews_count': len(aoty_album.get('critic_reviews', [])),
                    'album_user_reviews_count': len(aoty_album.get('popular_reviews', [])),
                    'album_is_must_hear': aoty_album.get('is_must_hear', False),
                    'match_confidence': aoty_match_data.get('match_details', {}).get('confidence', 'unknown')
                }
            })
        else:
            combined['aoty_data'] = {}
        
        return combined
    
    async def run_batch_modern(self, album_list: List[Tuple[str, str]], max_concurrent: int = 3) -> Dict[str, any]:
        """Run modern batch ingestion with concurrency control"""
        logger.info(f"Starting modern batch ingestion for {len(album_list)} albums")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_album(album_name: str, artist_name: str) -> bool:
            async with semaphore:
                return await self.ingest_album_modern(album_name, artist_name)
        
        # Process all albums
        tasks = [process_album(album, artist) for album, artist in album_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful = sum(1 for result in results if result is True)
        failed = len(results) - successful
        
        # Collect exceptions
        exceptions = [str(result) for result in results if isinstance(result, Exception)]
        self.stats["errors"].extend(exceptions)
        
        return {
            'total': len(album_list),
            'successful': successful,
            'failed': failed,
            'stats': self.stats,
            'errors': self.stats["errors"]
        }
    
    def print_statistics(self):
        """Print ingestion statistics"""
        print("\n" + "="*60)
        print("MODERN INGESTION STATISTICS")
        print("="*60)
        print(f"Albums processed: {self.stats['albums_processed']}")
        print(f"Tracks found: {self.stats['tracks_found']}")
        print(f"Tracks matched with AOTY: {self.stats['tracks_matched']}")
        print(f"Tracks stored: {self.stats['tracks_stored']}")
        
        if self.stats['tracks_found'] > 0:
            match_rate = (self.stats['tracks_matched'] / self.stats['tracks_found']) * 100
            print(f"AOTY match rate: {match_rate:.1f}%")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            print("Recent errors:")
            for error in self.stats['errors'][-3:]:
                print(f"  - {error}")
        
        # Get service statistics
        service_stats = self.matching_service.get_match_statistics()
        print(f"\nMatching service stats:")
        print(f"  Cache hit rate: {service_stats['cache_hit_rate_percent']}%")
        print(f"  AOTY requests made: {service_stats['aoty_requests_made']}")


# Convenience functions to replace old ones
async def run_modern_ingestion(album_name: str, artist_name: str) -> bool:
    """Modern replacement for run_ingestion()"""
    runner = ModernIngestionRunner()
    success = await runner.ingest_album_modern(album_name, artist_name)
    runner.print_statistics()
    return success

async def run_modern_batch(album_list: List[Tuple[str, str]]) -> Dict[str, any]:
    """Modern replacement for run_batch_ingestion()"""
    runner = ModernIngestionRunner()
    results = await runner.run_batch_modern(album_list)
    runner.print_statistics()
    return results

async def run_modern_csv_batch(csv_file_path: str) -> Dict[str, any]:
    """Modern replacement for run_batch_from_csv()"""
    try:
        album_list = []
        
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Skip header if present
            first_row = next(reader, None)
            if first_row and first_row[0].lower() not in ['album', 'album_name']:
                album_list.append((first_row[0], first_row[1]))
            
            # Read the rest of the rows
            for row in reader:
                if len(row) >= 2:
                    album_name = row[0].strip()
                    artist_name = row[1].strip()
                    if album_name and artist_name:
                        album_list.append((album_name, artist_name))
        
        logger.info(f"Loaded {len(album_list)} albums from CSV file")
        return await run_modern_batch(album_list)
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'errors': [f"Error reading CSV file: {e}"]
        }


if __name__ == "__main__":
    # Modern CLI interface
    if len(sys.argv) < 3:
        print("Modern Ingestion Runner - Uses CloudScraper + Fuzzy Matching")
        print("Usage:")
        print("  python updated_ingest_runner.py <album_name> <artist_name>")
        print("  python updated_ingest_runner.py --csv <csv_file_path>")
        print("  python updated_ingest_runner.py --batch album1,artist1 album2,artist2 ...")
        sys.exit(1)
    
    # Setup database
    setup_database()
    
    async def main():
        if sys.argv[1] == "--csv":
            csv_file = sys.argv[2]
            results = await run_modern_csv_batch(csv_file)
            print(f"\nBatch ingestion results: {results}")
        
        elif sys.argv[1] == "--batch":
            # Parse album,artist pairs
            album_list = []
            for i in range(2, len(sys.argv)):
                if ',' in sys.argv[i]:
                    parts = sys.argv[i].split(',')
                    if len(parts) >= 2:
                        album_list.append((parts[0].strip(), parts[1].strip()))
            
            if album_list:
                results = await run_modern_batch(album_list)
                print(f"\nBatch ingestion results: {results}")
            else:
                print("No valid album,artist pairs found")
        
        else:
            album_name = sys.argv[1]
            artist_name = sys.argv[2]
            success = await run_modern_ingestion(album_name, artist_name)
            print(f"\nIngestion {'successful' if success else 'failed'}")
    
    # Run the async main function
    asyncio.run(main())
    
    # Print final database stats
    total_tracks = get_track_count()
    print(f"\nTotal tracks in database: {total_tracks}")