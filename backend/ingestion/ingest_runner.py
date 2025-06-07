"""
Ingestion Runner - Pipeline runner to execute the complete ingestion process
"""
import asyncio
import logging
import csv
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from .spotify_fetcher import get_album_tracks, get_track_metadata, search_albums
from .lastfm_fetcher import enrich_with_tags, get_artist_tags, filter_relevant_tags
from .aoty_scraper import get_album_aoty_data_sync, get_album_aoty_data
from .normalizer import normalize_track, normalize_batch, TrackData, validate_track_data, deduplicate_tracks
from .insert_to_supabase import insert_tracks, setup_database, get_track_count

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ingestion(album_name: str, artist_name: str) -> bool:
    """
    Run the complete ingestion pipeline for a single album
    
    Args:
        album_name: Name of the album to ingest
        artist_name: Name of the artist
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting ingestion for album '{album_name}' by {artist_name}")
    
    try:
        # Step 1: Get album tracks from Spotify
        logger.info("Step 1: Fetching tracks from Spotify...")
        album_tracks = get_album_tracks(album_name, artist_name)
        
        if not album_tracks:
            logger.error(f"No tracks found for album '{album_name}' by {artist_name}")
            return False
        
        logger.info(f"Found {len(album_tracks)} tracks from Spotify")
        
        # Step 2: Get AOTY data for the album
        logger.info("Step 2: Fetching AOTY data...")
        aoty_data = get_album_aoty_data_sync(album_name, artist_name)
        
        # Step 3: Enrich tracks with Last.fm tags and normalize
        logger.info("Step 3: Enriching with Last.fm data and normalizing...")
        normalized_tracks = []
        
        for i, track in enumerate(album_tracks):
            logger.info(f"Processing track {i+1}/{len(album_tracks)}: {track['name']}")
            
            # Get Last.fm tags for the track
            try:
                lastfm_tags = enrich_with_tags(track['name'], track['artist'])
                # Filter to keep only relevant tags
                lastfm_tags = filter_relevant_tags(lastfm_tags)
            except Exception as e:
                logger.warning(f"Could not fetch Last.fm tags for track '{track['name']}': {e}")
                lastfm_tags = []
            
            # Normalize the track data
            try:
                normalized_track = normalize_track(track, lastfm_tags, aoty_data)
                if validate_track_data(normalized_track):
                    normalized_tracks.append(normalized_track)
                else:
                    logger.warning(f"Track failed validation: {track['name']}")
            except Exception as e:
                logger.error(f"Error normalizing track '{track['name']}': {e}")
        
        if not normalized_tracks:
            logger.error("No tracks were successfully normalized")
            return False
        
        # Step 4: Remove duplicates
        normalized_tracks = deduplicate_tracks(normalized_tracks)
        
        # Step 5: Insert into Supabase
        logger.info(f"Step 4: Inserting {len(normalized_tracks)} tracks into database...")
        success = insert_tracks(normalized_tracks)
        
        if success:
            logger.info(f"Successfully completed ingestion for album '{album_name}' by {artist_name}")
            return True
        else:
            logger.error(f"Failed to insert tracks for album '{album_name}' by {artist_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {e}")
        return False


async def run_ingestion_async(album_name: str, artist_name: str) -> bool:
    """
    Async version of the ingestion pipeline
    
    Args:
        album_name: Name of the album to ingest
        artist_name: Name of the artist
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting async ingestion for album '{album_name}' by {artist_name}")
    
    try:
        # Step 1: Get album tracks from Spotify
        logger.info("Step 1: Fetching tracks from Spotify...")
        album_tracks = get_album_tracks(album_name, artist_name)
        
        if not album_tracks:
            logger.error(f"No tracks found for album '{album_name}' by {artist_name}")
            return False
        
        logger.info(f"Found {len(album_tracks)} tracks from Spotify")
        
        # Step 2: Get AOTY data for the album (async)
        logger.info("Step 2: Fetching AOTY data...")
        aoty_data = await get_album_aoty_data(album_name, artist_name)
        
        # Step 3: Enrich tracks with Last.fm tags in parallel
        logger.info("Step 3: Enriching with Last.fm data...")
        
        async def enrich_track(track):
            try:
                # Run Last.fm fetching in thread pool since it's not async
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    lastfm_tags = await loop.run_in_executor(
                        executor, 
                        lambda: filter_relevant_tags(enrich_with_tags(track['name'], track['artist']))
                    )
                return track, lastfm_tags
            except Exception as e:
                logger.warning(f"Could not fetch Last.fm tags for track '{track['name']}': {e}")
                return track, []
        
        # Process all tracks in parallel
        enriched_results = await asyncio.gather(*[enrich_track(track) for track in album_tracks])
        
        # Step 4: Normalize all tracks
        logger.info("Step 4: Normalizing tracks...")
        normalized_tracks = []
        
        for track, lastfm_tags in enriched_results:
            try:
                normalized_track = normalize_track(track, lastfm_tags, aoty_data)
                if validate_track_data(normalized_track):
                    normalized_tracks.append(normalized_track)
                else:
                    logger.warning(f"Track failed validation: {track['name']}")
            except Exception as e:
                logger.error(f"Error normalizing track '{track['name']}': {e}")
        
        if not normalized_tracks:
            logger.error("No tracks were successfully normalized")
            return False
        
        # Step 5: Remove duplicates
        normalized_tracks = deduplicate_tracks(normalized_tracks)
        
        # Step 6: Insert into Supabase
        logger.info(f"Step 5: Inserting {len(normalized_tracks)} tracks into database...")
        success = insert_tracks(normalized_tracks)
        
        if success:
            logger.info(f"Successfully completed async ingestion for album '{album_name}' by {artist_name}")
            return True
        else:
            logger.error(f"Failed to insert tracks for album '{album_name}' by {artist_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error in async ingestion pipeline: {e}")
        return False


def run_batch_ingestion(album_list: List[Tuple[str, str]], use_async: bool = True) -> Dict[str, int]:
    """
    Run ingestion for multiple albums
    
    Args:
        album_list: List of (album_name, artist_name) tuples
        use_async: Whether to use async processing
        
    Returns:
        Dictionary with success/failure counts
    """
    logger.info(f"Starting batch ingestion for {len(album_list)} albums")
    
    # Setup database
    setup_database()
    
    results = {
        'total': len(album_list),
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    if use_async:
        async def batch_process():
            tasks = []
            for album_name, artist_name in album_list:
                task = run_ingestion_async(album_name, artist_name)
                tasks.append(task)
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            batch_results = asyncio.run(batch_process())
            
            for i, result in enumerate(batch_results):
                album_name, artist_name = album_list[i]
                
                if isinstance(result, Exception):
                    results['failed'] += 1
                    error_msg = f"Exception for '{album_name}' by {artist_name}: {str(result)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                elif result:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    error_msg = f"Failed ingestion for '{album_name}' by {artist_name}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
        
        except Exception as e:
            logger.error(f"Error in batch async processing: {e}")
            results['failed'] = len(album_list)
    
    else:
        # Sequential processing
        for i, (album_name, artist_name) in enumerate(album_list):
            logger.info(f"Processing album {i+1}/{len(album_list)}: '{album_name}' by {artist_name}")
            
            try:
                success = run_ingestion(album_name, artist_name)
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    error_msg = f"Failed ingestion for '{album_name}' by {artist_name}"
                    results['errors'].append(error_msg)
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Exception for '{album_name}' by {artist_name}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
            
            # Add delay between requests to be respectful to APIs
            time.sleep(1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Batch ingestion completed in {duration:.2f} seconds")
    logger.info(f"Results: {results['successful']} successful, {results['failed']} failed out of {results['total']} total")
    
    return results


def run_batch_from_csv(csv_file_path: str, use_async: bool = True) -> Dict[str, int]:
    """
    Run batch ingestion from a CSV file containing album and artist names
    
    CSV format: album_name,artist_name
    
    Args:
        csv_file_path: Path to the CSV file
        use_async: Whether to use async processing
        
    Returns:
        Dictionary with success/failure counts
    """
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
        return run_batch_ingestion(album_list, use_async=use_async)
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'errors': [f"Error reading CSV file: {e}"]
        }


def search_and_ingest(search_query: str, limit: int = 5) -> Dict[str, int]:
    """
    Search for albums using Spotify and ingest the results
    
    Args:
        search_query: Search query for albums
        limit: Maximum number of albums to search and ingest
        
    Returns:
        Dictionary with success/failure counts
    """
    logger.info(f"Searching for albums with query: '{search_query}'")
    
    try:
        # Search for albums on Spotify
        search_results = search_albums(search_query, limit=limit)
        
        if not search_results:
            logger.error(f"No albums found for query: '{search_query}'")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'errors': ['No albums found']
            }
        
        # Convert search results to album list
        album_list = []
        for album in search_results:
            album_list.append((album['name'], album['artist']))
        
        logger.info(f"Found {len(album_list)} albums, starting ingestion...")
        return run_batch_ingestion(album_list, use_async=True)
        
    except Exception as e:
        logger.error(f"Error in search and ingest: {e}")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'errors': [f"Error in search and ingest: {e}"]
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ingest_runner.py <album_name> <artist_name>")
        print("   or: python ingest_runner.py --csv <csv_file_path>")
        print("   or: python ingest_runner.py --search <search_query>")
        sys.exit(1)
    
    if sys.argv[1] == "--csv":
        csv_file = sys.argv[2]
        results = run_batch_from_csv(csv_file)
        print(f"Batch ingestion results: {results}")
    
    elif sys.argv[1] == "--search":
        search_query = " ".join(sys.argv[2:])
        results = search_and_ingest(search_query, limit=10)
        print(f"Search and ingest results: {results}")
    
    else:
        album_name = sys.argv[1]
        artist_name = sys.argv[2]
        success = run_ingestion(album_name, artist_name)
        print(f"Ingestion {'successful' if success else 'failed'}")
    
    # Print final database stats
    total_tracks = get_track_count()
    print(f"Total tracks in database: {total_tracks}") 