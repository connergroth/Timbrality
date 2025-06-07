"""
AOTY Scraper Wrapper - Reuse existing AOTY scraper for ingestion pipeline
"""
import logging
from typing import Dict, List, Optional
import asyncio
import sys
import os

# Add the parent directory to the path to import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scraper.aoty_scraper import (
    get_album_url, 
    scrape_album, 
    parse_tracks,
    new_page
)

logger = logging.getLogger(__name__)


async def get_album_aoty_data(album_name: str, artist_name: str) -> Dict:
    """
    Get album data from AOTY including ratings and genres
    
    Args:
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        Dictionary with AOTY album data
    """
    try:
        # Search for the album on AOTY
        album_search_result = await get_album_url(artist_name, album_name)
        
        if not album_search_result:
            logger.warning(f"Album not found on AOTY: {album_name} by {artist_name}")
            return {
                'album_score': None,
                'genres': [],
                'track_scores': {},
                'url': None,
                'metadata': {}
            }
        
        album_url, found_artist, found_album = album_search_result
        
        # Scrape the album page
        album_data = await scrape_album(album_url, found_artist, found_album)
        
        # Extract track scores
        track_scores = {}
        if album_data.tracks:
            for track in album_data.tracks:
                if track.rating:
                    track_scores[track.title] = float(track.rating)
        
        # Extract genres from metadata
        genres = []
        if album_data.metadata and album_data.metadata.genres:
            genres = album_data.metadata.genres
        
        result = {
            'album_score': float(album_data.rating) if album_data.rating else None,
            'genres': genres,
            'track_scores': track_scores,
            'url': album_url,
            'metadata': {
                'release_date': album_data.metadata.release_date if album_data.metadata else None,
                'record_label': album_data.metadata.record_label if album_data.metadata else None,
                'album_type': album_data.metadata.album_type if album_data.metadata else None,
                'runtime': album_data.metadata.runtime if album_data.metadata else None,
                'is_must_hear': album_data.metadata.is_must_hear if album_data.metadata else False
            }
        }
        
        logger.info(f"Successfully fetched AOTY data for album '{album_name}' by {artist_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching AOTY data for album '{album_name}' by {artist_name}: {e}")
        return {
            'album_score': None,
            'genres': [],
            'track_scores': {},
            'url': None,
            'metadata': {}
        }


async def get_track_aoty_score(track_name: str, album_name: str, artist_name: str) -> Optional[float]:
    """
    Get AOTY score for a specific track
    
    Args:
        track_name: Name of the track
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        Track score from AOTY or None if not found
    """
    try:
        album_data = await get_album_aoty_data(album_name, artist_name)
        
        if album_data and 'track_scores' in album_data:
            # Try exact match first
            if track_name in album_data['track_scores']:
                return album_data['track_scores'][track_name]
            
            # Try case-insensitive match
            for aoty_track_name, score in album_data['track_scores'].items():
                if track_name.lower() == aoty_track_name.lower():
                    return score
            
            # Try partial match
            for aoty_track_name, score in album_data['track_scores'].items():
                if track_name.lower() in aoty_track_name.lower() or aoty_track_name.lower() in track_name.lower():
                    return score
        
        logger.warning(f"Track score not found on AOTY: '{track_name}' from album '{album_name}' by {artist_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching track AOTY score: {e}")
        return None


def get_album_aoty_data_sync(album_name: str, artist_name: str) -> Dict:
    """
    Synchronous wrapper for get_album_aoty_data
    
    Args:
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        Dictionary with AOTY album data
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_album_aoty_data(album_name, artist_name))
                return future.result()
        else:
            # No running loop, can use asyncio.run
            return asyncio.run(get_album_aoty_data(album_name, artist_name))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return {
            'album_score': None,
            'genres': [],
            'track_scores': {},
            'url': None,
            'metadata': {}
        }


def get_track_aoty_score_sync(track_name: str, album_name: str, artist_name: str) -> Optional[float]:
    """
    Synchronous wrapper for get_track_aoty_score
    
    Args:
        track_name: Name of the track
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        Track score from AOTY or None if not found
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_track_aoty_score(track_name, album_name, artist_name))
                return future.result()
        else:
            # No running loop, can use asyncio.run
            return asyncio.run(get_track_aoty_score(track_name, album_name, artist_name))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return None


async def get_album_metadata_only(album_name: str, artist_name: str) -> Dict:
    """
    Get only metadata (genres, release info) from AOTY without full scraping
    
    Args:
        album_name: Name of the album
        artist_name: Name of the artist
        
    Returns:
        Dictionary with AOTY metadata
    """
    try:
        # Search for the album on AOTY
        album_search_result = await get_album_url(artist_name, album_name)
        
        if not album_search_result:
            logger.warning(f"Album not found on AOTY: {album_name} by {artist_name}")
            return {'genres': [], 'metadata': {}}
        
        album_url, found_artist, found_album = album_search_result
        
        # Create a new page to extract just metadata
        page = await new_page()
        try:
            await page.goto(album_url, timeout=30000, wait_until="domcontentloaded")
            
            # Extract genres
            genres = []
            try:
                genre_elements = await page.query_selector_all(".albumGenres a")
                for genre_elem in genre_elements:
                    genre_text = await genre_elem.text_content()
                    if genre_text:
                        genres.append(genre_text.strip().lower())
            except Exception as e:
                logger.debug(f"Could not extract genres: {e}")
            
            # Extract basic metadata
            metadata = {'genres': genres}
            
            # Try to get album score quickly
            try:
                score_elem = await page.query_selector(".albumScore")
                if score_elem:
                    score_text = await score_elem.text_content()
                    if score_text:
                        metadata['album_score'] = float(score_text.strip())
            except Exception as e:
                logger.debug(f"Could not extract album score: {e}")
            
            logger.info(f"Successfully fetched AOTY metadata for album '{album_name}' by {artist_name}")
            return metadata
            
        finally:
            await page.close()
        
    except Exception as e:
        logger.error(f"Error fetching AOTY metadata for album '{album_name}' by {artist_name}: {e}")
        return {'genres': [], 'metadata': {}}


def batch_get_aoty_data(album_list: List[tuple]) -> List[Dict]:
    """
    Batch process multiple albums for AOTY data
    
    Args:
        album_list: List of (album_name, artist_name) tuples
        
    Returns:
        List of AOTY data dictionaries
    """
    async def batch_process():
        tasks = []
        for album_name, artist_name in album_list:
            task = get_album_aoty_data(album_name, artist_name)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    try:
        return asyncio.run(batch_process())
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return [{
            'album_score': None,
            'genres': [],
            'track_scores': {},
            'url': None,
            'metadata': {}
        } for _ in album_list] 