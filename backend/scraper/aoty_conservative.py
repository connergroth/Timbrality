"""
Conservative AOTY Scraper - Designed to work around aggressive anti-bot measures.

This scraper:
1. Makes very few requests with long delays
2. Uses aggressive caching
3. Provides meaningful fallbacks
4. Only scrapes when absolutely necessary
"""

import asyncio
import random
import time
import json
import os
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

# Simple in-memory cache with file persistence
CACHE_FILE = "aoty_cache.json"
_cache = {}
_last_request_time = 0
MINIMUM_DELAY = 60.0  # 1 minute minimum between requests
CACHE_DURATION = 24 * 60 * 60  # 24 hours cache

def load_cache():
    """Load cache from file"""
    global _cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                _cache = json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
        _cache = {}

def save_cache():
    """Save cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(_cache, f, indent=2)
    except Exception as e:
        print(f"Cache save error: {e}")

def get_cache_key(artist: str, album: str) -> str:
    """Generate cache key for artist/album"""
    return f"{artist.lower().strip()}_{album.lower().strip()}"

def is_cache_valid(cache_entry: dict) -> bool:
    """Check if cache entry is still valid"""
    if 'timestamp' not in cache_entry:
        return False
    
    cache_time = cache_entry['timestamp']
    current_time = time.time()
    return (current_time - cache_time) < CACHE_DURATION

def should_allow_request() -> Tuple[bool, float]:
    """Check if we should allow a request and how long to wait"""
    global _last_request_time
    
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    if time_since_last < MINIMUM_DELAY:
        wait_time = MINIMUM_DELAY - time_since_last
        return False, wait_time
    
    return True, 0.0

async def conservative_search_album(artist: str, album: str, force: bool = False) -> Optional[dict]:
    """
    Conservative album search that respects rate limits and uses caching.
    
    Args:
        artist: Artist name
        album: Album title  
        force: If True, bypasses cache (use sparingly!)
    
    Returns:
        Dictionary with album data or None if not found/blocked
    """
    global _last_request_time
    
    # Load cache if not already loaded
    if not _cache:
        load_cache()
    
    cache_key = get_cache_key(artist, album)
    
    # Check cache first (unless forced)
    if not force and cache_key in _cache:
        cache_entry = _cache[cache_key]
        if is_cache_valid(cache_entry):
            print(f"🗄️  Using cached data for {artist} - {album}")
            return cache_entry.get('data')
    
    # Check if we should make a request
    can_request, wait_time = should_allow_request()
    
    if not can_request:
        print(f"⏰ Rate limited: need to wait {wait_time:.1f}s before AOTY request")
        if not force:
            # Return cached data even if expired, rather than making request
            if cache_key in _cache:
                print(f"🗄️  Returning expired cache for {artist} - {album}")
                return _cache[cache_key].get('data')
            return None
        else:
            print(f"⚠️  Forced request: waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    
    # Add extra random delay to appear more human
    extra_delay = random.uniform(5.0, 15.0)
    print(f"🤖 Making conservative AOTY request (waiting {extra_delay:.1f}s extra)")
    await asyncio.sleep(extra_delay)
    
    # Update last request time
    _last_request_time = time.time()
    
    # Try the actual request (this will likely fail due to blocking)
    try:
        # Import here to avoid circular imports
        from .aoty_scraper import get_album_url
        
        result = await get_album_url(artist, album, max_retries=1)
        
        if result:
            url, found_artist, found_album = result
            album_data = {
                "url": url,
                "artist": found_artist,
                "title": found_album,
                "source": "aoty_scraper",
                "success": True
            }
            
            # Cache the successful result
            _cache[cache_key] = {
                "data": album_data,
                "timestamp": time.time()
            }
            save_cache()
            
            print(f"✅ AOTY found: {found_artist} - {found_album}")
            return album_data
        else:
            # Cache the "not found" result to avoid repeat requests
            _cache[cache_key] = {
                "data": None,
                "timestamp": time.time()
            }
            save_cache()
            
            print(f"❌ AOTY: {artist} - {album} not found")
            return None
            
    except Exception as e:
        print(f"🚫 AOTY request blocked/failed: {str(e)[:100]}...")
        
        # Cache the failure to avoid immediate retry
        _cache[cache_key] = {
            "data": {"blocked": True, "error": str(e)[:100]},
            "timestamp": time.time()
        }
        save_cache()
        
        return None

def get_aoty_recommendations() -> dict:
    """Get practical recommendations for using AOTY"""
    return {
        "status": "AOTY is currently blocking automated requests",
        "recommendations": [
            "Use Spotify as primary data source",
            "Use Last.fm for additional metadata",
            "Only request AOTY data for high-priority albums",
            "Consider manual data entry for key albums",
            "Cache any AOTY data you do manage to get"
        ],
        "cache_stats": {
            "entries": len(_cache),
            "cache_file": CACHE_FILE
        }
    }

async def batch_conservative_search(album_pairs: List[Tuple[str, str]], max_requests: int = 3) -> List[dict]:
    """
    Conservatively search for multiple albums with strict limits.
    
    Args:
        album_pairs: List of (artist, album) tuples
        max_requests: Maximum number of actual web requests to make
    
    Returns:
        List of results (cached data preferred)
    """
    print(f"🔍 Batch search for {len(album_pairs)} albums (max {max_requests} web requests)")
    
    results = []
    requests_made = 0
    
    for artist, album in album_pairs:
        if requests_made >= max_requests:
            print(f"🛑 Reached request limit ({max_requests}), using cache only")
            result = await conservative_search_album(artist, album, force=False)
        else:
            result = await conservative_search_album(artist, album, force=False)
            if result and 'blocked' not in str(result):
                requests_made += 1
        
        results.append({
            "artist": artist,
            "album": album,
            "aoty_data": result,
            "status": "found" if result and 'blocked' not in str(result) else "unavailable"
        })
    
    return results

if __name__ == "__main__":
    async def test_conservative_scraper():
        print("Testing Conservative AOTY Scraper")
        print("=" * 40)
        
        # Test single search
        result = await conservative_search_album("Radiohead", "OK Computer")
        print(f"Single search result: {result}")
        
        # Test batch search
        albums = [
            ("Pink Floyd", "The Dark Side of the Moon"),
            ("The Beatles", "Abbey Road"),
            ("Nirvana", "Nevermind")
        ]
        
        batch_results = await batch_conservative_search(albums, max_requests=1)
        print(f"\nBatch results: {len(batch_results)} processed")
        
        # Show recommendations
        recs = get_aoty_recommendations()
        print(f"\nRecommendations: {recs}")
    
    asyncio.run(test_conservative_scraper()) 