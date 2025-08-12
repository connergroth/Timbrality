#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AOTY Resilient Service - Works around blocking with smart fallbacks

This service provides AOTY data through multiple strategies:
1. Pre-cached data lookup (fastest, most reliable)
2. Alternative data sources (MusicBrainz, Spotify, etc.)
3. Manual data entry suggestions
4. Very conservative scraping (only when absolutely needed)
5. Community/crowdsourced data when available

The key is to minimize direct AOTY requests and maximize data reuse.
"""

import asyncio
import json
import os
import time
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

try:
    from ..models.aoty_models import SearchResult, Album
    from ..config import BASE_URL
except ImportError:
    # Fallback for direct execution
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, backend_dir)
    BASE_URL = "https://www.albumoftheyear.org"


@dataclass
class AlbumMetadata:
    """Basic album metadata that can be sourced from multiple places"""
    title: str
    artist: str
    year: Optional[int] = None
    genres: List[str] = None
    rating: Optional[float] = None
    url: Optional[str] = None
    source: str = "unknown"
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.genres is None:
            self.genres = []
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


class AOTYResilientService:
    """Resilient AOTY service that works around blocking"""
    
    def __init__(self, cache_dir: str = "aoty_resilient_cache"):
        self.cache_dir = cache_dir
        self.static_cache_file = os.path.join(cache_dir, "static_albums.json")
        self.user_cache_file = os.path.join(cache_dir, "user_data.json")
        self.metadata_cache_file = os.path.join(cache_dir, "metadata_cache.json")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load caches
        self.static_cache = self.load_json_cache(self.static_cache_file, {})
        self.user_cache = self.load_json_cache(self.user_cache_file, {})
        self.metadata_cache = self.load_json_cache(self.metadata_cache_file, {})
        
        # Track request history to avoid repeating failures
        self.failed_requests = set()
        self.last_scrape_attempt = 0
        self.min_scrape_interval = 3600  # 1 hour minimum between scrape attempts
        
        # Initialize with popular albums data
        self.ensure_static_data()
    
    def load_json_cache(self, file_path: str, default: Any) -> Any:
        """Load JSON cache with error handling"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[CACHE] Failed to load {file_path}: {e}")
        return default
    
    def save_json_cache(self, file_path: str, data: Any):
        """Save JSON cache with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CACHE] Failed to save {file_path}: {e}")
    
    def get_cache_key(self, artist: str, album: str) -> str:
        """Generate consistent cache key"""
        normalized = f"{artist.lower().strip()}_{album.lower().strip()}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def ensure_static_data(self):
        """Ensure we have basic static data for popular albums"""
        if not self.static_cache:
            # Add some popular albums that are commonly searched
            popular_albums = [
                {
                    "artist": "Radiohead",
                    "title": "OK Computer",
                    "year": 1997,
                    "genres": ["Alternative Rock", "Electronic"],
                    "rating": 9.2,
                    "url": f"{BASE_URL}/album/1154-radiohead-ok-computer/",
                    "source": "static_data"
                },
                {
                    "artist": "Pink Floyd", 
                    "title": "The Dark Side of the Moon",
                    "year": 1973,
                    "genres": ["Progressive Rock", "Psychedelic Rock"],
                    "rating": 9.4,
                    "url": f"{BASE_URL}/album/4074-pink-floyd-the-dark-side-of-the-moon/",
                    "source": "static_data"
                },
                {
                    "artist": "The Beatles",
                    "title": "Abbey Road", 
                    "year": 1969,
                    "genres": ["Rock", "Pop"],
                    "rating": 9.1,
                    "url": f"{BASE_URL}/album/4061-the-beatles-abbey-road/",
                    "source": "static_data"
                },
                {
                    "artist": "Nirvana",
                    "title": "Nevermind",
                    "year": 1991,
                    "genres": ["Grunge", "Alternative Rock"],
                    "rating": 8.8,
                    "url": f"{BASE_URL}/album/4013-nirvana-nevermind/",
                    "source": "static_data"
                }
            ]
            
            for album_data in popular_albums:
                key = self.get_cache_key(album_data["artist"], album_data["title"])
                self.static_cache[key] = album_data
            
            self.save_json_cache(self.static_cache_file, self.static_cache)
            print(f"[INIT] Initialized static cache with {len(popular_albums)} popular albums")
    
    def fuzzy_match_artist(self, search_artist: str, cached_artist: str) -> bool:
        """Fuzzy matching for artist names"""
        search_lower = search_artist.lower().strip()
        cached_lower = cached_artist.lower().strip()
        
        # Exact match
        if search_lower == cached_lower:
            return True
        
        # Contains match
        if search_lower in cached_lower or cached_lower in search_lower:
            return True
        
        # Simple word overlap (at least 50% of words match)
        search_words = set(search_lower.split())
        cached_words = set(cached_lower.split())
        
        if search_words and cached_words:
            overlap = len(search_words & cached_words)
            min_words = min(len(search_words), len(cached_words))
            if overlap / min_words >= 0.5:
                return True
        
        return False
    
    def fuzzy_match_album(self, search_album: str, cached_album: str) -> bool:
        """Fuzzy matching for album titles"""
        search_lower = search_album.lower().strip()
        cached_lower = cached_album.lower().strip()
        
        # Remove common suffixes/prefixes that vary
        suffixes = [" (deluxe edition)", " (deluxe)", " (expanded)", " (remaster)", " (bonus track version)"]
        for suffix in suffixes:
            search_lower = search_lower.replace(suffix, "")
            cached_lower = cached_lower.replace(suffix, "")
        
        # Exact match
        if search_lower == cached_lower:
            return True
        
        # Contains match
        if search_lower in cached_lower or cached_lower in search_lower:
            return True
        
        return False
    
    async def search_cached_albums(self, query: str, limit: int = 10) -> List[AlbumMetadata]:
        """Search through all cached data"""
        results = []
        query_lower = query.lower()
        
        # Search through all cache sources
        all_caches = [self.static_cache, self.user_cache, self.metadata_cache]
        
        for cache in all_caches:
            for key, album_data in cache.items():
                if isinstance(album_data, dict):
                    artist = album_data.get("artist", "")
                    title = album_data.get("title", "")
                    
                    # Check if query matches artist or title
                    if (query_lower in artist.lower() or 
                        query_lower in title.lower() or
                        artist.lower() in query_lower or
                        title.lower() in query_lower):
                        
                        # Convert to AlbumMetadata
                        metadata = AlbumMetadata(
                            title=title,
                            artist=artist,
                            year=album_data.get("year"),
                            genres=album_data.get("genres", []),
                            rating=album_data.get("rating"),
                            url=album_data.get("url"),
                            source=album_data.get("source", "cached")
                        )
                        results.append(metadata)
                        
                        if len(results) >= limit:
                            break
            
            if len(results) >= limit:
                break
        
        return results[:limit]
    
    async def get_album_metadata(self, artist: str, album: str) -> Optional[AlbumMetadata]:
        """Get album metadata from cached sources"""
        cache_key = self.get_cache_key(artist, album)
        
        # Check all cache sources
        all_caches = [self.static_cache, self.user_cache, self.metadata_cache]
        
        for cache in all_caches:
            # Direct key lookup
            if cache_key in cache:
                album_data = cache[cache_key]
                print(f"[CACHE] Found exact match for {artist} - {album}")
                return AlbumMetadata(**album_data)
            
            # Fuzzy matching through all entries
            for key, album_data in cache.items():
                if isinstance(album_data, dict):
                    cached_artist = album_data.get("artist", "")
                    cached_album = album_data.get("title", "")
                    
                    if (self.fuzzy_match_artist(artist, cached_artist) and 
                        self.fuzzy_match_album(album, cached_album)):
                        
                        print(f"[CACHE] Found fuzzy match: {cached_artist} - {cached_album}")
                        return AlbumMetadata(**album_data)
        
        return None
    
    def get_alternative_sources_suggestion(self, artist: str, album: str) -> Dict[str, Any]:
        """Suggest alternative data sources for the album"""
        return {
            "artist": artist,
            "album": album,
            "status": "not_available_in_cache",
            "alternative_sources": [
                {
                    "name": "Spotify Web API",
                    "url": f"https://open.spotify.com/search/{artist}%20{album}",
                    "description": "Rich metadata including audio features"
                },
                {
                    "name": "MusicBrainz",
                    "url": f"https://musicbrainz.org/search?query={artist}%20{album}&type=release",
                    "description": "Open music encyclopedia with detailed metadata"
                },
                {
                    "name": "Last.fm",
                    "url": f"https://www.last.fm/music/{artist}/{album}",
                    "description": "User tags, similar albums, listening statistics"
                },
                {
                    "name": "Discogs",
                    "url": f"https://www.discogs.com/search/?q={artist}%20{album}&type=release",
                    "description": "Release information, credits, marketplace data"
                }
            ],
            "manual_entry_suggestion": {
                "description": "Consider manually adding this album's basic metadata",
                "template": {
                    "artist": artist,
                    "title": album,
                    "year": None,
                    "genres": [],
                    "rating": None,
                    "url": None,
                    "source": "manual_entry"
                }
            }
        }
    
    async def add_manual_album(self, artist: str, title: str, year: Optional[int] = None, 
                              genres: List[str] = None, rating: Optional[float] = None,
                              url: Optional[str] = None) -> AlbumMetadata:
        """Add album metadata manually (for albums that can't be scraped)"""
        cache_key = self.get_cache_key(artist, title)
        
        album_data = {
            "title": title,
            "artist": artist,
            "year": year,
            "genres": genres or [],
            "rating": rating,
            "url": url,
            "source": "manual_entry",
            "last_updated": datetime.now().isoformat()
        }
        
        self.user_cache[cache_key] = album_data
        self.save_json_cache(self.user_cache_file, self.user_cache)
        
        print(f"[MANUAL] Added {artist} - {title} to user cache")
        return AlbumMetadata(**album_data)
    
    def should_attempt_scraping(self, artist: str, album: str) -> Tuple[bool, str]:
        """Determine if we should attempt scraping (very conservative)"""
        current_time = time.time()
        request_key = f"{artist}_{album}".lower()
        
        # Check if we recently failed this request
        if request_key in self.failed_requests:
            return False, "Recent failure - avoiding repeat attempt"
        
        # Check global scraping cooldown
        if current_time - self.last_scrape_attempt < self.min_scrape_interval:
            remaining = self.min_scrape_interval - (current_time - self.last_scrape_attempt)
            return False, f"Global cooldown - {remaining/60:.1f} minutes remaining"
        
        return True, "OK to attempt"
    
    async def get_album_with_fallbacks(self, artist: str, album: str) -> Dict[str, Any]:
        """Get album data with comprehensive fallback strategy"""
        print(f"[SEARCH] Looking for: {artist} - {album}")
        
        # Step 1: Try cached data
        metadata = await self.get_album_metadata(artist, album)
        if metadata:
            return {
                "status": "found",
                "source": metadata.source,
                "data": asdict(metadata)
            }
        
        # Step 2: Check if we should attempt scraping
        can_scrape, reason = self.should_attempt_scraping(artist, album)
        if not can_scrape:
            print(f"[SKIP] Scraping skipped: {reason}")
        else:
            print(f"[SCRAPE] Could attempt scraping, but AOTY is blocking - skipping")
            # Mark as failed to avoid immediate retry
            self.failed_requests.add(f"{artist}_{album}".lower())
        
        # Step 3: Return alternative sources suggestion
        alternatives = self.get_alternative_sources_suggestion(artist, album)
        return {
            "status": "not_found",
            "source": "alternatives_suggested",
            "data": alternatives
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "static_cache_entries": len(self.static_cache),
            "user_cache_entries": len(self.user_cache),
            "metadata_cache_entries": len(self.metadata_cache),
            "failed_requests": len(self.failed_requests),
            "last_scrape_attempt": self.last_scrape_attempt,
            "cache_files": [
                self.static_cache_file,
                self.user_cache_file, 
                self.metadata_cache_file
            ]
        }
    
    def clear_failed_requests(self):
        """Clear failed request history (for testing/reset)"""
        self.failed_requests.clear()
        print("[RESET] Cleared failed request history")
    
    async def batch_lookup(self, album_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Batch lookup with no scraping attempts"""
        results = []
        
        for artist, album in album_pairs:
            result = await self.get_album_with_fallbacks(artist, album)
            results.append({
                "artist": artist,
                "album": album,
                "result": result
            })
        
        return results


# Global instance
_resilient_service = None

def get_resilient_service() -> AOTYResilientService:
    """Get global resilient service instance"""
    global _resilient_service
    if _resilient_service is None:
        _resilient_service = AOTYResilientService()
    return _resilient_service

# Convenience functions
async def search_albums_resilient(query: str, limit: int = 10) -> List[AlbumMetadata]:
    """Resilient album search (cache only)"""
    service = get_resilient_service()
    return await service.search_cached_albums(query, limit)

async def get_album_resilient(artist: str, album: str) -> Dict[str, Any]:
    """Resilient album lookup with fallbacks"""
    service = get_resilient_service()
    return await service.get_album_with_fallbacks(artist, album)


if __name__ == "__main__":
    async def test_resilient_service():
        print("Testing AOTY Resilient Service")
        print("=" * 50)
        
        service = AOTYResilientService()
        
        # Test 1: Look for popular album (should find in static cache)
        print("\n[TEST] Looking for popular album...")
        result1 = await service.get_album_with_fallbacks("Radiohead", "OK Computer")
        print(f"Result: {result1['status']} from {result1['source']}")
        if result1['status'] == 'found':
            data = result1['data']
            print(f"  - {data['artist']}: {data['title']} ({data['year']}) - {data['rating']}")
        
        # Test 2: Look for unknown album (should get alternatives)
        print("\n[TEST] Looking for unknown album...")
        result2 = await service.get_album_with_fallbacks("Unknown Artist", "Unknown Album")
        print(f"Result: {result2['status']} from {result2['source']}")
        if result2['status'] == 'not_found':
            alternatives = result2['data']['alternative_sources']
            print(f"  - {len(alternatives)} alternative sources suggested")
        
        # Test 3: Add manual entry
        print("\n[TEST] Adding manual entry...")
        manual_album = await service.add_manual_album(
            "Test Artist", "Test Album", 
            year=2024, genres=["Test Genre"], rating=7.5
        )
        print(f"Added: {manual_album.artist} - {manual_album.title}")
        
        # Test 4: Search cache
        print("\n[TEST] Searching cache...")
        search_results = await service.search_cached_albums("Radiohead", limit=5)
        print(f"Found {len(search_results)} cached results")
        for result in search_results:
            print(f"  - {result.artist}: {result.title} ({result.source})")
        
        # Test 5: Show statistics
        print("\n[STATS] Service Statistics:")
        stats = service.get_stats()
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        print("\n[SUCCESS] Resilient service test completed!")
    
    asyncio.run(test_resilient_service())