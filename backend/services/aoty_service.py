import httpx
from typing import Optional, Dict, Any, List

try:
    from config.settings import settings
    from services.aoty_scraper_service import get_album_url, scrape_album
    from utils.matching import find_best_match
except ImportError:
    # Fallback settings if services not available
    settings = type('Settings', (), {'aoty_api_url': 'https://albumoftheyear.org'})()
    
    async def get_album_url(*args, **kwargs):
        return None
    
    async def scrape_album(*args, **kwargs):
        return None
    
    def find_best_match(*args, **kwargs):
        return None

# Simple in-memory cache
_album_cache = {}

async def get_cached_album(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get album data from cache if available."""
    return _album_cache.get(cache_key)

async def cache_album_data(cache_key: str, data: Dict[str, Any]) -> None:
    """Store album data in cache."""
    _album_cache[cache_key] = data

class AOTYService:
    def __init__(self, base_url=None):
        settings = get_settings()
        self.base_url = base_url or getattr(settings, 'aoty_base_url', 'https://albumoftheyear.org')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def get_album(self, artist: str, album: str) -> Optional[Dict[str, Any]]:
        """Fetch album data from AOTY using scraper service with caching."""
        # Generate a cache key
        cache_key = f"{artist}:{album}".lower().replace(" ", "-")
        
        # Check cache first
        cached_data = await get_cached_album(cache_key)
        if cached_data:
            return cached_data
        
        # Fetch using scraper service if not in cache
        try:
            # First get the album URL
            album_url = await get_album_url(artist, album)
            if not album_url:
                return None
            
            # Then scrape the album data
            album_data = await scrape_album(album_url)
            if album_data:
                # Cache for future use
                await cache_album_data(cache_key, album_data)
                return album_data
            
            return None
        except Exception as e:
            print(f"Error fetching album data for {artist} - {album}: {str(e)}")
            return None
    
    async def get_similar_albums(self, artist: str, album: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch similar albums from AOTY API."""
        try:
            response = await self.client.get(
                f"{AOTY_API_URL}/album/similar/",
                params={"artist": artist, "album": album, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching similar albums for {artist} - {album}: {str(e)}")
            return []