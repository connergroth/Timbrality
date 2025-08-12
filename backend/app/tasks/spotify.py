"""
Enhanced Spotify API Module
Async track search and enrichment with caching and rate limiting
"""
import asyncio
import json
import logging
import random
import time
from typing import List, Optional, Dict, Any, Tuple
import httpx
import redis.asyncio as redis
from app.config import settings
from app.models import SongCore, SpotifyAttrs

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Async Spotify API client with OAuth and caching"""
    
    def __init__(self):
        self.client_id = settings.spotify_client_id
        self.client_secret = settings.spotify_client_secret
        self.base_url = "https://api.spotify.com/v1"
        self.token_url = "https://accounts.spotify.com/api/token"
        
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._redis_client: Optional[redis.Redis] = None
        
        # Rate limiting
        self.rate_limit_delay = 60.0 / settings.spotify_rate_limit
        self.last_request_time = 0
        
        # Circuit breaker for 502/503 errors
        self.consecutive_server_errors = 0
        self.max_consecutive_errors = 5
        self.circuit_breaker_until = 0
        
    async def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get or create Redis client for caching"""
        if self._redis_client is None:
            try:
                # Skip Redis entirely for now - it's optional for caching
                logger.info("Skipping Redis cache - operating without cache")
                self._redis_client = None
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self._redis_client = None
        return self._redis_client
    
    async def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token using Client Credentials flow"""
        
        # Check if current token is still valid
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        # Check Redis cache for token
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                cached_token = await redis_client.get("spotify_access_token")
                if cached_token:
                    token_data = json.loads(cached_token)
                    if time.time() < token_data["expires_at"]:
                        self._access_token = token_data["access_token"]
                        self._token_expires_at = token_data["expires_at"]
                        return self._access_token
            except Exception as e:
                logger.warning(f"Failed to retrieve cached token: {e}")
        
        # Request new token
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=settings.request_timeout
                )
                response.raise_for_status()
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                self._token_expires_at = time.time() + token_data["expires_in"] - 60  # 1 minute buffer
                
                # Cache token in Redis
                if redis_client:
                    try:
                        cache_data = {
                            "access_token": self._access_token,
                            "expires_at": self._token_expires_at
                        }
                        await redis_client.setex(
                            "spotify_access_token",
                            token_data["expires_in"] - 60,
                            json.dumps(cache_data)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache token: {e}")
                
                return self._access_token
                
            except Exception as e:
                logger.error(f"Failed to get Spotify access token: {e}")
                return None
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to Spotify API with retry logic"""
        
        # Check circuit breaker
        now = time.time()
        if now < self.circuit_breaker_until:
            logger.warning(f"Circuit breaker active, skipping request for {self.circuit_breaker_until - now:.1f}s")
            return None
        
        token = await self._get_access_token()
        if not token:
            return None
        
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        async with httpx.AsyncClient() as client:
            for attempt in range(3):  # Max 3 retries
                try:
                    response = await client.get(
                        f"{self.base_url}/{endpoint}",
                        params=params or {},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=settings.request_timeout
                    )
                    
                    self.last_request_time = time.time()
                    
                    if response.status_code == 429:
                        # Rate limit hit - exponential backoff
                        retry_after = int(response.headers.get("Retry-After", 1))
                        wait_time = min(retry_after * (2 ** attempt), 60)  # Max 60 seconds
                        logger.warning(f"Spotify rate limit hit, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    # Reset circuit breaker on successful request
                    self.consecutive_server_errors = 0
                    return response.json()
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        # Token expired, clear it and retry
                        self._access_token = None
                        self._token_expires_at = 0
                        token = await self._get_access_token()
                        if not token:
                            return None
                        continue
                    elif e.response.status_code == 404:
                        # Not found - don't retry
                        return None
                    elif e.response.status_code in [502, 503, 504]:
                        # Server errors - increment counter and check circuit breaker
                        self.consecutive_server_errors += 1
                        
                        if self.consecutive_server_errors >= self.max_consecutive_errors:
                            # Activate circuit breaker for 5 minutes
                            self.circuit_breaker_until = time.time() + 300
                            logger.error(f"Circuit breaker activated due to {self.consecutive_server_errors} consecutive server errors")
                            return None
                        
                        # Longer backoff with jitter
                        base_wait = min(10 * (2 ** attempt), 120)  # Up to 2 minutes
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = base_wait * jitter
                        logger.warning(f"Spotify server error {e.response.status_code}, waiting {wait_time:.1f}s (attempt {attempt + 1}/3, consecutive errors: {self.consecutive_server_errors})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Spotify API error {e.response.status_code}: {e}")
                        if attempt < 2:  # Retry for other errors
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
                        
                except Exception as e:
                    logger.error(f"Spotify request failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
            
            return None
    
    async def search_track(self, artist: str, title: str) -> Optional[SpotifyAttrs]:
        """
        Search for a track on Spotify and return attributes
        
        Args:
            artist: Artist name
            title: Track title
            
        Returns:
            SpotifyAttrs object or None if not found
        """
        
        # Check cache first
        cache_key = f"sp::{artist.lower()}::{title.lower()}"
        redis_client = await self._get_redis_client()
        
        if redis_client:
            try:
                cached = await redis_client.get(cache_key)
                if cached:
                    cached_data = json.loads(cached)
                    if cached_data:  # Non-null cached result
                        return SpotifyAttrs(**cached_data)
                    else:  # Cached "not found"
                        return None
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        # Search Spotify
        query = f'track:"{title}" artist:"{artist}"'
        response = await self._make_request("search", {
            "q": query,
            "type": "track",
            "limit": 1
        })
        
        if not response or not response.get("tracks", {}).get("items"):
            # Cache negative result
            if redis_client:
                try:
                    await redis_client.setex(cache_key, 3600, json.dumps(None))  # Cache for 1 hour
                except Exception:
                    pass
            return None
        
        track = response["tracks"]["items"][0]
        
        # Build SpotifyAttrs (basic metadata only)
        attrs = SpotifyAttrs(
            track_id=track.get("id"),  # Include Spotify track ID
            duration_ms=track.get("duration_ms"),
            popularity=track.get("popularity"),
            album_id=track["album"]["id"] if track.get("album") else None,
            artist_id=track["artists"][0]["id"] if track.get("artists") else None,
            album_name=track["album"]["name"] if track.get("album") else None,
            release_date=track["album"].get("release_date") if track.get("album") else None,
            explicit=track.get("explicit"),
            track_number=track.get("track_number")
        )
        
        # Cache result
        if redis_client:
            try:
                await redis_client.setex(cache_key, 86400, attrs.json())  # Cache for 24 hours
            except Exception:
                pass
        
        return attrs
    
    
    async def enrich_tracks_batch(self, songs: List[SongCore], batch_size: int = 50) -> List[Tuple[SongCore, Optional[SpotifyAttrs]]]:
        """
        Enrich a batch of songs with Spotify data using controlled concurrency
        
        Args:
            songs: List of SongCore objects to enrich
            batch_size: Number of concurrent requests
            
        Returns:
            List of tuples (SongCore, SpotifyAttrs or None)
        """
        semaphore = asyncio.Semaphore(batch_size)
        
        async def enrich_single(song: SongCore) -> Tuple[SongCore, Optional[SpotifyAttrs]]:
            async with semaphore:
                attrs = await self.search_track(song.artist, song.title)
                return (song, attrs)
        
        logger.info(f"Enriching {len(songs)} tracks with Spotify data (batch size: {batch_size})")
        
        tasks = [enrich_single(song) for song in songs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to enrich track {songs[i].artist} - {songs[i].title}: {result}")
                valid_results.append((songs[i], None))
            else:
                valid_results.append(result)
        
        successful = sum(1 for _, attrs in valid_results if attrs is not None)
        logger.info(f"Successfully enriched {successful}/{len(songs)} tracks with Spotify data")
        
        return valid_results


async def search_track(artist: str, title: str) -> Optional[SpotifyAttrs]:
    """
    Search for a single track on Spotify
    
    Args:
        artist: Artist name
        title: Track title
        
    Returns:
        SpotifyAttrs object or None if not found
    """
    client = SpotifyClient()
    return await client.search_track(artist, title)


async def enrich_tracks(songs: List[SongCore]) -> List[Tuple[SongCore, Optional[SpotifyAttrs]]]:
    """
    Enrich multiple tracks with Spotify data
    
    Args:
        songs: List of SongCore objects to enrich
        
    Returns:
        List of tuples (SongCore, SpotifyAttrs or None)
    """
    client = SpotifyClient()
    return await client.enrich_tracks_batch(songs, settings.max_concurrent_requests)