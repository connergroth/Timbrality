"""
Enhanced Last.fm API Module
Pulls user top tracks and recent tracks with rate limiting and caching
"""
import asyncio
import json
import logging
import random
from datetime import datetime
from typing import List, Optional, Dict, Any
import httpx
from app.config import settings
from app.models import SongCore
import os

logger = logging.getLogger(__name__)


class LastfmClient:
    """Async Last.fm API client with rate limiting"""
    
    def __init__(self):
        self.api_key = settings.lastfm_api_key
        self.username = settings.lastfm_username
        self.base_url = "https://ws.audioscrobbler.com/2.0/"
        self.rate_limit_delay = 60.0 / settings.lastfm_rate_limit  # Convert to seconds between requests
        
    async def _make_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to Last.fm API"""
        
        # Add required parameters
        params.update({
            "api_key": self.api_key,
            "method": method,
            "format": "json"
        })
        
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                response = await client.get(self.base_url, params=params)
                
                # Rate limiting with random jitter
                jitter = random.uniform(0.25, 0.4)  # 250-400ms jitter
                await asyncio.sleep(self.rate_limit_delay + jitter)
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Last.fm rate limit hit, backing off...")
                    await asyncio.sleep(5)  # Back off for 5 seconds
                    return None
                logger.error(f"Last.fm API error {e.response.status_code}: {e}")
                return None
            except Exception as e:
                logger.error(f"Last.fm request failed: {e}")
                return None
    
    async def get_top_tracks(self, max_tracks: int = 10000, period: str = "overall") -> List[SongCore]:
        """
        Get user's top tracks with pagination
        
        Args:
            max_tracks: Maximum number of tracks to fetch
            period: Time period (overall, 7day, 1month, 3month, 6month, 12month)
            
        Returns:
            List of SongCore objects ordered by playcount
        """
        tracks = []
        page = 1
        limit = min(1000, max_tracks)  # Max 1000 per page per Last.fm API
        
        logger.info(f"Fetching top {max_tracks} tracks for user {self.username}")
        
        while len(tracks) < max_tracks:
            params = {
                "user": self.username,
                "period": period,
                "limit": limit,
                "page": page
            }
            
            logger.info(f"Fetching page {page} (tracks {len(tracks)}/{max_tracks})")
            
            response = await self._make_request("user.getTopTracks", params)
            if not response:
                logger.error(f"Failed to fetch page {page}")
                break
            
            if "toptracks" not in response or "track" not in response["toptracks"]:
                logger.warning(f"No tracks in response for page {page}")
                break
                
            page_tracks = response["toptracks"]["track"]
            if not page_tracks:
                logger.info("No more tracks available")
                break
            
            # Convert to SongCore objects
            for track_data in page_tracks:
                try:
                    track = self._parse_track(track_data)
                    if track:
                        tracks.append(track)
                        
                    if len(tracks) >= max_tracks:
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to parse track: {e}")
                    continue
            
            page += 1
            
            # Check if we got fewer tracks than requested (last page)
            if len(page_tracks) < limit:
                break
        
        logger.info(f"Successfully fetched {len(tracks)} tracks from Last.fm")
        
        # Save raw data for replay
        await self._save_raw_data(tracks, "top_tracks")
        
        return tracks[:max_tracks]
    
    async def get_recent_tracks(self, max_tracks: int = 1000) -> List[SongCore]:
        """
        Get user's recent tracks
        
        Args:
            max_tracks: Maximum number of recent tracks to fetch
            
        Returns:
            List of SongCore objects
        """
        tracks = []
        page = 1
        limit = min(200, max_tracks)  # Max 200 per page for recent tracks
        
        logger.info(f"Fetching recent {max_tracks} tracks for user {self.username}")
        
        while len(tracks) < max_tracks:
            params = {
                "user": self.username,
                "limit": limit,
                "page": page
            }
            
            response = await self._make_request("user.getRecentTracks", params)
            if not response:
                break
            
            if "recenttracks" not in response or "track" not in response["recenttracks"]:
                break
                
            page_tracks = response["recenttracks"]["track"]
            if not page_tracks:
                break
            
            for track_data in page_tracks:
                try:
                    # Skip "now playing" tracks
                    if isinstance(track_data, dict) and track_data.get("@attr", {}).get("nowplaying"):
                        continue
                        
                    track = self._parse_track(track_data)
                    if track:
                        tracks.append(track)
                        
                    if len(tracks) >= max_tracks:
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to parse recent track: {e}")
                    continue
            
            page += 1
            
            if len(page_tracks) < limit:
                break
        
        logger.info(f"Successfully fetched {len(tracks)} recent tracks from Last.fm")
        
        # Save raw data for replay
        await self._save_raw_data(tracks, "recent_tracks")
        
        return tracks[:max_tracks]
    
    def _parse_track(self, track_data: Dict[str, Any]) -> Optional[SongCore]:
        """Parse Last.fm track data into SongCore object"""
        try:
            # Handle different response formats
            if isinstance(track_data.get("artist"), dict):
                artist = track_data["artist"]["name"]
            else:
                artist = track_data.get("artist", "")
            
            title = track_data.get("name", "")
            
            if not artist or not title:
                return None
            
            # Extract playcount
            playcount = 0
            if "playcount" in track_data:
                try:
                    playcount = int(track_data["playcount"])
                except (ValueError, TypeError):
                    pass
            
            return SongCore(
                artist=artist.strip(),
                title=title.strip(),
                playcount=playcount,
                source="lastfm"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse track data: {e}")
            return None
    
    async def _save_raw_data(self, tracks: List[SongCore], data_type: str):
        """Save raw track data for replay"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_lastfm_{timestamp}.json"
            filepath = os.path.join(settings.data_dir, filename)
            
            # Convert to serializable format
            data = [track.dict() for track in tracks]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "timestamp": timestamp,
                        "username": self.username,
                        "track_count": len(tracks),
                        "data_type": data_type
                    },
                    "tracks": data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved raw Last.fm data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")


async def pull_top_tracks(max_tracks: int) -> List[SongCore]:
    """
    Pull user's top tracks from Last.fm
    
    Args:
        max_tracks: Maximum number of tracks to fetch
        
    Returns:
        List of SongCore objects ordered by playcount
    """
    client = LastfmClient()
    return await client.get_top_tracks(max_tracks)


async def pull_recent_tracks(max_tracks: int = 1000) -> List[SongCore]:
    """
    Pull user's recent tracks from Last.fm
    
    Args:
        max_tracks: Maximum number of recent tracks to fetch
        
    Returns:
        List of SongCore objects
    """
    client = LastfmClient()
    return await client.get_recent_tracks(max_tracks)