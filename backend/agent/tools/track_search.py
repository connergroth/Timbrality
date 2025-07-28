from typing import Dict, Any, List
import asyncio
import httpx
from .base import BaseTool, ToolResult
from services.spotify_service import SpotifyService
from services.lastfm_service import LastFMService


class TrackSearchTool(BaseTool):
    """Tool for searching tracks via Spotify and Last.fm APIs."""
    
    def __init__(self):
        super().__init__(
            name="track_search",
            description="Search for tracks by name, artist, or other metadata"
        )
        self.spotify_service = SpotifyService()
        self.lastfm_service = LastFMService()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute track search based on intent results."""
        try:
            # Handle different intent_result formats
            if hasattr(intent_result, 'entities'):
                entities = intent_result.entities
                query = entities.get("search_query", "")
                artist = entities.get("artist", "")
                track_name = entities.get("track_name", "")
            elif hasattr(intent_result, 'raw_text'):
                # Use the raw text as the search query
                query = intent_result.raw_text
                artist = ""
                track_name = ""
            else:
                # Fallback: assume it's a mock intent object with raw_text
                query = getattr(intent_result, 'raw_text', str(intent_result))
                artist = ""
                track_name = ""
            
            if not any([query, artist, track_name]):
                return self._create_error_result("No search terms provided")
            
            # Build search query - for general queries, just use the text
            if query and not track_name and not artist:
                final_query = query
            else:
                search_terms = []
                if track_name:
                    search_terms.append(f"track:{track_name}")
                if artist:
                    search_terms.append(f"artist:{artist}")
                final_query = " ".join(search_terms) if search_terms else query
            
            # Search both services concurrently
            spotify_task = self._search_spotify(final_query)
            lastfm_task = self._search_lastfm(final_query)
            
            spotify_results, lastfm_results = await asyncio.gather(
                spotify_task, lastfm_task, return_exceptions=True
            )
            
            # Combine and deduplicate results
            tracks = self._combine_results(spotify_results, lastfm_results)
            
            if not tracks:
                return self._create_error_result("No tracks found")
            
            return self._create_success_result({
                "tracks": tracks[:20],  # Limit to top 20
                "explanation": f"Found {len(tracks)} tracks matching '{final_query}'"
            }, confidence=0.9)
            
        except Exception as e:
            return self._create_error_result(f"Search failed: {str(e)}")
    
    async def _search_spotify(self, query: str) -> List[Dict[str, Any]]:
        """Search Spotify for tracks."""
        try:
            # Use the spotipy search method directly
            results = self.spotify_service.sp.search(q=query, type="track", limit=20)
            return self._format_spotify_results(results)
        except Exception as e:
            print(f"Spotify search error: {e}")
            return []
    
    async def _search_lastfm(self, query: str) -> List[Dict[str, Any]]:
        """Search Last.fm for tracks."""
        try:
            results = await self.lastfm_service.search_tracks(query, limit=20)
            return self._format_lastfm_results(results)
        except Exception:
            return []
    
    def _format_spotify_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format Spotify search results."""
        tracks = []
        
        for item in results.get("tracks", {}).get("items", []):
            track = {
                "id": item["id"],
                "name": item["name"],
                "artist": ", ".join([artist["name"] for artist in item["artists"]]),
                "album": item["album"]["name"],
                "spotify_url": item["external_urls"]["spotify"],
                "preview_url": item.get("preview_url"),
                "popularity": item.get("popularity", 0),
                "source": "spotify",
                "duration_ms": item.get("duration_ms"),
                "explicit": item.get("explicit", False)
            }
            
            # Add album artwork
            if item["album"].get("images"):
                track["artwork_url"] = item["album"]["images"][0]["url"]
            
            tracks.append(track)
        
        return tracks
    
    def _format_lastfm_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format Last.fm search results."""
        tracks = []
        
        for item in results.get("results", {}).get("trackmatches", {}).get("track", []):
            track = {
                "id": f"lastfm_{item.get('mbid', '')}_{item.get('name', '')[:50]}",
                "name": item.get("name", ""),
                "artist": item.get("artist", ""),
                "lastfm_url": item.get("url", ""),
                "listeners": int(item.get("listeners", 0)),
                "source": "lastfm"
            }
            
            # Add album artwork if available
            if item.get("image"):
                for img in item["image"]:
                    if img.get("size") == "large":
                        track["artwork_url"] = img.get("#text")
                        break
            
            tracks.append(track)
        
        return tracks
    
    def _combine_results(self, spotify_results, lastfm_results) -> List[Dict[str, Any]]:
        """Combine and deduplicate results from both services."""
        if isinstance(spotify_results, Exception):
            spotify_results = []
        if isinstance(lastfm_results, Exception):
            lastfm_results = []
        
        # Start with Spotify results (usually higher quality)
        combined = list(spotify_results)
        
        # Add Last.fm results that don't duplicate Spotify tracks
        for lastfm_track in lastfm_results:
            is_duplicate = False
            
            for spotify_track in spotify_results:
                if self._is_duplicate_track(spotify_track, lastfm_track):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append(lastfm_track)
        
        # Sort by relevance (Spotify popularity, Last.fm listeners)
        return sorted(combined, key=self._get_track_score, reverse=True)
    
    def _is_duplicate_track(self, track1: Dict[str, Any], track2: Dict[str, Any]) -> bool:
        """Check if two tracks are duplicates."""
        name1 = track1.get("name", "").lower().strip()
        name2 = track2.get("name", "").lower().strip()
        artist1 = track1.get("artist", "").lower().strip()
        artist2 = track2.get("artist", "").lower().strip()
        
        # Simple similarity check
        name_similar = name1 == name2 or name1 in name2 or name2 in name1
        artist_similar = artist1 == artist2 or artist1 in artist2 or artist2 in artist1
        
        return name_similar and artist_similar
    
    def _get_track_score(self, track: Dict[str, Any]) -> float:
        """Calculate relevance score for sorting."""
        score = 0.0
        
        if track.get("source") == "spotify":
            score += track.get("popularity", 0) / 100  # Normalize to 0-1
            score += 0.1  # Bonus for Spotify (usually higher quality metadata)
        
        if track.get("source") == "lastfm":
            listeners = track.get("listeners", 0)
            score += min(listeners / 1000000, 1.0)  # Normalize listeners to 0-1
        
        return score