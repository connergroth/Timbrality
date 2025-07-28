from typing import Dict, Any, Optional, List
from .base import BaseTool, ToolResult
from services.spotify_service import SpotifyService
from agent.llm.service import LLMService


class SpotifySearchTool(BaseTool):
    """Tool for searching Spotify for tracks, albums, and artists."""
    
    def __init__(self):
        super().__init__(
            name="spotify_search",
            description="Search Spotify for tracks, albums, and artists. Returns metadata including cover art and Spotify IDs."
        )
        self.spotify_service = SpotifyService(user_auth=False)  # Use client credentials for search
        self.llm_service = LLMService()  # For intent extraction
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute Spotify search based on intent parameters."""
        try:
            search_type = "track"  # Default to track search
            limit = 10
            search_query = None
            original_query = None
            
            # Handle different intent_result formats
            if hasattr(intent_result, 'raw_text'):
                # Mock intent from agent core - use LLM extraction
                original_query = intent_result.raw_text.strip()
                search_query = await self._extract_search_query(original_query)
            elif isinstance(intent_result, dict):
                # Direct dictionary format
                original_query = intent_result.get("query", "").strip()
                search_type = intent_result.get("type", "track").lower()
                limit = min(intent_result.get("limit", 10), 50)
                search_query = await self._extract_search_query(original_query)
            else:
                # Fallback: treat as string
                original_query = str(intent_result).strip()
                search_query = await self._extract_search_query(original_query)
            
            if not search_query:
                return self._create_error_result("Search query is required")
            
            # Perform search based on type
            if search_type == "track":
                results = await self._search_tracks(search_query, limit)
            elif search_type == "album":
                results = await self._search_albums(search_query, limit)
            elif search_type == "artist":
                results = await self._search_artists(search_query, limit)
            else:
                return self._create_error_result(f"Invalid search type: {search_type}. Must be 'track', 'album', or 'artist'")
            
            # TODO: Add database insertion for caching search results to avoid repeated Spotify API calls
            
            return self._create_success_result({
                "query": original_query,
                "search_query": search_query,
                "type": search_type,
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            return self._create_error_result(f"Spotify search failed: {str(e)}")
    
    async def _search_tracks(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for tracks on Spotify."""
        results = self.spotify_service.sp.search(q=query, type="track", limit=limit)
        tracks = []
        
        for track in results["tracks"]["items"]:
            tracks.append({
                "id": track["id"],
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "artists": [artist["name"] for artist in track["artists"]],
                "album": track["album"]["name"],
                "album_id": track["album"]["id"],
                "duration_ms": track["duration_ms"],
                "popularity": track["popularity"],
                "preview_url": track.get("preview_url"),
                "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                "spotify_url": track["external_urls"]["spotify"],
                "release_date": track["album"]["release_date"],
                "explicit": track["explicit"]
            })
        
        return tracks
    
    async def _search_albums(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for albums on Spotify."""
        results = self.spotify_service.sp.search(q=query, type="album", limit=limit)
        albums = []
        
        for album in results["albums"]["items"]:
            albums.append({
                "id": album["id"],
                "name": album["name"],
                "artist": album["artists"][0]["name"],
                "artists": [artist["name"] for artist in album["artists"]],
                "total_tracks": album["total_tracks"],
                "release_date": album["release_date"],
                "cover_art": album["images"][0]["url"] if album["images"] else None,
                "spotify_url": album["external_urls"]["spotify"],
                "album_type": album["album_type"]
            })
        
        return albums
    
    async def _search_artists(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for artists on Spotify."""
        results = self.spotify_service.sp.search(q=query, type="artist", limit=limit)
        artists = []
        
        for artist in results["artists"]["items"]:
            artists.append({
                "id": artist["id"],
                "name": artist["name"],
                "genres": artist["genres"],
                "popularity": artist["popularity"],
                "followers": artist["followers"]["total"],
                "profile_picture": artist["images"][0]["url"] if artist["images"] else None,
                "spotify_url": artist["external_urls"]["spotify"]
            })
        
        return artists
    
    async def _extract_search_query(self, user_query: str) -> str:
        """Extract clean search query using LLM intent extraction."""
        try:
            # Use LLM to extract structured intent
            intent_data = await self.llm_service.extract_music_intent(user_query)
            
            # Use the cleaned search_query from LLM extraction
            if intent_data and intent_data.get("search_query"):
                return intent_data["search_query"]
            
            # Fallback to regex parsing if LLM extraction fails
            return self._parse_search_query_fallback(user_query)
            
        except Exception as e:
            print(f"Intent extraction failed: {e}")
            return self._parse_search_query_fallback(user_query)
    
    def _parse_search_query_fallback(self, query: str) -> str:
        """Fallback regex-based query parsing when LLM extraction fails."""
        import re
        
        # Remove common prefixes and stopwords
        query_lower = query.lower().strip()
        
        # Remove common command phrases
        prefixes_to_remove = [
            r'^find\s+(the\s+)?song\s+',
            r'^search\s+for\s+',
            r'^look\s+for\s+',
            r'^play\s+',
            r'^get\s+me\s+',
        ]
        
        for prefix in prefixes_to_remove:
            query_lower = re.sub(prefix, '', query_lower)
        
        # Handle "SONG by ARTIST" pattern specifically
        by_pattern = r'^(.+?)\s+by\s+(.+?)(?:\s+and\s+.*)?$'
        match = re.match(by_pattern, query_lower)
        if match:
            song_name = match.group(1).strip()
            artist_name = match.group(2).strip()
            # Remove common trailing phrases
            artist_name = re.sub(r'\s+(and\s+explain.*|on\s+spotify.*|from.*)', '', artist_name)
            return f"{song_name} {artist_name}"
        
        # Handle "ARTIST - SONG" pattern
        dash_pattern = r'^(.+?)\s*-\s*(.+)$'
        match = re.match(dash_pattern, query_lower)
        if match:
            # Assume first part is artist for dash pattern
            artist_name = match.group(1).strip()
            song_name = match.group(2).strip()
            return f"{song_name} {artist_name}"
        
        # Return cleaned query if no specific pattern found
        return query_lower.strip()