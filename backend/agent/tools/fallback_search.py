from typing import Dict, Any, List
import asyncio
import httpx
from .base import BaseTool, ToolResult
from services.aoty_service import AOTYService


class FallbackSearchTool(BaseTool):
    """Tool for fallback search when primary methods fail."""
    
    def __init__(self):
        super().__init__(
            name="fallback_search",
            description="Search AOTY, forums, and web when track metadata is insufficient"
        )
        self.aoty_service = AOTYService()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute fallback search across multiple sources."""
        try:
            entities = intent_result.entities
            search_query = entities.get("search_query", "")
            mood = entities.get("mood", "")
            
            if not search_query and not mood:
                return self._create_error_result("No search terms for fallback search")
            
            # Combine search terms
            full_query = " ".join(filter(None, [search_query, mood]))
            
            # Execute multiple search strategies concurrently
            aoty_task = self._search_aoty(full_query)
            web_task = self._search_web_forums(full_query)
            genre_task = self._search_by_genre_mood(mood) if mood else self._create_empty_result()
            
            aoty_results, web_results, genre_results = await asyncio.gather(
                aoty_task, web_task, genre_task, return_exceptions=True
            )
            
            # Combine results
            all_tracks = []
            explanations = []
            
            if not isinstance(aoty_results, Exception) and aoty_results:
                all_tracks.extend(aoty_results.get("tracks", []))
                explanations.append(f"Found {len(aoty_results.get('tracks', []))} albums from AOTY")
            
            if not isinstance(web_results, Exception) and web_results:
                all_tracks.extend(web_results.get("tracks", []))
                explanations.append(f"Found {len(web_results.get('tracks', []))} recommendations from web sources")
            
            if not isinstance(genre_results, Exception) and genre_results:
                all_tracks.extend(genre_results.get("tracks", []))
                explanations.append(f"Found {len(genre_results.get('tracks', []))} tracks by genre matching")
            
            if not all_tracks:
                return self._create_error_result("No tracks found through fallback search")
            
            # Deduplicate and rank results
            unique_tracks = self._deduplicate_tracks(all_tracks)
            ranked_tracks = self._rank_fallback_results(unique_tracks, full_query)
            
            return self._create_success_result({
                "tracks": ranked_tracks[:15],
                "explanation": "Used fallback search across multiple sources: " + ", ".join(explanations),
                "search_methods": ["aoty", "web_forums", "genre_matching"]
            }, confidence=0.6)
            
        except Exception as e:
            return self._create_error_result(f"Fallback search failed: {str(e)}")
    
    async def _search_aoty(self, query: str) -> Dict[str, Any]:
        """Search Album of the Year for relevant albums."""
        try:
            # Search for albums matching the query
            search_results = await self.aoty_service.search_albums(query, limit=10)
            
            tracks = []
            for album in search_results.get("albums", []):
                # Convert album to track format for consistency
                track = {
                    "id": f"aoty_album_{album.get('id', '')}",
                    "name": album.get("title", ""),
                    "artist": album.get("artist", ""),
                    "album": album.get("title", ""),
                    "source": "aoty",
                    "aoty_rating": album.get("rating", 0),
                    "aoty_url": album.get("url", ""),
                    "year": album.get("year"),
                    "genres": album.get("genres", []),
                    "tags": album.get("tags", [])
                }
                
                if album.get("cover_url"):
                    track["artwork_url"] = album["cover_url"]
                
                tracks.append(track)
            
            return {"tracks": tracks}
            
        except Exception:
            return {"tracks": []}
    
    async def _search_web_forums(self, query: str) -> Dict[str, Any]:
        """Search web forums and music discovery sites."""
        try:
            # This would integrate with music forums, Reddit, etc.
            # For now, return simulated results
            
            forum_searches = [
                self._search_reddit_music(query),
                self._search_music_forums(query),
                self._search_rym_lists(query)
            ]
            
            results = await asyncio.gather(*forum_searches, return_exceptions=True)
            
            tracks = []
            for result in results:
                if not isinstance(result, Exception) and result:
                    tracks.extend(result.get("tracks", []))
            
            return {"tracks": tracks}
            
        except Exception:
            return {"tracks": []}
    
    async def _search_reddit_music(self, query: str) -> Dict[str, Any]:
        """Search Reddit music communities."""
        # Placeholder for Reddit API integration
        return {"tracks": []}
    
    async def _search_music_forums(self, query: str) -> Dict[str, Any]:
        """Search music forums and communities."""
        # Placeholder for music forum integration
        return {"tracks": []}
    
    async def _search_rym_lists(self, query: str) -> Dict[str, Any]:
        """Search RateYourMusic lists and charts."""
        # Placeholder for RateYourMusic integration
        return {"tracks": []}
    
    async def _search_by_genre_mood(self, mood: str) -> Dict[str, Any]:
        """Search by genre/mood mapping."""
        try:
            # Map moods to genres and search patterns
            mood_genre_mapping = {
                "summer": ["indie pop", "surf rock", "tropical house", "reggae"],
                "night": ["ambient", "darkwave", "trip-hop", "neo-soul"],
                "chill": ["lo-fi", "chillwave", "downtempo", "jazz"],
                "energetic": ["electronic", "punk", "dance", "pop-rock"],
                "sad": ["folk", "indie folk", "slowcore", "ambient"],
                "happy": ["pop", "indie pop", "funk", "disco"],
                "dark": ["post-rock", "doom metal", "dark ambient", "gothic"],
                "upbeat": ["pop-punk", "ska", "electronic", "new wave"],
                "relaxed": ["soft rock", "acoustic", "folk", "jazz"],
                "intense": ["metal", "hardcore", "industrial", "noise rock"]
            }
            
            mood_lower = mood.lower()
            relevant_genres = []
            
            for mood_key, genres in mood_genre_mapping.items():
                if mood_key in mood_lower:
                    relevant_genres.extend(genres)
            
            if not relevant_genres:
                return {"tracks": []}
            
            # For each genre, get some representative tracks
            tracks = []
            for genre in relevant_genres[:3]:  # Limit to top 3 genres
                genre_tracks = await self._get_tracks_by_genre(genre)
                tracks.extend(genre_tracks[:5])  # 5 tracks per genre
            
            return {"tracks": tracks}
            
        except Exception:
            return {"tracks": []}
    
    async def _get_tracks_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """Get representative tracks for a genre."""
        # This would query a genre database or use curated lists
        # For now, return placeholder data
        
        genre_examples = {
            "indie pop": [
                {"name": "Electric Feel", "artist": "MGMT", "genre": "indie pop"},
                {"name": "Young Folks", "artist": "Peter Bjorn and John", "genre": "indie pop"}
            ],
            "ambient": [
                {"name": "Music for Airports", "artist": "Brian Eno", "genre": "ambient"},
                {"name": "Discreet Music", "artist": "Brian Eno", "genre": "ambient"}
            ],
            "lo-fi": [
                {"name": "Shiloh", "artist": "Shiloh Dynasty", "genre": "lo-fi"},
                {"name": "Cotton Candy", "artist": "SwuM", "genre": "lo-fi"}
            ]
        }
        
        examples = genre_examples.get(genre, [])
        
        # Format as track objects
        tracks = []
        for i, example in enumerate(examples):
            track = {
                "id": f"genre_{genre}_{i}",
                "name": example["name"],
                "artist": example["artist"],
                "source": "genre_fallback",
                "genres": [genre],
                "confidence": 0.5
            }
            tracks.append(track)
        
        return tracks
    
    async def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for async gather."""
        return {"tracks": []}
    
    def _deduplicate_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks from combined results."""
        seen = set()
        unique_tracks = []
        
        for track in tracks:
            # Create a key based on track name and artist
            key = (
                track.get("name", "").lower().strip(),
                track.get("artist", "").lower().strip()
            )
            
            if key not in seen and key != ("", ""):
                seen.add(key)
                unique_tracks.append(track)
        
        return unique_tracks
    
    def _rank_fallback_results(
        self, 
        tracks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Rank fallback search results by relevance."""
        query_lower = query.lower()
        
        def calculate_relevance_score(track):
            score = 0.0
            
            # Source priority
            source_scores = {
                "aoty": 0.8,
                "spotify": 0.9,
                "lastfm": 0.7,
                "genre_fallback": 0.5,
                "web_forums": 0.6
            }
            score += source_scores.get(track.get("source", ""), 0.3)
            
            # Text matching
            track_text = f"{track.get('name', '')} {track.get('artist', '')}".lower()
            
            # Exact matches
            for word in query_lower.split():
                if word in track_text:
                    score += 0.2
            
            # AOTY rating bonus
            if track.get("aoty_rating"):
                score += min(track["aoty_rating"] / 100, 0.3)
            
            # Spotify popularity bonus
            if track.get("popularity"):
                score += track["popularity"] / 1000  # Small bonus
            
            return score
        
        # Sort by relevance score
        return sorted(tracks, key=calculate_relevance_score, reverse=True)