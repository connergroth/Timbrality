from typing import Dict, Any, Optional, List
from .base import BaseTool, ToolResult
from .spotify_search import SpotifySearchTool


class SongRecommendationTool(BaseTool):
    """Tool for making specific song recommendations with full metadata from Spotify."""
    
    def __init__(self):
        super().__init__(
            name="song_recommendation",
            description="Recommend specific songs with full metadata including album art, Spotify links, and track details."
        )
        self.spotify_search = SpotifySearchTool()
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute song recommendation with Spotify metadata lookup."""
        try:
            # Extract parameters
            song_name = intent_result.get("song", "").strip()
            artist_name = intent_result.get("artist", "").strip()
            reason = intent_result.get("reason", "").strip()
            
            if not song_name:
                return self._create_error_result("Song name is required for recommendation")
            
            # Construct search query with better formatting for Spotify
            if artist_name:
                search_query = f'track:"{song_name}" artist:"{artist_name}"'
            else:
                search_query = f'track:"{song_name}"'
            
            # Search Spotify for the track
            search_result = await self.spotify_search.execute({
                "query": search_query,
                "type": "track",
                "limit": 5
            }, context)
            
            if not search_result.success or not search_result.data.get("results"):
                # Try a simpler search as fallback
                fallback_query = f"{song_name} {artist_name}" if artist_name else song_name
                fallback_result = await self.spotify_search.execute({
                    "query": fallback_query,
                    "type": "track",
                    "limit": 10
                }, context)
                
                if fallback_result.success and fallback_result.data.get("results"):
                    spotify_tracks = fallback_result.data["results"]
                else:
                    # Final fallback: return basic recommendation without Spotify data
                    return self._create_success_result({
                        "recommendation": {
                            "name": song_name,
                            "artist": artist_name or "Unknown Artist",
                            "reason": reason or f"Recommended track: {song_name}",
                            "source": "agent",
                            "has_spotify_data": False
                        }
                    })
            else:
                # Get the search results
                spotify_tracks = search_result.data["results"]
            
            # Get the best match (first result is usually most relevant)
            best_match = spotify_tracks[0]
            
            # Find exact match if possible - prioritize exact artist matches
            if artist_name:
                exact_matches = []
                partial_matches = []
                
                for track in spotify_tracks:
                    track_artists = [track["artist"]] + track.get("artists", [])
                    
                    # Check for exact artist name match (case insensitive)
                    if any(artist_name.lower().strip() == artist.lower().strip() for artist in track_artists):
                        exact_matches.append(track)
                    # Check for partial match as fallback
                    elif any(artist_name.lower() in artist.lower() for artist in track_artists):
                        partial_matches.append(track)
                
                # Prefer exact matches, then partial matches, then first result
                if exact_matches:
                    best_match = exact_matches[0]
                elif partial_matches:
                    best_match = partial_matches[0]
                # else keep the original best_match (first result)
            
            # Format recommendation with Spotify metadata
            recommendation = {
                "id": best_match["id"],
                "spotify_id": best_match["id"],
                "name": best_match["name"],
                "artist": best_match["artist"],
                "artists": best_match.get("artists", [best_match["artist"]]),
                "album": best_match.get("album"),
                "album_id": best_match.get("album_id"),
                "artwork_url": best_match.get("cover_art"),
                "spotify_url": best_match.get("spotify_url"),
                "preview_url": best_match.get("preview_url"),
                "duration_ms": best_match.get("duration_ms"),
                "popularity": best_match.get("popularity"),
                "explicit": best_match.get("explicit"),
                "release_date": best_match.get("release_date"),
                "source": "agent",
                "recommendation_reason": reason or f"I think you'll enjoy this track",
                "has_spotify_data": True,
                "confidence_score": 0.9  # High confidence for specific recommendations
            }
            
            # TODO: Add to database cache to avoid repeated Spotify API calls for the same recommendation
            
            return self._create_success_result({
                "recommendation": recommendation,
                "search_query": search_query,
                "spotify_matches": len(spotify_tracks)
            })
            
        except Exception as e:
            return self._create_error_result(f"Song recommendation failed: {str(e)}")
    
    def format_for_display(self, recommendation: Dict[str, Any], user_context: str = "") -> str:
        """Format the recommendation for natural language display."""
        song = recommendation["name"]
        artist = recommendation["artist"]
        reason = recommendation.get("recommendation_reason", "")
        
        if reason:
            return f"I recommend '{song}' by {artist}. {reason}"
        else:
            return f"You might like '{song}' by {artist}."