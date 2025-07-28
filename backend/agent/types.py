from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    SEARCH_TRACK = "search_track"
    FIND_SIMILAR = "find_similar"
    ANALYZE_PLAYLIST = "analyze_playlist"
    VIBE_DISCOVERY = "vibe_discovery"
    EXPLAIN_RECOMMENDATION = "explain_recommendation"
    UPDATE_PREFERENCES = "update_preferences"
    CONVERSATIONAL = "conversational"


@dataclass
class ToolInput:
    """Structured input contract for tools with validation."""
    query: str
    song_name: Optional[str] = None
    artist_name: Optional[str] = None
    album_name: Optional[str] = None
    spotify_id: Optional[str] = None
    previous_results: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    
    def validate_for_tool(self, tool_name: str) -> bool:
        """Validate that this input has required fields for specific tool."""
        requirements = {
            "spotify_search": ["query"],
            "vibe_discovery": ["song_name", "artist_name"],  # Force specific song analysis
            "song_recommendation": ["song_name", "artist_name"],
            "similarity": ["song_name", "artist_name"],
            "playlist_analyzer": ["query"],  # Can handle playlist URLs or IDs
            "hybrid_recommender": ["query"],  # Can work with general queries
            "track_search": ["query"],
            "web_feel": ["query"],
            "explainability": ["query"],
            "memory_embedder": ["query"]
        }
        
        required_fields = requirements.get(tool_name, ["query"])
        
        for field in required_fields:
            value = getattr(self, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                return False
        return True
    
    def get_validation_error(self, tool_name: str) -> str:
        """Get specific validation error message for tool."""
        if not self.validate_for_tool(tool_name):
            requirements = {
                "vibe_discovery": "requires specific song_name and artist_name (not general query)",
                "song_recommendation": "requires specific song_name and artist_name",
                "similarity": "requires specific song_name and artist_name"
            }
            return requirements.get(tool_name, f"missing required fields for {tool_name}")
        return ""


@dataclass
class ToolOutput:
    """Structured output contract for tools with validation."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    confidence: float = 0.0
    tool_name: Optional[str] = None
    
    def validate_schema(self) -> bool:
        """Validate that output contains expected structure."""
        if not self.success:
            return self.error is not None
        
        if not isinstance(self.data, dict):
            return False
            
        return True
    
    def get_tracks(self) -> List[Dict[str, Any]]:
        """Extract tracks from various data formats."""
        if not self.success or not self.data:
            return []
        
        # Handle different track data formats
        if 'tracks' in self.data:
            return self.data['tracks'] or []
        elif 'results' in self.data:
            # spotify_search format
            return self.data['results'] or []
        elif 'recommendation' in self.data:
            # song_recommendation format - single track
            return [self.data['recommendation']] if self.data['recommendation'] else []
        
        return []
    
    def get_primary_track(self) -> Optional[Dict[str, Any]]:
        """Get the first/primary track from results."""
        tracks = self.get_tracks()
        return tracks[0] if tracks else None


@dataclass
class ToolChainResult:
    """Result of executing a chain of tools."""
    success: bool
    tool_results: Dict[str, ToolOutput]
    final_tracks: List[Dict[str, Any]]
    explanations: List[str]
    tools_used: List[str]
    confidence: float
    errors: List[str]
    
    def validate_chain_consistency(self) -> bool:
        """Validate that chained tools operated on consistent data."""
        # Check spotify_search -> vibe_discovery consistency
        if "spotify_search" in self.tool_results and "vibe_discovery" in self.tool_results:
            search_result = self.tool_results["spotify_search"]
            vibe_result = self.tool_results["vibe_discovery"]
            
            if search_result.success and vibe_result.success:
                search_track = search_result.get_primary_track()
                vibe_data = vibe_result.data.get("vibe_analysis", {})
                
                if search_track and vibe_data:
                    search_song = search_track.get("name", "").lower()
                    analyzed_song = vibe_data.get("song", "").lower()
                    
                    # Check if the analyzed song matches the searched song
                    if search_song and analyzed_song:
                        if search_song not in analyzed_song and analyzed_song not in search_song:
                            self.errors.append(
                                f"Chain inconsistency: searched for '{search_song}' but analyzed '{analyzed_song}'"
                            )
                            return False
        
        return True


# Legacy compatibility - keep existing mock intent structure
class MockIntent:
    """Legacy compatibility for existing tool interfaces."""
    def __init__(self, tool_input: ToolInput):
        self.raw_text = tool_input.query
        self.processed_text = tool_input.query.lower()
        self.entities = tool_input.entities or {}
        self.confidence = 0.8
        self.song_name = tool_input.song_name
        self.artist_name = tool_input.artist_name
        self.album_name = tool_input.album_name
        self.spotify_id = tool_input.spotify_id
        self.previous_results = tool_input.previous_results 