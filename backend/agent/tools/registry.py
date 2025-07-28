from typing import Dict, Optional, Type
from .base import BaseTool
from .track_search import TrackSearchTool
from .similarity import SimilarityTool
from .playlist_analyzer import PlaylistAnalyzerTool
from .web_feel import WebFeelTool
from .fallback_search import FallbackSearchTool
from .explainability import ExplainabilityTool
from .memory_embedder import MemoryEmbedderTool
from .hybrid_recommender import HybridRecommenderTool
from .spotify_search import SpotifySearchTool
from .spotify_playlist import SpotifyPlaylistTool
from .song_recommendation import SongRecommendationTool
from .vibe_discovery import VibeDiscoveryTool


class ToolRegistry:
    """Registry for managing and accessing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_classes: Dict[str, Type[BaseTool]] = {
            "track_search": TrackSearchTool,
            "similarity": SimilarityTool,
            "playlist_analyzer": PlaylistAnalyzerTool,
            "web_feel": WebFeelTool,
            "fallback_search": FallbackSearchTool,
            "explainability": ExplainabilityTool,
            "memory_embedder": MemoryEmbedderTool,
            "hybrid_recommender": HybridRecommenderTool,
            "spotify_search": SpotifySearchTool,
            "spotify_playlist": SpotifyPlaylistTool,
            "song_recommendation": SongRecommendationTool,
            "vibe_discovery": VibeDiscoveryTool
        }
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name, creating it if not already instantiated."""
        if name not in self.tools:
            if name in self.tool_classes:
                self.tools[name] = self.tool_classes[name]()
            else:
                return None
        return self.tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        """List all available tools with descriptions."""
        tool_descriptions = {}
        for name, tool_class in self.tool_classes.items():
            # Create a temporary instance to get the description
            temp_tool = tool_class()
            tool_descriptions[name] = temp_tool.description
        return tool_descriptions
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool."""
        self.tools[name] = tool
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from registry."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False