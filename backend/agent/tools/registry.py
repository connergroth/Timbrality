from typing import Dict, Optional
from .base import BaseTool
from .track_search import TrackSearchTool
from .similarity import SimilarityTool
from .playlist_analyzer import PlaylistAnalyzerTool
from .web_feel import WebFeelTool
from .fallback_search import FallbackSearchTool
from .explainability import ExplainabilityTool
from .memory_embedder import MemoryEmbedderTool
from .hybrid_recommender import HybridRecommenderTool


class ToolRegistry:
    """Registry for managing and accessing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        tools = [
            TrackSearchTool(),
            SimilarityTool(),
            PlaylistAnalyzerTool(),
            WebFeelTool(),
            FallbackSearchTool(),
            ExplainabilityTool(),
            MemoryEmbedderTool(),
            HybridRecommenderTool()
        ]
        
        for tool in tools:
            self.register_tool(tool.name, tool)
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool."""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        """List all available tools with descriptions."""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from registry."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False