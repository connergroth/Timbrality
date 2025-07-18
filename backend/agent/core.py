from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime

from .tools.base import BaseTool, ToolResult
from .tools.registry import ToolRegistry
from .memory.store import MemoryStore
from .embedding.engine import EmbeddingEngine
from .nlp.processor import NaturalLanguageProcessor


class IntentType(Enum):
    SEARCH_TRACK = "search_track"
    FIND_SIMILAR = "find_similar"
    ANALYZE_PLAYLIST = "analyze_playlist"
    VIBE_DISCOVERY = "vibe_discovery"
    EXPLAIN_RECOMMENDATION = "explain_recommendation"
    UPDATE_PREFERENCES = "update_preferences"


@dataclass
class AgentContext:
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_mood: Optional[str] = None
    timestamp: datetime = datetime.now()


@dataclass
class AgentResponse:
    content: str
    tracks: List[Dict[str, Any]]
    explanations: List[str]
    confidence: float
    tools_used: List[str]
    context_updates: Dict[str, Any]
    metadata: Dict[str, Any]


class TimbreAgent:
    """
    Core AI Agent for Timbre music recommendation platform.
    
    Orchestrates natural language understanding, tool selection,
    memory management, and personalized music discovery.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        nlp_processor: NaturalLanguageProcessor,
        tool_registry: ToolRegistry
    ):
        self.memory_store = memory_store
        self.embedding_engine = embedding_engine
        self.nlp_processor = nlp_processor
        self.tool_registry = tool_registry
        
    async def process_query(
        self, 
        user_input: str, 
        context: AgentContext
    ) -> AgentResponse:
        """
        Main entry point for processing user queries.
        
        Args:
            user_input: Natural language input from user
            context: Current session context and user data
            
        Returns:
            AgentResponse with recommendations and explanations
        """
        # 1. Parse intent and extract entities
        intent_result = await self.nlp_processor.parse_intent(user_input, context)
        
        # 2. Select appropriate tools based on intent
        tools_to_use = self._select_tools(intent_result.intent, intent_result.entities)
        
        # 3. Execute tools in parallel where possible
        tool_results = await self._execute_tools(tools_to_use, intent_result, context)
        
        # 4. Synthesize results and generate response
        response = await self._synthesize_response(
            user_input, intent_result, tool_results, context
        )
        
        # 5. Update user memory and preferences
        await self._update_memory(user_input, response, context)
        
        return response
    
    def _select_tools(self, intent: IntentType, entities: Dict[str, Any]) -> List[str]:
        """Select which tools to invoke based on detected intent."""
        tool_mapping = {
            IntentType.SEARCH_TRACK: ["track_search"],
            IntentType.FIND_SIMILAR: ["similarity", "track_search"],
            IntentType.ANALYZE_PLAYLIST: ["playlist_analyzer", "similarity"],
            IntentType.VIBE_DISCOVERY: ["web_feel", "similarity", "hybrid_recommender"],
            IntentType.EXPLAIN_RECOMMENDATION: ["explainability"],
            IntentType.UPDATE_PREFERENCES: ["memory_embedder"]
        }
        
        tools = tool_mapping.get(intent, ["hybrid_recommender"])
        
        # Add fallback search if no specific tracks found
        if "track_id" not in entities and "playlist_id" not in entities:
            tools.append("fallback_search")
            
        return tools
    
    async def _execute_tools(
        self, 
        tool_names: List[str], 
        intent_result, 
        context: AgentContext
    ) -> Dict[str, ToolResult]:
        """Execute selected tools concurrently."""
        tasks = []
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                task = tool.execute(intent_result, context)
                tasks.append((tool_name, task))
        
        results = {}
        if tasks:
            completed = await asyncio.gather(
                *[task for _, task in tasks], 
                return_exceptions=True
            )
            
            for (tool_name, _), result in zip(tasks, completed):
                if isinstance(result, Exception):
                    results[tool_name] = ToolResult(
                        success=False, 
                        data={}, 
                        error=str(result)
                    )
                else:
                    results[tool_name] = result
        
        return results
    
    async def _synthesize_response(
        self,
        user_input: str,
        intent_result,
        tool_results: Dict[str, ToolResult],
        context: AgentContext
    ) -> AgentResponse:
        """Combine tool results into a coherent response."""
        tracks = []
        explanations = []
        tools_used = []
        confidence_scores = []
        
        # Collect successful results
        for tool_name, result in tool_results.items():
            if result.success:
                tools_used.append(tool_name)
                confidence_scores.append(result.confidence)
                
                if "tracks" in result.data:
                    tracks.extend(result.data["tracks"])
                    
                if "explanation" in result.data:
                    explanations.append(result.data["explanation"])
        
        # Generate natural language response
        response_text = await self.nlp_processor.generate_response(
            user_input, intent_result, tracks, explanations, context
        )
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return AgentResponse(
            content=response_text,
            tracks=tracks[:10],  # Limit to top 10
            explanations=explanations,
            confidence=avg_confidence,
            tools_used=tools_used,
            context_updates={},
            metadata={
                "intent": intent_result.intent.value,
                "entities": intent_result.entities,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def _update_memory(
        self, 
        user_input: str, 
        response: AgentResponse, 
        context: AgentContext
    ):
        """Update user memory with interaction data."""
        interaction = {
            "input": user_input,
            "response": response.content,
            "tracks": response.tracks,
            "confidence": response.confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.memory_store.add_interaction(context.user_id, interaction)
        
        # Update user embedding based on interaction
        if response.tracks:
            await self.embedding_engine.update_user_profile(
                context.user_id, response.tracks, user_input
            )