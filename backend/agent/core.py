from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import json
from datetime import datetime

from .tools.base import BaseTool, ToolResult
from .tools.registry import ToolRegistry
from .memory.store import MemoryStore
from .embedding.engine import EmbeddingEngine
from .llm.service import LLMService
from .types import IntentType, ToolInput, ToolOutput, ToolChainResult, MockIntent

# Memory system imports
from services.memory_service import memory_service, ChatTurn
from services.memory_processor import trigger_conversation_processing, process_and_diversify_recommendations


@dataclass
class AgentContext:
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_mood: Optional[str] = None
    timestamp: datetime = datetime.now()
    
    # Enhanced memory fields
    chat_id: Optional[str] = None
    assembled_context: Optional[Dict[str, Any]] = None
    turn_count: int = 0


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
    Enhanced with dual-store memory system (Redis + PostgreSQL).
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        llm_service: LLMService,
        tool_registry: ToolRegistry
    ):
        self.memory_store = memory_store
        self.embedding_engine = embedding_engine
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.turn_counter = {}  # Track turns per chat for processing triggers
        
    async def process_query(
        self, 
        user_input: str, 
        context: AgentContext,
        stream_callback=None
    ) -> AgentResponse:
        """
        Main entry point for processing user queries using LLM-driven decision making.
        
        Args:
            user_input: Natural language input from user
            context: Current session context and user data
            
        Returns:
            AgentResponse with recommendations and explanations
        """
        # 1. Use LLM to intelligently select tools
        context_dict = {
            "user_id": context.user_id,
            "user_preferences": context.user_preferences,
            "current_mood": context.current_mood,
            "recent_interactions": context.conversation_history[-3:] if context.conversation_history else []
        }
        
        tool_selection = await self.llm_service.select_tools(user_input, context_dict)
        
        # 2. Execute selected tools (skip if none needed)
        tool_results = {}
        if tool_selection.tools and tool_selection.tools != ["none"]:
            # Stream tool selection information
            if stream_callback:
                await stream_callback({
                    'type': 'tool_selection',
                    'tools': tool_selection.tools,
                    'reasoning': tool_selection.reasoning
                })
            
            tool_results = await self._execute_tools_by_name(
                tool_selection.tools, user_input, context, stream_callback
            )
        
        # 3. Use LLM to generate natural response
        llm_response = await self.llm_service.generate_response(
            user_input, tool_selection.tools, tool_results, context_dict
        )
        
        # 4. Extract tracks from tool results for response structure
        tracks = []
        explanations = [tool_selection.reasoning, llm_response.reasoning]
        
        for tool_name, result in tool_results.items():
            # Handle both ToolResult and legacy result formats
            if hasattr(result, 'data') and result.data:
                # Extract tracks from ToolResult.data - handle different formats
                if 'tracks' in result.data and result.data['tracks']:
                    tracks.extend(result.data['tracks'])
                elif 'results' in result.data and result.data['results']:
                    # Handle spotify_search format which uses 'results' instead of 'tracks'
                    if tool_name in ['spotify_search', 'spotify_playlist']:
                        tracks.extend(result.data['results'])
                elif 'recommendation' in result.data and result.data['recommendation']:
                    # Handle song_recommendation format which uses 'recommendation'
                    if tool_name == 'song_recommendation':
                        tracks.append(result.data['recommendation'])
                
                if 'explanation' in result.data and result.data['explanation']:
                    explanations.append(result.data['explanation'])
            elif hasattr(result, 'tracks') and result.tracks:
                # Legacy format
                tracks.extend(result.tracks)
            elif hasattr(result, 'explanation') and result.explanation:
                # Legacy format
                explanations.append(result.explanation)
        
        # 5. For sequential tool execution, prioritize the exact track that was analyzed
        if tracks and tool_selection.tools:
            if self._needs_sequential_execution(tool_selection.tools):
                # For sequential chains, use the primary track that was analyzed
                primary_track = self._get_primary_analyzed_track(tool_results, tracks)
                if primary_track:
                    # Validate track ID consistency
                    validated_track = self._validate_track_consistency(primary_track, tool_results)
                    tracks = [validated_track]
                else:
                    # Fallback to first track
                    tracks = tracks[:1]
            else:
                # For parallel execution, use LLM mention filtering
                mentioned_tracks = await self._filter_tracks_by_llm_mentions(
                    tracks, llm_response.content
                )
                if mentioned_tracks:
                    tracks = mentioned_tracks
        
        # 6. Post-process LLM response to detect and fetch song recommendations
        # Only do this if no tracks were found from tools
        if not tracks:
            additional_tracks = await self._detect_and_fetch_song_recommendations(
                llm_response.content, context, stream_callback
            )
            if additional_tracks:
                tracks.extend(additional_tracks)
        
        # 5.1. Deduplicate tracks by ID
        seen_ids = set()
        unique_tracks = []
        for track in tracks:
            track_id = track.get('id') or track.get('spotify_id')
            if track_id and track_id not in seen_ids:
                seen_ids.add(track_id)
                unique_tracks.append(track)
            elif not track_id:
                # Keep tracks without IDs (shouldn't happen but just in case)
                unique_tracks.append(track)
        
        tracks = unique_tracks
        
        # 6. Create response
        response = AgentResponse(
            content=llm_response.content,
            tracks=tracks,
            explanations=explanations,
            confidence=min(tool_selection.confidence, llm_response.confidence),
            tools_used=tool_selection.tools,
            context_updates={},
            metadata={
                "tool_selection_reasoning": tool_selection.reasoning,
                "response_reasoning": llm_response.reasoning
            }
        )
        
        # 7. Update user memory and preferences
        await self._update_memory(user_input, response, context)
        
        return response
    
    async def _execute_tools_by_name(
        self, 
        tool_names: List[str], 
        user_input: str,
        context: AgentContext,
        stream_callback=None
    ) -> Dict[str, ToolResult]:
        """Execute tools with sequential chaining and validation."""
        # Detect if this is a chain that needs sequential execution
        needs_chaining = self._needs_sequential_execution(tool_names)
        
        if needs_chaining:
            return await self._execute_tools_sequentially(
                tool_names, user_input, context, stream_callback
            )
        else:
            return await self._execute_tools_parallel(
                tool_names, user_input, context, stream_callback
            )
    
    def _needs_sequential_execution(self, tool_names: List[str]) -> bool:
        """Determine if tool combination requires sequential execution."""
        sequential_chains = [
            # Search -> Analysis chains
            ["spotify_search", "vibe_discovery"],
            ["spotify_search", "similarity"], 
            ["spotify_search", "song_recommendation"],
            ["track_search", "vibe_discovery"],
            ["track_search", "similarity"],
            ["track_search", "song_recommendation"],
            
            # Playlist -> Analysis chains  
            ["playlist_analyzer", "similarity"],
            ["playlist_analyzer", "vibe_discovery"],
            
            # Recommendation -> Explanation chains
            ["hybrid_recommender", "explainability"],
            ["song_recommendation", "explainability"],
            
            # Vibe/Mood -> Recommendation chains
            ["vibe_discovery", "hybrid_recommender"],
            ["web_feel", "hybrid_recommender"],
            
            # Fallback chains (conditional)
            ["track_search", "fallback_search"],
            ["spotify_search", "fallback_search"]
        ]
        
        for chain in sequential_chains:
            if all(tool in tool_names for tool in chain):
                return True
        return False
    
    async def _execute_tools_sequentially(
        self,
        tool_names: List[str],
        user_input: str,
        context: AgentContext,
        stream_callback=None
    ) -> Dict[str, ToolResult]:
        """Execute tools sequentially with result passing and validation."""
        results = {}
        current_input = ToolInput(query=user_input)
        
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                results[tool_name] = ToolResult(
                    success=False,
                    data={},
                    error=f"Tool {tool_name} not found"
                )
                continue
            
            # Stream tool start
            if stream_callback:
                await stream_callback({
                    'type': 'tool_start',
                    'tool': tool_name,
                    'description': self._get_tool_description(tool_name)
                })
            
            try:
                # Update input based on previous results
                if results:
                    current_input = self._prepare_chained_input(
                        current_input, tool_name, results
                    )
                
                # Validate input for this tool
                if not current_input.validate_for_tool(tool_name):
                    error_msg = current_input.get_validation_error(tool_name)
                    results[tool_name] = ToolResult(
                        success=False,
                        data={},
                        error=f"Input validation failed for {tool_name}: {error_msg}"
                    )
                    
                    if stream_callback:
                        await stream_callback({
                            'type': 'tool_error',
                            'tool': tool_name,
                            'error': error_msg
                        })
                    continue
                
                # Create mock intent for compatibility
                mock_intent = MockIntent(current_input)
                
                # Execute the tool
                result = await tool.execute(mock_intent, context)
                
                # Convert to ToolOutput for consistency validation
                tool_output = self._convert_to_tool_output(result, tool_name)
                results[tool_name] = result  # Keep original format for existing code
                
                # Stream tool completion
                if stream_callback:
                    tracks_found = len(tool_output.get_tracks())
                    await stream_callback({
                        'type': 'tool_complete',
                        'tool': tool_name,
                        'success': tool_output.success,
                        'tracks_found': tracks_found
                    })
                    
            except Exception as e:
                results[tool_name] = ToolResult(
                    success=False,
                    data={},
                    error=str(e)
                )
                
                if stream_callback:
                    await stream_callback({
                        'type': 'tool_error',
                        'tool': tool_name,
                        'error': str(e)
                    })
        
        # Validate chain consistency
        self._validate_chain_consistency(results, tool_names)
        
        return results
    
    async def _execute_tools_parallel(
        self,
        tool_names: List[str],
        user_input: str,
        context: AgentContext,
        stream_callback=None
    ) -> Dict[str, ToolResult]:
        """Execute tools in parallel (original behavior)."""
        results = {}
        
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                # Stream tool start
                if stream_callback:
                    await stream_callback({
                        'type': 'tool_start',
                        'tool': tool_name,
                        'description': self._get_tool_description(tool_name)
                    })
                
                try:
                    # Create mock intent for compatibility
                    tool_input = ToolInput(query=user_input)
                    mock_intent = MockIntent(tool_input)
                    
                    # Execute the tool
                    result = await tool.execute(mock_intent, context)
                    results[tool_name] = result
                    
                    # Stream tool completion
                    if stream_callback:
                        tool_output = self._convert_to_tool_output(result, tool_name)
                        tracks_found = len(tool_output.get_tracks())
                        await stream_callback({
                            'type': 'tool_complete',
                            'tool': tool_name,
                            'success': tool_output.success,
                            'tracks_found': tracks_found
                        })
                        
                except Exception as e:
                    results[tool_name] = ToolResult(
                        success=False,
                        data={'tracks': [], 'explanation': f"Error executing {tool_name}: {str(e)}"},
                        error=str(e)
                    )
                    
                    if stream_callback:
                        await stream_callback({
                            'type': 'tool_error',
                            'tool': tool_name,
                            'error': str(e)
                        })
        
        return results
    
    def _prepare_chained_input(
        self, 
        current_input: ToolInput, 
        tool_name: str, 
        previous_results: Dict[str, Any]
    ) -> ToolInput:
        """Prepare input for chained tool execution."""
        # Handle spotify_search -> vibe_discovery chain
        if tool_name == "vibe_discovery" and "spotify_search" in previous_results:
            search_result = previous_results["spotify_search"]
            if search_result.success and search_result.data.get("results"):
                first_track = search_result.data["results"][0]
                return ToolInput(
                    query=current_input.query,
                    song_name=first_track["name"],
                    artist_name=first_track["artist"],
                    spotify_id=first_track["id"],
                    previous_results=search_result.data,
                    entities=current_input.entities,
                    user_context=current_input.user_context
                )
        
        # Handle spotify_search -> similarity chain
        elif tool_name == "similarity" and "spotify_search" in previous_results:
            search_result = previous_results["spotify_search"]
            if search_result.success and search_result.data.get("results"):
                first_track = search_result.data["results"][0]
                return ToolInput(
                    query=current_input.query,
                    song_name=first_track["name"],
                    artist_name=first_track["artist"],
                    spotify_id=first_track["id"],
                    previous_results=search_result.data,
                    entities=current_input.entities,
                    user_context=current_input.user_context
                )
        
        # Handle track_search -> vibe_discovery chain
        elif tool_name == "vibe_discovery" and "track_search" in previous_results:
            search_result = previous_results["track_search"]
            if search_result.success and search_result.data.get("tracks"):
                first_track = search_result.data["tracks"][0]
                return ToolInput(
                    query=current_input.query,
                    song_name=first_track.get("name", first_track.get("title")),
                    artist_name=first_track.get("artist"),
                    previous_results=search_result.data,
                    entities=current_input.entities,
                    user_context=current_input.user_context
                )
        
        # Handle track_search -> similarity chain
        elif tool_name == "similarity" and "track_search" in previous_results:
            search_result = previous_results["track_search"]
            if search_result.success and search_result.data.get("tracks"):
                first_track = search_result.data["tracks"][0]
                return ToolInput(
                    query=current_input.query,
                    song_name=first_track.get("name", first_track.get("title")),
                    artist_name=first_track.get("artist"),
                    previous_results=search_result.data,
                    entities=current_input.entities,
                    user_context=current_input.user_context
                )
        
        # Handle playlist_analyzer -> similarity chain
        elif tool_name == "similarity" and "playlist_analyzer" in previous_results:
            playlist_result = previous_results["playlist_analyzer"]
            if playlist_result.success and playlist_result.data:
                # Use playlist characteristics for similarity
                return ToolInput(
                    query=current_input.query,
                    previous_results=playlist_result.data,
                    entities={**(current_input.entities or {}), "playlist_analysis": playlist_result.data},
                    user_context=current_input.user_context
                )
        
        # Handle recommendation -> explainability chains
        elif tool_name == "explainability" and any(tool in previous_results for tool in ["hybrid_recommender", "song_recommendation"]):
            for rec_tool in ["hybrid_recommender", "song_recommendation"]:
                if rec_tool in previous_results:
                    rec_result = previous_results[rec_tool]
                    if rec_result.success and rec_result.data:
                        return ToolInput(
                            query=current_input.query,
                            previous_results=rec_result.data,
                            entities={**(current_input.entities or {}), "recommendations": rec_result.data},
                            user_context=current_input.user_context
                        )
        
        # Handle vibe/mood -> recommendation chains
        elif tool_name == "hybrid_recommender" and any(tool in previous_results for tool in ["vibe_discovery", "web_feel"]):
            for vibe_tool in ["vibe_discovery", "web_feel"]:
                if vibe_tool in previous_results:
                    vibe_result = previous_results[vibe_tool]
                    if vibe_result.success and vibe_result.data:
                        # Extract mood information for recommendations
                        mood_data = vibe_result.data.get("vibe_analysis", {}) or vibe_result.data.get("mood_analysis", {})
                        return ToolInput(
                            query=current_input.query,
                            previous_results=vibe_result.data,
                            entities={**(current_input.entities or {}), "mood_analysis": mood_data},
                            user_context=current_input.user_context
                        )
        
        return current_input
    
    def _convert_to_tool_output(self, result: Any, tool_name: str) -> ToolOutput:
        """Convert various result formats to ToolOutput."""
        if hasattr(result, 'success'):
            return ToolOutput(
                success=result.success,
                data=result.data if hasattr(result, 'data') else {},
                error=result.error if hasattr(result, 'error') else None,
                confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
                tool_name=tool_name
            )
        else:
            return ToolOutput(
                success=True,
                data=result if isinstance(result, dict) else {},
                tool_name=tool_name
            )
    
    def _validate_chain_consistency(
        self, 
        results: Dict[str, Any], 
        tool_names: List[str]
    ) -> None:
        """Validate that chained tools operated on consistent data."""
        # Convert results to ToolOutput format for validation
        tool_outputs = {}
        for name, result in results.items():
            tool_outputs[name] = self._convert_to_tool_output(result, name)
        
        # Create chain result for validation
        chain_result = ToolChainResult(
            success=all(r.success for r in tool_outputs.values()),
            tool_results=tool_outputs,
            final_tracks=[],  # Will be populated later
            explanations=[],
            tools_used=tool_names,
            confidence=0.0,
            errors=[]
        )
        
        # Validate consistency
        if not chain_result.validate_chain_consistency():
            print(f"WARNING: Tool chain consistency validation failed: {chain_result.errors}")
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get a human-readable description of what a tool does."""
        descriptions = {
            "track_search": "Searching for specific tracks or artists",
            "similarity": "Finding similar music based on your preferences", 
            "playlist_analyzer": "Analyzing your Spotify playlist",
            "vibe_discovery": "Discovering music that matches your mood",
            "hybrid_recommender": "Generating personalized recommendations",
            "explainability": "Explaining why music was recommended",
            "memory_embedder": "Learning your music preferences",
            "fallback_search": "Searching with alternative methods"
        }
        return descriptions.get(tool_name, f"Running {tool_name}")
    
    def _select_tools(self, intent: IntentType, entities: Dict[str, Any]) -> List[str]:
        """Select which tools to invoke based on detected intent."""
        tool_mapping = {
            IntentType.SEARCH_TRACK: ["track_search"],
            IntentType.FIND_SIMILAR: ["similarity", "track_search"],
            IntentType.ANALYZE_PLAYLIST: ["playlist_analyzer", "similarity"],
            IntentType.VIBE_DISCOVERY: ["vibe_discovery", "hybrid_recommender"],
            IntentType.EXPLAIN_RECOMMENDATION: ["explainability"],
            IntentType.UPDATE_PREFERENCES: ["memory_embedder"],
            IntentType.CONVERSATIONAL: []
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
        """Execute selected tools sequentially to allow data flow between tools."""
        results = {}
        
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                try:
                    # If this is hybrid_recommender and we have vibe_discovery results,
                    # pass the mood information to it
                    if tool_name == "hybrid_recommender" and "vibe_discovery" in results:
                        vibe_result = results["vibe_discovery"]
                        if vibe_result.success and vibe_result.data:
                            # Extract mood information from vibe_discovery
                            vibe_data = vibe_result.data.get("vibe_analysis", {})
                            if "mood" in vibe_data:
                                # Update intent_result with mood information
                                if not hasattr(intent_result, 'entities'):
                                    intent_result.entities = {}
                                intent_result.entities["mood"] = vibe_data["mood"]
                            elif "search_keywords" in vibe_data:
                                # Use search keywords as mood
                                keywords = vibe_data.get("search_keywords", [])
                                if keywords:
                                    if not hasattr(intent_result, 'entities'):
                                        intent_result.entities = {}
                                    intent_result.entities["mood"] = " ".join(keywords[:3])
                    
                    result = await tool.execute(intent_result, context)
                    results[tool_name] = result
                except Exception as e:
                    results[tool_name] = ToolResult(
                        success=False, 
                        data={}, 
                        error=str(e)
                    )
            else:
                results[tool_name] = ToolResult(
                    success=False, 
                    data={}, 
                    error=f"Tool {tool_name} not found"
                )
        
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
    
    async def _detect_and_fetch_song_recommendations(
        self, 
        response_content: str, 
        context: AgentContext,
        stream_callback=None
    ) -> List[Dict[str, Any]]:
        """Detect specific song recommendations in LLM response and fetch their metadata."""
        import re
        
        # Pattern to detect song recommendations: "I recommend 'Song Name' by Artist"
        patterns = [
            r"(?:I recommend|try|listen to|check out|you might like)\s+['\"]([^'\"]+)['\"](?:\s+by\s+([^.!?,\n]+))?",
            r"['\"]([^'\"]+)['\"](?:\s+by\s+([^.!?,\n]+))(?:\s+is|\.|\s)",
            r"(?:song|track)\s+['\"]([^'\"]+)['\"](?:\s+by\s+([^.!?,\n]+))?",
        ]
        
        found_recommendations = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, response_content, re.IGNORECASE)
            for match in matches:
                song_name = match.group(1).strip()
                artist_name = match.group(2).strip() if match.group(2) else ""
                
                # Clean up artist name
                if artist_name:
                    artist_name = re.sub(r'\.$', '', artist_name)  # Remove trailing period
                    artist_name = artist_name.strip()
                
                if song_name:  # Only proceed if we have a song name
                    found_recommendations.append({
                        "song": song_name,
                        "artist": artist_name,
                        "reason": f"Recommended by AI agent"
                    })
        
        # Fetch metadata for found recommendations
        tracks = []
        if found_recommendations:
            song_rec_tool = self.tool_registry.get_tool("song_recommendation")
            if song_rec_tool:
                if stream_callback:
                    await stream_callback({
                        'type': 'song_detection',
                        'songs_found': len(found_recommendations)
                    })
                
                for rec in found_recommendations:
                    try:
                        # Create mock intent for the song recommendation tool
                        mock_intent = {
                            "song": rec["song"],
                            "artist": rec["artist"],
                            "reason": rec["reason"]
                        }
                        
                        result = await song_rec_tool.execute(mock_intent, context)
                        
                        if result.success and "recommendation" in result.data:
                            tracks.append(result.data["recommendation"])
                            
                            if stream_callback:
                                await stream_callback({
                                    'type': 'song_metadata_fetched',
                                    'song': rec["song"],
                                    'artist': rec["artist"]
                                })
                    except Exception as e:
                        print(f"Failed to fetch metadata for {rec['song']} by {rec['artist']}: {e}")
        
        return tracks
    
    async def _filter_tracks_by_llm_mentions(
        self, 
        tracks: List[Dict[str, Any]], 
        llm_response: str
    ) -> List[Dict[str, Any]]:
        """Filter tracks to only include those specifically mentioned by the LLM."""
        import re
        
        # Extract song names mentioned in the LLM response
        mentioned_songs = set()
        
        # Patterns to detect song mentions
        patterns = [
            r"(?:song|track)\s+['\"]([^'\"]+)['\"]",
            r"['\"]([^'\"]+)['\"](?:\s+by\s+[^.!?,\n]+)",
            r"(?:listen to|check out|recommend)\s+['\"]([^'\"]+)['\"]",
            r"(?:called|titled)\s+['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                song_name = match.group(1).strip().lower()
                mentioned_songs.add(song_name)
        
        if not mentioned_songs:
            # If no specific songs mentioned, limit to first 3 tracks to avoid overwhelming
            return tracks[:3]
        
        # Filter tracks to only those mentioned
        filtered_tracks = []
        for track in tracks:
            track_name = track.get('name', '').lower().strip()
            # Check if this track name matches any mentioned song
            if any(mentioned in track_name or track_name in mentioned 
                   for mentioned in mentioned_songs):
                filtered_tracks.append(track)
        
        # If we found matches, return them; otherwise return first few tracks
        return filtered_tracks if filtered_tracks else tracks[:2]
    
    def _get_primary_analyzed_track(
        self, 
        tool_results: Dict[str, Any], 
        available_tracks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get the primary track that was analyzed by the tool chain."""
        # For sequential chains, we want the exact track that was passed between tools
        
        # Check if vibe_discovery was used and get the analyzed song
        if "vibe_discovery" in tool_results:
            vibe_result = tool_results["vibe_discovery"]
            if (hasattr(vibe_result, 'success') and vibe_result.success and 
                hasattr(vibe_result, 'data') and vibe_result.data):
                
                analyzed_song = None
                analyzed_artist = None
                
                # Get the song that was actually analyzed
                if 'song_analyzed' in vibe_result.data:
                    analyzed_query = vibe_result.data['song_analyzed']  # "Maroon by Kevin Abstract"
                    if ' by ' in analyzed_query:
                        analyzed_song, analyzed_artist = analyzed_query.split(' by ', 1)
                        analyzed_song = analyzed_song.strip()
                        analyzed_artist = analyzed_artist.strip()
                
                # If we have the analyzed song info, find the matching track
                if analyzed_song and analyzed_artist:
                    matching_track = self._find_matching_track(
                        available_tracks, analyzed_song, analyzed_artist
                    )
                    if matching_track:
                        return matching_track
        
        # Fallback: if no vibe_discovery or no match found, check search results
        search_tools = ["spotify_search", "track_search"]
        for search_tool in search_tools:
            if search_tool in tool_results:
                search_result = tool_results[search_tool]
                if (hasattr(search_result, 'success') and search_result.success and 
                    hasattr(search_result, 'data') and search_result.data):
                    
                    # Get first result from search
                    if 'results' in search_result.data and search_result.data['results']:
                        return search_result.data['results'][0]
                    elif 'tracks' in search_result.data and search_result.data['tracks']:
                        return search_result.data['tracks'][0]
        
        return None
    
    def _find_matching_track(
        self, 
        tracks: List[Dict[str, Any]], 
        target_song: str, 
        target_artist: str
    ) -> Optional[Dict[str, Any]]:
        """Find a track that matches the target song and artist with fuzzy matching."""
        import re
        
        def normalize_name(name: str) -> str:
            """Normalize name for comparison by removing special chars and lowercasing."""
            if not name:
                return ""
            # Remove common suffixes and prefixes
            name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove (Remastered), (Deluxe), etc.
            name = re.sub(r'\s*\[.*?\]\s*', '', name)  # Remove [Explicit], [Clean], etc.
            name = re.sub(r'\s*-\s*(remaster|remix|edit|version).*$', '', name, flags=re.IGNORECASE)
            return name.lower().strip()
        
        target_song_norm = normalize_name(target_song)
        target_artist_norm = normalize_name(target_artist)
        
        best_match = None
        best_score = 0
        
        for track in tracks:
            track_name = track.get('name', '')
            track_artist = track.get('artist', '')
            
            track_name_norm = normalize_name(track_name)
            track_artist_norm = normalize_name(track_artist)
            
            # Exact match gets highest priority
            if (track_name_norm == target_song_norm and 
                track_artist_norm == target_artist_norm):
                return track
            
            # Calculate similarity score
            score = 0
            
            # Song name matching (most important)
            if target_song_norm in track_name_norm or track_name_norm in target_song_norm:
                score += 10
            elif target_song_norm and track_name_norm:
                # Fuzzy matching using simple character overlap
                common_chars = set(target_song_norm) & set(track_name_norm)
                if common_chars:
                    score += len(common_chars) / max(len(target_song_norm), len(track_name_norm)) * 5
            
            # Artist name matching (also important)
            if target_artist_norm in track_artist_norm or track_artist_norm in target_artist_norm:
                score += 8
            elif target_artist_norm and track_artist_norm:
                # Fuzzy matching for artist
                common_chars = set(target_artist_norm) & set(track_artist_norm)
                if common_chars:
                    score += len(common_chars) / max(len(target_artist_norm), len(track_artist_norm)) * 4
            
            if score > best_score:
                best_score = score
                best_match = track
        
        # Only return match if score is reasonable (minimum threshold)
        return best_match if best_score >= 5 else None
    
    def _validate_track_consistency(
        self, 
        track: Dict[str, Any], 
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and ensure track metadata consistency across the tool chain."""
        validated_track = track.copy()
        
        # If we have vibe_discovery results, cross-validate the metadata
        if "vibe_discovery" in tool_results:
            vibe_result = tool_results["vibe_discovery"]
            if (hasattr(vibe_result, 'success') and vibe_result.success and 
                hasattr(vibe_result, 'data') and vibe_result.data):
                
                # Get the analyzed song info for consistency check
                if 'song_analyzed' in vibe_result.data:
                    analyzed_query = vibe_result.data['song_analyzed']
                    if ' by ' in analyzed_query:
                        expected_song, expected_artist = analyzed_query.split(' by ', 1)
                        expected_song = expected_song.strip()
                        expected_artist = expected_artist.strip()
                        
                        # Ensure our track matches what was analyzed
                        track_name = validated_track.get('name', '')
                        track_artist = validated_track.get('artist', '')
                        
                        # Log potential mismatches for debugging
                        if (track_name.lower() != expected_song.lower() or 
                            track_artist.lower() != expected_artist.lower()):
                            print(f"WARNING: Track mismatch detected!")
                            print(f"  Expected: '{expected_song}' by '{expected_artist}'")
                            print(f"  Got: '{track_name}' by '{track_artist}'")
                            
                            # Force correction if names are close but not exact
                            if (expected_song.lower() in track_name.lower() and 
                                expected_artist.lower() in track_artist.lower()):
                                print(f"  Auto-correcting track metadata")
                                validated_track['name'] = expected_song
                                validated_track['artist'] = expected_artist
        
        # Add validation metadata
        validated_track['_validation'] = {
            'validated_at': 'agent_core',
            'tool_chain': list(tool_results.keys()),
            'consistency_check': True
        }
        
        return validated_track
    
    # ===== ENHANCED MEMORY INTEGRATION =====
    
    async def process_query_with_memory(
        self, 
        user_input: str, 
        user_id: str,
        chat_id: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_callback=None
    ) -> AgentResponse:
        """
        Enhanced query processing with dual-store memory integration.
        
        Args:
            user_input: Natural language input from user
            user_id: User identifier
            chat_id: Chat/conversation identifier
            session_id: Session identifier (fallback to chat_id)
            stream_callback: Optional streaming callback
            
        Returns:
            AgentResponse with recommendations and explanations
        """
        import uuid
        
        # Generate IDs if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
        if not session_id:
            session_id = chat_id
        
        # 1. Store user turn in working memory
        user_turn = ChatTurn(
            message_type="user",
            content=user_input,
            metadata={"source": "chat_interface"},
            timestamp=datetime.now()
        )
        
        await memory_service.add_chat_turn(user_id, chat_id, user_turn)
        
        # 2. Assemble context from both memory stores
        assembled_context = await memory_service.assemble_context(
            user_id=user_id,
            chat_id=chat_id,
            query=user_input,
            max_recent_turns=20,
            max_memories=10
        )
        
        # 3. Build enhanced context for agent
        enhanced_context = AgentContext(
            user_id=user_id,
            session_id=session_id,
            chat_id=chat_id,
            conversation_history=assembled_context.get("recent_turns", []),
            user_preferences=await self._extract_preferences_from_memories(
                assembled_context.get("relevant_memories", []),
                user_id
            ),
            current_mood=assembled_context.get("scratch_state", {}).get("current_mood"),
            assembled_context=assembled_context,
            turn_count=len(assembled_context.get("recent_turns", []))
        )
        
        # 4. Process query using original logic with enhanced context
        agent_response = await self.process_query(
            user_input, 
            enhanced_context, 
            stream_callback
        )
        
        # 4.5. Apply deduplication and diversity to track recommendations
        if agent_response.tracks:
            processed_tracks = await process_and_diversify_recommendations(
                user_id=user_id,
                raw_recommendations=agent_response.tracks,
                diversity_lambda=0.3,
                max_results=min(10, len(agent_response.tracks))
            )
            agent_response.tracks = processed_tracks
        
        # 5. Store agent response in working memory
        agent_turn = ChatTurn(
            message_type="agent",
            content=agent_response.content,
            metadata={
                "tracks": agent_response.tracks,
                "tools_used": agent_response.tools_used,
                "confidence": agent_response.confidence
            },
            timestamp=datetime.now()
        )
        
        await memory_service.add_chat_turn(user_id, chat_id, agent_turn)
        
        # 6. Update scratch state with any context updates
        if agent_response.context_updates:
            await memory_service.update_scratch_state(
                user_id, chat_id, agent_response.context_updates
            )
        
        # 7. Update recent topics based on conversation
        await self._update_topics_from_response(user_id, user_input, agent_response)
        
        # 8. Track turns and trigger background processing if needed
        await self._track_and_trigger_processing(user_id, chat_id)
        
        # 9. Enhance response with memory metadata
        agent_response.metadata.update({
            "chat_id": chat_id,
            "context_stats": assembled_context.get("context_stats", {}),
            "memory_insights": await self._generate_memory_insights(assembled_context)
        })
        
        return agent_response
    
    async def _extract_preferences_from_memories(
        self, 
        memories: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Extract user preferences from relevant memories and database."""
        # Get comprehensive taste profile from database
        taste_profile = await memory_service.get_user_taste_profile(user_id)
        
        preferences = {
            "genres": taste_profile.get("genres", []),
            "moods": taste_profile.get("moods", []),
            "artist_affinities": taste_profile.get("artist_affinities", {}),
            "depth_weight": taste_profile.get("depth_weight", 0.5),
            "novelty_weight": taste_profile.get("novelty_weight", 0.5),
            "dislikes": [],
            "discovery_preferences": {},
            "recent_tracks": taste_profile.get("recent_tracks", []),
            "blocked_tracks": taste_profile.get("blocked_tracks", [])
        }
        
        # Enhance with memory-based preferences
        for memory in memories:
            if memory["kind"] == "preference":
                content = memory["content"].lower()
                
                # Simple preference extraction
                if "like" in content or "love" in content:
                    if any(genre in content for genre in ["pop", "rock", "jazz", "electronic"]):
                        # Extract genre mentions
                        for genre in ["pop", "rock", "jazz", "electronic", "hip hop", "classical"]:
                            if genre in content and genre not in preferences["genres"]:
                                preferences["genres"].append(genre)
                
                elif "dislike" in content or "hate" in content:
                    preferences["dislikes"].append(content[:100])
            
            elif memory["kind"] == "fact":
                # Extract factual preferences
                content = memory["content"]
                preferences["discovery_preferences"]["last_updated"] = memory.get("created_at")
        
        return preferences
    
    async def _update_topics_from_response(
        self, 
        user_id: str, 
        user_input: str, 
        response: AgentResponse
    ):
        """Update recent topics based on user input and agent response."""
        topics_to_update = []
        
        # Extract topics from user input
        user_topics = await self._extract_topics_from_text(user_input)
        topics_to_update.extend(user_topics)
        
        # Extract topics from recommended tracks
        for track in response.tracks:
            if "genre" in track:
                topics_to_update.append(track["genre"])
            if "artist" in track:
                topics_to_update.append(f"artist:{track['artist']}")
        
        # Update topics in Redis
        current_score = datetime.now().timestamp()
        for topic in topics_to_update:
            await memory_service.update_recent_topics(user_id, topic, current_score)
    
    async def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract music-related topics from text."""
        topics = []
        text_lower = text.lower()
        
        # Music genre keywords
        genres = [
            "pop", "rock", "jazz", "classical", "hip hop", "r&b", "soul", "funk",
            "electronic", "ambient", "house", "techno", "drum and bass",
            "indie", "alternative", "punk", "metal", "country", "folk",
            "reggae", "blues", "disco", "new wave", "shoegaze", "dream pop"
        ]
        
        for genre in genres:
            if genre in text_lower:
                topics.append(genre)
        
        # Mood keywords
        moods = ["chill", "upbeat", "sad", "happy", "energetic", "relaxing", "dark", "bright"]
        for mood in moods:
            if mood in text_lower:
                topics.append(f"mood:{mood}")
        
        return topics
    
    async def _track_and_trigger_processing(self, user_id: str, chat_id: str):
        """Track conversation turns and trigger background processing when needed."""
        # Track turn count for this chat
        chat_key = f"{user_id}:{chat_id}"
        self.turn_counter[chat_key] = self.turn_counter.get(chat_key, 0) + 1
        
        # Trigger processing every 10 turns
        if self.turn_counter[chat_key] % 10 == 0:
            await trigger_conversation_processing(user_id, chat_id)
        
        # Trigger preference learning every 25 turns
        if self.turn_counter[chat_key] % 25 == 0:
            from services.memory_processor import memory_processor
            await memory_processor.queue_preference_learning(user_id)
    
    async def _generate_memory_insights(
        self, 
        assembled_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights about the user's memory and conversation patterns."""
        insights = {}
        
        # Context statistics
        stats = assembled_context.get("context_stats", {})
        insights["context_quality"] = {
            "recent_turns": stats.get("recent_turn_count", 0),
            "relevant_memories": stats.get("memory_count", 0),
            "avg_memory_relevance": round(stats.get("avg_memory_similarity", 0), 3)
        }
        
        # Memory patterns
        memories = assembled_context.get("relevant_memories", [])
        if memories:
            memory_kinds = {}
            for memory in memories:
                kind = memory["kind"]
                memory_kinds[kind] = memory_kinds.get(kind, 0) + 1
            
            insights["memory_patterns"] = memory_kinds
        
        # Topic diversity
        topics = assembled_context.get("recent_topics", [])
        insights["topic_diversity"] = {
            "total_topics": len(topics),
            "top_topics": topics[:5]
        }
        
        return insights
    
    async def clear_chat_memory(self, user_id: str, chat_id: str):
        """Clear working memory for a specific chat."""
        await memory_service.clear_session_memory(user_id, chat_id)
        
        # Reset turn counter
        chat_key = f"{user_id}:{chat_id}"
        if chat_key in self.turn_counter:
            del self.turn_counter[chat_key]
    
    async def get_chat_summary(self, user_id: str, chat_id: str) -> Optional[str]:
        """Get a summary of the chat conversation."""
        try:
            # Get memories related to this chat
            memories = await memory_service.retrieve_memories(
                query="conversation summary",
                user_id=user_id,
                chat_id=chat_id,
                match_count=1
            )
            
            # Find the most recent summary
            summaries = [m for m in memories if m.kind == "summary"]
            if summaries:
                return summaries[0].content
            
            # If no summary exists, create one from recent turns
            recent_turns = await memory_service.get_recent_turns(user_id, chat_id, limit=20)
            if len(recent_turns) >= 5:
                summary = await memory_service.create_conversation_summary(
                    user_id, chat_id, window_size=len(recent_turns)
                )
                return summary.content if summary else None
            
        except Exception as e:
            print(f"Error getting chat summary: {e}")
        
        return None
    
    async def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory statistics for a user."""
        try:
            # Get recent topics
            topics = await memory_service.get_recent_topics(user_id)
            
            # Get turn counts for active chats
            active_chats = {}
            for chat_key, count in self.turn_counter.items():
                if chat_key.startswith(f"{user_id}:"):
                    chat_id = chat_key.split(":", 1)[1]
                    active_chats[chat_id] = count
            
            return {
                "recent_topics": topics[:10],
                "active_chats": active_chats,
                "total_active_chats": len(active_chats),
                "total_turns_tracked": sum(active_chats.values())
            }
            
        except Exception as e:
            print(f"Error getting user memory stats: {e}")
            return {}
    
    async def get_user_recommendation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user recommendation statistics and patterns."""
        try:
            from datetime import timedelta
            
            # Get recent recommendation events
            recent_recs = await memory_service.get_recent_recommendations(
                user_id, days=30, limit=100
            )
            
            # Get user preferences
            prefs = await memory_service.get_cached_user_preferences(user_id)
            
            # Calculate stats
            stats = {
                "total_recommendations": len(recent_recs),
                "recommendations_last_week": len([
                    r for r in recent_recs 
                    if r.created_at and r.created_at > datetime.now() - timedelta(days=7)
                ]),
                "unique_tracks": len(set(r.item_id for r in recent_recs)),
                "recommendation_reasons": {},
                "user_preferences": {
                    "depth_weight": prefs.depth_weight if prefs else 0.5,
                    "novelty_weight": prefs.novelty_weight if prefs else 0.5,
                    "top_genres": prefs.top_genres if prefs else [],
                    "top_moods": prefs.top_moods if prefs else []
                }
            }
            
            # Count recommendation reasons
            for rec in recent_recs:
                reason = rec.reason or "unknown"
                stats["recommendation_reasons"][reason] = stats["recommendation_reasons"].get(reason, 0) + 1
            
            return stats
            
        except Exception as e:
            print(f"Error getting recommendation stats: {e}")
            return {}
    
    async def update_user_preference_weights(
        self,
        user_id: str,
        depth_weight: Optional[float] = None,
        novelty_weight: Optional[float] = None
    ) -> bool:
        """Update user preference weights based on feedback."""
        try:
            success = await memory_service.update_user_preferences(
                user_id=user_id,
                depth_weight=depth_weight,
                novelty_weight=novelty_weight
            )
            
            if success:
                # Store preference change as a memory
                from services.memory_service import MemoryEntry
                change_memory = MemoryEntry(
                    user_id=user_id,
                    kind="preference",
                    content=f"Updated preference weights - depth: {depth_weight}, novelty: {novelty_weight}",
                    importance=2
                )
                await memory_service.store_memory(change_memory)
            
            return success
            
        except Exception as e:
            print(f"Error updating preference weights: {e}")
            return False


# Convenience function to create enhanced agent (backward compatibility)
def create_enhanced_agent(
    memory_store: MemoryStore,
    embedding_engine: EmbeddingEngine,
    llm_service: LLMService,
    tool_registry: ToolRegistry
) -> TimbreAgent:
    """Create an enhanced agent with dual-store memory."""
    return TimbreAgent(
        memory_store=memory_store,
        embedding_engine=embedding_engine,
        llm_service=llm_service,
        tool_registry=tool_registry
    )