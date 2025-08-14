"""
Agent Routes with Dual-Store Memory Integration

API routes that use the Redis + PostgreSQL memory architecture.
Consolidated from enhanced_agent_routes.py to replace the legacy version.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
from dataclasses import asdict, is_dataclass

from agent.core import TimbreAgent, create_enhanced_agent
from agent.embedding.engine import EmbeddingEngine
from agent.memory.store import MemoryStore
from agent.llm.service import LLMService
from agent.tools.registry import ToolRegistry
from services.memory_service import memory_service
from services.memory_processor import memory_processor, start_memory_processing
from utils.metrics import metrics


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    tracks: List[Dict[str, Any]]
    explanations: List[str]
    confidence: float
    session_id: str
    chat_id: str
    metadata: Dict[str, Any]


class StreamChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str
    feedback_type: str  # "like", "dislike", "skip", "play_full", etc.
    feedback_data: Optional[Dict[str, Any]] = None


class PreferencesRequest(BaseModel):
    user_id: str
    top_genres: Optional[List[str]] = None
    top_moods: Optional[List[str]] = None
    artist_affinities: Optional[Dict[str, float]] = None
    depth_weight: Optional[float] = None
    novelty_weight: Optional[float] = None


class PreferencesResponse(BaseModel):
    user_id: str
    preferences: Dict[str, Any]


class RecommendationStatsResponse(BaseModel):
    user_id: str
    stats: Dict[str, Any]


class TasteProfileResponse(BaseModel):
    user_id: str
    taste_profile: Dict[str, Any]


class TitleGenerationRequest(BaseModel):
    conversation_text: str
    user_id: str


class TitleGenerationResponse(BaseModel):
    title: str


class MemoryStatsResponse(BaseModel):
    user_id: str
    stats: Dict[str, Any]


class ChatSummaryRequest(BaseModel):
    user_id: str
    chat_id: str


class ChatSummaryResponse(BaseModel):
    summary: Optional[str]
    chat_id: str


# Initialize router
router = APIRouter()


def safe_json_dumps(obj):
    """Safely serialize objects to JSON, handling dataclasses and complex types."""
    def default_serializer(obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)
    
    return json.dumps(obj, default=default_serializer)


# Initialize agent components
from config.settings import settings

memory_store = MemoryStore()  # Legacy store for compatibility
embedding_engine = EmbeddingEngine()
llm_service = LLMService(api_key=settings.openai_api_key)
tool_registry = ToolRegistry()

# Initialize agent with memory integration
agent = create_enhanced_agent(
    memory_store=memory_store,
    embedding_engine=embedding_engine,
    llm_service=llm_service,
    tool_registry=tool_registry
)


@router.on_event("startup")
async def startup_event():
    """Start memory processing service."""
    await start_memory_processing()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Chat endpoint with dual-store memory integration.
    
    Processes natural language input using Redis working memory and 
    PostgreSQL long-term memory for improved context and personalization.
    """
    try:
        metrics.record_request()
        
        # Generate session and chat IDs if not provided
        session_id = request.session_id or str(uuid.uuid4())
        chat_id = request.chat_id or str(uuid.uuid4())
        
        # Process query with enhanced memory system
        response = await agent.process_query_with_memory(
            user_input=request.message,
            user_id=request.user_id,
            chat_id=chat_id,
            session_id=session_id
        )
        
        return ChatResponse(
            response=response.content,
            tracks=response.tracks,
            explanations=response.explanations,
            confidence=response.confidence,
            session_id=session_id,
            chat_id=chat_id,
            metadata=response.metadata
        )
        
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_with_agent(request: StreamChatRequest):
    """
    Streaming chat endpoint with enhanced memory integration.
    
    Provides real-time streaming responses while building context
    in both Redis and PostgreSQL memory stores.
    """
    try:
        metrics.record_request()
        
        # Generate session and chat IDs if not provided
        session_id = request.session_id or str(uuid.uuid4())
        chat_id = request.chat_id or str(uuid.uuid4())
        
        async def generate_stream():
            try:
                chunks = []
                
                async def stream_callback(chunk):
                    chunks.append(chunk)
                    yield f"data: {safe_json_dumps(chunk)}\n\n"
                
                # Process query with streaming
                response = await agent.process_query_with_memory(
                    user_input=request.message,
                    user_id=request.user_id,
                    chat_id=chat_id,
                    session_id=session_id,
                    stream_callback=stream_callback
                )
                
                # Send final response
                final_chunk = {
                    'type': 'complete',
                    'response': response.content,
                    'tracks': response.tracks,
                    'explanations': response.explanations,
                    'confidence': response.confidence,
                    'session_id': session_id,
                    'chat_id': chat_id,
                    'metadata': response.metadata
                }
                
                yield f"data: {safe_json_dumps(final_chunk)}\n\n"
                
            except Exception as e:
                error_chunk = {
                    'type': 'error',
                    'error': str(e)
                }
                yield f"data: {safe_json_dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit user feedback for a track with memory integration.
    
    Stores feedback in both legacy system and new memory system
    for comprehensive tracking.
    """
    try:
        # Store in legacy memory system
        await memory_store.add_feedback(
            request.user_id,
            request.track_id,
            request.feedback_type,
            request.feedback_data or {}
        )
        
        # Store feedback as a memory entry in new system
        feedback_content = f"User {request.feedback_type} track {request.track_id}"
        if request.feedback_data:
            feedback_content += f" with data: {json.dumps(request.feedback_data)}"
        
        from services.memory_service import MemoryEntry
        feedback_memory = MemoryEntry(
            user_id=request.user_id,
            kind="preference",
            content=feedback_content,
            importance=3  # Medium importance for feedback
        )
        
        background_tasks.add_task(
            memory_service.store_memory,
            feedback_memory
        )
        
        # Record as recommendation event if it's a positive interaction
        if request.feedback_type in ["like", "play_full", "save"]:
            background_tasks.add_task(
                memory_service.record_recommendation,
                request.user_id,
                request.track_id,
                "track",
                f"user_{request.feedback_type}",
                1.0
            )
        
        return {
            "status": "success", 
            "message": "Feedback recorded",
            "feedback_type": request.feedback_type,
            "track_id": request.track_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@router.post("/generate-title", response_model=TitleGenerationResponse)
async def generate_chat_title(request: TitleGenerationRequest):
    """
    Generate a chat title from conversation text.
    
    Uses the LLM service to create concise, descriptive titles
    for chat conversations.
    """
    try:
        # Use the LLM service to generate a title
        title = await llm_service.generate_chat_title(
            request.conversation_text,
            request.user_id
        )
        
        return TitleGenerationResponse(
            title=title.get("title", "New Chat")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation failed: {str(e)}")


@router.get("/preferences/{user_id}", response_model=PreferencesResponse)
async def get_user_preferences(user_id: str):
    """
    Get user preferences and taste profile.
    
    Returns comprehensive preference data including genres,
    moods, artist affinities, and preference weights.
    """
    try:
        preferences = await memory_service.get_cached_user_preferences(user_id)
        taste_profile = await memory_service.get_user_taste_profile(user_id)
        
        combined_preferences = {
            "top_genres": preferences.top_genres if preferences else taste_profile.get("genres", []),
            "top_moods": preferences.top_moods if preferences else taste_profile.get("moods", []),
            "artist_affinities": preferences.artist_affinities if preferences else taste_profile.get("artist_affinities", {}),
            "depth_weight": preferences.depth_weight if preferences else taste_profile.get("depth_weight", 0.5),
            "novelty_weight": preferences.novelty_weight if preferences else taste_profile.get("novelty_weight", 0.5),
            "recent_tracks": taste_profile.get("recent_tracks", []),
            "blocked_tracks": taste_profile.get("blocked_tracks", [])
        }
        
        return PreferencesResponse(
            user_id=user_id,
            preferences=combined_preferences
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.post("/preferences", response_model=PreferencesResponse)
async def update_user_preferences(
    request: PreferencesRequest,
    background_tasks: BackgroundTasks
):
    """
    Update user preferences.
    
    Allows updating genres, moods, artist affinities,
    and preference weights for personalized recommendations.
    """
    try:
        success = await memory_service.update_user_preferences(
            user_id=request.user_id,
            top_genres=request.top_genres,
            top_moods=request.top_moods,
            artist_affinities=request.artist_affinities,
            depth_weight=request.depth_weight,
            novelty_weight=request.novelty_weight
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
        
        # Store preference update as memory
        update_content = f"User updated preferences: "
        if request.top_genres:
            update_content += f"genres={request.top_genres} "
        if request.top_moods:
            update_content += f"moods={request.top_moods} "
        if request.depth_weight is not None:
            update_content += f"depth_weight={request.depth_weight} "
        if request.novelty_weight is not None:
            update_content += f"novelty_weight={request.novelty_weight}"
        
        from services.memory_service import MemoryEntry
        preference_memory = MemoryEntry(
            user_id=request.user_id,
            kind="preference",
            content=update_content.strip(),
            importance=4  # High importance for explicit preference updates
        )
        
        background_tasks.add_task(
            memory_service.store_memory,
            preference_memory
        )
        
        # Get updated preferences to return
        updated_preferences = await memory_service.get_user_taste_profile(request.user_id)
        
        return PreferencesResponse(
            user_id=request.user_id,
            preferences=updated_preferences
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.get("/memory/stats/{user_id}", response_model=MemoryStatsResponse)
async def get_user_memory_stats(user_id: str):
    """
    Get comprehensive memory statistics for a user.
    
    Returns information about recent topics, active chats,
    and memory patterns from both storage systems.
    """
    try:
        stats = await agent.get_user_memory_stats(user_id)
        
        return MemoryStatsResponse(
            user_id=user_id,
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.get("/recommendations/stats/{user_id}", response_model=RecommendationStatsResponse)
async def get_recommendation_stats(user_id: str):
    """
    Get user recommendation statistics.
    
    Returns analytics about recent recommendations,
    preference patterns, and system performance.
    """
    try:
        stats = await agent.get_user_recommendation_stats(user_id)
        
        return RecommendationStatsResponse(
            user_id=user_id,
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation stats: {str(e)}")


@router.get("/taste-profile/{user_id}", response_model=TasteProfileResponse)
async def get_taste_profile(user_id: str):
    """
    Get comprehensive user taste profile.
    
    Returns detailed taste analysis including preferences,
    recent activity, and recommendation constraints.
    """
    try:
        taste_profile = await memory_service.get_user_taste_profile(user_id)
        
        return TasteProfileResponse(
            user_id=user_id,
            taste_profile=taste_profile
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get taste profile: {str(e)}")


@router.post("/chat/summary", response_model=ChatSummaryResponse)
async def get_chat_summary(request: ChatSummaryRequest):
    """
    Get a summary of a chat conversation.
    
    Returns an AI-generated summary of the conversation
    from the long-term memory system.
    """
    try:
        summary = await agent.get_chat_summary(
            request.user_id,
            request.chat_id
        )
        
        return ChatSummaryResponse(
            summary=summary,
            chat_id=request.chat_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat summary: {str(e)}")


@router.post("/recommendations/process")
async def process_recommendations(
    user_id: str,
    recommendations: List[Dict[str, Any]],
    diversity_lambda: float = 0.3,
    max_results: int = 10
):
    """
    Process raw recommendations with deduplication and diversity.
    
    Applies the complete recommendation processing pipeline
    including seen penalty, artist diversity, and MMR.
    """
    try:
        from services.memory_processor import process_and_diversify_recommendations
        
        processed_recs = await process_and_diversify_recommendations(
            user_id=user_id,
            raw_recommendations=recommendations,
            diversity_lambda=diversity_lambda,
            max_results=max_results
        )
        
        return {
            "processed_recommendations": processed_recs,
            "original_count": len(recommendations),
            "processed_count": len(processed_recs),
            "diversity_lambda": diversity_lambda
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process recommendations: {str(e)}")


@router.post("/preferences/weights")
async def update_preference_weights(
    user_id: str,
    depth_weight: Optional[float] = None,
    novelty_weight: Optional[float] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Update user preference weights.
    
    Allows fine-tuning of depth vs novelty preferences
    for personalized recommendation balancing.
    """
    try:
        # Validate weights
        if depth_weight is not None and not (0.0 <= depth_weight <= 1.0):
            raise HTTPException(status_code=400, detail="depth_weight must be between 0.0 and 1.0")
        if novelty_weight is not None and not (0.0 <= novelty_weight <= 1.0):
            raise HTTPException(status_code=400, detail="novelty_weight must be between 0.0 and 1.0")
        
        success = await agent.update_user_preference_weights(
            user_id=user_id,
            depth_weight=depth_weight,
            novelty_weight=novelty_weight
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update preference weights")
        
        return {
            "status": "success",
            "message": "Preference weights updated",
            "depth_weight": depth_weight,
            "novelty_weight": novelty_weight
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")


@router.delete("/chat/{chat_id}/memory")
async def clear_chat_memory(
    chat_id: str,
    user_id: str,
    background_tasks: BackgroundTasks
):
    """
    Clear working memory for a specific chat.
    
    Removes Redis working memory while preserving 
    long-term memories in PostgreSQL.
    """
    try:
        await agent.clear_chat_memory(user_id, chat_id)
        
        return {"status": "success", "message": f"Memory cleared for chat {chat_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@router.get("/memory/context/{user_id}/{chat_id}")
async def get_assembled_context(
    user_id: str,
    chat_id: str,
    query: str = "recent conversation context"
):
    """
    Get assembled context from both memory stores.
    
    Returns the complete context that would be used
    for processing a query, useful for debugging.
    """
    try:
        context = await memory_service.assemble_context(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            max_recent_turns=20,
            max_memories=10
        )
        
        return context
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")


@router.post("/memory/process/{user_id}/{chat_id}")
async def trigger_memory_processing(
    user_id: str,
    chat_id: str,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger background memory processing.
    
    Useful for testing or forcing summarization and
    fact extraction for a conversation.
    """
    try:
        from services.memory_processor import trigger_conversation_processing
        
        background_tasks.add_task(
            trigger_conversation_processing,
            user_id,
            chat_id
        )
        
        return {"status": "success", "message": "Memory processing triggered"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger processing: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for the agent service."""
    try:
        # Test memory service connectivity
        test_topics = await memory_service.get_recent_topics("health_check_user")
        
        return {
            "status": "healthy",
            "memory_service": "connected",
            "redis_cache": "connected" if test_topics is not None else "disconnected",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }