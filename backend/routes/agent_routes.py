from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime

from agent.core import TimbreAgent, AgentContext
from agent.embedding.engine import EmbeddingEngine
from agent.memory.store import MemoryStore
from agent.nlp.processor import NaturalLanguageProcessor
from agent.tools.registry import ToolRegistry
from utils.metrics import metrics


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    tracks: List[Dict[str, Any]]
    explanations: List[str]
    confidence: float
    session_id: str
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str
    feedback_type: str  # "like", "dislike", "skip", "play_full", etc.
    feedback_data: Optional[Dict[str, Any]] = None


class PlaylistAnalysisRequest(BaseModel):
    playlist_url: str
    user_id: str


# Initialize router
router = APIRouter()

# Initialize agent components (these would be dependency injected in production)
memory_store = MemoryStore()
embedding_engine = EmbeddingEngine()
nlp_processor = NaturalLanguageProcessor()
tool_registry = ToolRegistry()

# Initialize main agent
agent = TimbreAgent(
    memory_store=memory_store,
    embedding_engine=embedding_engine,
    nlp_processor=nlp_processor,
    tool_registry=tool_registry
)


async def get_agent_context(user_id: str, session_id: Optional[str] = None) -> AgentContext:
    """Create agent context from request data."""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get user preferences and conversation history
    user_preferences = await memory_store.get_user_profile(user_id)
    conversation_history = await memory_store.get_user_interactions(user_id, limit=10)
    
    return AgentContext(
        user_id=user_id,
        session_id=session_id,
        conversation_history=conversation_history,
        user_preferences=user_preferences,
        timestamp=datetime.now()
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Main chat endpoint for conversing with the Timbre agent.
    
    Processes natural language input and returns music recommendations
    with explanations.
    """
    try:
        metrics.record_request()
        
        # Create agent context
        context = await get_agent_context(request.user_id, request.session_id)
        
        # Add any additional context from request
        if request.context:
            context.current_mood = request.context.get('mood')
        
        # Process the query with the agent
        response = await agent.process_query(request.message, context)
        
        # Update conversation context in background
        background_tasks.add_task(
            memory_store.update_conversation_context,
            request.user_id,
            context.session_id,
            {
                "last_message": request.message,
                "last_response": response.content,
                "tracks_recommended": len(response.tracks)
            }
        )
        
        return ChatResponse(
            response=response.content,
            tracks=response.tracks,
            explanations=response.explanations,
            confidence=response.confidence,
            session_id=context.session_id,
            metadata={
                **response.metadata,
                "tools_used": response.tools_used,
                "context_updates": response.context_updates
            }
        )
        
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    
    Returns a streaming response with partial results as they're generated.
    """
    try:
        metrics.record_request()
        
        async def generate_stream():
            context = await get_agent_context(request.user_id, request.session_id)
            
            # Send initial acknowledgment
            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...'})}\n\n"
            
            # Process query and yield intermediate results
            try:
                response = await agent.process_query(request.message, context)
                
                # Send tool execution updates
                for tool in response.tools_used:
                    yield f"data: {json.dumps({'type': 'tool', 'tool': tool})}\n\n"
                
                # Send final response
                final_data = {
                    'type': 'complete',
                    'response': response.content,
                    'tracks': response.tracks,
                    'explanations': response.explanations,
                    'confidence': response.confidence,
                    'session_id': context.session_id
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    'type': 'error',
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a track recommendation.
    
    This helps improve future recommendations by learning user preferences.
    """
    try:
        # Store feedback in memory store
        await memory_store.add_feedback(
            request.user_id,
            request.track_id,
            request.feedback_type,
            request.feedback_data or {}
        )
        
        # Update user profile based on feedback
        if request.feedback_type in ["like", "dislike"]:
            # Get track metadata and update embedding
            # This would trigger the memory embedder tool
            pass
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@router.post("/analyze-playlist")
async def analyze_playlist(request: PlaylistAnalysisRequest):
    """
    Analyze a Spotify playlist and provide recommendations based on it.
    """
    try:
        metrics.record_request()
        
        context = await get_agent_context(request.user_id)
        
        # Create a specialized query for playlist analysis
        query = f"analyze this playlist: {request.playlist_url}"
        
        response = await agent.process_query(query, context)
        
        return {
            "analysis": response.content,
            "tracks": response.tracks,
            "explanations": response.explanations,
            "metadata": response.metadata
        }
        
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Playlist analysis failed: {str(e)}")


@router.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user's music profile and preferences."""
    try:
        profile = await memory_store.get_user_profile(user_id)
        stats = await memory_store.get_user_stats(user_id)
        
        return {
            "profile": profile,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")


@router.get("/user/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 50):
    """Get user's interaction history."""
    try:
        interactions = await memory_store.get_user_interactions(user_id, limit=limit)
        listening_history = await memory_store.get_user_listening_history(user_id, limit=limit)
        
        return {
            "interactions": interactions,
            "listening_history": listening_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@router.post("/user/{user_id}/listening-event")
async def add_listening_event(
    user_id: str,
    track_data: Dict[str, Any],
    event_type: str = "play"
):
    """Add a listening event to user's history."""
    try:
        await memory_store.add_listening_event(user_id, track_data, event_type)
        return {"status": "success", "message": "Listening event recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Listening event failed: {str(e)}")


@router.delete("/user/{user_id}/session/{session_id}")
async def clear_session(user_id: str, session_id: str):
    """Clear user's session data."""
    try:
        await memory_store.clear_user_session(user_id, session_id)
        return {"status": "success", "message": "Session cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session clearing failed: {str(e)}")


@router.get("/tools")
async def list_available_tools():
    """List all available agent tools and their descriptions."""
    try:
        tools = tool_registry.list_tools()
        return {"tools": tools}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool listing failed: {str(e)}")


@router.get("/health")
async def agent_health_check():
    """Health check for agent system."""
    try:
        # Test basic agent functionality
        test_context = AgentContext(
            user_id="health_check",
            session_id="health_check",
            conversation_history=[],
            user_preferences={}
        )
        
        # Simple test query
        test_response = await agent.process_query("test", test_context)
        
        return {
            "status": "healthy",
            "agent_responsive": True,
            "tools_available": len(tool_registry.list_tools()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }