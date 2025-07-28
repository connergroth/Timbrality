from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio

from agent.tools.registry import ToolRegistry
from agent.core import AgentContext
from datetime import datetime

router = APIRouter()
tool_registry = ToolRegistry()

class PlaylistRequest(BaseModel):
    action: str
    user_id: Optional[str] = "default_user"
    playlist_id: Optional[str] = None
    playlist_name: Optional[str] = ""
    description: Optional[str] = ""
    public: Optional[bool] = False
    track_ids: Optional[List[str]] = []
    query: Optional[str] = ""
    limit: Optional[int] = 20

class PlaylistResponse(BaseModel):
    success: bool
    action: str
    results: Any
    count: int
    message: Optional[str] = None

@router.post("/spotify-playlist", response_model=PlaylistResponse)
async def handle_playlist_operation(request: PlaylistRequest):
    """Handle Spotify playlist operations through the agent tool."""
    try:
        # Get the Spotify playlist tool
        playlist_tool = tool_registry.get_tool("spotify_playlist")
        if not playlist_tool:
            raise HTTPException(status_code=500, detail="Spotify playlist tool not available")
        
        # Create context with user ID
        context = AgentContext(
            user_id=request.user_id,
            session_id="playlist_session",
            conversation_history=[],
            user_preferences={},
            timestamp=datetime.now()
        )
        
        # Create intent from request
        intent_data = {
            "action": request.action,
            "playlist_id": request.playlist_id,
            "playlist_name": request.playlist_name,
            "description": request.description,
            "public": request.public,
            "track_ids": request.track_ids or [],
            "query": request.query,
            "limit": request.limit
        }
        
        # Execute the tool
        result = await playlist_tool.execute(intent_data, context)
        
        if result.success:
            return PlaylistResponse(
                success=True,
                action=request.action,
                results=result.data.get("results", []),
                count=result.data.get("count", 0),
                message="Operation completed successfully"
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=result.error or "Playlist operation failed"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-playlists/{user_id}")
async def get_user_playlists(user_id: str, limit: int = 20):
    """Get user's Spotify playlists."""
    try:
        playlist_tool = tool_registry.get_tool("spotify_playlist")
        if not playlist_tool:
            raise HTTPException(status_code=500, detail="Spotify playlist tool not available")
        
        # Create context
        context = AgentContext(
            user_id=user_id,
            session_id="playlist_fetch",
            conversation_history=[],
            user_preferences={},
            timestamp=datetime.now()
        )
        
        # Execute list playlists action
        result = await playlist_tool.execute({
            "action": "list_playlists",
            "limit": limit
        }, context)
        
        if result.success:
            return JSONResponse(content={
                "playlists": result.data.get("results", []),
                "count": result.data.get("count", 0)
            })
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/playlist/{playlist_id}/add-track")
async def add_track_to_playlist(
    playlist_id: str, 
    track_id: str,
    user_id: str = "default_user"
):
    """Add a single track to a playlist."""
    try:
        playlist_tool = tool_registry.get_tool("spotify_playlist")
        if not playlist_tool:
            raise HTTPException(status_code=500, detail="Spotify playlist tool not available")
        
        context = AgentContext(
            user_id=user_id,
            session_id="add_track",
            conversation_history=[],
            user_preferences={},
            timestamp=datetime.now()
        )
        
        result = await playlist_tool.execute({
            "action": "add_tracks_to_playlist",
            "playlist_id": playlist_id,
            "track_ids": [track_id]
        }, context)
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "message": f"Track added to playlist successfully",
                "result": result.data.get("results", {})
            })
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-playlist")
async def create_playlist_with_track(
    name: str,
    track_id: str,
    description: str = "",
    public: bool = False,
    user_id: str = "default_user"
):
    """Create a new playlist with an initial track."""
    try:
        playlist_tool = tool_registry.get_tool("spotify_playlist")
        if not playlist_tool:
            raise HTTPException(status_code=500, detail="Spotify playlist tool not available")
        
        context = AgentContext(
            user_id=user_id,
            session_id="create_playlist",
            conversation_history=[],
            user_preferences={},
            timestamp=datetime.now()
        )
        
        result = await playlist_tool.execute({
            "action": "create_playlist",
            "playlist_name": name,
            "description": description,
            "public": public,
            "track_ids": [track_id]
        }, context)
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "message": f"Playlist '{name}' created successfully",
                "playlist": result.data.get("results", {})
            })
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))