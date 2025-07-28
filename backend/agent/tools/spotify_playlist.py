from typing import Dict, Any, Optional, List
from .base import BaseTool, ToolResult
from services.spotify_service import SpotifyService


class SpotifyPlaylistTool(BaseTool):
    """Tool for accessing user's Spotify playlists and playlist tracks."""
    
    def __init__(self):
        super().__init__(
            name="spotify_playlist",
            description="Access user's Spotify playlists and retrieve playlist tracks with metadata including cover art."
        )
        self.spotify_service = SpotifyService(user_auth=True)  # Requires user authentication
    
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute playlist operations based on intent parameters."""
        try:
            # Handle different intent formats
            if hasattr(intent_result, 'raw_text'):
                # Mock intent from agent core - parse action from text
                action = "list_playlists"  # Default action
                playlist_id = None
                track_ids = []
                playlist_name = ""
                query = intent_result.raw_text
            elif isinstance(intent_result, dict):
                action = intent_result.get("action", "list_playlists").lower()
                playlist_id = intent_result.get("playlist_id")
                track_ids = intent_result.get("track_ids", [])
                playlist_name = intent_result.get("playlist_name", "")
                query = intent_result.get("query", "")
            else:
                action = "list_playlists"
                playlist_id = None
                track_ids = []
                playlist_name = ""
                query = str(intent_result)
            
            if action == "list_playlists":
                results = await self._get_user_playlists(
                    limit=intent_result.get("limit", 20) if isinstance(intent_result, dict) else 20
                )
            elif action == "get_playlist_tracks":
                if not playlist_id:
                    return self._create_error_result("playlist_id is required for get_playlist_tracks action")
                
                results = await self._get_playlist_tracks(
                    playlist_id=playlist_id,
                    limit=intent_result.get("limit", 50) if isinstance(intent_result, dict) else 50
                )
            elif action == "search_user_playlists":
                if not query:
                    return self._create_error_result("query is required for search_user_playlists action")
                
                results = await self._search_user_playlists(query)
            elif action == "create_playlist":
                if not playlist_name:
                    return self._create_error_result("playlist_name is required for create_playlist action")
                
                results = await self._create_playlist(
                    name=playlist_name,
                    description=intent_result.get("description", "") if isinstance(intent_result, dict) else "",
                    public=intent_result.get("public", False) if isinstance(intent_result, dict) else False,
                    track_ids=track_ids
                )
            elif action == "add_tracks_to_playlist":
                if not playlist_id:
                    return self._create_error_result("playlist_id is required for add_tracks_to_playlist action")
                if not track_ids:
                    return self._create_error_result("track_ids is required for add_tracks_to_playlist action")
                
                results = await self._add_tracks_to_playlist(playlist_id, track_ids)
            elif action == "remove_tracks_from_playlist":
                if not playlist_id:
                    return self._create_error_result("playlist_id is required for remove_tracks_from_playlist action")
                if not track_ids:
                    return self._create_error_result("track_ids is required for remove_tracks_from_playlist action")
                
                results = await self._remove_tracks_from_playlist(playlist_id, track_ids)
            else:
                return self._create_error_result(f"Invalid action: {action}. Supported actions: list_playlists, get_playlist_tracks, search_user_playlists, create_playlist, add_tracks_to_playlist, remove_tracks_from_playlist")
            
            # TODO: Add database insertion for caching playlist data to avoid repeated Spotify API calls
            
            return self._create_success_result({
                "action": action,
                "results": results,
                "count": len(results) if isinstance(results, list) else 1
            })
            
        except Exception as e:
            return self._create_error_result(f"Spotify playlist operation failed: {str(e)}")
    
    async def _get_user_playlists(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's playlists."""
        playlists = []
        offset = 0
        
        while len(playlists) < limit:
            results = self.spotify_service.sp.current_user_playlists(
                limit=min(50, limit - len(playlists)), 
                offset=offset
            )
            
            if not results["items"]:
                break
                
            for playlist in results["items"]:
                playlists.append({
                    "id": playlist["id"],
                    "name": playlist["name"],
                    "description": playlist.get("description", ""),
                    "public": playlist["public"],
                    "collaborative": playlist["collaborative"],
                    "owner": playlist["owner"]["display_name"],
                    "owner_id": playlist["owner"]["id"],
                    "total_tracks": playlist["tracks"]["total"],
                    "cover_art": playlist["images"][0]["url"] if playlist["images"] else None,
                    "spotify_url": playlist["external_urls"]["spotify"]
                })
            
            offset += 50
            if not results["next"]:
                break
        
        return playlists[:limit]
    
    async def _get_playlist_tracks(self, playlist_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get tracks from a specific playlist."""
        tracks = []
        offset = 0
        
        while len(tracks) < limit:
            results = self.spotify_service.sp.playlist_tracks(
                playlist_id=playlist_id,
                limit=min(100, limit - len(tracks)),
                offset=offset
            )
            
            if not results["items"]:
                break
                
            for item in results["items"]:
                if item["track"] and item["track"]["type"] == "track":  # Skip episodes/podcasts
                    track = item["track"]
                    tracks.append({
                        "id": track["id"],
                        "name": track["name"],
                        "artist": track["artists"][0]["name"],
                        "artists": [artist["name"] for artist in track["artists"]],
                        "album": track["album"]["name"],
                        "album_id": track["album"]["id"],
                        "duration_ms": track["duration_ms"],
                        "popularity": track["popularity"],
                        "preview_url": track.get("preview_url"),
                        "cover_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                        "spotify_url": track["external_urls"]["spotify"],
                        "added_at": item["added_at"],
                        "added_by": item["added_by"]["id"] if item["added_by"] else None,
                        "explicit": track["explicit"]
                    })
            
            offset += 100
            if not results["next"]:
                break
        
        return tracks[:limit]
    
    async def _search_user_playlists(self, query: str) -> List[Dict[str, Any]]:
        """Search through user's playlists by name."""
        all_playlists = await self._get_user_playlists(limit=200)  # Get more playlists for search
        
        # Filter playlists that match the query (case-insensitive)
        matching_playlists = [
            playlist for playlist in all_playlists
            if query.lower() in playlist["name"].lower() or 
               query.lower() in playlist.get("description", "").lower()
        ]
        
        return matching_playlists
    
    async def _create_playlist(
        self, 
        name: str, 
        description: str = "", 
        public: bool = False,
        track_ids: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new playlist for the user."""
        try:
            # Get current user info
            user_info = self.spotify_service.sp.me()
            user_id = user_info["id"]
            
            # Create the playlist
            playlist = self.spotify_service.sp.user_playlist_create(
                user=user_id,
                name=name,
                public=public,
                description=description
            )
            
            playlist_data = {
                "id": playlist["id"],
                "name": playlist["name"],
                "description": playlist.get("description", ""),
                "public": playlist["public"],
                "collaborative": playlist["collaborative"],
                "owner": playlist["owner"]["display_name"],
                "owner_id": playlist["owner"]["id"],
                "total_tracks": 0,
                "cover_art": playlist["images"][0]["url"] if playlist["images"] else None,
                "spotify_url": playlist["external_urls"]["spotify"],
                "created": True
            }
            
            # Add tracks if provided
            if track_ids:
                # Convert track IDs to Spotify URIs
                track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
                
                # Add tracks to playlist
                self.spotify_service.sp.playlist_add_items(playlist["id"], track_uris)
                playlist_data["total_tracks"] = len(track_ids)
                playlist_data["tracks_added"] = len(track_ids)
            
            return playlist_data
            
        except Exception as e:
            raise Exception(f"Failed to create playlist: {str(e)}")
    
    async def _add_tracks_to_playlist(
        self, 
        playlist_id: str, 
        track_ids: List[str]
    ) -> Dict[str, Any]:
        """Add tracks to an existing playlist."""
        try:
            # Convert track IDs to Spotify URIs
            track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
            
            # Add tracks to playlist
            result = self.spotify_service.sp.playlist_add_items(playlist_id, track_uris)
            
            # Get updated playlist info
            playlist = self.spotify_service.sp.playlist(playlist_id)
            
            return {
                "playlist_id": playlist_id,
                "playlist_name": playlist["name"],
                "tracks_added": len(track_ids),
                "total_tracks": playlist["tracks"]["total"],
                "snapshot_id": result["snapshot_id"],
                "success": True
            }
            
        except Exception as e:
            raise Exception(f"Failed to add tracks to playlist: {str(e)}")
    
    async def _remove_tracks_from_playlist(
        self, 
        playlist_id: str, 
        track_ids: List[str]
    ) -> Dict[str, Any]:
        """Remove tracks from an existing playlist."""
        try:
            # Convert track IDs to Spotify URIs
            track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
            
            # Remove tracks from playlist
            result = self.spotify_service.sp.playlist_remove_all_occurrences_of_items(
                playlist_id, track_uris
            )
            
            # Get updated playlist info
            playlist = self.spotify_service.sp.playlist(playlist_id)
            
            return {
                "playlist_id": playlist_id,
                "playlist_name": playlist["name"],
                "tracks_removed": len(track_ids),
                "total_tracks": playlist["tracks"]["total"],
                "snapshot_id": result["snapshot_id"],
                "success": True
            }
            
        except Exception as e:
            raise Exception(f"Failed to remove tracks from playlist: {str(e)}")