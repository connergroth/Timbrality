from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np

from utils.cache import cache_manager
from utils.database_utils import get_database_connection


class MemoryStore:
    """
    Manages user memories, preferences, and interaction history.
    Provides both short-term (session) and long-term (persistent) storage.
    """
    
    def __init__(self):
        self.session_cache = {}  # In-memory session storage
        
    async def add_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """Add a new interaction to user's memory."""
        # Store in session cache
        if user_id not in self.session_cache:
            self.session_cache[user_id] = {"interactions": [], "preferences": {}}
        
        interaction_data["timestamp"] = datetime.now().isoformat()
        self.session_cache[user_id]["interactions"].append(interaction_data)
        
        # Store in persistent cache
        cache_key = f"user_interactions:{user_id}"
        cached_interactions = await cache_manager.get(cache_key)
        
        if cached_interactions:
            interactions = json.loads(cached_interactions)
        else:
            interactions = []
        
        interactions.append(interaction_data)
        
        # Keep only recent interactions (last 100)
        interactions = interactions[-100:]
        
        await cache_manager.set(
            cache_key, 
            json.dumps(interactions),
            expire=86400 * 7  # 1 week
        )
        
        # Store in database for long-term persistence
        await self._store_interaction_db(user_id, interaction_data)
    
    async def get_user_interactions(
        self, 
        user_id: str, 
        limit: int = 50,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get user's interaction history."""
        # Check session cache first
        if user_id in self.session_cache:
            session_interactions = self.session_cache[user_id]["interactions"]
            if len(session_interactions) >= limit:
                return session_interactions[-limit:]
        
        # Check Redis cache
        cache_key = f"user_interactions:{user_id}"
        cached_interactions = await cache_manager.get(cache_key)
        
        if cached_interactions:
            interactions = json.loads(cached_interactions)
            
            # Filter by date if specified
            if since:
                since_str = since.isoformat()
                interactions = [
                    i for i in interactions 
                    if i.get("timestamp", "") >= since_str
                ]
            
            return interactions[-limit:]
        
        # Fallback to database
        return await self._get_interactions_db(user_id, limit, since)
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile with new data."""
        # Update session cache
        if user_id not in self.session_cache:
            self.session_cache[user_id] = {"interactions": [], "preferences": {}}
        
        self.session_cache[user_id]["preferences"].update(profile_data)
        
        # Update Redis cache
        cache_key = f"user_profile:{user_id}"
        current_profile = await self.get_user_profile(user_id)
        current_profile.update(profile_data)
        current_profile["last_updated"] = datetime.now().isoformat()
        
        await cache_manager.set(
            cache_key,
            json.dumps(current_profile),
            expire=86400 * 30  # 30 days
        )
        
        # Update database
        await self._update_profile_db(user_id, profile_data)
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile."""
        # Check session cache
        if user_id in self.session_cache:
            session_prefs = self.session_cache[user_id]["preferences"]
            if session_prefs:
                return session_prefs
        
        # Check Redis cache
        cache_key = f"user_profile:{user_id}"
        cached_profile = await cache_manager.get(cache_key)
        
        if cached_profile:
            return json.loads(cached_profile)
        
        # Fallback to database
        db_profile = await self._get_profile_db(user_id)
        
        # Cache the result
        if db_profile:
            await cache_manager.set(
                cache_key,
                json.dumps(db_profile),
                expire=86400 * 30
            )
        
        return db_profile or {}
    
    async def update_user_embedding(self, user_id: str, embedding: List[float]):
        """Update user's preference embedding vector."""
        profile_update = {
            "preference_embedding": embedding,
            "embedding_updated": datetime.now().isoformat()
        }
        
        await self.update_user_profile(user_id, profile_update)
    
    async def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user's preference embedding vector."""
        profile = await self.get_user_profile(user_id)
        embedding = profile.get("preference_embedding")
        
        if embedding:
            return np.array(embedding)
        
        return None
    
    async def get_similar_users(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find users with similar preferences."""
        user_embedding = await self.get_user_embedding(user_id)
        if user_embedding is None:
            return []
        
        # This would query the database for other users' embeddings
        # and compute similarity scores
        similar_users = await self._find_similar_users_db(user_embedding, limit)
        
        return similar_users
    
    async def get_conversation_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get conversation context for the current session."""
        cache_key = f"conversation:{user_id}:{session_id}"
        cached_context = await cache_manager.get(cache_key)
        
        if cached_context:
            return json.loads(cached_context)
        
        # Create new context
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "current_mood": None,
            "context_tracks": []
        }
        
        return context
    
    async def update_conversation_context(
        self, 
        user_id: str, 
        session_id: str, 
        context_update: Dict[str, Any]
    ):
        """Update conversation context."""
        context = await self.get_conversation_context(user_id, session_id)
        context.update(context_update)
        context["last_updated"] = datetime.now().isoformat()
        
        cache_key = f"conversation:{user_id}:{session_id}"
        await cache_manager.set(
            cache_key,
            json.dumps(context),
            expire=3600  # 1 hour session timeout
        )
    
    async def add_feedback(
        self, 
        user_id: str, 
        track_id: str, 
        feedback_type: str, 
        feedback_data: Dict[str, Any]
    ):
        """Add user feedback for a track."""
        feedback = {
            "user_id": user_id,
            "track_id": track_id,
            "feedback_type": feedback_type,
            "feedback_data": feedback_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in cache for quick access
        cache_key = f"user_feedback:{user_id}:{track_id}"
        await cache_manager.set(
            cache_key,
            json.dumps(feedback),
            expire=86400 * 30  # 30 days
        )
        
        # Store in database
        await self._store_feedback_db(feedback)
    
    async def get_track_feedback(self, user_id: str, track_id: str) -> Optional[Dict[str, Any]]:
        """Get user's feedback for a specific track."""
        cache_key = f"user_feedback:{user_id}:{track_id}"
        cached_feedback = await cache_manager.get(cache_key)
        
        if cached_feedback:
            return json.loads(cached_feedback)
        
        # Fallback to database
        return await self._get_feedback_db(user_id, track_id)
    
    async def get_user_listening_history(
        self, 
        user_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get user's listening history with play counts and timestamps."""
        cache_key = f"listening_history:{user_id}"
        cached_history = await cache_manager.get(cache_key)
        
        if cached_history:
            history = json.loads(cached_history)
            return history[-limit:]
        
        # Fallback to database
        history = await self._get_listening_history_db(user_id, limit)
        
        # Cache the result
        if history:
            await cache_manager.set(
                cache_key,
                json.dumps(history),
                expire=3600  # 1 hour
            )
        
        return history
    
    async def add_listening_event(
        self, 
        user_id: str, 
        track_data: Dict[str, Any],
        event_type: str = "play"
    ):
        """Add a listening event to user's history."""
        event = {
            "track_id": track_data.get("id"),
            "track_name": track_data.get("name"),
            "artist": track_data.get("artist"),
            "event_type": event_type,  # play, skip, like, etc.
            "timestamp": datetime.now().isoformat(),
            "duration_played": track_data.get("duration_played"),
            "source": track_data.get("source", "timbre_agent")
        }
        
        # Add to recent events cache
        cache_key = f"recent_events:{user_id}"
        cached_events = await cache_manager.get(cache_key)
        
        if cached_events:
            events = json.loads(cached_events)
        else:
            events = []
        
        events.append(event)
        events = events[-50:]  # Keep last 50 events
        
        await cache_manager.set(
            cache_key,
            json.dumps(events),
            expire=86400  # 24 hours
        )
        
        # Store in database
        await self._store_listening_event_db(user_id, event)
    
    async def clear_user_session(self, user_id: str, session_id: str):
        """Clear user's session data."""
        if user_id in self.session_cache:
            del self.session_cache[user_id]
        
        # Clear conversation context
        cache_key = f"conversation:{user_id}:{session_id}"
        await cache_manager.delete(cache_key)
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics and analytics."""
        stats = {
            "total_interactions": 0,
            "total_tracks_recommended": 0,
            "feedback_given": 0,
            "listening_time": 0,
            "favorite_genres": [],
            "discovery_rate": 0.0,
            "last_active": None
        }
        
        # Calculate from interactions
        interactions = await self.get_user_interactions(user_id, limit=1000)
        stats["total_interactions"] = len(interactions)
        
        total_tracks = sum(
            len(i.get("tracks", [])) for i in interactions
        )
        stats["total_tracks_recommended"] = total_tracks
        
        # Calculate from listening history
        history = await self.get_user_listening_history(user_id, limit=1000)
        stats["listening_time"] = sum(
            h.get("duration_played", 0) for h in history
        )
        
        if interactions:
            stats["last_active"] = interactions[-1].get("timestamp")
        
        return stats
    
    # Database methods (to be implemented based on actual DB schema)
    
    async def _store_interaction_db(self, user_id: str, interaction_data: Dict[str, Any]):
        """Store interaction in database."""
        # Implementation would depend on database schema
        pass
    
    async def _get_interactions_db(
        self, 
        user_id: str, 
        limit: int, 
        since: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Get interactions from database."""
        # Implementation would depend on database schema
        return []
    
    async def _update_profile_db(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile in database."""
        # Implementation would depend on database schema
        pass
    
    async def _get_profile_db(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from database."""
        # Implementation would depend on database schema
        return {}
    
    async def _find_similar_users_db(
        self, 
        user_embedding: np.ndarray, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar users in database."""
        # Implementation would use vector similarity search
        return []
    
    async def _store_feedback_db(self, feedback: Dict[str, Any]):
        """Store feedback in database."""
        # Implementation would depend on database schema
        pass
    
    async def _get_feedback_db(self, user_id: str, track_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback from database."""
        # Implementation would depend on database schema
        return None
    
    async def _get_listening_history_db(
        self, 
        user_id: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get listening history from database."""
        # Implementation would depend on database schema
        return []
    
    async def _store_listening_event_db(self, user_id: str, event: Dict[str, Any]):
        """Store listening event in database."""
        # Implementation would depend on database schema
        pass